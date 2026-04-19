from __future__ import annotations

import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "step02_input"
OUTPUT_DIR = BASE_DIR / "step03_outputs"

SEED = 42
LATENT_DIM = 8
HIDDEN_DIM_1 = 320
HIDDEN_DIM_2 = 96
DROPOUT = 0.15
INPUT_NOISE = 0.05
LEARNING_RATE = 2.5e-4
WEIGHT_DECAY = 2e-5
BATCH_SIZE = 32
MAX_EPOCHS = 40
MIN_EPOCHS = 25
PATIENCE = 14
TOTAL_CAP = 200.0
NEIGHBOR_K = 15
PCA_COMPONENTS = 8
PCA_OVERSAMPLE = 16
PREDICTION_SCATTER_MAX_POINTS = 50000
ALL_NLL_GUARDRAIL_MARGIN = 0.01
BACKGROUND_RATIO_THRESHOLD = 0.05
BACKGROUND_PRED_TARGET = 0.05
BACKGROUND_LAMBDA = 0.20

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_input(stem: str) -> Path:
    csv_path = INPUT_DIR / f"{stem}.csv"
    if csv_path.exists():
        return csv_path
    legacy_gz = INPUT_DIR / f"{stem}.csv.gz"
    if legacy_gz.exists():
        return legacy_gz
    raise FileNotFoundError(f"Missing Step2 input: {csv_path}")


def load_matrix(stem: str, dtype=np.float32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = resolve_input(stem)
    print(f"loading {path.name}", flush=True)
    df = pd.read_csv(path)
    row_ids = df.iloc[:, 0].astype(str).to_numpy()
    col_ids = df.columns[1:].astype(str).to_numpy()
    matrix = df.iloc[:, 1:].to_numpy(dtype=dtype, copy=True)
    del df
    print(f"loaded {path.name}: {matrix.shape}", flush=True)
    return row_ids, col_ids, matrix


def to_device(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=DEVICE)


def capped_total(total_t: torch.Tensor) -> torch.Tensor:
    return torch.clamp(total_t, max=TOTAL_CAP)


def make_input_features(
    ratio_t: torch.Tensor,
    mask_t: torch.Tensor,
    total_t: torch.Tensor,
    noise_rate: float = 0.0,
) -> torch.Tensor:
    if noise_rate > 0:
        dropped = (torch.rand_like(ratio_t) < noise_rate) & (mask_t > 0)
        ratio_t = ratio_t.masked_fill(dropped, 0.0)
        mask_t = mask_t.masked_fill(dropped, 0.0)
        total_t = total_t.masked_fill(dropped, 0.0)
    log_total_t = torch.log1p(capped_total(total_t)) / math.log1p(TOTAL_CAP)
    return torch.cat([ratio_t * mask_t, mask_t, log_total_t * mask_t], dim=1)


class BinomialAutoencoder(nn.Module):
    def __init__(self, n_sites: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_sites * 3, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM_2, LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM_1, n_sites),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        pred = self.decoder(latent).clamp(1e-6, 1.0 - 1e-6)
        return pred, latent


def masked_binomial_nll_weighted(
    pred_t: torch.Tensor,
    m6a_t: torch.Tensor,
    total_t: torch.Tensor,
    mask_t: torch.Tensor,
    weight_t: torch.Tensor,
) -> torch.Tensor:
    eff_total_t = capped_total(total_t)
    cap_weight_t = eff_total_t / total_t.clamp_min(1.0)
    nll_t = -(m6a_t * torch.log(pred_t) + (total_t - m6a_t) * torch.log1p(-pred_t))
    weighted_mask_t = mask_t * weight_t
    denom_t = (eff_total_t * weighted_mask_t).sum().clamp_min(1.0)
    return (nll_t * cap_weight_t * weighted_mask_t).sum() / denom_t


def evaluate_prediction_vector(
    pred_vector_t: torch.Tensor,
    eval_mask_t: torch.Tensor,
    ratio_t: torch.Tensor,
    m6a_t: torch.Tensor,
    total_t: torch.Tensor,
) -> tuple[float, float]:
    pred_t = pred_vector_t.reshape(1, -1).clamp(1e-6, 1.0 - 1e-6)
    eff_total_t = capped_total(total_t)
    cap_weight_t = eff_total_t / total_t.clamp_min(1.0)
    nll_t = -(m6a_t * torch.log(pred_t) + (total_t - m6a_t) * torch.log1p(-pred_t))
    denom_t = (eff_total_t * eval_mask_t).sum().clamp_min(1.0)
    nll = float(((nll_t * cap_weight_t * eval_mask_t).sum() / denom_t).detach().cpu())
    mae = float((((pred_t - ratio_t).abs() * eval_mask_t).sum() / eval_mask_t.sum().clamp_min(1.0)).detach().cpu())
    return nll, mae


def evaluate_model(
    model: BinomialAutoencoder,
    input_mask_t: torch.Tensor,
    eval_mask_t: torch.Tensor,
    ratio_t: torch.Tensor,
    m6a_t: torch.Tensor,
    total_t: torch.Tensor,
    batch_size: int = BATCH_SIZE,
) -> tuple[float, float]:
    model.eval()
    nll_num = 0.0
    nll_den = 0.0
    mae_num = 0.0
    mae_den = 0.0
    with torch.inference_mode():
        for start in range(0, ratio_t.shape[0], batch_size):
            sl = slice(start, min(start + batch_size, ratio_t.shape[0]))
            pred_t, _ = model(make_input_features(ratio_t[sl], input_mask_t[sl], total_t[sl], 0.0))
            eff_total_t = capped_total(total_t[sl])
            cap_weight_t = eff_total_t / total_t[sl].clamp_min(1.0)
            nll_t = -(m6a_t[sl] * torch.log(pred_t) + (total_t[sl] - m6a_t[sl]) * torch.log1p(-pred_t))
            nll_num += float((nll_t * cap_weight_t * eval_mask_t[sl]).sum().detach().cpu())
            nll_den += float((eff_total_t * eval_mask_t[sl]).sum().detach().cpu())
            mae_num += float(((pred_t - ratio_t[sl]).abs() * eval_mask_t[sl]).sum().detach().cpu())
            mae_den += float(eval_mask_t[sl].sum().detach().cpu())
    return nll_num / max(nll_den, 1.0), mae_num / max(mae_den, 1.0)


def background_overprediction_penalty(
    pred_t: torch.Tensor,
    ratio_t: torch.Tensor,
    total_t: torch.Tensor,
    train_t: torch.Tensor,
    significance_t: torch.Tensor,
) -> torch.Tensor:
    background_t = train_t * (1.0 - significance_t) * (ratio_t <= BACKGROUND_RATIO_THRESHOLD).float()
    if int(background_t.sum().detach().cpu()) == 0:
        return pred_t.new_tensor(0.0)
    eff_total_t = capped_total(total_t)
    over_pred_t = torch.relu(pred_t - BACKGROUND_PRED_TARGET)
    denom_t = (eff_total_t * background_t).sum().clamp_min(1.0)
    return ((over_pred_t * over_pred_t) * eff_total_t * background_t).sum() / denom_t


def site_mean_prediction(ratio_t: torch.Tensor, train_t: torch.Tensor) -> torch.Tensor:
    observed_sum = (ratio_t * train_t).sum(dim=0)
    observed_n = train_t.sum(dim=0).clamp_min(1.0)
    return (observed_sum / observed_n).clamp(1e-6, 1.0 - 1e-6)


def fit_randomized_pca(
    ratio_np: np.ndarray,
    train_np: np.ndarray,
    n_components: int = PCA_COMPONENTS,
    oversample: int = PCA_OVERSAMPLE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    observed_sum = (ratio_np * train_np).sum(axis=0)
    observed_n = np.maximum(train_np.sum(axis=0), 1.0)
    site_mean_np = np.clip(observed_sum / observed_n, 1e-6, 1.0 - 1e-6).astype(np.float32)
    filled_np = ratio_np * train_np + site_mean_np.reshape(1, -1) * (1.0 - train_np)
    centered_np = filled_np - site_mean_np.reshape(1, -1)
    rank = min(n_components + oversample, centered_np.shape[0] - 1, centered_np.shape[1])
    rng = np.random.default_rng(SEED)
    omega_np = rng.standard_normal(size=(centered_np.shape[1], rank), dtype=np.float32)
    y_np = centered_np @ omega_np
    q_np, _ = np.linalg.qr(y_np, mode="reduced")
    b_np = q_np.T @ centered_np
    _, _, vh_np = np.linalg.svd(b_np, full_matrices=False)
    components_np = np.ascontiguousarray(vh_np[:n_components].astype(np.float32))
    scores_np = (centered_np @ components_np.T).astype(np.float32)
    return scores_np, site_mean_np, components_np, f"randomized_pca_rank={rank}_components={n_components}"


def evaluate_pca_reconstruction(
    site_mean_np: np.ndarray,
    components_np: np.ndarray,
    train_np: np.ndarray,
    eval_mask_np: np.ndarray,
    ratio_np: np.ndarray,
    m6a_np: np.ndarray,
    total_np: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> tuple[float, float]:
    mean_row = site_mean_np.reshape(1, -1)
    nll_num = 0.0
    nll_den = 0.0
    mae_num = 0.0
    mae_den = 0.0
    for start in range(0, ratio_np.shape[0], batch_size):
        sl = slice(start, min(start + batch_size, ratio_np.shape[0]))
        filled = ratio_np[sl] * train_np[sl] + mean_row * (1.0 - train_np[sl])
        centered = filled - mean_row
        pred = np.clip(centered @ components_np.T @ components_np + mean_row, 1e-6, 1.0 - 1e-6)
        eff_total = np.minimum(total_np[sl], TOTAL_CAP)
        cap_weight = eff_total / np.maximum(total_np[sl], 1.0)
        nll = -(m6a_np[sl] * np.log(pred) + (total_np[sl] - m6a_np[sl]) * np.log1p(-pred)) * cap_weight
        mask = eval_mask_np[sl]
        nll_num += float((nll * mask).sum())
        nll_den += float((eff_total * mask).sum())
        mae_num += float((np.abs(pred - ratio_np[sl]) * mask).sum())
        mae_den += float(mask.sum())
    return nll_num / max(nll_den, 1.0), mae_num / max(mae_den, 1.0)


def pca_scores(matrix: np.ndarray, n_components: int) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return (x @ vt[:n_components].T).astype(np.float32)


def kmeans(matrix: np.ndarray, n_clusters: int, seed: int, n_init: int = 10, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(matrix, dtype=np.float32)
    best = None
    best_inertia = math.inf
    for _ in range(n_init):
        centers = x[rng.choice(x.shape[0], size=n_clusters, replace=False)].copy()
        labels = np.zeros(x.shape[0], dtype=np.int64)
        for _ in range(max_iter):
            distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for cluster_idx in range(n_clusters):
                points = x[labels == cluster_idx]
                centers[cluster_idx] = x[rng.integers(0, x.shape[0])] if len(points) == 0 else points.mean(axis=0)
        inertia = float(((x - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best = (labels.copy(), centers.copy(), inertia)
            best_inertia = inertia
    return best


def calinski_harabasz(matrix: np.ndarray, labels: np.ndarray, centers: np.ndarray, inertia: float) -> float:
    n_samples = matrix.shape[0]
    n_clusters = len(np.unique(labels))
    if n_clusters <= 1 or n_clusters >= n_samples:
        return float("-inf")
    overall_mean = matrix.mean(axis=0)
    between = 0.0
    for cluster_idx in range(n_clusters):
        points = matrix[labels == cluster_idx]
        if len(points) > 0:
            between += len(points) * float(((centers[cluster_idx] - overall_mean) ** 2).sum())
    return (between / (n_clusters - 1)) / (max(float(inertia), 1e-8) / (n_samples - n_clusters))


def choose_k(matrix: np.ndarray, seed: int) -> tuple[int, pd.DataFrame]:
    rows = []
    best_k = 2
    best_score = float("-inf")
    for k in (2, 3, 4, 5, 6):
        labels, centers, inertia = kmeans(matrix, k, seed + k)
        score = calinski_harabasz(matrix, labels, centers, inertia)
        rows.append({"k": k, "calinski_harabasz": score, "inertia": inertia})
        if score > best_score:
            best_k = k
            best_score = score
    return best_k, pd.DataFrame(rows)


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    contingency = pd.crosstab(labels_true, labels_pred).to_numpy(dtype=np.int64)

    def comb2(x: np.ndarray) -> float:
        x = x.astype(np.float64)
        return float((x * (x - 1) / 2).sum())

    sum_comb_cells = comb2(contingency.ravel())
    sum_comb_rows = comb2(contingency.sum(axis=1))
    sum_comb_cols = comb2(contingency.sum(axis=0))
    total_n = contingency.sum()
    total_comb = total_n * (total_n - 1) / 2
    if total_comb == 0:
        return 0.0
    expected = sum_comb_rows * sum_comb_cols / total_comb
    max_index = 0.5 * (sum_comb_rows + sum_comb_cols)
    denom = max_index - expected
    return 0.0 if denom == 0 else float((sum_comb_cells - expected) / denom)


def neighbor_overlap(reference: np.ndarray, query: np.ndarray, k: int) -> float:
    ref_dist = ((reference[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
    qry_dist = ((query[:, None, :] - query[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(ref_dist, np.inf)
    np.fill_diagonal(qry_dist, np.inf)
    ref_nn = np.argpartition(ref_dist, kth=np.arange(k), axis=1)[:, :k]
    qry_nn = np.argpartition(qry_dist, kth=np.arange(k), axis=1)[:, :k]
    overlaps = [len(set(ref_nn[i].tolist()) & set(qry_nn[i].tolist())) / k for i in range(reference.shape[0])]
    return float(np.mean(overlaps))


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float32)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float32)
    x0 = xr - xr.mean()
    y0 = yr - yr.mean()
    denom = float(np.sqrt((x0 * x0).sum()) * np.sqrt((y0 * y0).sum()))
    return 0.0 if denom == 0 else float((x0 * y0).sum() / denom)


def extract_latent(
    model: BinomialAutoencoder,
    ratio_t: torch.Tensor,
    coverage_t: torch.Tensor,
    total_t: torch.Tensor,
) -> np.ndarray:
    model.eval()
    zs = []
    with torch.inference_mode():
        for start in range(0, ratio_t.shape[0], BATCH_SIZE):
            sl = slice(start, min(start + BATCH_SIZE, ratio_t.shape[0]))
            _, z = model(make_input_features(ratio_t[sl], coverage_t[sl], total_t[sl], 0.0))
            zs.append(z.detach().cpu().numpy())
    z_np = np.vstack(zs).astype(np.float32)
    return (z_np - z_np.mean(axis=0, keepdims=True)) / np.maximum(z_np.std(axis=0, keepdims=True), 1e-6)


def sample_significant_predictions(
    model: BinomialAutoencoder,
    ratio_t: torch.Tensor,
    total_t: torch.Tensor,
    train_t: torch.Tensor,
    sig_heldout_t: torch.Tensor,
) -> pd.DataFrame:
    model.eval()
    obs_all = []
    pred_all = []
    with torch.inference_mode():
        for start in range(0, ratio_t.shape[0], BATCH_SIZE):
            sl = slice(start, min(start + BATCH_SIZE, ratio_t.shape[0]))
            pred, _ = model(make_input_features(ratio_t[sl], train_t[sl], total_t[sl], 0.0))
            mask = sig_heldout_t[sl] > 0
            if int(mask.sum().detach().cpu()) > 0:
                obs_all.append(ratio_t[sl][mask].detach().cpu().numpy())
                pred_all.append(pred[mask].detach().cpu().numpy())
    if not obs_all:
        return pd.DataFrame({"observed_ratio": [], "predicted_ratio": []})
    obs = np.concatenate(obs_all)
    pred = np.concatenate(pred_all)
    if len(obs) > PREDICTION_SCATTER_MAX_POINTS:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(obs), size=PREDICTION_SCATTER_MAX_POINTS, replace=False)
        obs = obs[idx]
        pred = pred[idx]
    return pd.DataFrame({"observed_ratio": obs, "predicted_ratio": pred})


def prediction_calibration(scatter_df: pd.DataFrame) -> pd.DataFrame:
    if scatter_df.empty:
        return pd.DataFrame(columns=["observed_bin", "n", "observed_mean", "predicted_mean"])
    bins = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.000001]
    labels = ["0-0.05", "0.05-0.10", "0.10-0.25", "0.25-0.50", "0.50-0.75", "0.75-0.90", "0.90-1.00"]
    df = scatter_df.copy()
    df["observed_bin"] = pd.cut(df["observed_ratio"], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for bin_name, group in df.groupby("observed_bin", observed=False):
        if not group.empty:
            rows.append(
                {
                    "observed_bin": str(bin_name),
                    "n": int(len(group)),
                    "observed_mean": float(group["observed_ratio"].mean()),
                    "predicted_mean": float(group["predicted_ratio"].mean()),
                    "predicted_median": float(group["predicted_ratio"].median()),
                    "predicted_p75": float(group["predicted_ratio"].quantile(0.75)),
                    "predicted_p90": float(group["predicted_ratio"].quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def train_autoencoder(tensors: dict[str, torch.Tensor], site_all_nll: float) -> tuple[BinomialAutoencoder, pd.DataFrame, int]:
    ratio_t = tensors["ratio"]
    m6a_t = tensors["m6a"]
    total_t = tensors["total"]
    train_t = tensors["train"]
    heldout_t = tensors["heldout"]
    sig_heldout_t = tensors["sig_heldout"]
    significance_t = tensors["significance"]
    weight_t = tensors["entry_weight"]

    model = BinomialAutoencoder(ratio_t.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(TensorDataset(torch.arange(ratio_t.shape[0], dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=True)
    rows = []
    best = {"score": math.inf, "epoch": 0, "state": None}
    stale = 0
    guardrail_target = site_all_nll + ALL_NLL_GUARDRAIL_MARGIN

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_losses = []
        for (idx_cpu,) in loader:
            idx = idx_cpu.to(DEVICE)
            br = ratio_t.index_select(0, idx)
            bm6a = m6a_t.index_select(0, idx)
            bt = total_t.index_select(0, idx)
            btrain = train_t.index_select(0, idx)
            bsig = significance_t.index_select(0, idx)
            bw = weight_t.index_select(0, idx)
            pred, _ = model(make_input_features(br, btrain, bt, INPUT_NOISE))
            main_loss = masked_binomial_nll_weighted(pred, bm6a, bt, btrain, bw)
            bg_penalty = background_overprediction_penalty(pred, br, bt, btrain, bsig)
            loss = main_loss + BACKGROUND_LAMBDA * bg_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        heldout_nll, heldout_mae = evaluate_model(model, train_t, heldout_t, ratio_t, m6a_t, total_t)
        sig_nll, sig_mae = evaluate_model(model, train_t, sig_heldout_t, ratio_t, m6a_t, total_t)
        overrun = max(0.0, heldout_nll - guardrail_target)
        selection_score = 0.65 * heldout_nll + 0.35 * sig_nll + 4.0 * overrun
        eligible = epoch >= MIN_EPOCHS
        row = {
            "epoch": epoch,
            "train_weighted_binomial_nll": float(np.mean(batch_losses)),
            "heldout_nll": heldout_nll,
            "heldout_mae": heldout_mae,
            "significant_heldout_nll": sig_nll,
            "significant_heldout_mae": sig_mae,
            "all_nll_guardrail_target": guardrail_target,
            "all_nll_overrun": overrun,
            "selection_score": selection_score,
            "eligible_for_selection": eligible,
        }
        rows.append(row)
        print(row, flush=True)
        if eligible and selection_score < best["score"]:
            best = {
                "score": selection_score,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            stale = 0
        elif eligible:
            stale += 1
            if stale >= PATIENCE:
                break

    if best["state"] is None:
        best = {
            "score": rows[-1]["selection_score"],
            "epoch": rows[-1]["epoch"],
            "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        }
    model.load_state_dict(best["state"])
    model.to(DEVICE)
    return model, pd.DataFrame(rows), int(best["epoch"])


def main() -> None:
    start_time = time.time()
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print({"device": str(DEVICE), "input_dir": str(INPUT_DIR), "output_dir": str(OUTPUT_DIR)}, flush=True)

    ratio_cell_ids, site_ids, ratio_np = load_matrix("step02_ratio_cells_by_sites")
    m6a_cell_ids, _, m6a_np = load_matrix("step02_m6a_cells_by_sites")
    total_cell_ids, _, total_np = load_matrix("step02_total_cells_by_sites")
    coverage_cell_ids, _, coverage_np = load_matrix("step02_coverage_mask_cells_by_sites")
    train_cell_ids, _, train_np = load_matrix("step02_train_mask_cells_by_sites")
    heldout_cell_ids, _, heldout_np = load_matrix("step02_heldout_mask_cells_by_sites")
    sig_cell_ids, _, sig_np = load_matrix("step02_significance_mask_cells_by_sites")
    expr_cell_ids, _, expression_np = load_matrix("step02_expression_anchor_cells_by_genes")
    cell_metadata = pd.read_csv(resolve_input("step02_cell_metadata"))
    regulator_df = pd.read_csv(resolve_input("step02_regulator_expression_by_cell"))
    site_metadata = pd.read_csv(resolve_input("step02_site_metadata"))

    for name, ids in [
        ("m6a", m6a_cell_ids),
        ("total", total_cell_ids),
        ("coverage", coverage_cell_ids),
        ("train", train_cell_ids),
        ("heldout", heldout_cell_ids),
        ("significance", sig_cell_ids),
        ("expression", expr_cell_ids),
    ]:
        if not np.array_equal(ratio_cell_ids, ids):
            raise ValueError(f"Cell order mismatch: ratio vs {name}")

    sig_heldout_np = heldout_np * sig_np
    input_summary = {
        "cells": int(ratio_np.shape[0]),
        "sites": int(ratio_np.shape[1]),
        "heldout_entries": int(heldout_np.sum()),
        "significant_heldout_entries": int(sig_heldout_np.sum()),
        "train_observed_entries": int(train_np.sum()),
        "site_metadata_rows": int(len(site_metadata)),
        "cell_metadata_rows": int(len(cell_metadata)),
    }
    (OUTPUT_DIR / "step03_input_summary.json").write_text(json.dumps(input_summary, indent=2), encoding="utf-8")
    print(input_summary, flush=True)

    ratio_t = to_device(ratio_np)
    m6a_t = to_device(m6a_np)
    total_t = to_device(total_np)
    coverage_t = to_device(coverage_np)
    train_t = to_device(train_np)
    heldout_t = to_device(heldout_np)
    sig_heldout_t = to_device(sig_heldout_np)
    coverage_conf_np = np.log1p(np.minimum(total_np, TOTAL_CAP)) / math.log1p(TOTAL_CAP)
    entry_weight_np = (1.0 + 0.50 * coverage_conf_np + 0.25 * sig_np).astype(np.float32)
    entry_weight_t = to_device(entry_weight_np)

    site_mean_t = site_mean_prediction(ratio_t, train_t)
    site_all_nll, site_all_mae = evaluate_prediction_vector(site_mean_t, heldout_t, ratio_t, m6a_t, total_t)
    site_sig_nll, site_sig_mae = evaluate_prediction_vector(site_mean_t, sig_heldout_t, ratio_t, m6a_t, total_t)
    print({"site_mean_all_nll": site_all_nll, "site_mean_significant_nll": site_sig_nll}, flush=True)

    print("fitting randomized m6A PCA baseline", flush=True)
    m6a_pca_scores, pca_site_mean, pca_components, pca_note = fit_randomized_pca(ratio_np, train_np)
    pca_all_nll, pca_all_mae = evaluate_pca_reconstruction(pca_site_mean, pca_components, train_np, heldout_np, ratio_np, m6a_np, total_np)
    pca_sig_nll, pca_sig_mae = evaluate_pca_reconstruction(pca_site_mean, pca_components, train_np, sig_heldout_np, ratio_np, m6a_np, total_np)
    print({"pca_note": pca_note, "m6a_pca_all_nll": pca_all_nll, "m6a_pca_significant_nll": pca_sig_nll}, flush=True)

    expression_pcs = pca_scores(expression_np.astype(np.float32), min(10, expression_np.shape[1], expression_np.shape[0] - 1))
    cluster_k, cluster_selection_df = choose_k(expression_pcs, SEED)
    expr_cluster, _, _ = kmeans(expression_pcs, cluster_k, SEED + 100)
    m6a_cluster, _, _ = kmeans(m6a_pca_scores, cluster_k, SEED + 200)
    expr_vs_m6a_pca_neighbor_overlap = neighbor_overlap(expression_pcs, m6a_pca_scores, NEIGHBOR_K)
    cluster_selection_df.to_csv(OUTPUT_DIR / "step03_cluster_selection_metrics.csv", index=False)

    tensors = {
        "ratio": ratio_t,
        "m6a": m6a_t,
        "total": total_t,
        "coverage": coverage_t,
        "train": train_t,
        "heldout": heldout_t,
        "sig_heldout": sig_heldout_t,
        "significance": to_device(sig_np),
        "entry_weight": entry_weight_t,
    }
    model, history_df, best_epoch = train_autoencoder(tensors, site_all_nll)
    history_df.to_csv(OUTPUT_DIR / "step03_training_history.csv", index=False)

    ae_all_nll, ae_all_mae = evaluate_model(model, train_t, heldout_t, ratio_t, m6a_t, total_t)
    ae_sig_nll, ae_sig_mae = evaluate_model(model, train_t, sig_heldout_t, ratio_t, m6a_t, total_t)
    method_metrics = pd.DataFrame(
        [
            {"method": "site_mean", "heldout_nll": site_all_nll, "heldout_mae": site_all_mae, "significant_heldout_nll": site_sig_nll, "significant_heldout_mae": site_sig_mae},
            {"method": "m6a_pca", "heldout_nll": pca_all_nll, "heldout_mae": pca_all_mae, "significant_heldout_nll": pca_sig_nll, "significant_heldout_mae": pca_sig_mae},
            {"method": "cell_state_ae", "heldout_nll": ae_all_nll, "heldout_mae": ae_all_mae, "significant_heldout_nll": ae_sig_nll, "significant_heldout_mae": ae_sig_mae},
        ]
    )
    method_metrics.to_csv(OUTPUT_DIR / "step03_method_metrics.csv", index=False)

    ae_latent = extract_latent(model, ratio_t, coverage_t, total_t)
    ae_2d = pca_scores(ae_latent, 2)
    m6a_pca_2d = pca_scores(m6a_pca_scores, 2) if m6a_pca_scores.shape[1] > 2 else m6a_pca_scores[:, :2]
    ae_cluster, _, _ = kmeans(ae_latent, cluster_k, SEED + 300)
    expr_vs_ae_ari = adjusted_rand_index(expr_cluster, ae_cluster)
    expr_vs_m6a_pca_ari = adjusted_rand_index(expr_cluster, m6a_cluster)
    expr_vs_ae_neighbor_overlap = neighbor_overlap(expression_pcs, ae_latent, NEIGHBOR_K)
    cluster_metrics = pd.DataFrame(
        [
            {"comparison": "expression_vs_ae", "ari": expr_vs_ae_ari, "neighbor_overlap": expr_vs_ae_neighbor_overlap},
            {"comparison": "expression_vs_m6a_pca", "ari": expr_vs_m6a_pca_ari, "neighbor_overlap": expr_vs_m6a_pca_neighbor_overlap},
        ]
    )
    cluster_metrics.to_csv(OUTPUT_DIR / "step03_cluster_comparison_metrics.csv", index=False)

    reg_rows = []
    for regulator in regulator_df.columns[1:]:
        values = regulator_df[regulator].to_numpy(dtype=np.float32)
        for dim_idx in range(ae_latent.shape[1]):
            corr = rank_corr(ae_latent[:, dim_idx], values)
            reg_rows.append({"pair": f"{regulator} vs latent_{dim_idx + 1}", "regulator": regulator, "latent_dim": dim_idx + 1, "spearman_like_corr": corr, "abs_corr": abs(corr)})
    regulator_corr_df = pd.DataFrame(reg_rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)
    regulator_corr_df.to_csv(OUTPUT_DIR / "step03_regulator_correlations.csv", index=False)

    cell_embeddings = cell_metadata.copy()
    cell_embeddings["expression_pc1"] = expression_pcs[:, 0]
    cell_embeddings["expression_pc2"] = expression_pcs[:, 1]
    cell_embeddings["expression_cluster"] = expr_cluster.astype(str)
    cell_embeddings["m6a_pca1"] = m6a_pca_2d[:, 0]
    cell_embeddings["m6a_pca2"] = m6a_pca_2d[:, 1]
    cell_embeddings["m6a_pca_cluster"] = m6a_cluster.astype(str)
    cell_embeddings["ae_pc1"] = ae_2d[:, 0]
    cell_embeddings["ae_pc2"] = ae_2d[:, 1]
    cell_embeddings["ae_cluster"] = ae_cluster.astype(str)
    for dim_idx in range(ae_latent.shape[1]):
        cell_embeddings[f"latent_{dim_idx + 1}"] = ae_latent[:, dim_idx]
    cell_embeddings.to_csv(OUTPUT_DIR / "step03_cell_embeddings.csv", index=False)

    scatter_df = sample_significant_predictions(model, ratio_t, total_t, train_t, sig_heldout_t)
    scatter_df.to_csv(OUTPUT_DIR / "step03_significant_heldout_prediction_sample.csv", index=False)
    prediction_calibration(scatter_df).to_csv(OUTPUT_DIR / "step03_prediction_calibration_by_bin.csv", index=False)

    np.savez_compressed(
        OUTPUT_DIR / "step03_embeddings.npz",
        ae_latent=ae_latent,
        m6a_pca_scores=m6a_pca_scores,
        expression_pcs=expression_pcs,
        cell_ids=ratio_cell_ids.astype(str),
        site_ids=site_ids.astype(str),
    )
    torch.save(
        {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "model_dims": {"n_sites": int(ratio_t.shape[1]), "latent_dim": LATENT_DIM, "hidden_dim_1": HIDDEN_DIM_1, "hidden_dim_2": HIDDEN_DIM_2},
        },
        OUTPUT_DIR / "step03_final_model.pt",
    )

    ae_counts = pd.Series(ae_cluster).value_counts().sort_index()
    score = {
        "run_id": "fypmain3_cell_state_ae",
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "device": str(DEVICE),
        "cells": int(ratio_t.shape[0]),
        "sites": int(ratio_t.shape[1]),
        "heldout_entries": int(heldout_np.sum()),
        "significant_heldout_entries": int(sig_heldout_np.sum()),
        "best_epoch": int(best_epoch),
        "selected_cluster_k": int(cluster_k),
        "site_mean_significant_nll": float(site_sig_nll),
        "m6a_pca_significant_nll": float(pca_sig_nll),
        "cell_state_ae_significant_nll": float(ae_sig_nll),
        "significant_delta_nll_vs_site_mean": float(site_sig_nll - ae_sig_nll),
        "site_mean_all_nll": float(site_all_nll),
        "m6a_pca_all_nll": float(pca_all_nll),
        "cell_state_ae_all_nll": float(ae_all_nll),
        "all_delta_nll_vs_site_mean": float(site_all_nll - ae_all_nll),
        "cell_state_ae_significant_mae": float(ae_sig_mae),
        "cell_state_ae_all_mae": float(ae_all_mae),
        "expression_vs_ae_ari": float(expr_vs_ae_ari),
        "expression_vs_m6a_pca_ari": float(expr_vs_m6a_pca_ari),
        "expression_vs_ae_neighbor_overlap": float(expr_vs_ae_neighbor_overlap),
        "expression_vs_m6a_pca_neighbor_overlap": float(expr_vs_m6a_pca_neighbor_overlap),
        "ae_cluster_counts": {str(k): int(v) for k, v in ae_counts.to_dict().items()},
        "ae_cluster_min_size": int(ae_counts.min()),
        "top_regulator_pair": str(regulator_corr_df.iloc[0]["pair"]) if not regulator_corr_df.empty else "NA",
        "top_regulator_corr": float(regulator_corr_df.iloc[0]["spearman_like_corr"]) if not regulator_corr_df.empty else float("nan"),
        "pca_note": pca_note,
        "elapsed_minutes": round((time.time() - start_time) / 60, 3),
        "loss": "coverage-capped weighted binomial NLL",
        "background_lambda": BACKGROUND_LAMBDA,
        "task": "cell-state / m6A regulatory-state autoencoder, not SigRM-style differential site calling",
        "passes_display_thresholds": bool(
            expr_vs_ae_ari >= 0.60
            and (not regulator_corr_df.empty and abs(float(regulator_corr_df.iloc[0]["spearman_like_corr"])) >= 0.35)
            and ae_all_nll <= site_all_nll + ALL_NLL_GUARDRAIL_MARGIN
        ),
    }
    (OUTPUT_DIR / "step03_score.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "step03_summary.txt").write_text("\n".join([f"{k}={v}" for k, v in score.items()]) + "\n", encoding="utf-8")
    print(json.dumps(score, indent=2), flush=True)
    gc.collect()


if __name__ == "__main__":
    main()
