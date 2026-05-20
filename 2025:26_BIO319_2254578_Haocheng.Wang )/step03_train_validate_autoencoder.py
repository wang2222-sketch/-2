#!/usr/bin/env python3
"""Self-contained Step 3 sparse count-aware m6A autoencoder.

This GitHub-facing script intentionally carries the previous external
``m6a_ae.data``, ``m6a_ae.models`` and ``m6a_ae.metrics`` logic inline so Step3
can be run from this repository with only Step2 outputs present. The model idea,
data handling, loss, and metrics are preserved from the current modular version.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_from_script(path: str | Path) -> Path:
    """Resolve relative paths from this script's folder, not the shell cwd."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (SCRIPT_DIR / path).resolve()

try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover
    PCA = None

try:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    brier_score_loss = None
    roc_auc_score = None


# ---------------------------------------------------------------------------
# Inlined m6a_ae.data
# ---------------------------------------------------------------------------
@dataclass
class DensePanel:
    """Dense tensors and metadata used by the autoencoder."""

    x: np.ndarray
    y: np.ndarray
    n: np.ndarray
    observed: np.ndarray
    significant: np.ndarray
    cell_metadata: pd.DataFrame
    site_metadata: pd.DataFrame
    observed_long: pd.DataFrame
    site_mean: np.ndarray
    site_logit: np.ndarray
    site_sig_prior: np.ndarray
    global_mean: float


def _find_file(step2_dir: Path, stem: str) -> Path:
    candidates = [
        step2_dir / f"{stem}.tsv.gz",
        step2_dir / f"{stem}.tsv",
        step2_dir / f"{stem}.csv.gz",
        step2_dir / f"{stem}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {stem}.tsv(.gz) or {stem}.csv(.gz) in {step2_dir}")


def _read_table(path: Path, **kwargs) -> pd.DataFrame:
    if path.name.endswith(".csv") or path.name.endswith(".csv.gz"):
        return pd.read_csv(path, **kwargs)
    return pd.read_csv(path, sep="\t", **kwargs)


def safe_logit(p: np.ndarray | float, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def choose_model_sites(step2_dir: Path, max_sites: Optional[int] = 0) -> pd.DataFrame:
    """Load Step2 model panel and select sites deterministically."""
    panel_path = step2_dir / "model_panel_sites.csv"
    fallback_path = step2_dir / "site_qc.csv"
    if panel_path.exists():
        sites = pd.read_csv(panel_path)
    elif fallback_path.exists():
        sites = pd.read_csv(fallback_path)
    else:
        raise FileNotFoundError("Neither model_panel_sites.csv nor site_qc.csv was found in Step2 output.")

    if "site_id" not in sites.columns:
        raise ValueError("model_panel_sites/site_qc must contain site_id")
    if "site_idx" not in sites.columns:
        sites = sites.copy()
        sites["site_idx"] = np.arange(len(sites), dtype=int)

    sort_cols = [c for c in [
        "significant_cells", "candidate_cells", "observed_cells",
        "var_ratio_observed", "mean_total_observed",
    ] if c in sites.columns]
    if sort_cols:
        sites = sites.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    else:
        sites = sites.sort_values("site_id")

    if max_sites is not None and max_sites > 0 and len(sites) > max_sites:
        sites = sites.head(max_sites).copy()
    return sites.drop_duplicates("site_idx").reset_index(drop=True)


def split_cells(
    n_cells: int,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Split by cells rather than by entries to test cell-level generalization."""
    if n_cells < 3:
        raise ValueError("At least three cells are required for train/val/test splitting.")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_cells)
    n_train = int(round(n_cells * train_fraction))
    n_val = int(round(n_cells * val_fraction))
    n_train = min(max(n_train, 1), n_cells - 2)
    n_val = min(max(n_val, 1), n_cells - n_train - 1)
    return {
        "train": np.sort(idx[:n_train]),
        "val": np.sort(idx[n_train:n_train + n_val]),
        "test": np.sort(idx[n_train + n_val:]),
    }


def _load_metadata(step2_dir: Path, observed: pd.DataFrame, kind: str) -> pd.DataFrame:
    path = step2_dir / f"{kind}_metadata.csv"
    if path.exists():
        meta = pd.read_csv(path)
    elif kind == "site" and (step2_dir / "site_qc.csv").exists():
        meta = pd.read_csv(step2_dir / "site_qc.csv")
    else:
        idx_col = f"{kind}_idx"
        id_col = f"{kind}_id"
        meta = observed[[idx_col, id_col]].drop_duplicates().copy()
    return meta


def load_sparse_step2_panel(
    step2_dir: str | Path,
    max_sites: Optional[int] = 0,
    min_total: int = 5,
) -> DensePanel:
    """Read Step2 outputs and build raw dense matrices.

    Training-derived priors and encoder input are finalized later by
    apply_training_priors(), after train/val/test split is known. This avoids
    validation/test leakage into site priors.
    """
    step2_dir = Path(step2_dir)
    obs_path = _find_file(step2_dir, "observed_entries")
    observed = _read_table(obs_path)
    if observed.empty:
        raise ValueError(f"{obs_path} is empty.")

    rename_map = {"m6A": "Y", "Total": "N", "significant": "is_significant"}
    observed = observed.rename(columns={k: v for k, v in rename_map.items() if k in observed.columns and v not in observed.columns})
    required = {"cell_id", "site_id", "Y", "N"}
    missing = required - set(observed.columns)
    if missing:
        raise ValueError(f"observed_entries is missing required columns: {sorted(missing)}")
    if "is_significant" not in observed.columns:
        observed["is_significant"] = 0
    if "cell_idx" not in observed.columns:
        cells_tmp = pd.Series(observed["cell_id"].astype(str).unique()).sort_values().reset_index(drop=True)
        observed["cell_idx"] = observed["cell_id"].astype(str).map({v: i for i, v in cells_tmp.items()})
    if "site_idx" not in observed.columns:
        sites_tmp = pd.Series(observed["site_id"].astype(str).unique()).sort_values().reset_index(drop=True)
        observed["site_idx"] = observed["site_id"].astype(str).map({v: i for i, v in sites_tmp.items()})

    observed["site_idx"] = pd.to_numeric(observed["site_idx"], errors="coerce")
    observed["cell_idx"] = pd.to_numeric(observed["cell_idx"], errors="coerce")
    observed = observed.dropna(subset=["site_idx", "cell_idx", "Y", "N", "cell_id", "site_id"]).copy()
    if max_sites is not None and int(max_sites) > 0:
        model_sites = choose_model_sites(step2_dir, max_sites=max_sites)
        keep_site_idx = set(pd.to_numeric(model_sites["site_idx"], errors="coerce").dropna().astype(int))
        observed = observed[observed["site_idx"].astype(int).isin(keep_site_idx)].copy()
    observed["Y"] = pd.to_numeric(observed["Y"], errors="coerce")
    observed["N"] = pd.to_numeric(observed["N"], errors="coerce")
    observed = observed.dropna(subset=["Y", "N"])
    observed = observed[observed["N"] >= min_total].copy()
    observed["Y"] = np.minimum(observed["Y"].astype(float), observed["N"].astype(float))
    observed["is_significant"] = pd.to_numeric(observed["is_significant"], errors="coerce").fillna(0).astype(int)

    if observed.empty:
        raise ValueError("No observed entries remained after optional site cap and min-total filtering.")

    cells = observed[["cell_idx", "cell_id"]].drop_duplicates().sort_values("cell_idx").reset_index(drop=True)
    sites = observed[["site_idx", "site_id"]].drop_duplicates().sort_values("site_idx").reset_index(drop=True)
    cell_code = {int(v): i for i, v in enumerate(cells["cell_idx"].astype(int))}
    site_code = {int(v): i for i, v in enumerate(sites["site_idx"].astype(int))}
    observed["cell_code"] = observed["cell_idx"].astype(int).map(cell_code).astype(int)
    observed["site_code"] = observed["site_idx"].astype(int).map(site_code).astype(int)

    n_cells = len(cells)
    n_sites = len(sites)
    y = np.zeros((n_cells, n_sites), dtype=np.float32)
    n = np.zeros((n_cells, n_sites), dtype=np.float32)
    obs_mask = np.zeros((n_cells, n_sites), dtype=bool)
    sig = np.zeros((n_cells, n_sites), dtype=bool)
    ci = observed["cell_code"].to_numpy(dtype=int)
    si = observed["site_code"].to_numpy(dtype=int)
    y[ci, si] = observed["Y"].to_numpy(dtype=np.float32)
    n[ci, si] = observed["N"].to_numpy(dtype=np.float32)
    obs_mask[ci, si] = True
    sig[ci, si] = observed["is_significant"].to_numpy(dtype=bool)

    cell_meta = _load_metadata(step2_dir, observed, "cell")
    site_meta = _load_metadata(step2_dir, observed, "site")
    cell_meta = cells.merge(cell_meta.drop_duplicates("cell_id"), on="cell_id", how="left", suffixes=("", "_meta"))
    site_meta = sites.merge(site_meta.drop_duplicates("site_id"), on="site_id", how="left", suffixes=("", "_meta"))

    global_mean = float(y.sum() / max(n.sum(), 1.0))
    site_mean = np.full(n_sites, global_mean, dtype=np.float32)
    site_logit = safe_logit(site_mean).astype(np.float32)
    site_sig_prior = np.full(n_sites, float(sig[obs_mask].mean()) if obs_mask.any() else 0.0, dtype=np.float32)
    x = np.tile(site_logit[None, :], (n_cells, 1)).astype(np.float32)

    return DensePanel(
        x=x,
        y=y,
        n=n,
        observed=obs_mask,
        significant=sig,
        cell_metadata=cell_meta,
        site_metadata=site_meta,
        observed_long=observed.reset_index(drop=True),
        site_mean=site_mean,
        site_logit=site_logit,
        site_sig_prior=site_sig_prior,
        global_mean=global_mean,
    )


def apply_training_priors(
    panel: DensePanel,
    train_cells: np.ndarray,
    eb_alpha: float = 1.0,
    site_mean_prior: float = 5.0,
    sig_prior_strength: float = 2.0,
    eps: float = 1e-6,
) -> DensePanel:
    """Recompute site priors from training cells only and rebuild encoder input."""
    train_cells = np.asarray(train_cells, dtype=int)
    train_mask = np.zeros(panel.observed.shape[0], dtype=bool)
    train_mask[train_cells] = True
    obs_train = panel.observed & train_mask[:, None]

    y_sum = (panel.y * obs_train).sum(axis=0)
    n_sum = (panel.n * obs_train).sum(axis=0)
    global_mean = float(y_sum.sum() / max(n_sum.sum(), 1.0))
    site_mean = (y_sum + site_mean_prior * global_mean) / np.maximum(n_sum + site_mean_prior, 1.0)
    site_mean = np.clip(site_mean, eps, 1.0 - eps).astype(np.float32)
    site_logit = safe_logit(site_mean, eps).astype(np.float32)

    sig_sum = (panel.significant & obs_train).sum(axis=0).astype(float)
    obs_count = obs_train.sum(axis=0).astype(float)
    global_sig = float(sig_sum.sum() / max(obs_count.sum(), 1.0))
    site_sig_prior = (sig_sum + sig_prior_strength * global_sig) / np.maximum(obs_count + sig_prior_strength, 1.0)
    site_sig_prior = np.clip(site_sig_prior, 0.0, 1.0).astype(np.float32)

    ratio_eb = (panel.y + eb_alpha * site_mean[None, :]) / np.maximum(panel.n + eb_alpha, 1.0)
    ratio_eb = np.clip(ratio_eb, eps, 1.0 - eps)
    x = safe_logit(ratio_eb, eps).astype(np.float32)
    neutral = np.broadcast_to(site_logit[None, :], x.shape)
    x[~panel.observed] = neutral[~panel.observed]

    return replace(panel, x=x, site_mean=site_mean, site_logit=site_logit, site_sig_prior=site_sig_prior, global_mean=global_mean)


def make_entry_holdout_masks(
    panel: DensePanel,
    splits: Dict[str, np.ndarray],
    holdout_fraction: float = 0.20,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Create deterministic entry masks for input hiding and validation."""
    holdout_fraction = float(holdout_fraction)
    rng = np.random.default_rng(seed)
    masks: Dict[str, np.ndarray] = {}
    for split_name, cell_idx in splits.items():
        mask = np.zeros_like(panel.observed, dtype=bool)
        split_rows = np.asarray(cell_idx, dtype=int)
        split_observed = panel.observed[split_rows]
        split_sig = panel.significant[split_rows]
        for label in [False, True]:
            local_pairs = np.argwhere(split_observed & (split_sig == label))
            if local_pairs.size == 0:
                continue
            n_choose = int(round(len(local_pairs) * holdout_fraction))
            if label and n_choose == 0:
                n_choose = 1
            n_choose = min(max(n_choose, 1), len(local_pairs))
            chosen = rng.choice(len(local_pairs), size=n_choose, replace=False)
            selected = local_pairs[chosen]
            actual_cells = split_rows[selected[:, 0]]
            actual_sites = selected[:, 1]
            mask[actual_cells, actual_sites] = True
        masks[split_name] = mask
    return masks


def split_name_by_cell(n_cells: int, splits: Dict[str, np.ndarray]) -> np.ndarray:
    out = np.array(["unused"] * n_cells, dtype=object)
    for name, idx in splits.items():
        out[np.asarray(idx, dtype=int)] = name
    return out


def make_observed_prediction_table(
    panel: DensePanel,
    pred_m6a: np.ndarray,
    pred_sig: np.ndarray,
    latent: np.ndarray,
    splits: Dict[str, np.ndarray],
    holdout_masks: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Return one row per observed cell-site entry with model predictions."""
    split_by_cell = split_name_by_cell(panel.x.shape[0], splits)
    rows = panel.observed_long.copy()
    ci = rows["cell_code"].to_numpy(dtype=int)
    si = rows["site_code"].to_numpy(dtype=int)
    rows["split"] = split_by_cell[ci]
    hidden = np.zeros(len(rows), dtype=bool)
    for split_name, hmask in holdout_masks.items():
        split_rows = rows["split"].to_numpy() == split_name
        if split_rows.any():
            hidden[split_rows] = hmask[ci[split_rows], si[split_rows]]
    rows["is_input_hidden"] = hidden
    rows["observed_ratio"] = rows["Y"].astype(float) / np.maximum(rows["N"].astype(float), 1.0)
    rows["pred_m6a_probability"] = pred_m6a[ci, si]
    rows["pred_significant_probability"] = pred_sig[ci, si]
    rows["site_mean_train_smoothed"] = panel.site_mean[si]
    rows["site_significance_train_prior"] = panel.site_sig_prior[si]
    rows["residual_pred_minus_site_mean"] = rows["pred_m6a_probability"] - rows["site_mean_train_smoothed"]
    for j in range(latent.shape[1]):
        rows[f"latent_{j+1}"] = latent[ci, j]
    return rows


# ---------------------------------------------------------------------------
# Inlined m6a_ae.metrics
# ---------------------------------------------------------------------------
def _clean_binary(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    n = min(len(y_true), len(y_score))
    y_true = (y_true[:n].astype(float) > 0).astype(int)
    y_score = y_score[:n]
    ok = np.isfinite(y_score)
    return y_true[ok], y_score[ok]


def _auc_pair(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    y_true, y_score = _clean_binary(y_true, y_score)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    if average_precision_score is None or roc_auc_score is None:
        return float("nan"), float("nan")
    return float(average_precision_score(y_true, y_score)), float(roc_auc_score(y_true, y_score))


def precision_lift_at_fraction(y_true: np.ndarray, y_score: np.ndarray, frac: float) -> Tuple[float, float]:
    y_true, y_score = _clean_binary(y_true, y_score)
    if len(y_true) == 0:
        return float("nan"), float("nan")
    k = max(1, int(np.ceil(len(y_true) * frac)))
    order = np.argsort(-y_score, kind="mergesort")[:k]
    precision = float(y_true[order].mean())
    prevalence = float(y_true.mean())
    return precision, precision / max(prevalence, 1e-12)


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    y_true, y_score = _clean_binary(y_true, y_score)
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (y_score >= lo) & ((y_score < hi) if i < n_bins - 1 else (y_score <= hi))
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(float(y_score[in_bin].mean()) - float(y_true[in_bin].mean()))
    return float(ece)


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, prefix: str = "") -> Dict[str, float]:
    y_true, y_score = _clean_binary(y_true, y_score)
    prevalence = float(y_true.mean()) if len(y_true) else float("nan")
    auprc, auroc = _auc_pair(y_true, y_score)
    if brier_score_loss is not None and len(np.unique(y_true)) >= 2:
        brier = float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0)))
    else:
        brier = float("nan")
    out: Dict[str, float] = {
        f"{prefix}n": int(len(y_true)),
        f"{prefix}positive_n": int(y_true.sum()) if len(y_true) else 0,
        f"{prefix}prevalence": prevalence,
        f"{prefix}auprc": auprc,
        f"{prefix}auroc": auroc,
        f"{prefix}auprc_lift_over_random": auprc / max(prevalence, 1e-12) if np.isfinite(auprc) else float("nan"),
        f"{prefix}brier": brier,
        f"{prefix}ece10": expected_calibration_error(y_true, y_score, 10),
    }
    for frac in [0.01, 0.05, 0.10]:
        precision, lift = precision_lift_at_fraction(y_true, y_score, frac)
        out[f"{prefix}precision_at_{int(frac * 100)}pct"] = precision
        out[f"{prefix}lift_at_{int(frac * 100)}pct"] = lift
    if len(y_score) > 0:
        out[f"{prefix}mean_score_positive"] = float(y_score[y_true == 1].mean()) if (y_true == 1).any() else float("nan")
        out[f"{prefix}mean_score_negative"] = float(y_score[y_true == 0].mean()) if (y_true == 0).any() else float("nan")
        out[f"{prefix}score_separation"] = out[f"{prefix}mean_score_positive"] - out[f"{prefix}mean_score_negative"]
    return out


def score_columns(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    scores: Dict[str, np.ndarray] = {
        "ae_significance_head": df["pred_significant_probability"].to_numpy(float),
        "ae_m6a_probability": df["pred_m6a_probability"].to_numpy(float),
        "coverage_only": np.log1p(df["N"].to_numpy(float)),
        "train_site_mean_m6a": df["site_mean_train_smoothed"].to_numpy(float),
        "train_site_significance_prior": df["site_significance_train_prior"].to_numpy(float),
    }
    # Upper bound only: the true observed ratio is not available for future hidden entries.
    scores["observed_ratio_upper_bound"] = df["observed_ratio"].to_numpy(float)
    return scores


def evaluate_prediction_scores(pred: pd.DataFrame, label_col: str = "is_significant") -> pd.DataFrame:
    """Evaluate AE and baseline scores by split and evaluation scope."""
    rows: List[Dict[str, float]] = []
    for split, split_df in pred.groupby("split"):
        scopes = {
            "all_observed": split_df,
            "heldout_input_entries": split_df[split_df["is_input_hidden"].astype(bool)],
            "visible_input_entries": split_df[~split_df["is_input_hidden"].astype(bool)],
        }
        for scope, g in scopes.items():
            if g.empty:
                continue
            labels = g[label_col].to_numpy(int)
            for method, score in score_columns(g).items():
                row = {"split": split, "scope": scope, "method": method}
                row.update(binary_metrics(labels, score))
                rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(
    pred: pd.DataFrame,
    score_col: str = "pred_significant_probability",
    label_col: str = "is_significant",
    scope_col: str = "is_input_hidden",
    n_bins: int = 10,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for split, split_df in pred.groupby("split"):
        for scope_name, g in {
            "all_observed": split_df,
            "heldout_input_entries": split_df[split_df[scope_col].astype(bool)],
        }.items():
            g = g[[score_col, label_col]].dropna().copy()
            if g.empty:
                continue
            g["bin"] = pd.qcut(g[score_col].rank(method="first"), q=min(n_bins, len(g)), labels=False, duplicates="drop")
            for b, sub in g.groupby("bin", dropna=False):
                rows.append({
                    "split": split,
                    "scope": scope_name,
                    "bin": int(b) if pd.notna(b) else -1,
                    "n": int(len(sub)),
                    "mean_score": float(sub[score_col].mean()),
                    "observed_positive_rate": float(sub[label_col].mean()),
                })
    return pd.DataFrame(rows)


def decile_lift_table(pred: pd.DataFrame, score_col: str = "pred_significant_probability", label_col: str = "is_significant") -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for split, split_df in pred.groupby("split"):
        for scope_name, g in {
            "all_observed": split_df,
            "heldout_input_entries": split_df[split_df["is_input_hidden"].astype(bool)],
        }.items():
            g = g[[score_col, label_col]].dropna().sort_values(score_col, ascending=False).reset_index(drop=True)
            if g.empty:
                continue
            prevalence = float(g[label_col].mean())
            g["decile"] = np.floor(np.arange(len(g)) / max(len(g) / 10.0, 1.0)).astype(int) + 1
            g["decile"] = g["decile"].clip(1, 10)
            for decile, sub in g.groupby("decile"):
                rate = float(sub[label_col].mean())
                rows.append({
                    "split": split,
                    "scope": scope_name,
                    "decile": int(decile),
                    "n": int(len(sub)),
                    "positive_rate": rate,
                    "lift_vs_random": rate / max(prevalence, 1e-12),
                })
    return pd.DataFrame(rows)


def overfit_diagnostics(metrics: pd.DataFrame) -> pd.DataFrame:
    ae = metrics[(metrics["method"] == "ae_significance_head") & (metrics["scope"] == "heldout_input_entries")]
    rows = []
    for metric in ["auprc", "auroc", "auprc_lift_over_random", "brier", "ece10"]:
        vals = ae.pivot_table(index="method", columns="split", values=metric, aggfunc="first")
        if vals.empty:
            continue
        row = {"metric": metric}
        for split in ["train", "val", "test"]:
            row[split] = float(vals.loc["ae_significance_head", split]) if split in vals.columns else float("nan")
        row["train_minus_val"] = row.get("train", np.nan) - row.get("val", np.nan)
        row["val_minus_test"] = row.get("val", np.nan) - row.get("test", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inlined m6a_ae.models
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    n_sites: int
    hidden_dim: int = 256
    latent_dim: int = 16
    dropout: float = 0.10
    kappa_min: float = 5.0
    kappa_init: float = 50.0


def beta_binomial_logpmf(y: torch.Tensor, n: torch.Tensor, p: torch.Tensor, kappa: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Beta-Binomial log-PMF parameterized by mean p and concentration kappa."""
    y = y.float()
    n = n.float().clamp_min(1.0)
    y = torch.minimum(torch.maximum(y, torch.zeros_like(y)), n)
    p = p.float().clamp(eps, 1.0 - eps)
    kappa = kappa.float().clamp_min(eps)
    alpha = p * kappa
    beta = (1.0 - p) * kappa
    return (
        torch.lgamma(n + 1.0)
        - torch.lgamma(y + 1.0)
        - torch.lgamma(n - y + 1.0)
        + torch.lgamma(y + alpha)
        + torch.lgamma(n - y + beta)
        + torch.lgamma(alpha + beta)
        - torch.lgamma(n + alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
    )


class SparseM6AAutoencoder(nn.Module):
    """Sparse count-aware autoencoder for cell-by-site m6A profiles.

    The encoder compresses one cell's sparse m6A profile into a latent vector.
    The decoder reconstructs two quantities for every cell-site entry:

    1. p_hat: m6A probability, trained with Beta-Binomial likelihood on visible
       observed entries;
    2. q_hat: Step2-defined significance probability, trained as a weak
       auxiliary head and evaluated on hidden observed entries.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        n = int(config.n_sites)
        h = int(config.hidden_dim)
        z = int(config.latent_dim)
        mid = max(h // 2, z)
        self.encoder = nn.Sequential(
            nn.Linear(n, h),
            nn.LayerNorm(h),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, mid),
            nn.LayerNorm(mid),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(mid, z),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z, mid),
            nn.LayerNorm(mid),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(mid, h),
            nn.LayerNorm(h),
            nn.ReLU(),
        )
        self.m6a_head = nn.Linear(h, n)
        self.significance_head = nn.Linear(h, n)
        raw_init = torch.log(torch.tensor(max(config.kappa_init - config.kappa_min, 1e-3)))
        self.raw_kappa = nn.Parameter(torch.full((n,), float(raw_init)))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        h = self.decoder(z)
        p = torch.sigmoid(self.m6a_head(h)).clamp(1e-6, 1.0 - 1e-6)
        q = torch.sigmoid(self.significance_head(h)).clamp(1e-6, 1.0 - 1e-6)
        kappa = F.softplus(self.raw_kappa) + float(self.config.kappa_min)
        return {"p": p, "q": q, "z": z, "kappa": kappa}


def masked_autoencoder_loss(
    model: SparseM6AAutoencoder,
    batch: Dict[str, torch.Tensor],
    lambda_sig: float = 0.15,
    lambda_latent_l2: float = 1e-4,
    pos_weight: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Count reconstruction plus weak significance supervision.

    The loss uses `loss_mask` if present. This allows the script to hide a subset
    of observed entries from both input and loss, then use those entries only for
    leakage-resistant evaluation.
    """
    out = model(batch["x"])
    mask = batch.get("loss_mask", batch["observed"]).bool()
    if int(mask.sum().detach().cpu()) == 0:
        zero = torch.zeros([], device=batch["x"].device)
        return zero, {"nll": 0.0, "sig_bce": 0.0, "latent_l2": 0.0}

    y = batch["y"].float()
    n = batch["n"].float()
    sig = batch["significant"].float()
    row_idx, site_idx = torch.nonzero(mask, as_tuple=True)
    nll = -beta_binomial_logpmf(
        y[row_idx, site_idx],
        n[row_idx, site_idx],
        out["p"][row_idx, site_idx],
        out["kappa"][site_idx],
    ).mean()
    q_logits = torch.logit(out["q"].clamp(1e-6, 1.0 - 1e-6))[row_idx, site_idx]
    sig_obs = sig[row_idx, site_idx]
    sig_bce = F.binary_cross_entropy_with_logits(q_logits, sig_obs, pos_weight=pos_weight)
    latent_l2 = out["z"].pow(2).mean()
    loss = nll + float(lambda_sig) * sig_bce + float(lambda_latent_l2) * latent_l2
    return loss, {
        "nll": float(nll.detach().cpu()),
        "sig_bce": float(sig_bce.detach().cpu()),
        "latent_l2": float(latent_l2.detach().cpu()),
    }


# ---------------------------------------------------------------------------
# Step3 training and validation entrypoint
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    root: str = "."
    step2_dir: str = "step2-output"
    outdir: str = "step3-output"
    max_sites: int = 0
    min_total: int = 5
    hidden_dim: int = 256
    latent_dim: int = 16
    dropout: float = 0.10
    max_epochs: int = 100
    patience: int = 15
    batch_cells: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_sig: float = 0.15
    lambda_latent_l2: float = 1e-4
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    entry_holdout_fraction: float = 0.20
    seed: int = 66
    device: str = "auto"
    run_permutation_control: bool = True
    permutation_epochs: int = 40
    save_train_predictions: bool = True


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train sparse m6A autoencoder and validate significance learning")
    p.add_argument("--root", default=".")
    p.add_argument("--step2-dir", default="step2-output")
    p.add_argument("--outdir", default="step3-output")
    p.add_argument("--max-sites", type=int, default=0, help="Optional site cap for memory control. Default 0 keeps all observed Step2 sites.")
    p.add_argument("--min-total", type=int, default=5)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-cells", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-sig", type=float, default=0.15)
    p.add_argument("--lambda-latent-l2", type=float, default=1e-4)
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--entry-holdout-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--run-permutation-control", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--permutation-epochs", type=int, default=40)
    p.add_argument("--save-train-predictions", action=argparse.BooleanOptionalAction, default=True)
    return TrainConfig(**vars(p.parse_args()))


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but unavailable")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def merge_holdout_masks(panel: DensePanel, holdout_masks: Dict[str, np.ndarray]) -> np.ndarray:
    merged = np.zeros_like(panel.observed, dtype=bool)
    for mask in holdout_masks.values():
        merged |= mask
    return merged


class CellDataset(Dataset):
    """Cell-level dataset with split-specific entry hiding."""

    def __init__(
        self,
        cell_indices: np.ndarray,
        panel: DensePanel,
        hide_mask: np.ndarray,
        significant_override: Optional[np.ndarray] = None,
    ):
        self.indices = np.asarray(cell_indices, dtype=int)
        self.panel = panel
        self.hide_mask = hide_mask
        self.significant = significant_override if significant_override is not None else panel.significant

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        idx = int(self.indices[i])
        x = self.panel.x[idx].copy()
        hidden = self.hide_mask[idx]
        if hidden.any():
            x[hidden] = self.panel.site_logit[hidden]
        observed = self.panel.observed[idx]
        loss_mask = observed & (~hidden)
        return {
            "cell_index": torch.tensor(idx, dtype=torch.long),
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(self.panel.y[idx], dtype=torch.float32),
            "n": torch.tensor(self.panel.n[idx], dtype=torch.float32),
            "observed": torch.tensor(observed, dtype=torch.bool),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
            "significant": torch.tensor(self.significant[idx], dtype=torch.float32),
        }


def compute_pos_weight(panel: DensePanel, train_idx: np.ndarray, sig_matrix: np.ndarray, hide_mask: np.ndarray, device: torch.device) -> torch.Tensor:
    visible = panel.observed[train_idx] & (~hide_mask[train_idx])
    sig = sig_matrix[train_idx][visible]
    pos = float(sig.sum())
    neg = float(len(sig) - pos)
    value = min(max(neg / max(pos, 1.0), 1.0), 100.0)
    return torch.tensor(value, dtype=torch.float32, device=device)


@torch.no_grad()
def predict_all(
    model: SparseM6AAutoencoder,
    panel: DensePanel,
    hide_mask: np.ndarray,
    device: torch.device,
    batch_cells: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    mu_parts, sig_parts, z_parts = [], [], []
    n_cells = panel.x.shape[0]
    for start in range(0, n_cells, batch_cells):
        stop = min(start + batch_cells, n_cells)
        x = panel.x[start:stop].copy()
        local_hide = hide_mask[start:stop]
        if local_hide.any():
            neutral = np.broadcast_to(panel.site_logit[None, :], x.shape)
            x[local_hide] = neutral[local_hide]
        xb = torch.tensor(x, dtype=torch.float32, device=device)
        out = model(xb)
        mu_parts.append(out["p"].detach().cpu().numpy())
        sig_parts.append(out["q"].detach().cpu().numpy())
        z_parts.append(out["z"].detach().cpu().numpy())
    return np.vstack(mu_parts), np.vstack(sig_parts), np.vstack(z_parts)


@torch.no_grad()
def eval_hidden_entries(
    model: SparseM6AAutoencoder,
    panel: DensePanel,
    cell_idx: np.ndarray,
    hide_mask: np.ndarray,
    device: torch.device,
    batch_cells: int,
) -> Dict[str, float]:
    """Evaluate NLL/MAE/AUPRC on hidden observed entries for selected cells."""
    model.eval()
    ds = CellDataset(cell_idx, panel, hide_mask)
    loader = DataLoader(ds, batch_size=batch_cells, shuffle=False)
    nll_sum, mae_sum, total_n = 0.0, 0.0, 0
    labels, scores = [], []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch["x"])
        hidden_eval = batch["observed"].bool() & (~batch["loss_mask"].bool())
        if int(hidden_eval.sum().detach().cpu()) == 0:
            continue
        row_idx, site_idx = torch.nonzero(hidden_eval, as_tuple=True)
        logp = beta_binomial_logpmf(
            batch["y"][row_idx, site_idx],
            batch["n"][row_idx, site_idx],
            out["p"][row_idx, site_idx],
            out["kappa"][site_idx],
        )
        ratio = batch["y"][row_idx, site_idx] / torch.clamp(batch["n"][row_idx, site_idx], min=1.0)
        nll_sum += float((-logp).sum().detach().cpu())
        mae_sum += float(torch.abs(out["p"][row_idx, site_idx] - ratio).sum().detach().cpu())
        total_n += int(hidden_eval.sum().detach().cpu())
        labels.append(batch["significant"][row_idx, site_idx].detach().cpu().numpy())
        scores.append(out["q"][row_idx, site_idx].detach().cpu().numpy())
    if total_n == 0:
        return {"heldout_n": 0, "heldout_nll": float("nan"), "heldout_mae": float("nan")}
    y_true = np.concatenate(labels).astype(int)
    q = np.concatenate(scores)
    out = {"heldout_n": int(total_n), "heldout_nll": nll_sum / total_n, "heldout_mae": mae_sum / total_n}
    out.update(binary_metrics(y_true, q))
    return out


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    """Load a PyTorch checkpoint while staying compatible with older PyTorch."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older PyTorch versions do not support weights_only
        return torch.load(path, map_location=device)


def train_one_model(
    panel: DensePanel,
    cfg: TrainConfig,
    splits: Dict[str, np.ndarray],
    hide_mask: np.ndarray,
    outdir: Path,
    device: torch.device,
    model_label: str = "real_labels",
    significant_override: Optional[np.ndarray] = None,
    max_epochs_override: Optional[int] = None,
) -> Tuple[SparseM6AAutoencoder, pd.DataFrame, Dict[str, float]]:
    set_seed(cfg.seed + (0 if model_label == "real_labels" else 1000))
    sig_matrix = significant_override if significant_override is not None else panel.significant
    model_cfg = ModelConfig(panel.x.shape[1], cfg.hidden_dim, cfg.latent_dim, cfg.dropout)
    model = SparseM6AAutoencoder(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    pos_weight = compute_pos_weight(panel, splits["train"], sig_matrix, hide_mask, device)
    train_ds = CellDataset(splits["train"], panel, hide_mask, significant_override=sig_matrix)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_cells, shuffle=True)
    max_epochs = int(max_epochs_override or cfg.max_epochs)
    history = []
    best_metric = float("inf")
    best_epoch = -1
    bad = 0
    best_path = outdir / f"{model_label}_model.pt"

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum, train_cells = 0.0, 0
        comp = {"nll": 0.0, "sig_bce": 0.0, "latent_l2": 0.0}
        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, c = masked_autoencoder_loss(model, batch, cfg.lambda_sig, cfg.lambda_latent_l2, pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            bs = int(batch["x"].shape[0])
            train_loss_sum += float(loss.detach().cpu()) * bs
            train_cells += bs
            for k in comp:
                comp[k] += c[k] * bs

        train_eval = eval_hidden_entries(model, panel, splits["train"], hide_mask, device, cfg.batch_cells)
        val_eval = eval_hidden_entries(model, panel, splits["val"], hide_mask, device, cfg.batch_cells)
        test_eval = eval_hidden_entries(model, panel, splits["test"], hide_mask, device, cfg.batch_cells)
        scheduler_metric = val_eval.get("heldout_nll", float("inf"))
        scheduler.step(scheduler_metric)
        if significant_override is None:
            metric = scheduler_metric - 0.01 * val_eval.get("auprc_lift_over_random", 0.0)
        else:
            metric = scheduler_metric
        row = {
            "model": model_label,
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_cells, 1),
            "train_nll_component": comp["nll"] / max(train_cells, 1),
            "train_sig_bce_component": comp["sig_bce"] / max(train_cells, 1),
            "lr": optimizer.param_groups[0]["lr"],
        }
        for prefix, metrics in [("train", train_eval), ("val", val_eval), ("test", test_eval)]:
            for k, v in metrics.items():
                row[f"{prefix}_{k}"] = v
        history.append(row)
        if metric < best_metric - 1e-5:
            best_metric = metric
            best_epoch = epoch
            bad = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model_cfg),
                "train_config": asdict(cfg),
                "best_epoch": best_epoch,
                "model_label": model_label,
            }, best_path)
        else:
            bad += 1
        print(
            f"[{model_label}] epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
            f"val_hidden_nll={val_eval.get('heldout_nll', math.nan):.4f} "
            f"val_auprc={val_eval.get('auprc', math.nan):.4f} "
            f"val_lift={val_eval.get('auprc_lift_over_random', math.nan):.3f}",
            flush=True,
        )
        if bad >= cfg.patience:
            break

    ckpt = load_checkpoint(best_path, device)
    model.load_state_dict(ckpt["model_state_dict"])
    hist = pd.DataFrame(history)
    hist.to_csv(outdir / f"{model_label}_train_history.csv", index=False)
    return model, hist, {"best_epoch": best_epoch, "best_metric": best_metric, "pos_weight": float(pos_weight.detach().cpu())}


def make_permuted_sig(panel: DensePanel, train_idx: np.ndarray, hide_mask: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = panel.significant.copy()
    visible = panel.observed[train_idx] & (~hide_mask[train_idx])
    vals = out[train_idx][visible].copy()
    rng.shuffle(vals)
    sub = out[train_idx].copy()
    sub[visible] = vals
    out[train_idx] = sub
    return out


def cell_score_table(panel: DensePanel, pred: pd.DataFrame, latent: np.ndarray, splits: Dict[str, np.ndarray]) -> pd.DataFrame:
    split_by_cell = np.array(["unused"] * panel.x.shape[0], dtype=object)
    for name, idx in splits.items():
        split_by_cell[np.asarray(idx, dtype=int)] = name
    rows = []
    for cell_id, g in pred.groupby("cell_id"):
        ci = int(g["cell_code"].iloc[0])
        row = {
            "cell_id": cell_id,
            "cell_code": ci,
            "split": split_by_cell[ci],
            "n_observed_entries": int(len(g)),
            "n_hidden_entries": int(g["is_input_hidden"].sum()),
            "observed_significant_rate": float(g["is_significant"].mean()),
            "mean_observed_ratio": float(g["observed_ratio"].mean()),
            "mean_pred_m6a_probability": float(g["pred_m6a_probability"].mean()),
            "mean_pred_significant_probability": float(g["pred_significant_probability"].mean()),
            "predicted_significance_burden": float(g["pred_significant_probability"].sum()),
            "mean_total_coverage": float(g["N"].mean()),
        }
        for j in range(latent.shape[1]):
            row[f"latent_{j+1}"] = float(latent[ci, j])
        rows.append(row)
    cell_scores = pd.DataFrame(rows)
    if PCA is not None and latent.shape[0] >= 3 and latent.shape[1] >= 2:
        pcs = PCA(n_components=2, random_state=0).fit_transform(latent)
        pc_df = pd.DataFrame({"cell_code": np.arange(latent.shape[0]), "PC1": pcs[:, 0], "PC2": pcs[:, 1]})
        cell_scores = cell_scores.merge(pc_df, on="cell_code", how="left")
    return cell_scores


def site_score_table(pred: pd.DataFrame) -> pd.DataFrame:
    agg = pred.groupby("site_id").agg(
        n_entries=("site_id", "size"),
        n_cells=("cell_id", "nunique"),
        n_hidden_entries=("is_input_hidden", "sum"),
        observed_significant_rate=("is_significant", "mean"),
        mean_observed_ratio=("observed_ratio", "mean"),
        mean_pred_m6a_probability=("pred_m6a_probability", "mean"),
        mean_pred_significant_probability=("pred_significant_probability", "mean"),
        site_mean_train_smoothed=("site_mean_train_smoothed", "mean"),
        site_significance_train_prior=("site_significance_train_prior", "mean"),
        mean_total_coverage=("N", "mean"),
    ).reset_index()
    for c in ["reference_site_source", "orthogonal_validation_status", "seqnames", "start", "end", "strand"]:
        if c in pred.columns:
            agg = agg.merge(pred[["site_id", c]].drop_duplicates("site_id"), on="site_id", how="left")
    return agg.sort_values("mean_pred_significant_probability", ascending=False)


def metric_value(metrics: pd.DataFrame, split: str, method: str, metric: str, scope: str = "heldout_input_entries") -> float:
    row = metrics[(metrics["split"] == split) & (metrics["scope"] == scope) & (metrics["method"] == method)]
    if row.empty or metric not in row.columns:
        return float("nan")
    return float(row[metric].iloc[0])


def build_validation_decision(metrics: pd.DataFrame, perm_summary: pd.DataFrame, run_permutation: bool) -> Dict[str, object]:
    test_lift = metric_value(metrics, "test", "ae_significance_head", "auprc_lift_over_random")
    val_lift = metric_value(metrics, "val", "ae_significance_head", "auprc_lift_over_random")
    train_lift = metric_value(metrics, "train", "ae_significance_head", "auprc_lift_over_random")
    test_auprc = metric_value(metrics, "test", "ae_significance_head", "auprc")
    test_prevalence = metric_value(metrics, "test", "ae_significance_head", "prevalence")
    site_prior_lift = metric_value(metrics, "test", "train_site_significance_prior", "auprc_lift_over_random")
    coverage_lift = metric_value(metrics, "test", "coverage_only", "auprc_lift_over_random")
    site_mean_lift = metric_value(metrics, "test", "train_site_mean_m6a", "auprc_lift_over_random")
    observed_ratio_lift = metric_value(metrics, "test", "observed_ratio_upper_bound", "auprc_lift_over_random")

    non_ae = metrics[(metrics["split"] == "test") & (metrics["scope"] == "heldout_input_entries") & (metrics["method"] != "ae_significance_head")]
    best_non_ae_lift = float(non_ae["auprc_lift_over_random"].max()) if not non_ae.empty else float("nan")
    best_non_ae_method = str(non_ae.sort_values("auprc_lift_over_random", ascending=False)["method"].iloc[0]) if not non_ae.empty else "NA"

    perm_lift = float("nan")
    perm_auprc = float("nan")
    if not perm_summary.empty and "model" in perm_summary.columns:
        pr = perm_summary[perm_summary["model"] == "permuted_train_labels"]
        if not pr.empty:
            perm_lift = float(pr["test_auprc_lift_over_random"].iloc[0]) if "test_auprc_lift_over_random" in pr.columns else float("nan")
            perm_auprc = float(pr["test_auprc"].iloc[0]) if "test_auprc" in pr.columns else float("nan")

    overfit_gap = train_lift - val_lift if np.isfinite(train_lift) and np.isfinite(val_lift) else float("nan")
    val_test_gap = abs(val_lift - test_lift) if np.isfinite(val_lift) and np.isfinite(test_lift) else float("nan")
    criteria = {
        "heldout_test_lift_ge_2": bool(np.isfinite(test_lift) and test_lift >= 2.0),
        "beats_site_prior_and_coverage": bool(
            np.isfinite(test_lift)
            and (not np.isfinite(site_prior_lift) or test_lift >= site_prior_lift)
            and (not np.isfinite(coverage_lift) or test_lift >= coverage_lift)
            and (not np.isfinite(site_mean_lift) or test_lift >= site_mean_lift)
        ),
        "beats_permutation_control": bool(
            (not run_permutation)
            or (np.isfinite(test_lift) and np.isfinite(perm_lift) and test_lift >= max(perm_lift * 1.25, perm_lift + 0.5))
        ),
        "no_obvious_train_val_overfit": bool((not np.isfinite(overfit_gap)) or overfit_gap <= max(5.0, 2.0 * max(val_lift, 1.0))),
        "val_test_reasonably_consistent": bool((not np.isfinite(val_test_gap)) or val_test_gap <= max(5.0, 2.0 * max(test_lift, 1.0))),
    }
    status = "defensible_significance_learning" if all(criteria.values()) else "weak_or_overfit"
    return {
        "status": status,
        "criterion_pass_count": int(sum(criteria.values())),
        "criterion_total": int(len(criteria)),
        **criteria,
        "test_ae_auprc": test_auprc,
        "test_prevalence": test_prevalence,
        "test_ae_lift": test_lift,
        "val_ae_lift": val_lift,
        "train_ae_lift": train_lift,
        "train_minus_val_lift": overfit_gap,
        "val_minus_test_lift_abs": val_test_gap,
        "test_best_non_ae_lift": best_non_ae_lift,
        "test_best_non_ae_method": best_non_ae_method,
        "test_site_prior_lift": site_prior_lift,
        "test_coverage_lift": coverage_lift,
        "test_site_mean_lift": site_mean_lift,
        "test_observed_ratio_upper_bound_lift": observed_ratio_lift,
        "permutation_test_auprc": perm_auprc,
        "permutation_test_lift": perm_lift,
    }


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    root = resolve_from_script(cfg.root)
    outdir = root / cfg.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg.device)
    start = time.time()

    print(f"[Step03] device={device}; loading Step2", flush=True)
    panel = load_sparse_step2_panel(root / cfg.step2_dir, max_sites=cfg.max_sites, min_total=cfg.min_total)
    splits = split_cells(panel.x.shape[0], cfg.train_fraction, cfg.val_fraction, cfg.seed)
    panel = apply_training_priors(panel, splits["train"])
    holdout_masks = make_entry_holdout_masks(panel, splits, holdout_fraction=cfg.entry_holdout_fraction, seed=cfg.seed + 17)
    merged_hide = merge_holdout_masks(panel, holdout_masks)

    split_summary = []
    for name, idx in splits.items():
        idx = np.asarray(idx, dtype=int)
        obs = panel.observed[idx]
        hidden = merged_hide[idx]
        split_summary.append({
            "split": name,
            "cells": int(len(idx)),
            "observed_entries": int(obs.sum()),
            "hidden_entries": int(hidden.sum()),
            "visible_training_entries": int((obs & ~hidden).sum()),
            "significant_entries": int(panel.significant[idx][obs].sum()),
            "hidden_significant_entries": int(panel.significant[idx][hidden].sum()),
        })
    pd.DataFrame(split_summary).to_csv(outdir / "split_summary.csv", index=False)

    config_record = {
        **asdict(cfg),
        "root_resolved": str(root),
        "device_used": str(device),
        "n_cells": int(panel.x.shape[0]),
        "n_sites": int(panel.x.shape[1]),
        "n_observed": int(panel.observed.sum()),
        "n_significant": int(panel.significant.sum()),
        "site_mean_fitted_from_train_cells_only": True,
        "entry_holdout_used_for_input_hiding": True,
    }
    (outdir / "step03_config.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")

    print(f"[Step03] cells={panel.x.shape[0]:,}; sites={panel.x.shape[1]:,}; observed={int(panel.observed.sum()):,}; significant={int(panel.significant.sum()):,}", flush=True)
    model, hist, info = train_one_model(panel, cfg, splits, merged_hide, outdir, device, "real_labels")
    hist.to_csv(outdir / "training_history.csv", index=False)

    mu, sigp, latent = predict_all(model, panel, merged_hide, device, cfg.batch_cells)
    pred = make_observed_prediction_table(panel, mu, sigp, latent, splits, holdout_masks)
    meta_cols = [c for c in ["site_id", "reference_site_source", "orthogonal_validation_status", "seqnames", "start", "end", "strand"] if c in panel.site_metadata.columns]
    if meta_cols:
        pred = pred.merge(panel.site_metadata[meta_cols].drop_duplicates("site_id"), on="site_id", how="left")
    if not cfg.save_train_predictions:
        pred = pred[pred["split"].isin(["val", "test"])].copy()
    pred.to_csv(outdir / "observed_predictions.tsv.gz", sep="\t", index=False, compression="gzip")

    cell_scores = cell_score_table(panel, pred, latent, splits)
    cell_scores.to_csv(outdir / "cell_scores.csv", index=False)
    site_scores = site_score_table(pred)
    site_scores.to_csv(outdir / "site_scores.csv", index=False)
    latent_cols = [c for c in cell_scores.columns if c in {"cell_id", "cell_code", "split", "PC1", "PC2"} or c.startswith("latent_")]
    cell_scores[latent_cols].to_csv(outdir / "latent.csv", index=False)

    metrics = evaluate_prediction_scores(pred)
    metrics.to_csv(outdir / "baseline_comparison.csv", index=False)
    calibration_table(pred).to_csv(outdir / "calibration_table.csv", index=False)
    decile_lift_table(pred).to_csv(outdir / "decile_lift_table.csv", index=False)
    overfit_diagnostics(metrics).to_csv(outdir / "overfit_diagnostics.csv", index=False)

    perm_rows = []
    real_test = metrics[(metrics["split"] == "test") & (metrics["scope"] == "heldout_input_entries") & (metrics["method"] == "ae_significance_head")]
    if not real_test.empty:
        d = real_test.iloc[0].to_dict()
        perm_rows.append({"model": "real_labels", **{f"test_{k}": v for k, v in d.items() if isinstance(v, (int, float, np.integer, np.floating))}})
    if cfg.run_permutation_control:
        print("[Step03] running label-permutation control", flush=True)
        perm_sig = make_permuted_sig(panel, splits["train"], merged_hide, cfg.seed + 999)
        perm_cfg = TrainConfig(**{**asdict(cfg), "max_epochs": cfg.permutation_epochs, "patience": max(5, min(cfg.patience, 8))})
        perm_model, perm_hist, _ = train_one_model(panel, perm_cfg, splits, merged_hide, outdir, device, "permuted_train_labels", significant_override=perm_sig, max_epochs_override=cfg.permutation_epochs)
        perm_hist.to_csv(outdir / "permuted_train_history.csv", index=False)
        p_mu, p_sigp, p_latent = predict_all(perm_model, panel, merged_hide, device, cfg.batch_cells)
        perm_pred = make_observed_prediction_table(panel, p_mu, p_sigp, p_latent, splits, holdout_masks)
        perm_metrics = evaluate_prediction_scores(perm_pred)
        perm_metrics.to_csv(outdir / "permuted_baseline_comparison.csv", index=False)
        ptest = perm_metrics[(perm_metrics["split"] == "test") & (perm_metrics["scope"] == "heldout_input_entries") & (perm_metrics["method"] == "ae_significance_head")]
        if not ptest.empty:
            d = ptest.iloc[0].to_dict()
            perm_rows.append({"model": "permuted_train_labels", **{f"test_{k}": v for k, v in d.items() if isinstance(v, (int, float, np.integer, np.floating))}})
    perm_summary = pd.DataFrame(perm_rows)
    perm_summary.to_csv(outdir / "permutation_control_summary.csv", index=False)

    decision = build_validation_decision(metrics, perm_summary, cfg.run_permutation_control)
    pd.DataFrame([decision]).to_csv(outdir / "validation_decision.csv", index=False)
    (outdir / "validation_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    summary = {
        **decision,
        "runtime_seconds": time.time() - start,
        "device_used": str(device),
        "best_epoch": info.get("best_epoch"),
        "pos_weight": info.get("pos_weight"),
        "site_mean_fitted_from_train_cells_only": True,
        "entry_holdout_used_for_input_hiding": True,
        "n_cells": int(panel.x.shape[0]),
        "n_sites": int(panel.x.shape[1]),
        "n_observed_entries": int(panel.observed.sum()),
        "n_significant_entries": int(panel.significant.sum()),
        "train_cells": int(len(splits["train"])),
        "val_cells": int(len(splits["val"])),
        "test_cells": int(len(splits["test"])),
    }
    test_row = metrics[(metrics["split"] == "test") & (metrics["scope"] == "heldout_input_entries") & (metrics["method"] == "ae_significance_head")]
    if not test_row.empty:
        for k, v in test_row.iloc[0].items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                summary[f"test_{k}"] = float(v)
    pd.DataFrame([summary]).to_csv(outdir / "model_validation_summary.csv", index=False)
    print(f"[Step03] completed: {outdir}; status={decision['status']}", flush=True)


if __name__ == "__main__":
    main()
