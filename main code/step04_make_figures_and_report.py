from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
STEP2_DIR = BASE_DIR / "step02_input"
STEP3_DIR = BASE_DIR / "step03_outputs"
OUTPUT_DIR = BASE_DIR / "step04_outputs"

SEED = 42
TOP_M6A_SITES = 96
TOP_TSNE_M6A_SITES = 128
TOP_EXPR_GENES = 300
CLIP_RANGE = 2.5
POINT_SIZE = 10


def resolve_input(base_dir: Path, stem: str) -> Path:
    csv_path = base_dir / f"{stem}.csv"
    if csv_path.exists():
        return csv_path
    legacy_gz = base_dir / f"{stem}.csv.gz"
    if legacy_gz.exists():
        return legacy_gz
    raise FileNotFoundError(f"Missing input: {csv_path}")


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def pca_reduce(x: np.ndarray, n_components: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return (x @ vt[:n_components].T).astype(np.float32)


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (x - mu) / sigma


def tsne_or_pca(x: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    x = standardize(x)
    if x.shape[1] > 30:
        x = pca_reduce(x, 30)
    try:
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=seed,
            max_iter=750,
        )
        return tsne.fit_transform(x).astype(np.float32), "sklearn_tsne"
    except Exception as exc:
        return tsne_exact(x, perplexity=30.0, n_iter=450, learning_rate=200.0, seed=seed), f"exact_tsne_fallback:{type(exc).__name__}"


def pairwise_sq_dists(x: np.ndarray) -> np.ndarray:
    sum_x = np.sum(np.square(x), axis=1)
    d = np.add(np.add(-2.0 * np.dot(x, x.T), sum_x).T, sum_x)
    np.fill_diagonal(d, 0.0)
    return np.maximum(d, 0.0)


def hbeta(dist_row: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    p = np.exp(-dist_row * beta)
    sum_p = max(float(np.sum(p)), 1e-12)
    h = np.log(sum_p) + beta * float(np.sum(dist_row * p)) / sum_p
    return h, p / sum_p


def x2p(distances: np.ndarray, perplexity: float = 30.0, tol: float = 1e-5) -> np.ndarray:
    n = distances.shape[0]
    p = np.zeros((n, n), dtype=np.float64)
    beta = np.ones((n, 1), dtype=np.float64)
    log_u = np.log(perplexity)
    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        di = np.concatenate((distances[i, :i], distances[i, i + 1 :]))
        h, this_p = hbeta(di, beta[i, 0])
        hdiff = h - log_u
        tries = 0
        while abs(hdiff) > tol and tries < 60:
            if hdiff > 0:
                betamin = beta[i, 0]
                beta[i, 0] = beta[i, 0] * 2.0 if np.isinf(betamax) else (beta[i, 0] + betamax) / 2.0
            else:
                betamax = beta[i, 0]
                beta[i, 0] = beta[i, 0] / 2.0 if np.isinf(betamin) else (beta[i, 0] + betamin) / 2.0
            h, this_p = hbeta(di, beta[i, 0])
            hdiff = h - log_u
            tries += 1
        p[i, np.r_[0:i, i + 1 : n]] = this_p
    return p


def tsne_exact(x: np.ndarray, perplexity: float, n_iter: int, learning_rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    distances = pairwise_sq_dists(x)
    p = x2p(distances, perplexity=perplexity)
    p = (p + p.T) / (2.0 * p.shape[0])
    p = np.maximum(p, 1e-12)
    p *= 4.0

    y = 1e-4 * rng.standard_normal((x.shape[0], 2))
    i_y = np.zeros_like(y)
    gains = np.ones_like(y)
    for iter_idx in range(n_iter):
        sum_y = np.sum(np.square(y), axis=1)
        num = 1.0 / (1.0 + np.add(np.add(-2.0 * np.dot(y, y.T), sum_y).T, sum_y))
        np.fill_diagonal(num, 0.0)
        q = num / max(float(np.sum(num)), 1e-12)
        q = np.maximum(q, 1e-12)
        pq = p - q
        grad = np.zeros_like(y)
        for i in range(y.shape[0]):
            grad[i, :] = np.sum((pq[:, i] * num[:, i])[:, None] * (y[i, :] - y), axis=0) * 4.0
        momentum = 0.5 if iter_idx < 100 else 0.8
        gains = (gains + 0.2) * ((grad > 0.0) != (i_y > 0.0)) + (gains * 0.8) * ((grad > 0.0) == (i_y > 0.0))
        gains = np.maximum(gains, 0.01)
        i_y = momentum * i_y - learning_rate * gains * grad
        y += i_y
        y -= y.mean(axis=0, keepdims=True)
        if iter_idx == 100:
            p /= 4.0
    return y.astype(np.float32)


def choose_top_sites(site_metadata: pd.DataFrame, n_sites: int) -> pd.DataFrame:
    score_df = site_metadata.copy()
    for col in ["significant_cells", "ratio_sd_detected", "coverage_cells", "start"]:
        if col not in score_df:
            score_df[col] = 0
    score_df["selection_score"] = (
        4.0 * score_df["significant_cells"].fillna(0)
        + 2.0 * score_df["ratio_sd_detected"].fillna(0)
        + 0.25 * score_df["coverage_cells"].fillna(0)
    )
    return score_df.sort_values(
        ["selection_score", "significant_cells", "ratio_sd_detected", "coverage_cells", "start"],
        ascending=[False, False, False, False, True],
    ).head(n_sites).reset_index(drop=True)


def choose_top_variable_columns(df: pd.DataFrame, n_cols: int) -> list[str]:
    vals = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    variances = vals.var(axis=0)
    order = np.argsort(-variances, kind="mergesort")[:n_cols]
    return df.columns[1:][order].tolist()


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    center = matrix.mean(axis=1, keepdims=True)
    scale = matrix.std(axis=1, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return np.clip((matrix - center) / scale, -CLIP_RANGE, CLIP_RANGE)


def cluster_palette(clusters: np.ndarray) -> tuple[dict[str, str], np.ndarray]:
    unique = [str(x) for x in pd.unique(clusters)]
    base = ["#4E79A7", "#E15759", "#59A14F", "#F28E2B", "#B07AA1", "#76B7B2", "#EDC948", "#FF9DA7"]
    palette = {cluster: base[idx % len(base)] for idx, cluster in enumerate(sorted(unique))}
    color_arr = np.array([mcolors.to_rgb(palette[str(c)]) for c in clusters], dtype=np.float32)
    return palette, color_arr


def plot_training(history_df: pd.DataFrame) -> Path:
    plt.figure(figsize=(8.4, 4.8))
    sns.lineplot(data=history_df, x="epoch", y="train_weighted_binomial_nll", label="train weighted binomial NLL", linewidth=2)
    sns.lineplot(data=history_df, x="epoch", y="heldout_nll", label="all heldout NLL", linewidth=2)
    sns.lineplot(data=history_df, x="epoch", y="significant_heldout_nll", label="significant heldout NLL", linewidth=2)
    if "all_nll_guardrail_target" in history_df:
        plt.axhline(float(history_df["all_nll_guardrail_target"].iloc[0]), color="crimson", linestyle="--", linewidth=1, label="all-NLL guardrail")
    plt.title("Step3 cell-state AE training history")
    plt.xlabel("epoch")
    plt.ylabel("NLL")
    plt.grid(alpha=0.3)
    out = OUTPUT_DIR / "plot_training_nll_curves.png"
    savefig(out)
    return out


def plot_method_comparison(method_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    sns.barplot(data=method_df, x="method", y="heldout_nll", hue="method", legend=False, ax=axes[0])
    axes[0].set_title("All heldout NLL")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.3)
    sns.barplot(data=method_df, x="method", y="significant_heldout_nll", hue="method", legend=False, ax=axes[1])
    axes[1].set_title("Significant heldout NLL")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.3)
    out = OUTPUT_DIR / "plot_method_nll_comparison.png"
    savefig(out)
    return out


def plot_embedding_panels(cell_df: pd.DataFrame) -> Path:
    clusters = cell_df["expression_cluster"].astype(str).to_numpy()
    palette, _ = cluster_palette(clusters)
    panels = [
        ("Expression PCA", "expression_pc1", "expression_pc2"),
        ("m6A randomized PCA", "m6a_pca1", "m6a_pca2"),
        ("AE latent PCA", "ae_pc1", "ae_pc2"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, (title, x_col, y_col) in zip(axes, panels):
        for cluster, color in palette.items():
            mask = clusters == cluster
            ax.scatter(cell_df.loc[mask, x_col], cell_df.loc[mask, y_col], s=POINT_SIZE, c=color, alpha=0.75, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
    handles = [plt.Line2D([0], [0], marker="o", color=color, linestyle="", markersize=6, label=f"expr cluster {cluster}") for cluster, color in palette.items()]
    fig.legend(handles=handles, loc="upper center", ncol=max(2, len(handles)), frameon=False, bbox_to_anchor=(0.5, 1.03))
    out = OUTPUT_DIR / "plot_embedding_panels_by_expression_cluster.png"
    savefig(out)
    return out


def plot_neighbor_overlap(cluster_df: pd.DataFrame) -> Path:
    plt.figure(figsize=(6.4, 4.2))
    sns.barplot(data=cluster_df, x="comparison", y="neighbor_overlap", hue="comparison", legend=False)
    plt.title("Expression-neighborhood overlap")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    out = OUTPUT_DIR / "plot_neighbor_overlap.png"
    savefig(out)
    return out


def plot_regulator_correlations(reg_df: pd.DataFrame) -> tuple[Path, Path]:
    top = reg_df.head(15).copy()
    plt.figure(figsize=(8.5, 5.4))
    sns.barplot(data=top, y="pair", x="spearman_like_corr", hue=(top["spearman_like_corr"] > 0), dodge=False, legend=False)
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Top m6A regulator vs AE latent correlations")
    plt.xlabel("rank correlation")
    plt.ylabel("")
    bar_out = OUTPUT_DIR / "plot_regulator_correlations.png"
    savefig(bar_out)

    heat = reg_df.pivot(index="regulator", columns="latent_dim", values="spearman_like_corr")
    plt.figure(figsize=(10, 6))
    sns.heatmap(heat, cmap="vlag", center=0)
    plt.title("m6A regulator expression vs AE latent dimensions")
    heat_out = OUTPUT_DIR / "plot_regulator_correlation_heatmap.png"
    savefig(heat_out)
    return bar_out, heat_out


def plot_failure_reflection(method_df: pd.DataFrame, score: dict) -> Path:
    plt.figure(figsize=(6.0, 4.4))
    sns.barplot(data=method_df, x="method", y="heldout_nll", hue="method", legend=False)
    site_mean = float(score["site_mean_all_nll"])
    margin = site_mean + 0.01
    plt.axhline(site_mean, color="black", linestyle=":", linewidth=1, label="site-mean baseline")
    plt.axhline(margin, color="crimson", linestyle="--", linewidth=1, label="guardrail margin")
    plt.title("Reflection: whole-matrix reconstruction guardrail")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("all heldout NLL")
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.3)
    out = OUTPUT_DIR / "plot_failure_reflection_all_nll.png"
    savefig(out)
    return out


def plot_dca_style_heatmap(cell_df: pd.DataFrame, site_meta: pd.DataFrame, ratio_path: Path) -> Path:
    top_sites = choose_top_sites(site_meta, TOP_M6A_SITES)
    site_ids = top_sites["site_id"].astype(str).tolist()
    ratio_df = pd.read_csv(ratio_path, usecols=["cell_id", *site_ids])
    ordered_cell_df = cell_df.set_index("cell_id").loc[ratio_df["cell_id"].astype(str)].reset_index()
    matrix_cells_by_sites = ratio_df[site_ids].to_numpy(dtype=np.float32)
    matrix_sites_by_cells = zscore_rows(matrix_cells_by_sites.T)
    clusters = ordered_cell_df["expression_cluster"].astype(str).to_numpy()
    palette, cluster_colors = cluster_palette(clusters)
    panel_specs = [
        ("Expression PC1 order", "expression_pc1"),
        ("m6A PCA1 order", "m6a_pca1"),
        ("AE PC1 order", "ae_pc1"),
    ]

    fig = plt.figure(figsize=(15, 7.5), dpi=200)
    outer = GridSpec(1, 3, wspace=0.12, left=0.05, right=0.98, top=0.90, bottom=0.08)
    for panel_idx, (title, order_col) in enumerate(panel_specs):
        sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[panel_idx], height_ratios=[0.05, 0.95], hspace=0.02)
        ax_bar = fig.add_subplot(sub[0])
        ax_heat = fig.add_subplot(sub[1])
        order = np.argsort(ordered_cell_df[order_col].to_numpy(dtype=np.float32), kind="mergesort")
        ordered_clusters = clusters[order]
        ax_bar.imshow(cluster_colors[order].reshape(1, -1, 3), aspect="auto", interpolation="nearest")
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        ax_bar.set_title(title, fontsize=13)
        ax_heat.imshow(matrix_sites_by_cells[:, order], aspect="auto", interpolation="nearest", cmap="cividis", vmin=-CLIP_RANGE, vmax=CLIP_RANGE)
        changes = np.flatnonzero(ordered_clusters[1:] != ordered_clusters[:-1]) + 1
        for pos in changes:
            ax_heat.axvline(pos - 0.5, color="black", linewidth=0.8, alpha=0.55)
        ax_heat.set_xticks([])
        ax_heat.set_xlabel("cells")
        if panel_idx == 0:
            ax_heat.set_ylabel("selected m6A sites")
        else:
            ax_heat.set_yticks([])
    handles = [plt.Line2D([0], [0], marker="s", color=color, linestyle="", markersize=8, label=f"expr cluster {cluster}") for cluster, color in palette.items()]
    fig.legend(handles=handles, loc="upper center", ncol=max(2, len(handles)), frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("DCA-style heatmap: cell ordering changes m6A site pattern visibility", fontsize=16, y=0.98)
    out = OUTPUT_DIR / "plot_dca_style_heatmap_panel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_dca_style_tsne(cell_df: pd.DataFrame, site_meta: pd.DataFrame, ratio_path: Path, expression_path: Path) -> tuple[Path, str]:
    top_sites = choose_top_sites(site_meta, TOP_TSNE_M6A_SITES)
    site_ids = top_sites["site_id"].astype(str).tolist()
    expr_df = pd.read_csv(expression_path)
    expr_cols = choose_top_variable_columns(expr_df, TOP_EXPR_GENES)
    ratio_df = pd.read_csv(ratio_path, usecols=["cell_id", *site_ids])
    common_ids = ratio_df["cell_id"].astype(str).tolist()
    expr_df["cell_id"] = expr_df["cell_id"].astype(str)
    expr_df = expr_df.set_index("cell_id").loc[common_ids].reset_index()
    cell_ordered = cell_df.set_index("cell_id").loc[common_ids].reset_index()
    latent_cols = [c for c in cell_ordered.columns if c.startswith("latent_")]

    expr_mat = np.log1p(np.maximum(expr_df[expr_cols].to_numpy(dtype=np.float64), 0.0))
    m6a_mat = ratio_df[site_ids].to_numpy(dtype=np.float64)
    latent_mat = cell_ordered[latent_cols].to_numpy(dtype=np.float64)
    expr_xy, expr_note = tsne_or_pca(expr_mat, SEED + 1)
    m6a_xy, m6a_note = tsne_or_pca(m6a_mat, SEED + 2)
    ae_xy, ae_note = tsne_or_pca(latent_mat, SEED + 3)

    clusters = cell_ordered["expression_cluster"].astype(str).to_numpy()
    palette, _ = cluster_palette(clusters)
    panels = [
        ("Expression anchors", expr_xy),
        ("Raw m6A site matrix", m6a_xy),
        ("AE latent space", ae_xy),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), dpi=220)
    for ax, (title, xy) in zip(axes, panels):
        for cluster, color in palette.items():
            mask = clusters == cluster
            ax.scatter(xy[mask, 0], xy[mask, 1], s=9, c=color, alpha=0.75, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("tSNE1")
        ax.set_ylabel("tSNE2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
    handles = [plt.Line2D([0], [0], marker="o", color=color, linestyle="", markersize=6, label=f"expr cluster {cluster}") for cluster, color in palette.items()]
    fig.legend(handles=handles, loc="upper center", ncol=max(2, len(handles)), frameon=False, bbox_to_anchor=(0.5, 1.03))
    out = OUTPUT_DIR / "plot_dca_style_tsne_panel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, ";".join([expr_note, m6a_note, ae_note])


def write_report(score: dict, step2_summary: str, figures: dict[str, Path]) -> Path:
    ari = float(score["expression_vs_ae_ari"])
    top_corr = float(score["top_regulator_corr"])
    all_nll = float(score["cell_state_ae_all_nll"])
    site_all = float(score["site_mean_all_nll"])
    sig_nll = float(score["cell_state_ae_significant_nll"])
    site_sig = float(score["site_mean_significant_nll"])
    passes = bool(score.get("passes_display_thresholds", False))
    status = "display-ready preliminary result" if passes else "exploratory result with limitation"
    limitation = (
        "The current result passes the planned display thresholds, but NB/Beta-binomial objectives remain the next method-level improvement for over-dispersion."
        if passes
        else "The result is not presented as a final success. The most likely next step is NB/Beta-binomial modelling or direct DCA/scVI baseline comparison, because sparse m6A site-by-cell matrices remain difficult for whole-matrix reconstruction."
    )
    lines = [
        "# FYPmain-3 Overall IDEA Report",
        "",
        "## Task Decision",
        "",
        "- The project now follows the cell-clustering / m6A regulatory-state autoencoder route.",
        "- This is not a SigRM-style differential site calling workflow. SigRM informs the count-aware statistical mindset, but its test formula is not copied as the AE loss.",
        "- This route was chosen because the existing code is already AE-based, earlier results showed usable expression-vs-AE ARI and YTHDF2-latent correlation, and DCA/scVI provide clear future baselines.",
        "",
        "## Data Basis",
        "",
        "Step1 was not rerun. Step2 uses the existing validated scDART/m6AConquer RDS inputs and writes plain `.csv` matrices for m6A count, Total coverage, ratio, masks, metadata, expression anchors, and regulator expression.",
        "",
        "```text",
        step2_summary.strip(),
        "```",
        "",
        "## Step3 Model",
        "",
        "- Model: `BinomialAutoencoder` trained on cell-by-site m6A matrices.",
        "- Input channels: `ratio x mask`, `mask`, and `log1p(capped Total) x mask`.",
        "- Loss: coverage-capped weighted binomial NLL.",
        "",
        "```text",
        "L_main = sum_ij m_ij w_ij [-y_ij log(p_ij) - (n_ij-y_ij) log(1-p_ij)] / sum_ij m_ij w_ij n_eff_ij",
        "```",
        "",
        "## Current Result",
        "",
        f"- Status: `{status}`.",
        f"- Cells/sites: `{score['cells']}` cells x `{score['sites']}` sites.",
        f"- Expression vs AE ARI: `{ari:.4f}`.",
        f"- Expression-neighborhood overlap: `{float(score['expression_vs_ae_neighbor_overlap']):.4f}`.",
        f"- Top regulator-latent association: `{score['top_regulator_pair']}`, corr `{top_corr:.4f}`.",
        f"- Significant-site NLL: AE `{sig_nll:.4f}` vs site-mean `{site_sig:.4f}`.",
        f"- All-site NLL guardrail: AE `{all_nll:.4f}` vs site-mean `{site_all:.4f}`.",
        "",
        "## Figure Outputs",
        "",
        *[f"- `{name}`: `{path}`" for name, path in figures.items()],
        "",
        "## Limitation and Next Step",
        "",
        limitation,
        "",
    ]
    out = OUTPUT_DIR / "整体IDEA报告.MD"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score = json.loads((STEP3_DIR / "step03_score.json").read_text(encoding="utf-8"))
    history_df = pd.read_csv(STEP3_DIR / "step03_training_history.csv")
    method_df = pd.read_csv(STEP3_DIR / "step03_method_metrics.csv")
    cell_df = pd.read_csv(STEP3_DIR / "step03_cell_embeddings.csv")
    cluster_df = pd.read_csv(STEP3_DIR / "step03_cluster_comparison_metrics.csv")
    reg_df = pd.read_csv(STEP3_DIR / "step03_regulator_correlations.csv")
    site_meta = pd.read_csv(resolve_input(STEP2_DIR, "step02_site_metadata"))
    step2_summary = (STEP2_DIR / "step02_summary.txt").read_text(encoding="utf-8")
    ratio_path = resolve_input(STEP2_DIR, "step02_ratio_cells_by_sites")
    expression_path = resolve_input(STEP2_DIR, "step02_expression_anchor_cells_by_genes")

    figures = {
        "training_nll_curves": plot_training(history_df),
        "method_nll_comparison": plot_method_comparison(method_df),
        "embedding_panel": plot_embedding_panels(cell_df),
        "neighbor_overlap": plot_neighbor_overlap(cluster_df),
        "failure_reflection": plot_failure_reflection(method_df, score),
    }
    reg_bar, reg_heat = plot_regulator_correlations(reg_df)
    figures["regulator_correlations"] = reg_bar
    figures["regulator_heatmap"] = reg_heat
    figures["dca_style_heatmap"] = plot_dca_style_heatmap(cell_df, site_meta, ratio_path)
    tsne_path, tsne_note = plot_dca_style_tsne(cell_df, site_meta, ratio_path, expression_path)
    figures["dca_style_tsne"] = tsne_path

    report_path = write_report(score, step2_summary, figures)
    summary = {
        "step": "04_make_figures_and_report",
        "task": "cell-state / m6A regulatory-state autoencoder",
        "output_dir": str(OUTPUT_DIR),
        "figures": {name: str(path) for name, path in figures.items()},
        "report": str(report_path),
        "tsne_note": tsne_note,
        "cells": int(score["cells"]),
        "sites": int(score["sites"]),
        "expression_vs_ae_ari": float(score["expression_vs_ae_ari"]),
        "top_regulator_pair": str(score["top_regulator_pair"]),
        "top_regulator_corr": float(score["top_regulator_corr"]),
        "passes_display_thresholds": bool(score.get("passes_display_thresholds", False)),
        "note": "Step4 reads Step2 and Step3 outputs only; it does not train or resplit data.",
    }
    (OUTPUT_DIR / "step04_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
