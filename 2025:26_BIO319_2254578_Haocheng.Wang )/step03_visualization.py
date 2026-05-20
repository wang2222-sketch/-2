#!/usr/bin/env python3
"""Step3-only visualizations for sparse m6A autoencoder outputs.

Default layout:
  SCRIPT_DIR/step3-output/
  SCRIPT_DIR/step3-visualization-output/

Relative paths are resolved from this script's folder, not the shell cwd.
The script intentionally uses Step3 aggregate CSV outputs and does not load the
large observed_predictions.tsv.gz table by default.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent

SPLIT_COLORS = {
    "train": "#2F5D8C",
    "val": "#1B9E77",
    "test": "#D95F02",
    "permuted": "#B279A2",
    "real": "#2F5D8C",
}
MODEL_COLORS = {
    "ae_significance_head": "#2F5D8C",
    "ae_m6a_probability": "#6FA8DC",
    "train_site_mean_m6a": "#9AA7B7",
    "train_site_significance_prior": "#D95F02",
    "coverage_only": "#B8B8B8",
}
ACCENT_RED = "#C44E52"
TEXT_COLOR = "#222222"
GRID_COLOR = "#E6E6E6"


def set_publication_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.7,
        "axes.edgecolor": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "lines.linewidth": 1.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


set_publication_style()


def clean_axis(ax: plt.Axes, grid_axis: Optional[str] = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.6, colors=TEXT_COLOR)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.45, alpha=0.9)
    else:
        ax.grid(False)


def split_color(split: object, fallback: str = "#666666") -> str:
    return SPLIT_COLORS.get(str(split).lower(), fallback)


def method_label(method: object) -> str:
    labels = {
        "ae_significance_head": "AE significance",
        "ae_m6a_probability": "AE m6A prob",
        "train_site_mean_m6a": "site mean",
        "train_site_significance_prior": "site prior",
        "coverage_only": "coverage",
        "observed_rate_upper_bound": "observed-rate upper bound",
    }
    return labels.get(str(method), str(method).replace("_", " "))


def bar_label(ax: plt.Axes, values: Sequence[float], orient: str = "v", fmt: str = "{:.2f}") -> None:
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if orient == "h":
            ax.text(float(value) + max(abs(float(value)) * 0.015, 0.015), i, fmt.format(float(value)), va="center", fontsize=7)
        else:
            ax.text(i, float(value) + max(abs(float(value)) * 0.018, 0.02), fmt.format(float(value)), ha="center", va="bottom", fontsize=7)


def metric_value_label(metric: object, value: object, signed: bool = False) -> str:
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(val):
        return "NA"
    metric_name = str(metric)
    if metric_name == "auprc_lift_over_random":
        fmt = "{:+.2f}" if signed else "{:.2f}"
    else:
        fmt = "{:+.4f}" if signed else "{:.3f}"
    return fmt.format(float(val))


def add_decile_value_labels(ax: plt.Axes, sub: pd.DataFrame, split: object, value_col: str = "lift_vs_random") -> None:
    offsets = {
        "train": (0, 8),
        "val": (0, -11),
        "test": (0, 20),
    }
    dx, dy = offsets.get(str(split).lower(), (0, 8))
    color = split_color(split)
    for _, row in sub.iterrows():
        x = pd.to_numeric(pd.Series([row.get("decile")]), errors="coerce").iloc[0]
        y = pd.to_numeric(pd.Series([row.get(value_col)]), errors="coerce").iloc[0]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        ax.annotate(
            f"{float(y):.2f}",
            xy=(float(x), float(y)),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=6.2,
            color=color,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.74),
            zorder=6,
        )


def resolve_from_script(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (SCRIPT_DIR / path).resolve()


def resolve_under_root(path: Optional[str], root: Path, default_rel: str) -> Path:
    if not path:
        return (root / default_rel).resolve()
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create Step3-only visualization outputs.")
    p.add_argument("--root", default=".", help="Project root; relative paths resolve from this script folder.")
    p.add_argument("--step3-dir", default=None, help="Step3 output directory; default root/step3-output.")
    p.add_argument("--outdir", default=None, help="Visualization output directory; default root/step3-visualization-output.")
    p.add_argument("--top-sites", type=int, default=100, help="Number of top sites to export.")
    p.add_argument("--max-site-scatter", type=int, default=5000, help="Maximum sites sampled for scatter plots.")
    p.add_argument("--random-state", type=int, default=666, help="Random state for deterministic plotting samples.")
    return p.parse_args()


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Step3 output: {path}")
    return pd.read_csv(path)


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def numeric_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def get_best_epoch(real_hist: pd.DataFrame, model_summary: Optional[pd.DataFrame] = None) -> int:
    if model_summary is not None and not model_summary.empty and "best_epoch" in model_summary.columns:
        best = pd.to_numeric(model_summary["best_epoch"], errors="coerce").dropna()
        if not best.empty:
            return int(best.iloc[0])
    if "val_heldout_nll" in real_hist.columns:
        idx = pd.to_numeric(real_hist["val_heldout_nll"], errors="coerce").idxmin()
        if pd.notna(idx):
            return int(real_hist.loc[idx, "epoch"])
    return int(real_hist["epoch"].iloc[-1])


def annotate_best_epoch(ax: plt.Axes, hist: pd.DataFrame, best_epoch: int, y_col: str, color: str, label: str) -> None:
    if y_col not in hist.columns:
        return
    rows = hist.loc[hist["epoch"].astype(int) == int(best_epoch)]
    if rows.empty:
        rows = hist.iloc[[int(np.argmin(np.abs(hist["epoch"].to_numpy(float) - best_epoch)))]]
    y = float(rows[y_col].iloc[0])
    ax.axvline(best_epoch, color="#666666", linewidth=0.9, linestyle="--", alpha=0.9)
    ax.scatter([best_epoch], [y], s=34, color=color, edgecolor="white", linewidth=0.7, zorder=5)
    ax.annotate(
        f"Best epoch {best_epoch}\n{label} = {y:.4f}",
        xy=(best_epoch, y),
        xytext=(10, 14),
        textcoords="offset points",
        fontsize=7.2,
        color=TEXT_COLOR,
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#B7C0C7", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="#666666", linewidth=0.7),
    )


def plot_training_loss_zoom(real_hist: pd.DataFrame, fig_dir: Path, model_summary: Optional[pd.DataFrame] = None) -> None:
    required = {"epoch", "train_loss", "val_heldout_nll"}
    if not required.issubset(real_hist.columns):
        return
    hist = real_hist.copy()
    hist["epoch"] = pd.to_numeric(hist["epoch"], errors="coerce")
    hist["train_loss"] = pd.to_numeric(hist["train_loss"], errors="coerce")
    hist["val_heldout_nll"] = pd.to_numeric(hist["val_heldout_nll"], errors="coerce")
    hist = hist.dropna(subset=["epoch", "train_loss", "val_heldout_nll"]).sort_values("epoch")
    if hist.empty:
        return

    best_epoch = get_best_epoch(hist, model_summary)
    best_rows = hist.loc[hist["epoch"].astype(int) == int(best_epoch)]
    if best_rows.empty:
        best_rows = hist.iloc[[int(np.argmin(np.abs(hist["epoch"].to_numpy(float) - best_epoch)))]]
        best_epoch = int(best_rows["epoch"].iloc[0])
    best_val = float(best_rows["val_heldout_nll"].iloc[0])

    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    train_color = split_color("train")
    val_color = split_color("val")
    best_color = ACCENT_RED
    ax.plot(hist["epoch"], hist["train_loss"], color=train_color, linewidth=1.7, label="Train loss")
    ax.plot(hist["epoch"], hist["val_heldout_nll"], color=val_color, linewidth=1.7, label="Validation NLL")
    annotate_best_epoch(ax, hist, best_epoch, "val_heldout_nll", best_color, "Val NLL")
    ax.set_title("Training and validation losses", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    clean_axis(ax, "both")

    min_epoch = int(hist["epoch"].min())
    max_epoch = int(hist["epoch"].max())
    zoom_width = min(45, max(12, max_epoch - min_epoch))
    zoom_start = max(min_epoch, best_epoch - zoom_width)
    zoom_end = min(max_epoch, best_epoch + max(3, zoom_width // 8))
    if zoom_end - zoom_start < 8:
        zoom_start = max(min_epoch, zoom_end - 8)
    zoom = hist[(hist["epoch"] >= zoom_start) & (hist["epoch"] <= zoom_end)].copy()
    if len(zoom) >= 3:
        axins = inset_axes(ax, width="42%", height="38%", loc="center right", borderpad=1.4)
        axins.plot(zoom["epoch"], zoom["train_loss"], color=train_color, linewidth=1.1)
        axins.plot(zoom["epoch"], zoom["val_heldout_nll"], color=val_color, linewidth=1.1)
        axins.axvline(best_epoch, color="#666666", linewidth=0.8, linestyle="--", alpha=0.9)
        axins.scatter([best_epoch], [best_val], s=24, color=best_color, edgecolor="white", linewidth=0.7, zorder=5)
        axins.set_title("Best-epoch window", fontsize=7.2, pad=3)
        yvals = np.r_[zoom["train_loss"].to_numpy(float), zoom["val_heldout_nll"].to_numpy(float)]
        yvals = yvals[np.isfinite(yvals)]
        if len(yvals):
            pad = max((float(np.nanmax(yvals)) - float(np.nanmin(yvals))) * 0.18, 0.01)
            axins.set_ylim(float(np.nanmin(yvals)) - pad, float(np.nanmax(yvals)) + pad)
        axins.set_xlim(float(zoom_start), float(zoom_end))
        clean_axis(axins, "both")
        axins.tick_params(labelsize=6.5)
        for spine in axins.spines.values():
            spine.set_color("#4A4A4A")
            spine.set_linewidth(0.7)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="#B8B8B8", lw=0.7, alpha=0.65)

    savefig(fig, fig_dir / "training_loss_zoom_best.png")


def plot_training_curves(
    real_hist: pd.DataFrame,
    perm_hist: Optional[pd.DataFrame],
    fig_dir: Path,
    model_summary: Optional[pd.DataFrame] = None,
) -> None:
    best_epoch = get_best_epoch(real_hist, model_summary)
    series = [
        ("heldout_nll", "Heldout NLL"),
        ("heldout_mae", "Heldout MAE"),
        ("auprc", "AUPRC"),
        ("auroc", "AUROC"),
        ("auprc_lift_over_random", "AUPRC lift"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 7.2))
    axes = axes.ravel()
    for ax, (suffix, title) in zip(axes, series):
        for split in ["train", "val", "test"]:
            col = f"{split}_{suffix}"
            if col in real_hist.columns:
                ax.plot(real_hist["epoch"], real_hist[col], label=str(split), color=split_color(split), linewidth=1.35)
        if suffix == "heldout_nll" and "val_heldout_nll" in real_hist.columns:
            annotate_best_epoch(ax, real_hist, best_epoch, "val_heldout_nll", ACCENT_RED, "Val NLL")
        if perm_hist is not None:
            col = f"val_{suffix}"
            if col in perm_hist.columns:
                ax.plot(perm_hist["epoch"], perm_hist[col], label="permuted val", color=split_color("permuted"), linewidth=1.0, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        clean_axis(ax, "both")
    ax = axes[-1]
    if "train_loss" in real_hist.columns:
        ax.plot(real_hist["epoch"], real_hist["train_loss"], label="train loss", color=split_color("train"), linewidth=1.35)
    if "lr" in real_hist.columns:
        ax2 = ax.twinx()
        ax2.plot(real_hist["epoch"], real_hist["lr"], label="lr", color=ACCENT_RED, linewidth=1.0, linestyle=":")
        ax2.set_ylabel("LR")
        clean_axis(ax2, None)
    ax.set_title("Training loss and LR")
    ax.set_xlabel("Epoch")
    clean_axis(ax, "both")
    handles, labels = [], []
    for a in axes:
        h, l = a.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    savefig(fig, fig_dir / "training_curves.png")


def plot_split_summary(split_summary: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    x = np.arange(len(split_summary))
    axes[0].bar(x, split_summary["cells"], color=[split_color(s) for s in split_summary["split"]], edgecolor=TEXT_COLOR, linewidth=0.4)
    axes[0].set_xticks(x, split_summary["split"])
    axes[0].set_title("Cell counts by split")
    axes[0].set_ylabel("Cells")
    clean_axis(axes[0])
    for col, color in [("observed_entries", split_color("train")), ("hidden_entries", split_color("val")), ("significant_entries", split_color("test"))]:
        if col in split_summary.columns:
            axes[1].bar(x, split_summary[col], label=col, alpha=0.8, color=color)
            x = x + 0.0
    width = 0.25
    ax = axes[1]
    ax.clear()
    base = np.arange(len(split_summary))
    bars = [("observed_entries", split_color("train")), ("hidden_entries", split_color("val")), ("significant_entries", split_color("test"))]
    for i, (col, color) in enumerate(bars):
        if col in split_summary.columns:
            ax.bar(base + (i - 1) * width, split_summary[col], width=width, label=col.replace("_", " "), color=color, edgecolor=TEXT_COLOR, linewidth=0.25)
    ax.set_xticks(base, split_summary["split"])
    ax.set_title("Entry counts by split")
    ax.set_ylabel("Entries")
    ax.legend(frameon=False, fontsize=7)
    clean_axis(ax)
    savefig(fig, fig_dir / "split_summary.png")


def plot_baseline_comparison(metrics: pd.DataFrame, perm_metrics: Optional[pd.DataFrame], fig_dir: Path) -> None:
    scope = "heldout_input_entries"
    g = metrics[(metrics["scope"] == scope) & (metrics["split"] == "test")].copy()
    if g.empty:
        g = metrics[(metrics["split"] == "test")].copy()
    methods = g.sort_values("auprc_lift_over_random", ascending=False)["method"].astype(str).tolist()
    labels = [method_label(m) for m in methods]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.35))
    for ax, metric, title in [
        (axes[0], "auprc_lift_over_random", "Test AUPRC lift"),
        (axes[1], "auprc", "Test AUPRC"),
        (axes[2], "auroc", "Test AUROC"),
    ]:
        if metric not in g.columns:
            ax.axis("off")
            continue
        vals = g.set_index("method").reindex(methods)[metric].astype(float)
        colors = [MODEL_COLORS.get(m, "#9AA7B7") for m in methods]
        ax.barh(np.arange(len(vals)), vals.values, color=colors, edgecolor=TEXT_COLOR, linewidth=0.35)
        if metric == "auprc_lift_over_random":
            ax.axvline(1, color="#666666", linestyle="--", linewidth=0.8)
        ax.set_yticks(np.arange(len(vals)), labels if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_title(title)
        clean_axis(ax, "x")
        bar_label(ax, vals.values, orient="h", fmt="{:.2f}" if metric != "auprc" else "{:.3f}")
    savefig(fig, fig_dir / "baseline_comparison_test.png")

    if perm_metrics is not None:
        combined = []
        for label, df in [("real", metrics), ("permuted", perm_metrics)]:
            sub = df[(df["scope"] == scope) & (df["split"] == "test") & (df["method"] == "ae_significance_head")]
            if not sub.empty:
                row = sub.iloc[0].to_dict()
                row["model"] = label
                combined.append(row)
        if combined:
            comp = pd.DataFrame(combined)
            fig, ax = plt.subplots(figsize=(3.4, 2.8))
            vals = comp["auprc_lift_over_random"].astype(float).to_numpy()
            ax.bar(comp["model"], vals, color=[split_color(x) for x in comp["model"]], edgecolor=TEXT_COLOR, linewidth=0.35)
            ax.axhline(1, color="#666666", linestyle="--", linewidth=0.8)
            ax.set_title("Real vs permuted label control")
            ax.set_ylabel("Test AUPRC lift")
            clean_axis(ax)
            bar_label(ax, vals, fmt="{:.2f}")
            savefig(fig, fig_dir / "permutation_control_lift.png")


def plot_calibration(cal: Optional[pd.DataFrame], fig_dir: Path) -> None:
    if cal is None or cal.empty:
        return
    g = cal[cal["scope"] == "heldout_input_entries"].copy()
    if g.empty:
        g = cal.copy()
    fig, ax = plt.subplots(figsize=(4.7, 3.6))
    for split, sub in g.groupby("split"):
        sub = sub.sort_values("mean_score")
        ax.plot(sub["mean_score"], sub["observed_positive_rate"], marker="o", markersize=3.2, linewidth=1.35, label=str(split), color=split_color(split))
    lim = float(np.nanmax([g["mean_score"].max(), g["observed_positive_rate"].max(), 0.1]))
    ax.plot([0, lim], [0, lim], color="#666666", linestyle="--", linewidth=0.8, label="ideal")
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Calibration by split")
    ax.text(0.03, 0.92, "Use for ranking;\nabsolute probability is not calibrated", transform=ax.transAxes, fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#B7C0C7", alpha=0.9))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    clean_axis(ax, "both")
    savefig(fig, fig_dir / "calibration_heldout.png")


def plot_decile_lift(decile: Optional[pd.DataFrame], fig_dir: Path) -> None:
    if decile is None or decile.empty:
        return
    g = decile[decile["scope"] == "heldout_input_entries"].copy()
    if g.empty:
        g = decile.copy()
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for split, sub in g.groupby("split"):
        sub = sub.sort_values("decile")
        ax.plot(sub["decile"], sub["lift_vs_random"], marker="o", markersize=3.5, linewidth=1.4, label=str(split), color=split_color(split))
        add_decile_value_labels(ax, sub, split)
    ax.set_xlabel("Predicted score decile (1 = highest)")
    ax.set_ylabel("Lift vs random")
    ax.set_title("Decile lift")
    ax.set_xticks(sorted(g["decile"].dropna().unique()))
    ax.axhline(1, color="#666666", linestyle="--", linewidth=0.8)
    yvals = pd.to_numeric(g["lift_vs_random"], errors="coerce").dropna()
    if not yvals.empty:
        ymin = min(0.0, float(yvals.min()))
        ymax = float(yvals.max())
        ax.set_ylim(ymin - 0.05, ymax + max((ymax - ymin) * 0.18, 0.25))
    ax.legend(frameon=False, ncol=3)
    clean_axis(ax, "both")
    savefig(fig, fig_dir / "decile_lift.png")


def plot_overfit(overfit: Optional[pd.DataFrame], fig_dir: Path) -> None:
    if overfit is None or overfit.empty:
        return
    metrics = overfit["metric"].astype(str).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4))
    x = np.arange(len(metrics))
    width = 0.25
    for i, split in enumerate(["train", "val", "test"]):
        if split in overfit.columns:
            vals = pd.to_numeric(overfit[split], errors="coerce").to_numpy(float)
            bars = axes[0].bar(x + (i - 1) * width, vals, width=width, label=split, color=split_color(split), edgecolor=TEXT_COLOR, linewidth=0.25)
            for bar, metric, value in zip(bars, metrics, vals):
                if not np.isfinite(value):
                    continue
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    value + max(abs(value) * 0.018, 0.035) + i * 0.09,
                    metric_value_label(metric, value),
                    ha="center",
                    va="bottom",
                    fontsize=5.7,
                    color=TEXT_COLOR,
                )
    bar_values = overfit[[c for c in ["train", "val", "test"] if c in overfit.columns]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if bar_values.size and np.isfinite(bar_values).any():
        axes[0].set_ylim(0, float(np.nanmax(bar_values)) * 1.23)
    axes[0].set_xticks(x, metrics, rotation=30, ha="right")
    axes[0].set_title("Heldout metrics by split")
    axes[0].legend(frameon=False)
    clean_axis(axes[0])
    gap_cols = [("train_minus_val", ACCENT_RED), ("val_minus_test", split_color("val"))]
    gap_width = 0.34
    for i, (col, color) in enumerate(gap_cols):
        if col in overfit.columns:
            vals = pd.to_numeric(overfit[col], errors="coerce").to_numpy(float)
            xpos = x + (i - 0.5) * gap_width
            bars = axes[1].bar(xpos, vals, width=gap_width, label=col, alpha=0.75, color=color)
            for bar, metric, value in zip(bars, metrics, vals):
                if not np.isfinite(value):
                    continue
                va = "bottom" if value >= 0 else "top"
                y_offset = max(abs(value) * 0.05, 0.0025)
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (y_offset if value >= 0 else -y_offset),
                    metric_value_label(metric, value, signed=True),
                    ha="center",
                    va=va,
                    fontsize=5.7,
                    color=TEXT_COLOR,
                )
    gap_values = overfit[[c for c, _ in gap_cols if c in overfit.columns]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if gap_values.size and np.isfinite(gap_values).any():
        ymin = float(np.nanmin(gap_values))
        ymax = float(np.nanmax(gap_values))
        pad = max((ymax - ymin) * 0.25, 0.015)
        axes[1].set_ylim(ymin - pad, ymax + pad)
    axes[1].axhline(0, color="#666666", linewidth=0.8)
    axes[1].set_xticks(x, metrics, rotation=30, ha="right")
    axes[1].set_title("Generalization gaps")
    axes[1].legend(frameon=False)
    clean_axis(axes[1])
    savefig(fig, fig_dir / "overfit_diagnostics.png")


def plot_cell_latent(cell_scores: Optional[pd.DataFrame], fig_dir: Path) -> None:
    if cell_scores is None or cell_scores.empty or not {"PC1", "PC2"}.issubset(cell_scores.columns):
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for split, sub in cell_scores.groupby("split"):
        axes[0].scatter(sub["PC1"], sub["PC2"], s=10, alpha=0.72, label=str(split), color=split_color(split), linewidths=0)
    axes[0].set_title("Cell latent PCA by split")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(frameon=False)
    clean_axis(axes[0], None)
    color_col = "predicted_significance_burden" if "predicted_significance_burden" in cell_scores.columns else "mean_pred_significant_probability"
    sc = axes[1].scatter(cell_scores["PC1"], cell_scores["PC2"], c=cell_scores[color_col], s=10, cmap="viridis", alpha=0.78, linewidths=0)
    axes[1].set_title(f"Cell latent PCA by {color_col.replace('_', ' ')}")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    clean_axis(axes[1], None)
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04, label=color_col.replace("_", " "))
    savefig(fig, fig_dir / "cell_latent_pca.png")


def plot_cell_site_distributions(cell_scores: Optional[pd.DataFrame], site_scores: Optional[pd.DataFrame], fig_dir: Path, max_site_scatter: int, random_state: int) -> None:
    if cell_scores is not None and not cell_scores.empty:
        cols = numeric_cols(cell_scores, ["n_observed_entries", "observed_significant_rate", "mean_pred_significant_probability", "predicted_significance_burden"])
        if cols:
            fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6))
            for ax, col in zip(axes.ravel(), cols):
                values = cell_scores[col].dropna()
                ax.hist(values, bins=min(34, max(12, int(np.sqrt(len(values))))), color=split_color("train"), alpha=0.88, edgecolor="white", linewidth=0.25)
                ax.set_title(col.replace("_", " "))
                clean_axis(ax)
            for ax in axes.ravel()[len(cols):]:
                ax.axis("off")
            savefig(fig, fig_dir / "cell_score_distributions.png")

    if site_scores is not None and not site_scores.empty:
        cols = numeric_cols(site_scores, ["n_entries", "observed_significant_rate", "mean_pred_significant_probability", "mean_pred_m6a_probability"])
        if cols:
            fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6))
            for ax, col in zip(axes.ravel(), cols):
                values = site_scores[col].dropna()
                ax.hist(values, bins=min(45, max(16, int(np.sqrt(len(values)) // 2))), color=split_color("val"), alpha=0.88, edgecolor="white", linewidth=0.2)
                if values.min() >= 0 and values.quantile(0.99) > max(values.quantile(0.50) * 8, 1e-9):
                    ax.set_yscale("log")
                    ax.set_ylabel("Count (log)")
                ax.set_title(col.replace("_", " "))
                clean_axis(ax)
            for ax in axes.ravel()[len(cols):]:
                ax.axis("off")
            savefig(fig, fig_dir / "site_score_distributions.png")

        if {"observed_significant_rate", "mean_pred_significant_probability"}.issubset(site_scores.columns):
            sample = site_scores.dropna(subset=["observed_significant_rate", "mean_pred_significant_probability"]).copy()
            if len(sample) > max_site_scatter:
                sample = sample.sample(max_site_scatter, random_state=random_state)
            fig, ax = plt.subplots(figsize=(4.7, 3.9))
            c = sample["n_entries"] if "n_entries" in sample.columns else None
            sc = ax.scatter(sample["observed_significant_rate"], sample["mean_pred_significant_probability"], c=c, s=8, alpha=0.52, cmap="viridis", linewidths=0)
            ax.set_xlabel("Observed significant rate")
            ax.set_ylabel("Mean predicted significant probability")
            ax.set_title("Site-level observed vs predicted significance")
            clean_axis(ax, "both")
            if c is not None:
                fig.colorbar(sc, ax=ax, label="n_entries")
            savefig(fig, fig_dir / "site_observed_vs_predicted.png")


def export_tables(step3_dir: Path, out_dir: Path, top_sites: int) -> Dict[str, object]:
    summary: Dict[str, object] = {"step3_dir": str(step3_dir), "outputs": {}}
    validation = read_csv_if_exists(step3_dir / "validation_decision.csv")
    model_summary = read_csv_if_exists(step3_dir / "model_validation_summary.csv")
    perm_summary = read_csv_if_exists(step3_dir / "permutation_control_summary.csv")
    site_scores = read_csv_if_exists(step3_dir / "site_scores.csv")
    baseline = read_csv_if_exists(step3_dir / "baseline_comparison.csv")

    key_rows = []
    for label, df in [("validation_decision", validation), ("model_validation_summary", model_summary)]:
        if df is not None and not df.empty:
            row = df.iloc[0].to_dict()
            row["source"] = label
            key_rows.append(row)
    if key_rows:
        key_df = pd.DataFrame(key_rows)
        key_path = out_dir / "key_validation_summary.csv"
        key_df.to_csv(key_path, index=False)
        summary["outputs"]["key_validation_summary"] = str(key_path)

    if baseline is not None and not baseline.empty:
        scope = "heldout_input_entries"
        g = baseline[(baseline["split"] == "test") & (baseline["scope"] == scope)].copy()
        if g.empty:
            g = baseline[baseline["split"] == "test"].copy()
        g = g.sort_values("auprc_lift_over_random", ascending=False)
        path = out_dir / "test_method_ranking.csv"
        g.to_csv(path, index=False)
        summary["outputs"]["test_method_ranking"] = str(path)

    if perm_summary is not None and not perm_summary.empty:
        path = out_dir / "permutation_summary.csv"
        perm_summary.to_csv(path, index=False)
        summary["outputs"]["permutation_summary"] = str(path)

    if site_scores is not None and not site_scores.empty:
        sort_col = "mean_pred_significant_probability"
        top = site_scores.sort_values(sort_col, ascending=False).head(top_sites)
        path = out_dir / f"top_{top_sites}_predicted_sites.csv"
        top.to_csv(path, index=False)
        summary["outputs"]["top_predicted_sites"] = str(path)

    if model_summary is not None and not model_summary.empty:
        row = model_summary.iloc[0].to_dict()
        for k, v in row.items():
            if isinstance(v, (np.integer, np.floating)):
                row[k] = float(v)
        summary["model_validation_summary"] = row

    summary_path = out_dir / "step3_visualization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def write_report(out_dir: Path, fig_dir: Path, summary: Dict[str, object]) -> None:
    model = summary.get("model_validation_summary", {})
    lines = [
        "# Step3 Visualization Report",
        "",
        f"Step3 input: `{summary.get('step3_dir', '')}`",
        "",
        "## Key Validation",
    ]
    if model:
        for key in [
            "status",
            "criterion_pass_count",
            "criterion_total",
            "test_ae_auprc",
            "test_ae_lift",
            "test_auroc",
            "permutation_test_lift",
            "best_epoch",
            "n_sites",
            "n_observed_entries",
        ]:
            if key in model:
                lines.append(f"- {key}: {model[key]}")
    else:
        lines.append("- No model_validation_summary.csv available.")
    lines += ["", "## Figures"]
    for fig in sorted(fig_dir.glob("*.png")):
        lines.append(f"- `figures/{fig.name}`")
    lines += ["", "## Tables"]
    for name, path in sorted(summary.get("outputs", {}).items()):
        rel = Path(path).name
        lines.append(f"- `{rel}` ({name})")
    (out_dir / "step3_visualization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = resolve_from_script(args.root)
    step3_dir = resolve_under_root(args.step3_dir, root, "step3-output")
    out_dir = resolve_under_root(args.outdir, root, "step3-visualization-output")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    real_hist = require_csv(step3_dir / "training_history.csv")
    split_summary = require_csv(step3_dir / "split_summary.csv")
    metrics = require_csv(step3_dir / "baseline_comparison.csv")
    perm_hist = read_csv_if_exists(step3_dir / "permuted_train_history.csv")
    perm_metrics = read_csv_if_exists(step3_dir / "permuted_baseline_comparison.csv")
    calibration = read_csv_if_exists(step3_dir / "calibration_table.csv")
    decile = read_csv_if_exists(step3_dir / "decile_lift_table.csv")
    overfit = read_csv_if_exists(step3_dir / "overfit_diagnostics.csv")
    cell_scores = read_csv_if_exists(step3_dir / "cell_scores.csv")
    site_scores = read_csv_if_exists(step3_dir / "site_scores.csv")
    model_summary = read_csv_if_exists(step3_dir / "model_validation_summary.csv")

    plot_training_loss_zoom(real_hist, fig_dir, model_summary)
    plot_training_curves(real_hist, perm_hist, fig_dir, model_summary)
    plot_split_summary(split_summary, fig_dir)
    plot_baseline_comparison(metrics, perm_metrics, fig_dir)
    plot_calibration(calibration, fig_dir)
    plot_decile_lift(decile, fig_dir)
    plot_overfit(overfit, fig_dir)
    plot_cell_latent(cell_scores, fig_dir)
    plot_cell_site_distributions(cell_scores, site_scores, fig_dir, args.max_site_scatter, args.random_state)

    summary = export_tables(step3_dir, out_dir, args.top_sites)
    write_report(out_dir, fig_dir, summary)
    print(f"[Step3 visualization] wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
