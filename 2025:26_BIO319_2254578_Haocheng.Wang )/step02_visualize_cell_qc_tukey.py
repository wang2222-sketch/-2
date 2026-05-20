#!/usr/bin/env python3
"""Visualize Step2 two-metric log10 Tukey cell QC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize Step2 two-metric log10 Tukey cell QC")
    p.add_argument("--step2-dir", default=str(script_dir() / "step2-output"), help="Step2 output directory")
    p.add_argument("--figdir", default=None, help="Figure output directory; default: <step2-dir>/figures")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--png", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pdf", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def load_qc(step2_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    before_path = step2_dir / "cell_qc_all_before_filter.csv"
    removed_path = step2_dir / "cell_qc_removed_low_observed_sites.csv"
    summary_path = step2_dir / "step02_summary.json"
    if not before_path.exists():
        raise FileNotFoundError(f"Missing {before_path}. Run Step2 first.")
    before = pd.read_csv(before_path)
    removed = pd.read_csv(removed_path) if removed_path.exists() else pd.DataFrame()
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return before, removed, summary


def lower_fence_log10(x: np.ndarray, k: float) -> dict:
    lx = np.log10(np.clip(np.asarray(x, dtype=float), 1.0, None))
    q1 = float(np.quantile(lx, 0.25))
    q3 = float(np.quantile(lx, 0.75))
    iqr = q3 - q1
    log_threshold = q1 - float(k) * iqr
    return {
        "q1_log10": q1,
        "q3_log10": q3,
        "iqr_log10": iqr,
        "threshold_log10": log_threshold,
        "threshold": 10.0 ** log_threshold,
    }


def qc_values(before: pd.DataFrame, removed: pd.DataFrame, summary: dict) -> dict:
    thresholds = summary.get("thresholds", {})
    k = float(thresholds.get("cell_filter_tukey_k", 1.5))
    obs = before["observed_sites"].to_numpy(float)
    total = before["total_coverage_sum"].to_numpy(float)
    obs_vals = lower_fence_log10(obs, k)
    total_vals = lower_fence_log10(total, k)
    obs_threshold = float(thresholds.get("cell_filter_observed_sites_threshold", obs_vals["threshold"]))
    total_threshold = float(thresholds.get("cell_filter_total_coverage_sum_threshold", total_vals["threshold"]))
    removed_mask = (obs < obs_threshold) & (total < total_threshold)
    removed_n = int(len(removed)) if not removed.empty else int(removed_mask.sum())
    return {
        "k": k,
        "n": int(len(before)),
        "removed_n": removed_n,
        "obs_threshold": obs_threshold,
        "total_threshold": total_threshold,
        **{f"obs_{k2}": v for k2, v in obs_vals.items()},
        **{f"total_{k2}": v for k2, v in total_vals.items()},
    }


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig: plt.Figure, figdir: Path, name: str, dpi: int, png: bool, pdf: bool) -> None:
    fig.tight_layout()
    if png:
        fig.savefig(figdir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    if pdf:
        fig.savefig(figdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def density_curve(ax: plt.Axes, values: np.ndarray, threshold: float, xlabel: str, title: str, k: float) -> None:
    x = np.asarray(values, dtype=float)
    kde = stats.gaussian_kde(x)
    grid = np.linspace(0, max(x) * 1.05, 800)
    y = kde(grid)
    ax.hist(x, bins=36, density=True, color="#d7dde8", edgecolor="white", alpha=0.85, label="Cells")
    ax.plot(grid, y, color="#203040", linewidth=2.2, label="KDE density curve")
    ax.fill_between(grid, 0, y, where=grid < threshold, color="#e45756", alpha=0.16, label="Below metric threshold")
    ax.axvline(threshold, color="#c5161d", linewidth=2.8, label=f"log10 Tukey k={k:.1f}")
    ax.annotate(
        f"{threshold:,.0f}",
        xy=(threshold, kde(threshold)[0]),
        xytext=(threshold + max(x) * 0.08, max(y) * 0.72),
        arrowprops={"arrowstyle": "->", "color": "#c5161d", "lw": 1.4},
        color="#8a0f14",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#c5161d", "alpha": 0.95},
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_xlim(0, max(x) * 1.05)
    ax.legend(frameon=True, loc="upper left")


def density_panel(before: pd.DataFrame, vals: dict, figdir: Path, dpi: int, png: bool, pdf: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    density_curve(
        axes[0],
        before["observed_sites"].to_numpy(float),
        vals["obs_threshold"],
        "Observed m6A sites per cell (Total >= 5)",
        "Observed-Site Breadth",
        vals["k"],
    )
    density_curve(
        axes[1],
        before["total_coverage_sum"].to_numpy(float),
        vals["total_threshold"],
        "Total m6A coverage per cell",
        "Total Coverage",
        vals["k"],
    )
    fig.suptitle(
        f"Two-Metric Cell QC: remove only cells below both red thresholds ({vals['removed_n']}/{vals['n']} cells)",
        y=1.02,
        fontsize=15,
    )
    save(fig, figdir, "figure01_two_metric_tukey_density", dpi, png, pdf)


def scatter_plot(before: pd.DataFrame, removed: pd.DataFrame, vals: dict, figdir: Path, dpi: int, png: bool, pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    is_removed = before["cell_id"].astype(str).isin(removed["cell_id"].astype(str)) if not removed.empty else np.zeros(len(before), dtype=bool)
    ax.scatter(before.loc[~is_removed, "observed_sites"], before.loc[~is_removed, "total_coverage_sum"], s=22, color="#9aa7b7", alpha=0.72, label="Retained")
    if is_removed.any():
        ax.scatter(before.loc[is_removed, "observed_sites"], before.loc[is_removed, "total_coverage_sum"], s=42, color="#c5161d", alpha=0.95, label="Removed by AND rule")
    ax.axvline(vals["obs_threshold"], color="#c5161d", linewidth=2.4, label=f"observed threshold = {vals['obs_threshold']:,.0f}")
    ax.axhline(vals["total_threshold"], color="#c5161d", linewidth=2.4, linestyle="--", label=f"coverage threshold = {vals['total_threshold']:,.0f}")
    ax.fill_between(
        [0, vals["obs_threshold"]],
        0,
        vals["total_threshold"],
        color="#e45756",
        alpha=0.12,
        label="Removed region: both low",
    )
    ax.set_title("Cells Removed Only When Both Coverage Metrics Are Low")
    ax.set_xlabel("Observed m6A sites per cell (Total >= 5)")
    ax.set_ylabel("Total m6A coverage per cell")
    ax.set_xlim(0, before["observed_sites"].max() * 1.05)
    ax.set_ylim(0, before["total_coverage_sum"].max() * 1.05)
    ax.legend(frameon=True, loc="upper left")
    save(fig, figdir, "figure02_two_metric_tukey_scatter", dpi, png, pdf)


def write_report(before: pd.DataFrame, removed: pd.DataFrame, vals: dict, figdir: Path, step2_dir: Path) -> None:
    lines = [
        "# Step2 Two-Metric Tukey Cell QC Visualization",
        "",
        "## Rule",
        "- Metrics: `observed_sites` and `total_coverage_sum`.",
        "- Transform: log10 before computing Tukey lower fences.",
        f"- Tukey multiplier: k = {vals['k']:.1f}.",
        f"- Observed-site threshold: {vals['obs_threshold']:,.2f}.",
        f"- Total-coverage threshold: {vals['total_threshold']:,.2f}.",
        f"- Removed cells: {vals['removed_n']}/{vals['n']} ({vals['removed_n'] / vals['n']:.1%}).",
        "- Removal rule: a cell is removed only if both metrics fall below their red threshold lines.",
        "",
        "## Interpretation",
        "The two red threshold lines mark the log10 Tukey lower fences. The scatter plot shows the actual removal region: only cells in the lower-left rectangle are removed. This avoids removing cells that have low observed-site breadth but adequate total m6A coverage.",
        "",
        "## Figures",
        f"- `{figdir / 'figure01_two_metric_tukey_density.png'}`",
        f"- `{figdir / 'figure02_two_metric_tukey_scatter.png'}`",
    ]
    if not removed.empty:
        lines.extend(["", "## Removed Cell IDs"])
        lines.extend([f"- `{v}`" for v in removed["cell_id"].astype(str).tolist()])
    (step2_dir / "step02_cell_qc_tukey_visualization_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    step2_dir = Path(args.step2_dir).resolve()
    figdir = Path(args.figdir).resolve() if args.figdir else step2_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    before, removed, summary = load_qc(step2_dir)
    vals = qc_values(before, removed, summary)
    set_style()
    density_panel(before, vals, figdir, args.dpi, args.png, args.pdf)
    scatter_plot(before, removed, vals, figdir, args.dpi, args.png, args.pdf)
    write_report(before, removed, vals, figdir, step2_dir)
    print(f"[Step2 visualization] wrote figures to {figdir}")
    print(
        f"[Step2 visualization] observed_threshold={vals['obs_threshold']:.2f}; "
        f"total_threshold={vals['total_threshold']:.2f}; removed={vals['removed_n']}/{vals['n']} cells"
    )


if __name__ == "__main__":
    main()
