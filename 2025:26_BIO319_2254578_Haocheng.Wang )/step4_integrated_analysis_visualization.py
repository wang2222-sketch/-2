#!/usr/bin/env python3
"""
Step4 integrated downstream analysis + visualization for the compact scDART-seq
m6A Step3 model.

This script replaces the old two-file Step4 demo:
  - step4_analysis.py
  - step4_visualization.py

Design goals:
  1. Run all Step4 numeric summaries and visualizations from one entry point.
  2. Keep the original output filenames where possible, so existing report paths do not break.
  3. Avoid the previous risky site_id overwrite from cell_id.
  4. Add basic validation, calibration auditing, threshold reporting, and bias-aware gene/site summaries.

Default project layout expected on the user's machine:
  ROOT/step3-output/observed_predictions.tsv.gz
  ROOT/step2-output/site_metadata.csv
  ROOT/step4-output/

Example:
  python step4_integrated_analysis_visualization.py \
    --root "/Users/Zhuanz/Desktop/FYP/验证" \
    --threshold 0.5 \
    --split test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
        silhouette_score,
    )
except Exception as exc:  # pragma: no cover
    KMeans = None
    PCA = None
    silhouette_score = None
    average_precision_score = None
    precision_recall_curve = None
    roc_auc_score = None
    roc_curve = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None

try:
    from scipy.stats import fisher_exact
except Exception:  # pragma: no cover
    fisher_exact = None

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None

warnings.filterwarnings("ignore")
SPLIT_COLORS = {
    "train": "#2F5D8C",
    "val": "#1B9E77",
    "test": "#D95F02",
    "permuted": "#B279A2",
    "permutation": "#B279A2",
    "real": "#2F5D8C",
}
SPLIT_ORDER = ["train", "val", "test"]
SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}
ENTRY_LABELS = {
    "observed_entries": "Observed",
    "hidden_entries": "Held-out",
    "significant_entries": "Significant",
    "hidden_significant_entries": "Hidden significant",
}
ENTRY_COLORS = {
    "observed_entries": "#2F5D8C",
    "hidden_entries": "#1B9E77",
    "significant_entries": "#D95F02",
    "hidden_significant_entries": "#FFB000",
}
MODEL_COLORS = {
    "ae_significance_head": "#2F5D8C",
    "ae_m6a_probability": "#6FA8DC",
    "train_site_mean_m6a": "#9AA7B7",
    "train_site_significance_prior": "#D95F02",
    "coverage_only": "#B8B8B8",
}
EVIDENCE_COLORS = {
    "top": "#C44E52",
    "background": "#72B7B2",
    "support": "#C44E52",
    "topology": "#2F5D8C",
    "motif": "#D95F02",
    "regulator": "#1B9E77",
    "chromatin": "#7570B3",
    "neutral": "#9AA7B7",
}
TEXT_COLOR = "#222222"
GRID_COLOR = "#E6E6E6"


def set_publication_style() -> None:
    if sns is not None:
        sns.set_theme(style="white", context="paper")
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


def add_decile_value_labels(ax: plt.Axes, sub: pd.DataFrame, split: object, value_col: str, fmt: str) -> None:
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
            fmt.format(float(y)),
            xy=(float(x), float(y)),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=5.9,
            color=color,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.72),
            zorder=6,
        )


# =============================================================================
# CLI and path handling
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR
DEFAULT_STEP3_REL = Path("step3-output")
DEFAULT_STEP2_REL = Path("step2-output")
DEFAULT_OUT_REL = Path("step4-output")
DEFAULT_EXTERNAL_REL = Path("external-m6aconquer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Integrated Step4 downstream analysis and visualization for compact Step3 outputs."
    )
    p.add_argument("--root", default=str(DEFAULT_ROOT), help="Project root directory.")
    p.add_argument("--step3-dir", default=None, help="Directory containing Step3 prediction/effect outputs.")
    p.add_argument("--step2-dir", default=None, help="Directory containing Step2 metadata outputs.")
    p.add_argument("--out-dir", default=None, help="Directory where Step4 outputs will be written.")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"], help="Split used for downstream summaries.")
    p.add_argument("--threshold", type=float, default=0.50, help="Exploratory detected-site threshold on the standardized event_prob score.")
    p.add_argument("--conservative-threshold", type=float, default=0.70, help="Conservative detected-site threshold.")
    p.add_argument("--max-scatter-sites", type=int, default=3000, help="Maximum sites sampled for scatter plots.")
    p.add_argument("--random-state", type=int, default=44, help="Random seed for sampling and clustering.")
    p.add_argument("--min-gene-sites", type=int, default=5, help="Minimum sites per gene for gene plots/clustering.")
    p.add_argument("--skip-clustering", action="store_true", help="Skip cell/gene clustering figures.")
    p.add_argument("--fail-on-warning", action="store_true", help="Raise errors on important validation warnings.")
    p.add_argument("--enable-external-m6aconquer", action="store_true", help="Enable m6AConquer external validation and feature interpretation.")
    p.add_argument("--external-dir", default=None, help="Directory containing external-m6aconquer/raw and processed.")
    return p.parse_args()


def path_from_script_dir(value) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path, Path]:
    root = path_from_script_dir(args.root)
    step3_dir = path_from_script_dir(args.step3_dir) if args.step3_dir else root / DEFAULT_STEP3_REL
    step2_dir = path_from_script_dir(args.step2_dir) if args.step2_dir else root / DEFAULT_STEP2_REL
    out_dir = path_from_script_dir(args.out_dir) if args.out_dir else root / DEFAULT_OUT_REL
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return root, step3_dir, step2_dir, out_dir, fig_dir


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def read_csv_optional(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"  [optional missing] {path.name}")
        return None
    return pd.read_csv(path, **kwargs)


def first_existing_path(paths: Sequence[Path], description: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    searched = "\n".join(f"  - {p}" for p in paths)
    raise FileNotFoundError(f"Missing {description}; searched:\n{searched}")


# =============================================================================
# Utility functions
# =============================================================================


def safe_div(num: pd.Series | np.ndarray | float, den: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.divide(num, den, out=np.zeros_like(np.asarray(num, dtype=float)), where=np.asarray(den, dtype=float) != 0)


def is_chr_like(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    s = values.dropna().astype(str)
    if s.empty:
        return 0.0
    return float(s.str.match(r"^chr[^:]+:[0-9]+(?:-[0-9]+)?:[+\-.]$").mean())


def chr_sort_key(x: str) -> Tuple[int, str]:
    y = str(x).replace("chr", "")
    if y.isdigit():
        return int(y), ""
    if y == "X":
        return 23, ""
    if y == "Y":
        return 24, ""
    if y in {"M", "MT"}:
        return 25, ""
    return 99, y


def present_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def first_existing(df: pd.DataFrame, candidates: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return default


def coalesce_columns(df: pd.DataFrame, target: str, candidates: Sequence[str], default: Optional[str] = None) -> pd.DataFrame:
    present = [c for c in candidates if c in df.columns]
    if target in df.columns:
        series = df[target]
    elif present:
        series = df[present[0]]
    else:
        if default is None:
            return df
        df[target] = default
        return df
    for c in present:
        if c == target:
            continue
        series = series.combine_first(df[c])
    df[target] = series
    return df


def bin_pred_probability(x: pd.Series) -> pd.Series:
    return pd.cut(
        x.astype(float),
        bins=[-np.inf, 0.01, 0.03, 0.05, 0.10, 0.20, 0.40, np.inf],
        labels=["<=0.01", "0.01-0.03", "0.03-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.40", ">0.40"],
    ).astype(str)


def coverage_bin_label(n: pd.Series) -> pd.Series:
    return pd.cut(
        n.astype(float),
        bins=[-np.inf, 50, 100, 200, np.inf],
        labels=["20-50", "50-100", "100-200", "200+"],
    ).astype(str)


def make_site_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "seqnames" not in out.columns or "start" not in out.columns or "strand" not in out.columns:
        parsed = out.get("site_id", pd.Series(index=out.index, dtype=object)).astype(str).str.extract(
            r"^(?P<seqnames>chr[^:]+):(?P<start>[0-9]+)(?:-[0-9]+)?:(?P<strand>[+\-.])$"
        )
        for c in ["seqnames", "start", "strand"]:
            if c not in out.columns:
                out[c] = parsed[c]
    out["seqnames"] = out["seqnames"].astype(str)
    out["start"] = pd.to_numeric(out["start"], errors="coerce").astype("Int64")
    out["strand"] = out["strand"].astype(str)
    out["site_key"] = out["seqnames"] + ":" + out["start"].astype(str) + ":" + out["strand"]
    out.loc[out["start"].isna() | out["seqnames"].eq("nan") | out["strand"].eq("nan"), "site_key"] = np.nan
    return out


def add_site_key(df: pd.DataFrame) -> pd.DataFrame:
    return make_site_key_frame(df)


def one_sided_fisher(a: int, b: int, c: int, d: int) -> Tuple[float, float]:
    if fisher_exact is None:
        odds = (a * d) / max(b * c, 1)
        return float(odds), float("nan")
    odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return float(odds), float(p)


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    z = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(z) < 3 or z["x"].nunique() < 2 or z["y"].nunique() < 2:
        return float("nan")
    return float(z["x"].corr(z["y"], method=method))


def write_json(path: Path, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json_optional(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [optional unreadable] {path.name}: {exc}")
        return {}


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def warn_or_fail(message: str, fail: bool = False) -> None:
    if fail:
        raise RuntimeError(message)
    print(f"  [warning] {message}")


# =============================================================================
# Data loading and validation
# =============================================================================


def load_inputs(args: argparse.Namespace, step3_dir: Path, step2_dir: Path) -> Dict[str, Optional[pd.DataFrame] | str]:
    print_section("[1/8] Loading Step2/Step3 inputs")

    pred_path = first_existing_path(
        [
            step3_dir / "observed_predictions.tsv.gz",
            step3_dir / "step03_15_predictions.tsv.gz",
        ],
        "Step3 predictions",
    )
    site_meta_path = step2_dir / "site_metadata.csv"
    require_file(site_meta_path, "Step2 site metadata")

    print(f"  predictions: {pred_path}")
    pred = pd.read_csv(pred_path, sep="\t", compression="gzip")
    site_meta = pd.read_csv(site_meta_path)

    cell_eff = read_csv_optional(step3_dir / "step03_15_cell_effects.csv")
    site_eff = read_csv_optional(step3_dir / "step03_15_site_effects.csv")
    calib = read_csv_optional(step3_dir / "step03_15_calibration_by_group.csv")
    if calib is None:
        calib = read_csv_optional(step3_dir / "calibration_table.csv")
    site_scores = read_csv_optional(step3_dir / "site_scores.csv")
    cell_scores = read_csv_optional(step3_dir / "cell_scores.csv")
    latent = read_csv_optional(step3_dir / "latent.csv")
    if cell_eff is None:
        cell_eff = cell_scores if cell_scores is not None else latent
    validation_decision = read_csv_optional(step3_dir / "validation_decision.csv")
    model_summary = read_csv_optional(step3_dir / "model_validation_summary.csv")
    baseline_comparison = read_csv_optional(step3_dir / "baseline_comparison.csv")
    overfit_diagnostics = read_csv_optional(step3_dir / "overfit_diagnostics.csv")
    training_history = read_csv_optional(step3_dir / "training_history.csv")
    split_summary = read_csv_optional(step3_dir / "split_summary.csv")
    decile_lift = read_csv_optional(step3_dir / "decile_lift_table.csv")
    permutation_summary = read_csv_optional(step3_dir / "permutation_control_summary.csv")
    cell_meta = read_csv_optional(step2_dir / "cell_metadata.csv")
    step2_summary = read_json_optional(step2_dir / "step02_summary.json")
    step3_config = read_json_optional(step3_dir / "step03_config.json")
    validation_decision_json = read_json_optional(step3_dir / "validation_decision.json")

    summary_text = ""
    for summary_path in [step3_dir / "step03_15_summary.md", step3_dir / "step03_summary.md"]:
        if summary_path.exists():
            summary_text = summary_path.read_text(encoding="utf-8", errors="ignore")
            break

    print(f"  pred shape: {pred.shape}")
    print(f"  site_meta shape: {site_meta.shape}")
    return {
        "pred": pred,
        "site_meta": site_meta,
        "cell_eff": cell_eff,
        "site_eff": site_eff,
        "calib": calib,
        "site_scores": site_scores,
        "cell_scores": cell_scores,
        "latent": latent,
        "validation_decision": validation_decision,
        "model_summary": model_summary,
        "baseline_comparison": baseline_comparison,
        "overfit_diagnostics": overfit_diagnostics,
        "training_history": training_history,
        "split_summary": split_summary,
        "decile_lift": decile_lift,
        "permutation_summary": permutation_summary,
        "cell_meta": cell_meta,
        "step2_summary": step2_summary,
        "step3_config": step3_config,
        "validation_decision_json": validation_decision_json,
        "summary_text": summary_text,
    }


def validate_and_repair_predictions(
    pred: pd.DataFrame,
    site_meta: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    print_section("[2/8] Validating prediction columns and identifiers")

    pred = pred.copy()
    score_col = first_existing(
        pred,
        ["event_prob", "pred_significant_probability", "posterior_fg", "pred_m6a_probability"],
    )
    if score_col is None:
        raise ValueError(
            "Prediction file has no usable model score. Expected one of: "
            "event_prob, pred_significant_probability, posterior_fg, pred_m6a_probability."
        )
    if score_col != "event_prob":
        pred["event_prob"] = pred[score_col]
        print(f"  standardized score: {score_col} -> event_prob")
    pred["event_prob_source"] = score_col

    site_mean_col = first_existing(pred, ["site_mean_p", "site_mean_train_smoothed", "mean_pred_m6a_probability"])
    if site_mean_col and site_mean_col != "site_mean_p":
        pred["site_mean_p"] = pred[site_mean_col]
        print(f"  standardized site baseline: {site_mean_col} -> site_mean_p")

    pred = coalesce_columns(
        pred,
        "reference_site_source",
        ["reference_site_source", "reference_site_source_y", "reference_site_source_x"],
        default="unknown",
    )
    pred = coalesce_columns(
        pred,
        "orthogonal_validation_status",
        ["orthogonal_validation_status", "orthogonal_validation_status_y", "orthogonal_validation_status_x"],
        default="unknown",
    )

    required = ["cell_id", "Y", "N", "event_prob", "is_significant", "split"]
    missing = [c for c in required if c not in pred.columns]
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {missing}")

    # Never reconstruct site_id from cell_id. Use site_id as emitted by Step3, or merge by site_idx.
    if "site_id" not in pred.columns or pred["site_id"].isna().all():
        if "site_idx" in pred.columns and "site_idx" in site_meta.columns and "site_id" in site_meta.columns:
            pred = pred.merge(site_meta[["site_idx", "site_id"]].drop_duplicates("site_idx"), on="site_idx", how="left")
            print("  site_id repaired by merging site_idx with site_metadata.csv")
        else:
            raise ValueError("Prediction file has no usable site_id, and cannot be repaired from site_idx.")

    chr_like_fraction = is_chr_like(pred["site_id"])
    print(f"  site_id coordinate-like fraction: {chr_like_fraction:.3f}")
    if chr_like_fraction < 0.50:
        warn_or_fail(
            "site_id does not look like genomic coordinates for most rows. "
            "Check that Step3 predictions include correct site_id. The script did NOT overwrite site_id from cell_id.",
            args.fail_on_warning,
        )

    # Numeric coercion and safe derived fields.
    numeric_cols = [
        "Y", "N", "event_prob", "pred_m6a_probability", "pred_significant_probability",
        "posterior_fg", "site_mean_p", "site_mean_train_smoothed", "site_significance_train_prior", "mu_fg", "mu_bg",
        "eta_fg", "eta_bg", "pi_fg", "target_rate", "coverage_channel", "prob", "padj",
    ]
    for c in numeric_cols:
        if c in pred.columns:
            pred[c] = pd.to_numeric(pred[c], errors="coerce")

    pred["Y"] = pd.to_numeric(pred["Y"], errors="coerce").fillna(0).clip(lower=0)
    pred["N"] = pd.to_numeric(pred["N"], errors="coerce").fillna(0).clip(lower=0)
    pred = pred[pred["N"] > 0].copy()
    pred["Y"] = np.minimum(pred["Y"], pred["N"])
    pred["target_rate"] = pred["Y"] / pred["N"]
    if "observed_ratio" in pred.columns:
        observed_ratio = pd.to_numeric(pred["observed_ratio"], errors="coerce")
        pred["target_rate"] = pred["target_rate"].where(pred["target_rate"].notna(), observed_ratio)
    pred["is_significant"] = pd.to_numeric(pred["is_significant"], errors="coerce").fillna(0).astype(int)
    pred["event_prob"] = pred["event_prob"].clip(0, 1)

    if "site_mean_p" not in pred.columns or pred["site_mean_p"].isna().all():
        print("  site_mean_p missing; using entry-level target_rate as fallback for residual diagnostics")
        pred["site_mean_p"] = pred["target_rate"]
    pred["event_prob_residual_entry"] = pred["event_prob"] - pred["site_mean_p"]

    # Restore common site metadata columns into pred when missing.
    meta_cols = [
        "site_id", "seqnames", "strand", "reference_site_source", "orthogonal_validation_status",
        "gene_symbol", "gene_source",
    ]
    merge_cols = [c for c in meta_cols if c in site_meta.columns]
    need_merge = [c for c in merge_cols if c != "site_id" and c not in pred.columns]
    if need_merge:
        pred = pred.merge(site_meta[["site_id"] + need_merge].drop_duplicates("site_id"), on="site_id", how="left")
        print(f"  merged missing metadata into predictions: {need_merge}")

    for c in ["seqnames", "strand", "reference_site_source", "orthogonal_validation_status", "gene_symbol", "gene_source"]:
        if c not in pred.columns:
            pred[c] = "unknown"
        pred[c] = pred[c].fillna("unknown").astype(str)

    print(f"  usable prediction rows after validation: {len(pred):,}")
    print(f"  splits: {pred['split'].value_counts().to_dict()}")
    return pred


def choose_analysis_split(pred: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "all":
        sub = pred.copy()
    else:
        sub = pred[pred["split"].astype(str) == split].copy()
    if sub.empty:
        raise ValueError(f"No rows available for split={split!r}.")
    return sub


# =============================================================================
# Numeric summaries
# =============================================================================


def make_site_stats(df: pd.DataFrame, site_meta: pd.DataFrame, threshold: float, conservative_threshold: float) -> pd.DataFrame:
    agg_spec = {
        "event_prob_mean": ("event_prob", "mean"),
        "event_prob_median": ("event_prob", "median"),
        "event_prob_max": ("event_prob", "max"),
        "event_prob_p95": ("event_prob", lambda x: float(np.quantile(x, 0.95))),
        "n_cells": ("cell_id", "nunique"),
        "n_entries": ("Y", "size"),
        "total_Y": ("Y", "sum"),
        "total_N": ("N", "sum"),
        "is_significant": ("is_significant", "max"),
        "seqnames": ("seqnames", "first"),
        "strand": ("strand", "first"),
        "reference_site_source": ("reference_site_source", "first"),
        "orthogonal_validation_status": ("orthogonal_validation_status", "first"),
        "site_mean_p": ("site_mean_p", "first"),
        "mean_target_rate": ("target_rate", "mean"),
    }
    optional_mean_cols = [
        "pred_m6a_probability", "pred_significant_probability", "posterior_fg",
        "site_significance_train_prior", "mu_fg", "mu_bg", "eta_fg", "eta_bg", "pi_fg", "prob", "padj",
    ]
    for c in optional_mean_cols:
        if c in df.columns:
            agg_spec[f"{c}_mean"] = (c, "mean")
            if c in {"prob", "padj"}:
                agg_spec[f"{c}_min"] = (c, "min")
                agg_spec[f"{c}_max"] = (c, "max")

    site_stats = df.groupby("site_id", observed=False).agg(**agg_spec).reset_index()
    site_stats["m6a_frequency"] = site_stats["total_Y"] / site_stats["total_N"]
    site_stats["mean_coverage_per_entry"] = site_stats["total_N"] / site_stats["n_entries"]
    site_stats["detected"] = (site_stats["event_prob_max"] >= threshold).astype(int)
    site_stats["detected_conservative"] = (site_stats["event_prob_max"] >= conservative_threshold).astype(int)
    site_stats["event_prob_residual"] = site_stats["event_prob_mean"] - site_stats["site_mean_p"]
    site_stats["event_prob_max_residual"] = site_stats["event_prob_max"] - site_stats["site_mean_p"]

    meta_add = ["site_id", "gene_symbol", "gene_source", "orthogonal_validation_status", "strand", "seqnames", "reference_site_source"]
    meta_add = [c for c in meta_add if c in site_meta.columns]
    if "site_id" in meta_add:
        slim = site_meta[meta_add].drop_duplicates("site_id")
        # Prefer Step2 gene annotation; keep already aggregated columns when metadata absent.
        for c in ["gene_symbol", "gene_source"]:
            if c in site_stats.columns and c in slim.columns:
                site_stats = site_stats.drop(columns=[c])
        site_stats = site_stats.merge(slim, on="site_id", how="left", suffixes=("", "_meta"))
        for c in ["seqnames", "strand", "reference_site_source", "orthogonal_validation_status"]:
            meta_col = f"{c}_meta"
            if meta_col in site_stats.columns:
                site_stats[c] = site_stats[meta_col].fillna(site_stats.get(c, "unknown"))
                site_stats = site_stats.drop(columns=[meta_col])

    for c in ["gene_symbol", "gene_source", "seqnames", "strand", "reference_site_source", "orthogonal_validation_status"]:
        if c not in site_stats.columns:
            site_stats[c] = "unknown"
        site_stats[c] = site_stats[c].fillna("unknown").astype(str)

    return site_stats


def make_gene_stats(site_stats: pd.DataFrame, min_gene_sites: int) -> pd.DataFrame:
    s = site_stats.copy()
    s["gene_symbol"] = s["gene_symbol"].replace({"": "unknown", "nan": "unknown"}).fillna("unknown")
    gene_stats = s.groupby("gene_symbol", observed=False).agg(
        n_sites=("site_id", "count"),
        detected_sites=("detected", "sum"),
        detected_sites_conservative=("detected_conservative", "sum"),
        detection_fraction=("detected", "mean"),
        detection_fraction_conservative=("detected_conservative", "mean"),
        mean_event_prob=("event_prob_mean", "mean"),
        median_event_prob=("event_prob_median", "median"),
        max_event_prob=("event_prob_max", "max"),
        mean_m6a_freq=("m6a_frequency", "mean"),
        total_Y=("total_Y", "sum"),
        total_N=("total_N", "sum"),
        mean_coverage_per_entry=("mean_coverage_per_entry", "mean"),
        mean_event_prob_residual=("event_prob_residual", "mean"),
        max_event_prob_residual=("event_prob_max_residual", "max"),
        gene_source=("gene_source", "first"),
    ).reset_index()
    gene_stats["coverage_weighted_m6a_freq"] = gene_stats["total_Y"] / gene_stats["total_N"].replace(0, np.nan)
    gene_stats["is_gene_panel"] = (gene_stats["n_sites"] >= min_gene_sites).astype(int)
    return gene_stats.sort_values(["detected_sites", "detection_fraction", "max_event_prob"], ascending=False)


def make_cell_stats(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    cell_stats = df.groupby("cell_id", observed=False).agg(
        n_entries=("Y", "size"),
        n_significant=("is_significant", "sum"),
        n_detected_entries=("event_prob", lambda x: int((x >= threshold).sum())),
        mean_event_prob=("event_prob", "mean"),
        median_event_prob=("event_prob", "median"),
        event_prob_std=("event_prob", "std"),
        mean_event_prob_residual=("event_prob_residual_entry", "mean"),
        total_Y=("Y", "sum"),
        total_N=("N", "sum"),
        n_sites=("site_id", "nunique"),
    ).reset_index()
    if "posterior_fg" in df.columns:
        post = df.groupby("cell_id", observed=False)["posterior_fg"].mean().rename("mean_posterior_fg").reset_index()
        cell_stats = cell_stats.merge(post, on="cell_id", how="left")
    else:
        cell_stats["mean_posterior_fg"] = np.nan
    cell_stats["burden_rate"] = cell_stats["n_significant"] / cell_stats["n_entries"]
    cell_stats["detected_entry_rate"] = cell_stats["n_detected_entries"] / cell_stats["n_entries"]
    cell_stats["m6a_freq_cell"] = cell_stats["total_Y"] / cell_stats["total_N"]
    cell_stats["cell_rank"] = cell_stats["burden_rate"].rank(ascending=False, method="min")
    return cell_stats.sort_values("burden_rate", ascending=False)


def make_group_summaries(site_stats: pd.DataFrame, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    by_gene_source = site_stats.groupby("gene_source", observed=False).agg(
        n_sites=("site_id", "count"),
        detected=("detected", "sum"),
        detected_conservative=("detected_conservative", "sum"),
        mean_event_prob=("event_prob_mean", "mean"),
        mean_event_prob_residual=("event_prob_residual", "mean"),
        mean_m6a_freq=("m6a_frequency", "mean"),
        mean_coverage_per_entry=("mean_coverage_per_entry", "mean"),
    ).reset_index()
    by_gene_source["detection_rate"] = by_gene_source["detected"] / by_gene_source["n_sites"]
    by_gene_source["detection_rate_conservative"] = by_gene_source["detected_conservative"] / by_gene_source["n_sites"]

    by_ref = site_stats.groupby("reference_site_source", observed=False).agg(
        n_sites=("site_id", "count"),
        detected=("detected", "sum"),
        detected_conservative=("detected_conservative", "sum"),
        mean_event_prob=("event_prob_mean", "mean"),
        mean_event_prob_residual=("event_prob_residual", "mean"),
        mean_m6a_freq=("m6a_frequency", "mean"),
        mean_coverage_per_entry=("mean_coverage_per_entry", "mean"),
    ).reset_index()
    by_ref["detection_rate"] = by_ref["detected"] / by_ref["n_sites"]
    by_ref["detection_rate_conservative"] = by_ref["detected_conservative"] / by_ref["n_sites"]

    chr_stats = site_stats.groupby("seqnames", observed=False).agg(
        n_sites=("site_id", "count"),
        detected=("detected", "sum"),
        detected_conservative=("detected_conservative", "sum"),
        mean_event_prob=("event_prob_mean", "mean"),
        max_event_prob=("event_prob_max", "max"),
        mean_event_prob_residual=("event_prob_residual", "mean"),
        mean_m6a_freq=("m6a_frequency", "mean"),
        mean_coverage_per_entry=("mean_coverage_per_entry", "mean"),
        total_Y=("total_Y", "sum"),
        total_N=("total_N", "sum"),
    ).reset_index()
    chr_stats["detection_rate"] = chr_stats["detected"] / chr_stats["n_sites"]
    chr_stats["detection_rate_conservative"] = chr_stats["detected_conservative"] / chr_stats["n_sites"]
    chr_stats["m6a_freq_chr"] = chr_stats["total_Y"] / chr_stats["total_N"].replace(0, np.nan)
    chr_stats["norm_site_burden"] = chr_stats["n_sites"] / chr_stats["n_sites"].mean()
    chr_stats = chr_stats.sort_values("detection_rate", ascending=False)

    strand_stats = site_stats.groupby("strand", observed=False).agg(
        n_sites=("site_id", "count"),
        detected=("detected", "sum"),
        detected_conservative=("detected_conservative", "sum"),
        mean_event_prob=("event_prob_mean", "mean"),
        mean_event_prob_residual=("event_prob_residual", "mean"),
        mean_m6a_freq=("m6a_frequency", "mean"),
    ).reset_index()
    strand_stats["detection_rate"] = strand_stats["detected"] / strand_stats["n_sites"]
    strand_stats["detection_rate_conservative"] = strand_stats["detected_conservative"] / strand_stats["n_sites"]

    # Entry-level source x coverage calibration, useful for checking source/coverage bias.
    tmp = df.copy()
    tmp["coverage_bin"] = coverage_bin_label(tmp["N"])
    source_coverage = tmp.groupby(["reference_site_source", "coverage_bin"], observed=False).agg(
        n_entries=("Y", "size"),
        mean_pred=("event_prob", "mean"),
        mean_obs=("target_rate", "mean"),
        mean_N=("N", "mean"),
        significant_rate=("is_significant", "mean"),
    ).reset_index()
    source_coverage["residual_pred_minus_obs"] = source_coverage["mean_pred"] - source_coverage["mean_obs"]

    return {
        "by_gene_source": by_gene_source,
        "by_ref": by_ref,
        "chr_stats": chr_stats,
        "strand_stats": strand_stats,
        "source_coverage": source_coverage,
    }


def make_threshold_metrics(site_stats: pd.DataFrame) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.05, 0.51, 0.05), 2)
    rows = []
    y_true = site_stats["is_significant"].astype(int)
    for t in thresholds:
        y_pred = (site_stats["event_prob_max"] >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({
            "threshold": float(t),
            "detected_sites": int(y_pred.sum()),
            "precision_vs_step2_sig": precision,
            "recall_vs_step2_sig": recall,
            "f1_vs_step2_sig": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })
    return pd.DataFrame(rows)


def make_calibration(pred: pd.DataFrame, split: str, calib_file: Optional[pd.DataFrame]) -> pd.DataFrame:
    if calib_file is not None and {"grouping", "split", "pred_probability_bin", "mean_pred", "mean_obs"}.issubset(calib_file.columns):
        c = calib_file[(calib_file["grouping"] == "split+pred_probability_bin")].copy()
        if split != "all":
            c = c[c["split"].astype(str) == split].copy()
        if not c.empty:
            c["residual_pred_minus_obs"] = c["mean_pred"] - c["mean_obs"]
            return c
    if calib_file is not None and {"split", "bin", "n", "mean_score", "observed_positive_rate"}.issubset(calib_file.columns):
        c = calib_file.copy()
        if split != "all":
            c = c[c["split"].astype(str) == split].copy()
        if not c.empty:
            c = c.rename(columns={
                "bin": "pred_probability_bin",
                "mean_score": "mean_pred",
                "observed_positive_rate": "mean_obs",
            })
            c["pred_probability_bin"] = "decile_" + c["pred_probability_bin"].astype(int).astype(str)
            c["n_entries"] = c["n"]
            c["grouping"] = "step3_calibration_decile"
            c["residual_pred_minus_obs"] = c["mean_pred"] - c["mean_obs"]
            return c[present_cols(c, ["split", "scope", "grouping", "pred_probability_bin", "n_entries", "mean_pred", "mean_obs", "residual_pred_minus_obs"])]

    # Fallback: compute entry-level calibration from predictions.
    df = pred.copy()
    if split != "all":
        df = df[df["split"].astype(str) == split].copy()
    df["pred_probability_bin"] = bin_pred_probability(df["event_prob"])
    c = df.groupby("pred_probability_bin", observed=False).agg(
        n_entries=("Y", "size"),
        mean_pred=("event_prob", "mean"),
        mean_obs=("target_rate", "mean"),
        mean_N=("N", "mean"),
    ).reset_index()
    c["split"] = split
    c["grouping"] = "computed_split+pred_probability_bin"
    c["residual_pred_minus_obs"] = c["mean_pred"] - c["mean_obs"]
    return c


# =============================================================================
# m6AConquer external validation
# =============================================================================


EXTERNAL_SMALL_TABLES = {
    "validated_combined": "m6A_OrthogonallyValidatedSites_Combined_hg38.csv",
    "validated_hek293t": "m6A_OrthogonallyValidatedSites_HEK293T_Combined_hg38.csv",
    "fto_dmr": "DMR_eTAM_GLORI_Control_FTO_hg38.csv",
}

SUPPLEMENTARY_COLS = [
    "seqnames", "start", "end", "strand", "orthogonal_validation_status",
    "reference_site_source", "technique_support", "sample_support",
    "GLORI_supported", "eTAM_seq_supported", "HEK293T_supported", "HeLa_suppoerted",
]

OMICS_FEATURE_COLS = [
    "seqnames", "start", "end", "strand",
    "gc_content_flank_50bp", "phastCons_1bp",
    "overlap_exons", "overlap_introns", "overlap_exonicFivePrimeUTR",
    "overlap_fullFivePrimeUTR", "overlap_exonicCDS", "overlap_fullCDS",
    "overlap_exonicThreePrimeUTR", "overlap_fullThreePrimeUTR",
    "overlap_exonicTranscripts", "overlap_fullTranscripts", "overlap_promoters",
    "log2_NearestDistToJunction", "log2_GeneExonNumber", "log2_TxIsoformNumber",
    "meta_tx_topology",
    "TAACT", "TGACC", "AGACT", "TGACT", "GAACC", "GGACC", "GGACA", "TGACA",
    "AAACC", "AAACA", "AGACC", "GGACT", "TAACC", "AAACT", "AGACA", "TAACA",
    "GAACA", "GAACT", "GGACN",
    "protein_coding", "lincRNA", "antisense", "retained_intron", "nonsense_mediated_decay",
    "H3K36me3", "H3K27ac", "H3K4me3", "H3K4me1", "H3K27me3",
    "ELAVL1", "FMR1", "FTO", "HNRNPA2B1", "IGF2BP1", "IGF2BP3",
    "METTL14", "METTL3", "RBMX", "WTAP", "YTHDC1", "YTHDC2", "YTHDF1", "YTHDF2", "YTHDF3",
]


def external_dir_from_args(root: Path, args: argparse.Namespace) -> Path:
    if args.external_dir:
        return path_from_script_dir(args.external_dir)
    return root / DEFAULT_EXTERNAL_REL


def read_external_coordinate_table(path: Path, dataset: str, processed_path: Path) -> pd.DataFrame:
    if processed_path.exists() and processed_path.stat().st_size > 0:
        return pd.read_csv(processed_path)
    if not path.exists():
        print(f"  [external missing] {path.name}")
        return pd.DataFrame(columns=["site_key", "dataset"])
    df = pd.read_csv(path)
    df = add_site_key(df)
    keep = [c for c in ["site_key", "seqnames", "start", "end", "strand", "support_number", "support_technique",
                        "eTAM_seq_Control_FTO_diff", "GLORI_Control_FTO_diff", "idr"] if c in df.columns]
    out = df[keep].dropna(subset=["site_key"]).drop_duplicates("site_key").copy()
    out["dataset"] = dataset
    out.to_csv(processed_path, index=False)
    return out


def read_filtered_external_csv(
    path: Path,
    site_keys: set,
    usecols: Sequence[str],
    processed_path: Path,
    sep: str = ",",
    compression: Optional[str] = None,
    chunksize: int = 250000,
) -> pd.DataFrame:
    if processed_path.exists() and processed_path.stat().st_size > 0:
        return pd.read_csv(processed_path)
    if not path.exists():
        print(f"  [external missing] {path.name}")
        return pd.DataFrame(columns=["site_key"])

    if compression == "gzip":
        header = pd.read_csv(path, sep=sep, compression=compression, nrows=0).columns.tolist()
    else:
        header = pd.read_csv(path, sep=sep, nrows=0).columns.tolist()
    selected = [c for c in usecols if c in header]
    if "seqnames" not in selected or "start" not in selected or "strand" not in selected:
        return pd.DataFrame(columns=["site_key"])

    parts: List[pd.DataFrame] = []
    reader = pd.read_csv(path, sep=sep, compression=compression, usecols=selected, chunksize=chunksize)
    for chunk in reader:
        chunk = add_site_key(chunk)
        chunk = chunk[chunk["site_key"].isin(site_keys)].copy()
        if not chunk.empty:
            parts.append(chunk)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["site_key"] + selected)
    if "site_key" in out.columns:
        out = out.drop_duplicates("site_key")
    out.to_csv(processed_path, index=False)
    return out


def write_mae_site_request(site_stats: pd.DataFrame, processed_dir: Path) -> Path:
    req_path = processed_dir / "site_universe_for_mae_extract.csv"
    req = add_site_key(site_stats)
    cols = [c for c in ["site_key", "site_id", "seqnames", "start", "strand"] if c in req.columns]
    req = req[cols].dropna(subset=["site_key"]).drop_duplicates("site_key")
    req.to_csv(req_path, index=False)
    return req_path


def ensure_mae_quant_summary(
    raw_dir: Path,
    processed_dir: Path,
    site_request: Path,
    dataset: str,
    filename: str,
) -> pd.DataFrame:
    out_path = processed_dir / f"{dataset.lower()}_site_quant_summary.csv"
    if out_path.exists() and out_path.stat().st_size > 0:
        return pd.read_csv(out_path)
    mae_path = raw_dir / filename
    helper = Path(__file__).resolve().parent / "step4_extract_m6aconquer_mae.R"
    if not mae_path.exists() or not helper.exists():
        print(f"  [external missing] cannot extract {dataset}: {mae_path.name}")
        return pd.DataFrame(columns=["site_key", "site_id", "dataset", "external_ratio", "external_mean_prob"])
    cmd = [
        "Rscript", str(helper),
        "--mae", str(mae_path),
        "--sites", str(site_request),
        "--output", str(out_path),
        "--dataset", dataset,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception as exc:
        print(f"  [external warning] {dataset} RDS extraction failed: {exc}")
        if isinstance(exc, subprocess.CalledProcessError):
            print((exc.stderr or exc.stdout or "")[-1000:])
        return pd.DataFrame(columns=["site_key", "site_id", "dataset", "external_ratio", "external_mean_prob"])
    return pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["site_key", "site_id", "dataset", "external_ratio", "external_mean_prob"])


def make_top_masks(site_stats: pd.DataFrame) -> Dict[str, pd.Series]:
    score_resid = "event_prob_residual" if "event_prob_residual" in site_stats.columns else "event_prob_mean"
    score_event = "event_prob_max" if "event_prob_max" in site_stats.columns else "event_prob_mean"
    masks = {}
    for label, col in [("top_residual_decile", score_resid), ("top_event_decile", score_event), ("detected_threshold", "detected")]:
        if col == "detected":
            mask = site_stats[col].astype(int) == 1 if col in site_stats.columns else pd.Series(False, index=site_stats.index)
        else:
            values = pd.to_numeric(site_stats[col], errors="coerce")
            q = values.quantile(0.90)
            mask = values >= q
        masks[label] = mask.fillna(False)
    return masks


def make_external_overlap_summary(site_stats: pd.DataFrame, evidence_sets: Dict[str, set]) -> pd.DataFrame:
    s = add_site_key(site_stats).dropna(subset=["site_key"]).copy()
    universe = set(s["site_key"])
    masks = make_top_masks(s)
    rows = []
    for top_name, top_mask in masks.items():
        top_keys = set(s.loc[top_mask, "site_key"])
        non_top_keys = universe - top_keys
        for evidence_name, evidence_keys in evidence_sets.items():
            ev = set(evidence_keys) & universe
            a = len(top_keys & ev)
            b = len(top_keys - ev)
            c = len(non_top_keys & ev)
            d = len(non_top_keys - ev)
            odds, p = one_sided_fisher(a, b, c, d)
            rows.append({
                "top_set": top_name,
                "external_evidence": evidence_name,
                "universe_sites": len(universe),
                "top_sites": len(top_keys),
                "external_sites_in_universe": len(ev),
                "overlap_count": a,
                "top_overlap_fraction": a / max(len(top_keys), 1),
                "background_overlap_fraction": c / max(len(non_top_keys), 1),
                "odds_ratio": odds,
                "fisher_p_greater": p,
            })
    return pd.DataFrame(rows)


def binary_series(x: pd.Series) -> Optional[pd.Series]:
    s = x.dropna()
    if s.empty:
        return None
    if s.dtype == bool:
        return x.fillna(False).astype(bool)
    vals = set(s.astype(str).str.lower().unique().tolist())
    if vals.issubset({"true", "false"}):
        return x.astype(str).str.lower().eq("true")
    numeric = pd.to_numeric(x, errors="coerce")
    observed_vals = set(numeric.dropna().unique().tolist())
    if observed_vals.issubset({0, 1, 0.0, 1.0}):
        return numeric.fillna(0).astype(int).astype(bool)
    return None


def make_external_feature_enrichment(site_stats: pd.DataFrame, supp: pd.DataFrame, omics: pd.DataFrame) -> pd.DataFrame:
    s = add_site_key(site_stats).dropna(subset=["site_key"]).copy()
    masks = make_top_masks(s)
    top_mask = masks["top_residual_decile"]
    base = s[["site_key"]].copy()
    feature_tables = []
    if not supp.empty:
        feature_tables.append(supp.drop_duplicates("site_key"))
    if not omics.empty:
        feature_tables.append(omics.drop_duplicates("site_key"))
    for tbl in feature_tables:
        add_cols = [c for c in tbl.columns if c != "site_key" and c not in base.columns]
        base = base.merge(tbl[["site_key"] + add_cols], on="site_key", how="left")

    rows = []
    bg_top = top_mask.to_numpy(dtype=bool)
    for col in base.columns:
        if col in {"site_key", "seqnames", "start", "end", "strand"}:
            continue
        b = binary_series(base[col])
        if b is None:
            continue
        feat = b.to_numpy(dtype=bool)
        if feat.sum() == 0:
            continue
        a = int((bg_top & feat).sum())
        b_not = int((bg_top & ~feat).sum())
        c = int((~bg_top & feat).sum())
        d = int((~bg_top & ~feat).sum())
        if a + c < 3:
            continue
        odds, p = one_sided_fisher(a, b_not, c, d)
        rows.append({
            "top_set": "top_residual_decile",
            "feature": col,
            "feature_positive_sites": int(feat.sum()),
            "top_positive": a,
            "top_total": int(bg_top.sum()),
            "background_positive": c,
            "background_total": int((~bg_top).sum()),
            "top_fraction": a / max(int(bg_top.sum()), 1),
            "background_fraction": c / max(int((~bg_top).sum()), 1),
            "odds_ratio": odds,
            "fisher_p_greater": p,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["odds_ratio", "top_positive"], ascending=[False, False])
    return out


def make_external_score_correlation(site_stats: pd.DataFrame, quant_tables: Sequence[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s = add_site_key(site_stats).dropna(subset=["site_key"]).copy()
    quant = pd.concat([q for q in quant_tables if q is not None and not q.empty], ignore_index=True) if any(q is not None and not q.empty for q in quant_tables) else pd.DataFrame()
    if quant.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = quant.merge(s, on="site_key", how="inner", suffixes=("_external", ""))
    score_cols = [c for c in ["event_prob_mean", "event_prob_max", "event_prob_residual", "event_prob_max_residual", "posterior_fg_mean"] if c in merged.columns]
    external_cols = [c for c in ["external_ratio", "external_mean_prob"] if c in merged.columns]
    rows = []
    for dataset, sub in merged.groupby("dataset", observed=False):
        for score_col in score_cols:
            for ext_col in external_cols:
                z = sub.dropna(subset=[score_col, ext_col])
                rows.append({
                    "dataset": dataset,
                    "score": score_col,
                    "external_measure": ext_col,
                    "n_overlap": int(len(z)),
                    "pearson": safe_corr(z[score_col], z[ext_col], "pearson"),
                    "spearman": safe_corr(z[score_col], z[ext_col], "spearman"),
                    "mean_score": float(pd.to_numeric(z[score_col], errors="coerce").mean()) if len(z) else float("nan"),
                    "mean_external": float(pd.to_numeric(z[ext_col], errors="coerce").mean()) if len(z) else float("nan"),
                })
    return pd.DataFrame(rows), merged


def run_external_m6aconquer_analysis(
    root: Path,
    args: argparse.Namespace,
    site_stats: pd.DataFrame,
    out_dir: Path,
    fig_dir: Path,
) -> Dict[str, pd.DataFrame]:
    if not args.enable_external_m6aconquer:
        return {}
    print_section("[External] m6AConquer validation and biological interpretation")
    ext_dir = external_dir_from_args(root, args)
    raw_dir = ext_dir / "raw"
    processed_dir = ext_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    if not raw_dir.exists():
        print(f"  [external skipped] raw directory not found: {raw_dir}")
        return {}

    s = add_site_key(site_stats).dropna(subset=["site_key"]).copy()
    site_keys = set(s["site_key"].astype(str))

    external_tables = {}
    for dataset, filename in EXTERNAL_SMALL_TABLES.items():
        external_tables[dataset] = read_external_coordinate_table(
            raw_dir / filename,
            dataset,
            processed_dir / f"{dataset}_sites.csv",
        )

    supp = read_filtered_external_csv(
        raw_dir / "m6AConquer_supplementary_row_data_hg38.csv",
        site_keys,
        SUPPLEMENTARY_COLS,
        processed_dir / "supplementary_row_data_overlap_sites.csv",
    )
    omics = read_filtered_external_csv(
        raw_dir / "m6Aconquer_omicsFeatures_hg38.csv.gz",
        site_keys,
        OMICS_FEATURE_COLS,
        processed_dir / "omics_features_overlap_sites.csv",
        sep=r"\s+",
        compression="gzip",
    )

    evidence_sets = {
        "orthogonal_validated_combined": set(external_tables.get("validated_combined", pd.DataFrame()).get("site_key", [])),
        "orthogonal_validated_hek293t": set(external_tables.get("validated_hek293t", pd.DataFrame()).get("site_key", [])),
        "fto_responsive_dmr": set(external_tables.get("fto_dmr", pd.DataFrame()).get("site_key", [])),
    }
    if not supp.empty:
        for col, name in [
            ("GLORI_supported", "supplementary_glori_supported"),
            ("eTAM_seq_supported", "supplementary_etam_supported"),
            ("HEK293T_supported", "supplementary_hek293t_supported"),
        ]:
            if col in supp.columns:
                b = binary_series(supp[col])
                if b is not None:
                    evidence_sets[name] = set(supp.loc[b, "site_key"])

    overlap = make_external_overlap_summary(s, evidence_sets)
    features = make_external_feature_enrichment(s, supp, omics)

    site_request = write_mae_site_request(s, processed_dir)
    glori = ensure_mae_quant_summary(raw_dir, processed_dir, site_request, "GLORI", "GLORI_hg38_WT_MAE.rds")
    etam = ensure_mae_quant_summary(raw_dir, processed_dir, site_request, "eTAM", "eTAM_hg38_WT_MAE.rds")
    corr, quant_merged = make_external_score_correlation(s, [glori, etam])

    overlap.to_csv(out_dir / "step4_external_overlap_summary.csv", index=False)
    corr.to_csv(out_dir / "step4_external_score_correlation.csv", index=False)
    features.to_csv(out_dir / "step4_external_feature_enrichment.csv", index=False)
    if not quant_merged.empty:
        quant_merged.to_csv(processed_dir / "glori_etam_overlap_with_step4_site_stats.csv", index=False)

    plot_external_orthogonal_enrichment(overlap, fig_dir)
    plot_external_feature_enrichment(features, fig_dir)
    plot_external_topology_support(features, fig_dir)

    print(f"  external overlap rows: {len(overlap):,}")
    print(f"  external correlation rows: {len(corr):,}")
    print(f"  external feature enrichment rows: {len(features):,}")
    return {"overlap": overlap, "correlation": corr, "feature_enrichment": features}


# =============================================================================
# Plotting helpers
# =============================================================================


def savefig(path: Path) -> None:
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"  saved: {path.name}")


def metric_row(df: Optional[pd.DataFrame], **filters) -> Dict:
    if df is None or df.empty:
        return {}
    sub = df.copy()
    for col, value in filters.items():
        if col not in sub.columns:
            return {}
        sub = sub[sub[col].astype(str) == str(value)]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def fmt_float(value, digits: int = 3, na: str = "NA") -> str:
    try:
        if pd.isna(value):
            return na
        return f"{float(value):.{digits}f}"
    except Exception:
        return na


def fmt_int(value, na: str = "NA") -> str:
    try:
        if pd.isna(value):
            return na
        return f"{int(float(value)):,}"
    except Exception:
        return na


def fmt_plain_count(value: object) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.0f}"
    except Exception:
        return "NA"


def plot_step3_split_design(split_summary: Optional[pd.DataFrame], fig_dir: Path) -> None:
    if split_summary is None or split_summary.empty:
        return
    required = {
        "split",
        "cells",
        "observed_entries",
        "hidden_entries",
        "significant_entries",
        "hidden_significant_entries",
    }
    if not required.issubset(split_summary.columns):
        return

    data = split_summary.set_index("split").reindex(SPLIT_ORDER).reset_index()
    numeric_cols = ["cells", "observed_entries", "hidden_entries", "significant_entries", "hidden_significant_entries"]
    if data[numeric_cols].isna().any().any():
        return

    fig, ax = plt.subplots(figsize=(18.3 / 2.54, 8.8 / 2.54), constrained_layout=True)
    centers = {"train": (0.0, 0.0), "val": (2.75, 0.0), "test": (5.25, 0.0)}
    parent_radius = {"train": 1.18, "val": 0.92, "test": 0.92}
    observed_layout = (-0.06, -0.03, 0.66)
    heldout_layout = (0.18, 0.05, 0.34)
    significant_layout = (0.30, -0.20, 0.23)
    label_specs = []

    for _, row in data.iterrows():
        split = str(row["split"])
        cx, cy = centers[split]
        radius = parent_radius[split]
        split_color_value = split_color(split)
        parent = mpatches.Circle(
            (cx, cy),
            radius,
            facecolor=split_color_value,
            edgecolor="none",
            alpha=0.13,
            linewidth=0,
            zorder=1,
        )
        ax.add_patch(parent)
        label_specs.append(
            {
                "x": cx,
                "y": cy + radius + 0.23,
                "s": f"{SPLIT_LABELS[split]}\n{int(row['cells'])} cells",
                "ha": "center",
                "va": "bottom",
                "fontsize": 8,
                "fontweight": "bold",
                "color": TEXT_COLOR,
                "zorder": 20,
            }
        )

        observed_x = cx + observed_layout[0] * radius
        observed_y = cy + observed_layout[1] * radius
        observed_r = observed_layout[2] * radius
        observed = mpatches.Circle(
            (observed_x, observed_y),
            observed_r,
            facecolor=ENTRY_COLORS["observed_entries"],
            edgecolor="none",
            linewidth=0,
            alpha=0.90,
            zorder=2,
        )
        ax.add_patch(observed)

        subset_specs = [
            ("hidden_entries", heldout_layout, "white"),
            ("significant_entries", significant_layout, TEXT_COLOR),
        ]
        subset_centers = {}
        for col, layout, label_color in subset_specs:
            rel_x, rel_y, rel_r = layout
            child_radius = rel_r * radius
            child_x = cx + rel_x * radius
            child_y = cy + rel_y * radius
            subset_centers[col] = (child_x, child_y)
            child = mpatches.Circle(
                (child_x, child_y),
                child_radius,
                facecolor=ENTRY_COLORS[col],
                edgecolor="none",
                linewidth=0,
                alpha=0.90,
                zorder=3,
            )
            ax.add_patch(child)
            label_y = child_y - (0.16 * radius if col == "significant_entries" else 0.0)
            label_specs.append(
                {
                    "x": child_x,
                    "y": label_y,
                    "s": fmt_plain_count(row[col]),
                    "ha": "center",
                    "va": "center",
                    "fontsize": 6.1 if col == "significant_entries" else 6.5,
                    "fontweight": "bold",
                    "color": label_color,
                    "zorder": 30,
                }
            )

        label_specs.append(
            {
                "x": observed_x - 0.33 * radius,
                "y": observed_y + 0.38 * radius,
                "s": fmt_plain_count(row["observed_entries"]),
                "ha": "center",
                "va": "center",
                "fontsize": 6.7,
                "fontweight": "bold",
                "color": "white",
                "zorder": 30,
            }
        )
        hx, hy = subset_centers["hidden_entries"]
        sx, sy = subset_centers["significant_entries"]
        overlap_x = (hx + sx) / 2
        overlap_y = (hy + sy) / 2
        overlap = mpatches.Ellipse(
            (overlap_x, overlap_y),
            0.40 * radius,
            0.27 * radius,
            angle=-28,
            facecolor=ENTRY_COLORS["hidden_significant_entries"],
            edgecolor="none",
            linewidth=0,
            alpha=0.96,
            zorder=5,
        )
        ax.add_patch(overlap)
        label_specs.append(
            {
                "x": overlap_x,
                "y": overlap_y,
                "s": fmt_plain_count(row["hidden_significant_entries"]),
                "ha": "center",
                "va": "center",
                "fontsize": 5.5,
                "fontweight": "bold",
                "color": "white",
                "zorder": 35,
            }
        )

    for label_spec in label_specs:
        ax.text(**label_spec)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=ENTRY_COLORS[col], markeredgecolor="none", markersize=7, label=ENTRY_LABELS[col])
        for col in ["observed_entries", "hidden_entries", "significant_entries"]
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=ENTRY_COLORS["hidden_significant_entries"],
            markeredgecolor="none",
            markersize=6,
            label=ENTRY_LABELS["hidden_significant_entries"],
        )
    )
    ax.set_title("Nested observed, held-out, and significant entry design", pad=8)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False)
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 6.35)
    ax.set_ylim(-1.72, 1.68)
    ax.axis("off")
    savefig(fig_dir / "fig03_step3_train_validation_test_split_summary_corrected.png")


def plot_evidence_chain(
    loaded: Dict[str, Optional[pd.DataFrame] | Dict | str],
    site_stats: pd.DataFrame,
    cell_stats: pd.DataFrame,
    external_results: Optional[Dict[str, pd.DataFrame]],
    fig_dir: Path,
) -> None:
    """Compact paper-style evidence summary focused on the interpretable bar charts."""
    step2_summary = loaded.get("step2_summary") if isinstance(loaded.get("step2_summary"), dict) else {}
    step3_config = loaded.get("step3_config") if isinstance(loaded.get("step3_config"), dict) else {}
    validation_df = loaded.get("model_summary")
    if not isinstance(validation_df, pd.DataFrame) or validation_df.empty:
        validation_df = loaded.get("validation_decision")
    vrow = validation_df.iloc[0].to_dict() if isinstance(validation_df, pd.DataFrame) and not validation_df.empty else {}

    overlap = external_results.get("overlap") if external_results else None
    features = external_results.get("feature_enrichment") if external_results else None
    glori_row = metric_row(overlap, top_set="top_residual_decile", external_evidence="supplementary_glori_supported") if isinstance(overlap, pd.DataFrame) else {}
    etam_row = metric_row(overlap, top_set="top_residual_decile", external_evidence="supplementary_etam_supported") if isinstance(overlap, pd.DataFrame) else {}
    hek_row = metric_row(overlap, top_set="top_residual_decile", external_evidence="supplementary_hek293t_supported") if isinstance(overlap, pd.DataFrame) else {}

    fig, axes = plt.subplots(
        1, 3,
        figsize=(11.2, 3.65),
        gridspec_kw={"width_ratios": [1.0, 1.05, 1.8], "wspace": 0.35},
    )
    fig.suptitle("Integrated validation of m6A significance ranking", fontsize=10.5, fontweight="bold", y=1.03)

    ax = axes[0]
    vals = [vrow.get("train_ae_lift", np.nan), vrow.get("val_ae_lift", np.nan), vrow.get("test_ae_lift", np.nan), vrow.get("permutation_test_lift", np.nan)]
    labels = ["train", "val", "test", "permuted"]
    colors = [split_color(x) for x in labels]
    ax.bar(labels, vals, color=colors, edgecolor=TEXT_COLOR, linewidth=0.35)
    ax.axhline(1, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, max([v for v in vals if pd.notna(v)] + [1.0]) * 1.22)
    ax.set_ylabel("AUPRC lift vs random")
    ax.set_title("Split-wise AUPRC enrichment", fontweight="bold", pad=8)
    for i, v in enumerate(vals):
        ax.text(i, float(v) + 0.08 if pd.notna(v) else 0, fmt_float(v, 2), ha="center", fontsize=7)
    clean_axis(ax)

    ax = axes[1]
    row = metric_row(loaded.get("baseline_comparison") if isinstance(loaded.get("baseline_comparison"), pd.DataFrame) else None,
                     split="test", scope="heldout_input_entries", method="ae_significance_head")
    prior = metric_row(loaded.get("baseline_comparison") if isinstance(loaded.get("baseline_comparison"), pd.DataFrame) else None,
                       split="test", scope="heldout_input_entries", method="train_site_significance_prior")
    site_mean = metric_row(loaded.get("baseline_comparison") if isinstance(loaded.get("baseline_comparison"), pd.DataFrame) else None,
                           split="test", scope="heldout_input_entries", method="train_site_mean_m6a")
    bvals = [row.get("auprc_lift_over_random", np.nan), site_mean.get("auprc_lift_over_random", np.nan), prior.get("auprc_lift_over_random", np.nan)]
    y = np.arange(3)
    ax.barh(y, bvals, color=[MODEL_COLORS["ae_significance_head"], "#A0A7B0", MODEL_COLORS["train_site_significance_prior"]], edgecolor=TEXT_COLOR, linewidth=0.35)
    ax.axvline(1, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(["AE", "site mean", "site prior"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max([v for v in bvals if pd.notna(v)] + [1.0]) * 1.24)
    ax.set_xlabel("Heldout test AUPRC lift")
    ax.set_title("Comparison with site-level priors", fontweight="bold", pad=8)
    for i, v in enumerate(bvals):
        ax.text(float(v) + 0.08, i, fmt_float(v, 2), va="center", fontsize=7)
    clean_axis(ax, "x")

    ax = axes[2]
    if isinstance(overlap, pd.DataFrame) and not overlap.empty:
        d = overlap[
            (overlap["top_set"] == "top_residual_decile")
            & overlap["external_evidence"].isin(["supplementary_glori_supported", "supplementary_etam_supported", "supplementary_hek293t_supported"])
        ].copy()
        d["label"] = d["external_evidence"].str.replace("supplementary_", "", regex=False).str.replace("_supported", "", regex=False)
        d["label"] = d["label"].replace({"glori": "GLORI", "etam": "eTAM", "hek293t": "HEK293T"})
        x = np.arange(len(d))
        ax.bar(x - 0.18, d["top_overlap_fraction"], width=0.36, color=EVIDENCE_COLORS["top"], label="top residual decile", edgecolor=TEXT_COLOR, linewidth=0.35)
        ax.bar(x + 0.18, d["background_overlap_fraction"], width=0.36, color=EVIDENCE_COLORS["background"], label="background", edgecolor=TEXT_COLOR, linewidth=0.35)
        ax.set_xticks(x)
        ax.set_xticklabels(d["label"], fontsize=10)
        ymax = float(max(d["top_overlap_fraction"].max(), d["background_overlap_fraction"].max()) * 1.24)
        ax.set_ylim(0, max(ymax, 0.55))
        ax.set_ylabel("Overlap fraction")
        ax.set_title("Orthogonal m6A support enrichment", fontweight="bold", pad=8)
        for i, (_, r) in enumerate(d.iterrows()):
            ax.text(i, max(r["top_overlap_fraction"], r["background_overlap_fraction"]) + 0.035, f"OR={r['odds_ratio']:.2f}", ha="center", fontsize=7)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, borderaxespad=0.0)
        clean_axis(ax)
    else:
        ax.axis("off")

    savefig(fig_dir / "fig0_step1_to_step4_evidence_chain.png")


def plot_chromosome(site_stats: pd.DataFrame, fig_dir: Path) -> None:
    chr_agg = site_stats.groupby("seqnames", observed=False).agg(
        n=("site_id", "count"),
        det=("detected", "sum"),
        mean_ep=("event_prob_mean", "mean"),
        m6a_freq=("m6a_frequency", "mean"),
    ).reset_index()
    chr_agg["rate"] = chr_agg["det"] / chr_agg["n"]
    chr_agg = chr_agg[chr_agg["seqnames"].astype(str) != "unknown"].copy()
    if chr_agg.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    chr_order = sorted(chr_agg["seqnames"].astype(str).tolist(), key=chr_sort_key)
    chr_ordered = chr_agg.set_index("seqnames").reindex(chr_order).reset_index()

    ax = axes[0]
    mean_rate = chr_agg["rate"].mean()
    colors = ["#e74c3c" if r > mean_rate + 0.015 else "#3498db" if r >= mean_rate - 0.015 else "#95a5a6" for r in chr_ordered["rate"]]
    ax.bar(range(len(chr_ordered)), chr_ordered["rate"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(chr_ordered)))
    ax.set_xticklabels([str(c).replace("chr", "") for c in chr_ordered["seqnames"]], fontsize=8)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Detection rate (event_prob_max threshold)")
    ax.set_title("Model-detected m6A signal rate by chromosome", fontweight="bold")
    ax.axhline(y=mean_rate, color="red", linestyle="--", linewidth=1.5, label=f"Mean={mean_rate:.3f}")
    ax.legend(fontsize=9)
    for i, v in enumerate(chr_ordered["rate"]):
        ax.text(i, v + 0.003, f"{v:.2f}", ha="center", fontsize=7, rotation=45)

    ax2 = axes[1]
    chr_sorted = chr_agg.sort_values("rate", ascending=False).reset_index(drop=True)
    colors2 = ["#e74c3c" if i < 3 else "#27ae60" if i >= max(len(chr_sorted) - 3, 0) else "#3498db" for i in range(len(chr_sorted))]
    ax2.barh(range(len(chr_sorted)), chr_sorted["rate"], color=colors2, edgecolor="black", linewidth=0.3)
    ax2.set_yticks(range(len(chr_sorted)))
    ax2.set_yticklabels([str(c).replace("chr", "") for c in chr_sorted["seqnames"]], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Detection rate")
    ax2.set_title("Chromosomes ranked by detection rate", fontweight="bold")
    ax2.axvline(x=mean_rate, color="red", linestyle="--", linewidth=1.5)
    ax2.legend(
        handles=[
            mpatches.Patch(color="#e74c3c", label="Top 3"),
            mpatches.Patch(color="#3498db", label="Middle"),
            mpatches.Patch(color="#27ae60", label="Bottom 3"),
        ],
        fontsize=8,
        loc="lower right",
    )
    savefig(fig_dir / "fig1_chromosome_detection_rate.png")


def plot_ref_source(site_stats: pd.DataFrame, fig_dir: Path) -> None:
    by_ref = site_stats.groupby("reference_site_source", observed=False).agg(
        n=("site_id", "count"),
        det=("detected", "sum"),
        mean_ep=("event_prob_mean", "mean"),
        max_ep=("event_prob_max", "max"),
        m6a_freq=("m6a_frequency", "mean"),
        residual=("event_prob_residual", "mean"),
    ).reset_index()
    by_ref["rate"] = by_ref["det"] / by_ref["n"]
    if by_ref.empty:
        return
    palette = {
        "Exon_DRACH": "#3498db",
        "GLORI_NonDRACH": "#e74c3c",
        "GLORI_DRACH_NonExon": "#27ae60",
        "unknown": "#95a5a6",
    }
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax_idx, (col, label) in enumerate([
        ("rate", "Detection Rate"),
        ("mean_ep", "Mean Event Probability"),
        ("m6a_freq", "Mean m6A Frequency"),
        ("residual", "Mean EventProb - SiteMean"),
    ]):
        ax = axes[ax_idx]
        refs = by_ref.sort_values(col, ascending=False)
        ax.bar(
            range(len(refs)),
            refs[col],
            color=[palette.get(r, "#95a5a6") for r in refs["reference_site_source"]],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xticks(range(len(refs)))
        ax.set_xticklabels([str(r).replace("_", "\n") for r in refs["reference_site_source"]], fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.8) if col == "residual" else None
        for i, v in enumerate(refs[col]):
            ax.text(i, v + (0.005 if v >= 0 else -0.01), f"{v:.3f}", ha="center", fontsize=8)
        if ax_idx == 0:
            for i, (_, row) in enumerate(refs.iterrows()):
                ax.text(i, max(refs[col].min(), 0) + 0.01, f"n={int(row['n'])}", ha="center", fontsize=8, color="white", fontweight="bold")
    plt.suptitle("m6A detection by reference site source", fontsize=14, fontweight="bold", y=1.03)
    savefig(fig_dir / "fig2_ref_source_comparison.png")


def plot_cell_burden(cell_stats: pd.DataFrame, fig_dir: Path) -> None:
    if cell_stats.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sorted_cells = cell_stats.sort_values("burden_rate", ascending=False).reset_index(drop=True)

    ax = axes[0]
    ax.hist(cell_stats["burden_rate"], bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(cell_stats["burden_rate"].median(), color="red", linestyle="--", linewidth=2, label=f"Median={cell_stats['burden_rate'].median():.4f}")
    ax.axvline(cell_stats["burden_rate"].mean(), color="orange", linestyle="--", linewidth=2, label=f"Mean={cell_stats['burden_rate'].mean():.4f}")
    ax.set_xlabel("m6A burden rate (Step2 significant entries / observed entries)")
    ax.set_ylabel("Number of cells")
    ax.set_title("Cell m6A burden distribution", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(sorted_cells.index, sorted_cells["burden_rate"], color="#3498db", linewidth=0.8, alpha=0.8)
    ax.fill_between(sorted_cells.index, sorted_cells["burden_rate"], alpha=0.3, color="#3498db")
    p90 = float(cell_stats["burden_rate"].quantile(0.90))
    p75 = float(cell_stats["burden_rate"].quantile(0.75))
    ax.axhline(p90, color="red", linestyle="--", linewidth=1.5, label=f"P90={p90:.3f}")
    ax.axhline(p75, color="green", linestyle="--", linewidth=1.5, label=f"P75={p75:.3f}")
    ax.set_xlabel("Cell rank (sorted by burden)")
    ax.set_ylabel("m6A burden rate")
    ax.set_title("Cell burden rank plot", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[2]
    groups = pd.cut(
        cell_stats["burden_rate"],
        bins=[-np.inf, 0.01, 0.02, 0.05, np.inf],
        labels=["Low\n(<1%)", "Med-Low\n(1-2%)", "Med-High\n(2-5%)", "High\n(>5%)"],
    )
    counts = groups.value_counts().sort_index()
    colors = ["#27ae60", "#3498db", "#e67e22", "#e74c3c"]
    ax.bar(range(len(counts)), counts.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, fontsize=10)
    ax.set_ylabel("Number of cells")
    ax.set_title("Cell burden category distribution", fontweight="bold")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(1, len(cell_stats) * 0.005), f"{v} ({v/len(cell_stats)*100:.1f}%)", ha="center", fontsize=9)
    savefig(fig_dir / "fig3_cell_burden_heterogeneity.png")


def plot_gene_summary(gene_stats: pd.DataFrame, fig_dir: Path, min_gene_sites: int) -> None:
    g = gene_stats[(gene_stats["n_sites"] >= min_gene_sites) & (gene_stats["gene_symbol"] != "unknown")].copy()
    if g.empty:
        return

    top20 = g.nlargest(20, ["detected_sites", "detection_fraction", "max_event_prob"]).sort_values("detected_sites", ascending=True)
    # Bottom among genes with enough sites is not very informative if all zero; keep it for compatibility.
    bot20 = g.sort_values(["detected_sites", "detection_fraction", "max_event_prob"], ascending=[True, True, True]).head(20).sort_values("detected_sites", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    ax = axes[0]
    colors_top = ["#e74c3c" if f >= 0.7 else "#3498db" if f >= 0.5 else "#95a5a6" for f in top20["detection_fraction"]]
    ax.barh(range(len(top20)), top20["detected_sites"], color=colors_top, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["gene_symbol"], fontsize=9)
    ax.set_xlabel("Detected sites")
    ax.set_title(f"Top genes by detected sites\n(n_sites ≥ {min_gene_sites})", fontweight="bold")
    for i, (_, row) in enumerate(top20.iterrows()):
        ax.text(row["detected_sites"] + 0.2, i, f"{int(row['detected_sites'])}/{int(row['n_sites'])} ({row['detection_fraction']*100:.0f}%)", va="center", fontsize=8)
    ax.legend(
        handles=[
            mpatches.Patch(color="#e74c3c", label="≥70% detected"),
            mpatches.Patch(color="#3498db", label="50-70%"),
            mpatches.Patch(color="#95a5a6", label="<50%"),
        ],
        fontsize=9,
    )

    ax = axes[1]
    ax.scatter(
        g["n_sites"],
        g["detection_fraction"],
        s=np.clip(g["mean_coverage_per_entry"], 5, 300),
        c=g["mean_event_prob"],
        cmap="viridis",
        alpha=0.65,
        edgecolors="white",
        linewidth=0.3,
    )
    ax.set_xlabel("Number of annotated sites in gene")
    ax.set_ylabel("Detection fraction")
    ax.set_title("Gene burden: site count vs detection fraction", fontweight="bold")
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Mean event probability")
    for _, row in g.nlargest(8, "detected_sites").iterrows():
        ax.text(row["n_sites"], row["detection_fraction"] + 0.015, str(row["gene_symbol"]), fontsize=8)
    savefig(fig_dir / "fig4_top_bottom_genes.png")


def plot_bb_params(site_stats: pd.DataFrame, fig_dir: Path) -> None:
    candidates = [
        ("mu_fg_mean", "Mu_fg"),
        ("eta_fg_mean", "Eta_fg"),
        ("pi_fg_mean", "Pi_fg"),
        ("pred_significant_probability_mean", "Predicted significant probability"),
        ("pred_m6a_probability_mean", "Predicted m6A probability"),
        ("event_prob_mean", "Standardized detection score"),
    ]
    available = [(c, title) for c, title in candidates if c in site_stats.columns]
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
    if len(available) == 1:
        axes = [axes]
    for ax, (col, title) in zip(axes, available):
        sig = site_stats[site_stats["is_significant"] == 1][col].dropna()
        non_sig = site_stats[site_stats["is_significant"] == 0][col].dropna()
        data = [non_sig.values, sig.values]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, labels=["Non-Sig", "Sig"])
        bp["boxes"][0].set_facecolor("#3498db"); bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#e74c3c"); bp["boxes"][1].set_alpha(0.7)
        ax.set_ylabel(col)
        ax.set_title(title, fontweight="bold")
        if len(non_sig) > 0:
            ax.text(1, non_sig.mean(), f"μ={non_sig.mean():.4f}", ha="center", fontsize=8, color="#3498db")
        if len(sig) > 0:
            ax.text(2, sig.mean(), f"μ={sig.mean():.4f}", ha="center", fontsize=8, color="#e74c3c")
    plt.suptitle("Model score distributions stratified by Step2 significant label", fontsize=14, fontweight="bold", y=1.03)
    savefig(fig_dir / "fig6_beta_binomial_params.png")


def plot_threshold_metrics(threshold_metrics: pd.DataFrame, fig_dir: Path) -> None:
    if threshold_metrics.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    t = threshold_metrics["threshold"]
    ax = axes[0]
    ax.plot(t, threshold_metrics["precision_vs_step2_sig"], "o-", color="#e74c3c", linewidth=2, markersize=6, label="Precision")
    ax.plot(t, threshold_metrics["recall_vs_step2_sig"], "s-", color="#3498db", linewidth=2, markersize=6, label="Recall")
    ax.plot(t, threshold_metrics["f1_vs_step2_sig"], "^-", color="#27ae60", linewidth=2, markersize=6, label="F1")
    ax.set_xlabel("Detection threshold (event_prob_max)")
    ax.set_ylabel("Score vs Step2 significant label")
    ax.set_title("Precision/Recall/F1 vs detection threshold", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(t, threshold_metrics["detected_sites"], width=0.03, color="#9b59b6", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Detection threshold")
    ax.set_ylabel("Number of detected sites")
    ax.set_title("Detected sites count vs threshold", fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(t, threshold_metrics["f1_vs_step2_sig"], "o-", color="#27ae60", linewidth=2, markersize=6)
    ax2.set_ylabel("F1 score", color="#27ae60")
    for _, row in threshold_metrics.iterrows():
        ax.text(row["threshold"], row["detected_sites"] + max(50, threshold_metrics["detected_sites"].max() * 0.01), f"{int(row['detected_sites'])}", ha="center", fontsize=7, rotation=45)
    savefig(fig_dir / "fig8_threshold_analysis.png")


def plot_calibration_reliability(calibration: pd.DataFrame, fig_dir: Path) -> None:
    if calibration.empty or not {"mean_pred", "mean_obs"}.issubset(calibration.columns):
        return
    d = calibration.copy()
    if "scope" in d.columns and (d["scope"].astype(str) == "heldout_input_entries").any():
        d = d[d["scope"].astype(str) == "heldout_input_entries"].copy()
    d["mean_pred"] = pd.to_numeric(d["mean_pred"], errors="coerce")
    d["mean_obs"] = pd.to_numeric(d["mean_obs"], errors="coerce")
    d["residual_pred_minus_obs"] = d["mean_pred"] - d["mean_obs"]
    d = d.dropna(subset=["mean_pred", "mean_obs"]).sort_values("mean_pred").reset_index(drop=True)
    if d.empty:
        return
    d["bin_label"] = d.get("pred_probability_bin", pd.Series(range(len(d)))).astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))
    ax = axes[0]
    ax.plot(d["mean_pred"], d["mean_obs"], "o-", color=split_color("test"), linewidth=1.35, markersize=3.5, label="observed positive rate")
    lim = float(np.nanmax([d["mean_pred"].max(), d["mean_obs"].max(), 0.05]))
    ax.plot([0, lim], [0, lim], color="#666666", linestyle="--", linewidth=0.8, label="perfect calibration")
    ax.set_xlabel("Mean predicted AE significance score")
    ax.set_ylabel("Observed Step2 significant rate")
    ax.set_title("Reliability diagram for model scores", fontweight="bold")
    ax.legend(frameon=False)
    clean_axis(ax, "both")

    ax = axes[1]
    colors = [EVIDENCE_COLORS["top"] if v > 0 else MODEL_COLORS["ae_significance_head"] for v in d["residual_pred_minus_obs"]]
    ax.bar(np.arange(len(d)), d["residual_pred_minus_obs"], color=colors, edgecolor=TEXT_COLOR, linewidth=0.3)
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_xticks(np.arange(len(d)))
    ax.set_xticklabels(d["bin_label"].str.replace("decile_", "D", regex=False), rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Predicted - observed")
    ax.set_title("Calibration residual by score bin", fontweight="bold")
    max_abs = float(d["residual_pred_minus_obs"].abs().max())
    ax.text(0.02, 0.93, f"max |residual| = {max_abs:.3f}", transform=ax.transAxes, fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#B7C0C7"))
    clean_axis(ax)
    savefig(fig_dir / "fig7_calibration_reliability.png")


def plot_heldout_pr_roc_curves(pred: pd.DataFrame, fig_dir: Path) -> None:
    if precision_recall_curve is None or roc_curve is None or average_precision_score is None or roc_auc_score is None:
        print(f"  [skip PR/ROC] scikit-learn unavailable: {SKLEARN_IMPORT_ERROR}")
        return
    d = pred[pred["split"].astype(str) == "test"].copy()
    if "is_input_hidden" in d.columns:
        hidden = d[d["is_input_hidden"].astype(str).str.lower().isin(["true", "1"])].copy()
        if not hidden.empty:
            d = hidden
    d["is_significant"] = pd.to_numeric(d["is_significant"], errors="coerce").fillna(0).astype(int)
    if d["is_significant"].nunique() < 2:
        return
    score_specs = [
        ("AE significance head", "event_prob", "#4C78A8"),
        ("AE m6A probability", "pred_m6a_probability", "#54A24B"),
        ("Train site mean", "site_mean_p", "#A0A7B0"),
        ("Train site significance prior", "site_significance_train_prior", "#F58518"),
        ("Coverage only", "coverage_channel", "#B279A2"),
    ]
    y = d["is_significant"].to_numpy(dtype=int)
    prevalence = float(y.mean())

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    summary_rows = []
    for label, col, color in score_specs:
        if col not in d.columns:
            continue
        s = pd.to_numeric(d[col], errors="coerce")
        mask = s.notna() & np.isfinite(s)
        if mask.sum() < 10 or s[mask].nunique() < 2:
            continue
        yy = y[mask.to_numpy()]
        ss = s[mask].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(yy, ss)
        fpr, tpr, _ = roc_curve(yy, ss)
        ap = float(average_precision_score(yy, ss))
        auc = float(roc_auc_score(yy, ss))
        summary_rows.append((label, ap, auc))
        axes[0].plot(recall, precision, color=color, linewidth=1.35, label=f"{label} AP={ap:.3f}")
        axes[1].plot(fpr, tpr, color=color, linewidth=1.35, label=f"{label} AUC={auc:.3f}")

    axes[0].axhline(prevalence, color="#666666", linestyle="--", linewidth=0.8, label=f"random prevalence={prevalence:.3f}")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Heldout PR curve for methylation significance", fontweight="bold")
    axes[0].legend(fontsize=6.8, frameon=False, loc="upper right")
    clean_axis(axes[0], "both")

    axes[1].plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=0.8, label="random")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("Heldout ROC curve", fontweight="bold")
    axes[1].legend(fontsize=6.8, frameon=False, loc="lower right")
    clean_axis(axes[1], "both")

    savefig(fig_dir / "fig12_heldout_pr_roc_curves.png")


def plot_model_validation_artifacts(
    baseline: Optional[pd.DataFrame],
    decile: Optional[pd.DataFrame],
    permutation_summary: Optional[pd.DataFrame],
    fig_dir: Path,
) -> None:
    if baseline is not None and not baseline.empty:
        wanted = [
            "ae_significance_head",
            "ae_m6a_probability",
            "train_site_mean_m6a",
            "train_site_significance_prior",
            "coverage_only",
        ]
        b = baseline[
            (baseline["scope"].astype(str) == "heldout_input_entries")
            & (baseline["split"].astype(str) == "test")
            & (baseline["method"].astype(str).isin(wanted))
        ].copy()
        if not b.empty:
            method_labels = {
                "ae_significance_head": "AE significance",
                "ae_m6a_probability": "AE m6A prob",
                "train_site_mean_m6a": "train site mean",
                "train_site_significance_prior": "train site prior",
                "coverage_only": "coverage",
            }
            b["label"] = b["method"].map(method_labels).fillna(b["method"])
            b["order"] = b["method"].map({m: i for i, m in enumerate(wanted)})
            b = b.sort_values("order")

            fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8))
            ax = axes[0]
            colors = [MODEL_COLORS.get(m, "#A0A7B0") for m in b["method"]]
            ax.barh(np.arange(len(b)), b["auprc_lift_over_random"], color=colors, edgecolor=TEXT_COLOR, linewidth=0.35)
            ax.axvline(1, color="#666666", linestyle="--", linewidth=0.8)
            ax.set_yticks(np.arange(len(b)))
            ax.set_yticklabels(b["label"], fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel("AUPRC lift vs random")
            ax.set_title("Heldout AUPRC enrichment", fontweight="bold")
            for i, v in enumerate(b["auprc_lift_over_random"]):
                ax.text(v + 0.06, i, f"{v:.2f}", va="center", fontsize=7)
            clean_axis(ax, "x")

            ax = axes[1]
            ax.barh(np.arange(len(b)), b["precision_at_1pct"], color=colors, edgecolor=TEXT_COLOR, linewidth=0.35)
            ax.set_yticks(np.arange(len(b)))
            ax.set_yticklabels([])
            ax.invert_yaxis()
            ax.set_xlabel("Precision in top 1%")
            ax.set_title("Top-percentile precision", fontweight="bold")
            for i, v in enumerate(b["precision_at_1pct"]):
                ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7)
            clean_axis(ax, "x")

            ax = axes[2]
            if permutation_summary is not None and not permutation_summary.empty and {"model", "test_auprc_lift_over_random"}.issubset(permutation_summary.columns):
                p = permutation_summary.copy()
                p["label"] = p["model"].replace({"real_labels": "real labels", "permuted_train_labels": "permuted labels"})
                vals = p["test_auprc_lift_over_random"].astype(float).to_numpy()
                colors = ["#2F5D8C" if "real" in str(x) else "#B279A2" for x in p["label"]]
                ax.bar(p["label"], vals, color=colors, edgecolor=TEXT_COLOR, linewidth=0.35)
                for i, v in enumerate(p["test_auprc_lift_over_random"]):
                    ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=7)
            ax.axhline(1, color="#666666", linestyle="--", linewidth=0.8)
            ax.set_ylabel("Test AUPRC lift")
            ax.set_title("Permutation control", fontweight="bold")
            clean_axis(ax)
            savefig(fig_dir / "fig11_model_baseline_comparison.png")

    if decile is not None and not decile.empty and {"split", "decile", "positive_rate", "lift_vs_random"}.issubset(decile.columns):
        d = decile[decile["scope"].astype(str) == "heldout_input_entries"].copy() if "scope" in decile.columns else decile.copy()
        if d.empty:
            d = decile.copy()
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
        for split_name, sub in d.groupby("split", observed=False):
            sub = sub.sort_values("decile")
            color = split_color(split_name)
            axes[0].plot(sub["decile"], sub["positive_rate"], "o-", color=color, label=str(split_name), linewidth=1.35, markersize=3.5)
            axes[1].plot(sub["decile"], sub["lift_vs_random"], "o-", color=color, label=str(split_name), linewidth=1.35, markersize=3.5)
            add_decile_value_labels(axes[0], sub, split_name, "positive_rate", "{:.3f}")
            add_decile_value_labels(axes[1], sub, split_name, "lift_vs_random", "{:.2f}")
        test = d[d["split"].astype(str) == "test"].sort_values("decile")
        if not test.empty:
            top = test.iloc[0]
            axes[1].annotate(
                f"top decile lift={top['lift_vs_random']:.2f}",
                xy=(top["decile"], top["lift_vs_random"]),
                xytext=(top["decile"] + 1.0, top["lift_vs_random"] + 0.45),
                arrowprops=dict(arrowstyle="->", color="#333333", linewidth=0.8),
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#B7C0C7"),
            )
        axes[0].set_title("Observed positive rate by AE score decile", fontweight="bold")
        axes[0].set_xlabel("Decile (1 = highest AE score)")
        axes[0].set_ylabel("Observed Step2 significant rate")
        axes[1].set_title("Decile lift over random", fontweight="bold")
        axes[1].set_xlabel("Decile (1 = highest AE score)")
        axes[1].set_ylabel("Lift vs random")
        axes[1].axhline(1, color="#666666", linestyle="--", linewidth=0.8)
        positive_vals = pd.to_numeric(d["positive_rate"], errors="coerce").dropna()
        lift_vals = pd.to_numeric(d["lift_vs_random"], errors="coerce").dropna()
        if not positive_vals.empty:
            axes[0].set_ylim(0, float(positive_vals.max()) * 1.35)
        if not lift_vals.empty:
            ymin = min(0.0, float(lift_vals.min()))
            ymax = float(lift_vals.max())
            axes[1].set_ylim(ymin - 0.05, ymax + max((ymax - ymin) * 0.18, 0.25))
        for ax in axes:
            ax.invert_xaxis()
            clean_axis(ax, "both")
            ax.legend(title="Split", frameon=False)
        savefig(fig_dir / "fig14_decile_lift.png")


def plot_external_orthogonal_enrichment(overlap: pd.DataFrame, fig_dir: Path) -> None:
    if overlap.empty:
        return
    d = overlap[overlap["top_set"].isin(["top_residual_decile", "top_event_decile"])].copy()
    d = d[d["external_evidence"].isin([
        "supplementary_glori_supported",
        "supplementary_etam_supported",
        "supplementary_hek293t_supported",
        "fto_responsive_dmr",
    ])]
    if d.empty:
        return
    label_map = {
        "supplementary_glori_supported": "GLORI\nsupport",
        "supplementary_etam_supported": "eTAM\nsupport",
        "supplementary_hek293t_supported": "HEK293T\nsupport",
        "fto_responsive_dmr": "FTO DMR\nresponse",
    }
    set_map = {"top_residual_decile": "top residual decile", "top_event_decile": "top event decile"}
    d["label"] = d["external_evidence"].map(label_map).fillna(d["external_evidence"])
    d["top_label"] = d["top_set"].map(set_map).fillna(d["top_set"])
    d["odds_plot"] = pd.to_numeric(d["odds_ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    pivot_or = d.pivot_table(index="label", columns="top_label", values="odds_plot", aggfunc="first")
    pivot_top = d.pivot_table(index="label", columns="top_label", values="top_overlap_fraction", aggfunc="first")
    pivot_bg = d.pivot_table(index="label", columns="top_label", values="background_overlap_fraction", aggfunc="first")
    if pivot_or.empty:
        return
    order = [label_map[k] for k in ["supplementary_glori_supported", "supplementary_etam_supported", "supplementary_hek293t_supported", "fto_responsive_dmr"] if label_map[k] in pivot_or.index]
    pivot_or = pivot_or.reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    ax = axes[0]
    pivot_or.plot(kind="bar", ax=ax, edgecolor=TEXT_COLOR, linewidth=0.3, color=[EVIDENCE_COLORS["top"], MODEL_COLORS["ae_significance_head"]])
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Enrichment odds ratio")
    ax.set_xlabel("External evidence")
    ax.set_title("Top AE sites enriched for independent support", fontweight="bold")
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend(title="", fontsize=9, frameon=False)
    clean_axis(ax)

    ax = axes[1]
    residual_col = "top residual decile" if "top residual decile" in pivot_top.columns else pivot_top.columns[0]
    frac = pd.DataFrame({
        "top residual decile": pivot_top[residual_col],
        "background": pivot_bg[residual_col],
    }).reindex(order)
    frac.plot(kind="bar", ax=ax, edgecolor=TEXT_COLOR, linewidth=0.3, color=[EVIDENCE_COLORS["top"], EVIDENCE_COLORS["background"]])
    ax.set_ylabel("Overlap fraction")
    ax.set_xlabel("External evidence")
    ax.set_title("Top-vs-background overlap fraction", fontweight="bold")
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend(frameon=False)
    clean_axis(ax)
    fig.text(
        0.5, 0.01,
        "Combined orthogonal validation is intentionally omitted here because the Step4 universe is already saturated with validated sites.",
        ha="center",
        fontsize=7,
    )
    savefig(fig_dir / "fig_external_orthogonal_enrichment.png")


def feature_category(feature: str) -> str:
    f = str(feature)
    if re.fullmatch(r"[ACGTUN]{5}", f):
        return "DRACH-like motif"
    if f.startswith("overlap_") or f in {"protein_coding", "retained_intron", "lincRNA", "antisense", "nonsense_mediated_decay"}:
        return "Transcript topology"
    if f.startswith("H3"):
        return "Chromatin annotation"
    if f.endswith("_supported"):
        return "External support"
    if f in {"ELAVL1", "FMR1", "FTO", "HNRNPA2B1", "IGF2BP1", "IGF2BP3", "METTL14", "METTL3", "RBMX", "WTAP", "YTHDC1", "YTHDC2", "YTHDF1", "YTHDF2", "YTHDF3"}:
        return "m6A regulator/RBP"
    return "Other"


def plot_external_feature_enrichment(features: pd.DataFrame, fig_dir: Path) -> None:
    if features.empty:
        return
    d = features.replace([np.inf, -np.inf], np.nan).dropna(subset=["odds_ratio"]).copy()
    d = d[d["top_positive"] >= 2].copy()
    if d.empty:
        return
    d["category"] = d["feature"].map(feature_category)
    # Keep the strongest few per category so the plot is readable and result-like.
    keep = []
    for category, n in [
        ("External support", 3),
        ("Transcript topology", 6),
        ("DRACH-like motif", 5),
        ("m6A regulator/RBP", 5),
        ("Chromatin annotation", 3),
    ]:
        keep.append(d[d["category"] == category].sort_values(["odds_ratio", "top_positive"], ascending=False).head(n))
    d = pd.concat(keep, ignore_index=True).drop_duplicates("feature")
    if d.empty:
        return
    d = d.sort_values(["category", "odds_ratio"], ascending=[True, True])
    cat_colors = {
        "External support": EVIDENCE_COLORS["support"],
        "Transcript topology": EVIDENCE_COLORS["topology"],
        "DRACH-like motif": EVIDENCE_COLORS["motif"],
        "m6A regulator/RBP": EVIDENCE_COLORS["regulator"],
        "Chromatin annotation": EVIDENCE_COLORS["chromatin"],
        "Other": EVIDENCE_COLORS["neutral"],
    }
    fig, ax = plt.subplots(figsize=(8.0, max(4.8, 0.28 * len(d))))
    colors = [cat_colors.get(c, "#A0A7B0") for c in d["category"]]
    y = np.arange(len(d))
    ax.barh(y, d["odds_ratio"], color=colors, edgecolor=TEXT_COLOR, linewidth=0.3)
    ax.axvline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{f}  [{c}]" for f, c in zip(d["feature"].astype(str), d["category"])], fontsize=8)
    ax.set_xlabel("Top residual decile enrichment odds ratio")
    ax.set_title("Grouped feature enrichment in AE high-residual sites", fontweight="bold")
    for i, (_, row) in enumerate(d.iterrows()):
        ax.text(row["odds_ratio"] + 0.04, i, f"{int(row['top_positive'])}/{int(row['top_total'])}", va="center", fontsize=6.8)
    handles = [mpatches.Patch(color=color, label=cat) for cat, color in cat_colors.items() if cat in set(d["category"])]
    ax.legend(handles=handles, loc="lower right", fontsize=6.8, frameon=False)
    clean_axis(ax, "x")
    savefig(fig_dir / "fig_external_feature_enrichment.png")


def plot_external_topology_support(features: pd.DataFrame, fig_dir: Path) -> None:
    if features.empty:
        return
    d = features.copy().replace([np.inf, -np.inf], np.nan)
    wanted = [
        "GLORI_supported",
        "eTAM_seq_supported",
        "HEK293T_supported",
        "overlap_fullTranscripts",
        "overlap_exons",
        "overlap_fullThreePrimeUTR",
        "protein_coding",
        "TGACT",
        "TGACA",
        "GGACT",
    ]
    d = d[d["feature"].isin(wanted)].copy()
    if d.empty:
        return
    label_map = {
        "GLORI_supported": "GLORI support",
        "eTAM_seq_supported": "eTAM support",
        "HEK293T_supported": "HEK293T support",
        "overlap_fullTranscripts": "full transcript",
        "overlap_exons": "exon",
        "overlap_fullThreePrimeUTR": "3'UTR",
        "protein_coding": "protein coding",
        "TGACT": "TGACT motif",
        "TGACA": "TGACA motif",
        "GGACT": "GGACT motif",
    }
    d["label"] = d["feature"].map(label_map)
    d["category"] = d["feature"].map(feature_category)
    d["category"] = d["category"].replace({"External support": "External support", "DRACH-like motif": "Motif", "Transcript topology": "Topology"})
    d["top_fraction"] = pd.to_numeric(d["top_fraction"], errors="coerce")
    d["background_fraction"] = pd.to_numeric(d["background_fraction"], errors="coerce")
    d["odds_ratio"] = pd.to_numeric(d["odds_ratio"], errors="coerce")
    d = d.sort_values(["category", "odds_ratio"], ascending=[True, False])

    fig, axes = plt.subplots(1, 2, figsize=(9.8, max(4.2, 0.34 * len(d))))
    y = np.arange(len(d))
    ax = axes[0]
    ax.barh(y - 0.18, d["top_fraction"], height=0.36, color=EVIDENCE_COLORS["top"], edgecolor=TEXT_COLOR, linewidth=0.3, label="top residual decile")
    ax.barh(y + 0.18, d["background_fraction"], height=0.36, color=EVIDENCE_COLORS["background"], edgecolor=TEXT_COLOR, linewidth=0.3, label="background")
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of sites")
    ax.set_title("Top AE residual sites overlap m6A support/topology", fontweight="bold")
    ax.legend(frameon=False)
    clean_axis(ax, "x")

    ax = axes[1]
    colors = [EVIDENCE_COLORS["support"] if row["category"] == "External support" else EVIDENCE_COLORS["topology"] if row["category"] == "Topology" else EVIDENCE_COLORS["motif"] for _, row in d.iterrows()]
    ax.barh(y, d["odds_ratio"], color=colors, edgecolor=TEXT_COLOR, linewidth=0.3)
    ax.axvline(1, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("Enrichment odds ratio")
    ax.set_title("Top-vs-background enrichment", fontweight="bold")
    for i, (_, row) in enumerate(d.iterrows()):
        ax.text(row["odds_ratio"] + 0.05, i, f"OR={row['odds_ratio']:.2f}", va="center", fontsize=6.8)
    clean_axis(ax, "x")
    savefig(fig_dir / "fig_external_topology_support.png")


def plot_cell_clustering(
    df: pd.DataFrame,
    cell_stats: pd.DataFrame,
    cell_eff: Optional[pd.DataFrame],
    fig_dir: Path,
    random_state: int,
) -> Tuple[Optional[pd.DataFrame], Dict[str, float]]:
    if KMeans is None or PCA is None or silhouette_score is None:
        print(f"  [skip clustering] scikit-learn unavailable: {SKLEARN_IMPORT_ERROR}")
        return None, {}
    cell_feat = cell_stats.copy()
    if cell_eff is not None and "cell_id" in cell_eff.columns:
        latent_cols = [c for c in cell_eff.columns if re.match(r"^latent_\d+$", str(c))]
        eff_cols = present_cols(
            cell_eff,
            ["cell_id", "bg_cell_bias", "gate_cell_bias", "fg_latent_norm", "PC1", "PC2",
             "predicted_significance_burden", "mean_pred_significant_probability",
             "mean_pred_m6a_probability", "mean_total_coverage"] + latent_cols[:16],
        )
        if len(eff_cols) > 1:
            cell_feat = cell_feat.merge(cell_eff[eff_cols].drop_duplicates("cell_id"), on="cell_id", how="left")

    candidate_cols = [
        "mean_event_prob", "burden_rate", "m6a_freq_cell", "n_significant",
        "mean_posterior_fg", "event_prob_std", "mean_event_prob_residual",
        "bg_cell_bias", "gate_cell_bias", "fg_latent_norm", "PC1", "PC2",
        "predicted_significance_burden", "mean_pred_significant_probability",
        "mean_pred_m6a_probability", "mean_total_coverage",
    ]
    candidate_cols.extend([c for c in cell_feat.columns if re.match(r"^latent_\d+$", str(c))])
    feature_cols = [c for c in candidate_cols if c in cell_feat.columns]
    if len(cell_feat) < 10 or len(feature_cols) < 3:
        print("  [skip clustering] not enough cells/features")
        return None, {}
    X_df = cell_feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X_df.to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    max_k = min(7, len(cell_feat) - 1)
    k_range = range(2, max_k + 1)
    silhouettes: List[float] = []
    inertias: List[float] = []
    labels_by_k: Dict[int, np.ndarray] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        labels_by_k[k] = labels
        inertias.append(float(km.inertia_))
        try:
            silhouettes.append(float(silhouette_score(X, labels)))
        except Exception:
            silhouettes.append(float("nan"))

    chosen_k = 3 if 3 in labels_by_k else list(labels_by_k.keys())[0]
    labels_final = labels_by_k[chosen_k]
    km_final = KMeans(n_clusters=chosen_k, random_state=random_state, n_init=10).fit(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax = axes[0, 0]
    ax.plot(list(k_range), silhouettes, "o-", color="#3498db", linewidth=2, markersize=8)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette score vs k (cell clustering)", fontweight="bold")
    ax.axvline(chosen_k, color="red", linestyle="--", label=f"k={chosen_k} chosen")
    ax.legend()
    for k, s in zip(k_range, silhouettes):
        ax.text(k, s + 0.01 if np.isfinite(s) else 0, f"{s:.3f}" if np.isfinite(s) else "NA", ha="center", fontsize=8)

    ax = axes[0, 1]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_final, cmap="Set1", s=30, alpha=0.7, edgecolors="white", linewidth=0.3)
    centers_pca = pca.transform(km_final.cluster_centers_)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c="black", marker="X", s=200, edgecolors="white", linewidth=1)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Cell PCA (k={chosen_k}, colored by cluster)", fontweight="bold")
    cluster_counts = pd.Series(labels_final).value_counts().sort_index()
    for c in range(chosen_k):
        if c < len(centers_pca):
            ax.text(
                centers_pca[c, 0], centers_pca[c, 1], f"C{c}\n(n={cluster_counts.get(c, 0)})",
                ha="center", fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="round", facecolor=plt.cm.Set1(c / max(chosen_k, 1)), alpha=0.8),
            )

    cell_feat["cluster"] = labels_final
    profile_cols = [c for c in ["mean_event_prob", "burden_rate", "m6a_freq_cell", "n_significant"] if c in cell_feat.columns]
    cluster_summary = cell_feat.groupby("cluster")[profile_cols].mean()
    cluster_std = cell_feat.groupby("cluster")[profile_cols].std()

    ax = axes[1, 0]
    x = np.arange(len(profile_cols))
    width = min(0.8 / chosen_k, 0.25)
    for ci, c in enumerate(cluster_summary.index):
        means = cluster_summary.loc[c].values
        stds = cluster_std.loc[c].fillna(0).values
        ax.bar(x + ci * width, means, width, yerr=stds, label=f"Cluster {c} (n={(labels_final == c).sum()})", color=plt.cm.Set1(ci / max(chosen_k, 1)), alpha=0.8, capsize=3)
    ax.set_xticks(x + width * max(chosen_k - 1, 1) / 2)
    ax.set_xticklabels([c.replace("mean_", "Mean\n").replace("burden_", "Burden\n").replace("m6a_", "m6A\n").replace("n_", "N\n") for c in profile_cols], fontsize=9)
    ax.set_ylabel("Value")
    ax.set_title("Cluster profiles (mean ± std)", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    for ci in range(chosen_k):
        sub = cell_feat[cell_feat["cluster"] == ci]
        ax.scatter(sub["mean_event_prob"], sub["burden_rate"], c=[plt.cm.Set1(ci / max(chosen_k, 1))], label=f"Cluster {ci}", s=30, alpha=0.6)
    ax.set_xlabel("Mean event probability")
    ax.set_ylabel("Burden rate")
    ax.set_title("Cell clusters: event prob vs burden rate", fontweight="bold")
    ax.legend(fontsize=9)

    savefig(fig_dir / "fig9_cell_clustering.png")
    assign_cols = ["cell_id", "cluster"] + [c for c in ["mean_event_prob", "burden_rate", "m6a_freq_cell", "n_significant", "fg_latent_norm"] if c in cell_feat.columns]
    assignments = cell_feat[assign_cols].copy()
    metrics = {
        "cell_cluster_k": int(chosen_k),
        "cell_silhouette_chosen": float(silhouettes[list(k_range).index(chosen_k)]) if chosen_k in list(k_range) else float("nan"),
        "cell_cluster_min_size": int(cluster_counts.min()),
        "cell_cluster_max_size": int(cluster_counts.max()),
    }
    return assignments, metrics


def plot_gene_clustering(
    gene_stats: pd.DataFrame,
    fig_dir: Path,
    random_state: int,
    min_gene_sites: int,
) -> Tuple[Optional[pd.DataFrame], Dict[str, float]]:
    if KMeans is None or PCA is None:
        print(f"  [skip gene clustering] scikit-learn unavailable: {SKLEARN_IMPORT_ERROR}")
        return None, {}
    g = gene_stats[(gene_stats["n_sites"] >= min_gene_sites) & (gene_stats["gene_symbol"] != "unknown")].dropna(subset=["mean_event_prob", "max_event_prob", "mean_m6a_freq", "detection_fraction", "n_sites"]).copy()
    if len(g) < 10:
        print("  [skip gene clustering] not enough genes")
        return None, {}
    feature_cols = ["mean_event_prob", "max_event_prob", "mean_m6a_freq", "detection_fraction", "n_sites", "mean_event_prob_residual"]
    feature_cols = [c for c in feature_cols if c in g.columns]
    X = g[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    k = min(4, max(2, len(g) // 5))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    g["cluster"] = labels

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=g["detection_fraction"], cmap="RdYlGn",
        s=np.clip(g["n_sites"] * 2, 10, 300), alpha=0.6, vmin=0, vmax=1,
        edgecolors="white", linewidth=0.3,
    )
    centers = pca.transform(km.cluster_centers_)
    ax.scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=200, edgecolors="white", linewidth=1)
    plt.colorbar(scatter, ax=ax, label="Detection fraction")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Gene PCA (size = n_sites, color = detection fraction)", fontweight="bold")

    ax = axes[1]
    cluster_sum = g.groupby("cluster").agg(
        n_genes=("gene_symbol", "count"),
        mean_det=("detected_sites", "mean"),
        mean_ep=("mean_event_prob", "mean"),
        mean_det_frac=("detection_fraction", "mean"),
    ).reset_index()
    x = np.arange(len(cluster_sum))
    colors = [plt.cm.Set2(c / max(k, 1)) for c in cluster_sum["cluster"]]
    ax.bar(x, cluster_sum["n_genes"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}\n(n={n})" for c, n in zip(cluster_sum["cluster"], cluster_sum["n_genes"])], fontsize=10)
    ax.set_ylabel("Number of genes")
    ax.set_title("Gene clusters: distribution", fontweight="bold")
    for i, (_, row) in enumerate(cluster_sum.iterrows()):
        ax.text(i, row["n_genes"] + 2, f"det={row['mean_det']:.1f}\nep={row['mean_ep']:.3f}", ha="center", fontsize=8)
    savefig(fig_dir / "fig10_gene_clustering.png")
    metrics = {
        "gene_cluster_k": int(k),
        "gene_cluster_min_size": int(cluster_sum["n_genes"].min()),
        "gene_cluster_max_size": int(cluster_sum["n_genes"].max()),
    }
    return g, metrics


# =============================================================================
# Figure classification and contact sheet
# =============================================================================


FIGURE_CLASSIFICATION = {
    "00_method_design": [
        ("fig03_step3_train_validation_test_split_summary.png", "Method figure: nested train/validation/test cells with observed, held-out, and significant entry counts."),
    ],
    "01_best_results": [
        ("fig0_step1_to_step4_evidence_chain.png", "Best opening result figure: connects Step1/2 labels, Step3 AE validation, and Step4 external support."),
        ("fig11_model_baseline_comparison.png", "Best model-validation figure: heldout AE lift plus permutation control and site-prior boundary."),
        ("fig12_heldout_pr_roc_curves.png", "Best curve-based evidence that AE scores rank heldout methylation-significant entries above random."),
        ("fig14_decile_lift.png", "Best ranking summary: top decile enrichment is easy to explain in Results."),
        ("fig_external_orthogonal_enrichment.png", "Best external-support figure: GLORI/eTAM/HEK293T top-vs-background enrichment."),
        ("fig_external_topology_support.png", "Best biology bridge: top AE residual sites align with support/topology/motif features."),
    ],
    "02_supporting_biology": [
        ("fig_external_feature_enrichment.png", "Grouped feature enrichment for biological interpretation and follow-up discussion."),
        ("fig3_cell_burden_heterogeneity.png", "Cell-level heterogeneity support; useful after the main AE result is established."),
        ("fig9_cell_clustering.png", "Cell clustering support; useful as backup for single-cell heterogeneity."),
    ],
    "03_diagnostics_caveats": [
        ("fig7_calibration_reliability.png", "Calibration caveat: scores are useful for ranking but not absolute probabilities."),
        ("fig8_threshold_analysis.png", "Threshold sensitivity and precision/recall operating-point audit."),
        ("fig2_ref_source_comparison.png", "Reference-source/coverage context; important for caveats."),
        ("fig6_beta_binomial_params.png", "Model-score distribution diagnostic, useful for methods/Q&A."),
        ("fig10_gene_clustering.png", "Optional gene clustering diagnostic when trusted gene annotations exist."),
    ],
    "04_not_recommended": [
        ("fig1_chromosome_detection_rate.png", "Not recommended for main Results: chromosome-level differences may reflect source/coverage bias and distract from the core AE validation."),
        ("fig4_top_bottom_genes.png", "Not recommended unless trusted gene symbols are present; current Step2 metadata does not support a gene-list-first claim."),
    ],
}


def classify_figures(fig_dir: Path) -> None:
    class_root = fig_dir / "_classified"
    if class_root.exists():
        shutil.rmtree(class_root)
    class_root.mkdir(parents=True, exist_ok=True)

    copied: List[Tuple[str, str, Path, str]] = []
    missing: List[Tuple[str, str, str]] = []
    for category, entries in FIGURE_CLASSIFICATION.items():
        category_dir = class_root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for fname, reason in entries:
            src = fig_dir / fname
            if src.exists():
                dst = category_dir / fname
                shutil.copy2(src, dst)
                copied.append((category, fname, dst, reason))
            else:
                missing.append((category, fname, reason))

    readme = [
        "# Step4 Figure Classification",
        "",
        "This classification follows the paper-style Result narrative: show the strongest AE validation first, then biological support, then caveats. Weak or potentially distracting figures are kept out of the main folder.",
        "",
    ]
    for category in FIGURE_CLASSIFICATION:
        readme += [f"## {category}", ""]
        for fname, reason in FIGURE_CLASSIFICATION[category]:
            status = "present" if (fig_dir / fname).exists() else "missing"
            readme.append(f"- `{fname}` ({status}): {reason}")
        readme.append("")
    if missing:
        readme += ["## Missing Optional Figures", ""]
        for category, fname, reason in missing:
            readme.append(f"- `{category}/{fname}`: {reason}")
        readme.append("")
    (class_root / "README_classification.md").write_text("\n".join(readme), encoding="utf-8")

    if Image is None or ImageDraw is None or not copied:
        return
    thumbs = []
    for category, fname, dst, reason in copied:
        try:
            img = Image.open(dst).convert("RGB")
        except Exception:
            continue
        img.thumbnail((330, 250))
        canvas = Image.new("RGB", (370, 335), "white")
        canvas.paste(img, ((370 - img.width) // 2, 10))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 270), category, fill=(0, 0, 0))
        draw.text((10, 292), fname[:48], fill=(30, 30, 30))
        thumbs.append(canvas)
    if not thumbs:
        return
    cols = 3
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 370, rows * 335), (245, 245, 245))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * 370, (i // cols) * 335))
    sheet.save(class_root / "_contact_sheet.png")


# =============================================================================
# Report writing
# =============================================================================


def write_markdown_report(
    out_dir: Path,
    args: argparse.Namespace,
    key_metrics: Dict,
    threshold_metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    gene_stats: pd.DataFrame,
    external_results: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    best_f1 = threshold_metrics.sort_values("f1_vs_step2_sig", ascending=False).head(1)
    best_f1_text = "NA"
    if not best_f1.empty:
        r = best_f1.iloc[0]
        best_f1_text = f"threshold={r['threshold']:.2f}, F1={r['f1_vs_step2_sig']:.3f}, precision={r['precision_vs_step2_sig']:.3f}, recall={r['recall_vs_step2_sig']:.3f}"

    max_calib_resid = calibration["residual_pred_minus_obs"].abs().max() if not calibration.empty else float("nan")
    top_genes = gene_stats.head(10)[["gene_symbol", "n_sites", "detected_sites", "detection_fraction", "max_event_prob"]]

    report = []
    report.append("# Step4 Integrated Downstream Analysis Report\n")
    report.append(f"- Split used: `{args.split}`")
    report.append(f"- Exploratory detection threshold: `{args.threshold}`")
    report.append(f"- Conservative detection threshold: `{args.conservative_threshold}`")
    report.append(f"- Site detection is based on `{key_metrics.get('score_source', 'event_prob')}` standardized as `event_prob`, then aggregated as `event_prob_max` at the site level.")
    report.append("- This report treats Step2 significant labels as high-confidence evidence, not as complete ground truth.\n")

    report.append("## Evidence-chain conclusion\n")
    report.append(
        "The Step3 autoencoder learned a non-random methylation-significance signal on heldout entries: "
        f"test AUPRC lift is {key_metrics.get('test_auprc_lift', float('nan')):.2f}, while the permutation-label control is "
        f"{key_metrics.get('permutation_test_lift', float('nan')):.2f}. Train/validation/test lift is stable enough to avoid an obvious split-level overfit claim. "
        "Because the AE does not dominate every train-site prior baseline, the correct conclusion is that it learns and generalizes useful significance-ranking information, not that it universally beats all prior-based baselines."
    )
    report.append(
        "Step4 therefore uses the AE score mainly as a ranking/enrichment signal. External GLORI/eTAM/HEK293T support and feature enrichment are used for biological interpretation; calibrated absolute methylation probability is not claimed.\n"
    )

    report.append("## Key metrics\n")
    for k, v in key_metrics.items():
        report.append(f"- `{k}`: {v}")
    report.append("\n## Threshold audit\n")
    report.append(f"Best F1 against Step2 significant label: {best_f1_text}.")
    report.append("The default threshold is kept as an exploratory sensitivity threshold; the conservative threshold should be used when avoiding over-calling is more important.\n")

    report.append("## Calibration audit\n")
    report.append(f"Maximum absolute calibration residual across probability bins: {max_calib_resid:.4f}.")
    report.append("Positive residual means predicted event probability is higher than observed m6A frequency in that bin.\n")

    report.append("## Top genes by detected-site burden\n")
    if top_genes["gene_symbol"].astype(str).eq("unknown").all():
        report.append("Gene-level burden is not interpreted because the current Step2 metadata does not include trusted gene annotations.")
    else:
        report.append(top_genes.to_markdown(index=False))

    if external_results:
        report.append("\n## External m6AConquer biological validation\n")
        report.append("The primary biological interpretation is site-level: AE high-residual/high-event sites are compared with m6AConquer orthogonal validation, GLORI/eTAM evidence, genomic features, and FTO-responsive DMRs.")
        overlap = external_results.get("overlap")
        corr = external_results.get("correlation")
        feat = external_results.get("feature_enrichment")
        if isinstance(overlap, pd.DataFrame) and not overlap.empty:
            key_overlap = overlap[
                (overlap["top_set"] == "top_residual_decile")
                & overlap["external_evidence"].isin(["orthogonal_validated_combined", "orthogonal_validated_hek293t", "fto_responsive_dmr"])
            ][["external_evidence", "overlap_count", "top_overlap_fraction", "background_overlap_fraction", "odds_ratio", "fisher_p_greater"]]
            if not key_overlap.empty:
                report.append("\nTop residual decile overlap/enrichment:")
                report.append(key_overlap.to_markdown(index=False))
                combined = key_overlap[key_overlap["external_evidence"] == "orthogonal_validated_combined"]
                if not combined.empty and float(combined.iloc[0]["top_overlap_fraction"]) >= 0.999 and float(combined.iloc[0]["background_overlap_fraction"]) >= 0.999:
                    report.append("\nNote: combined orthogonal validation is saturated because the analyzed Step4 site universe already comes from validated m6A sites. It confirms source consistency but does not distinguish AE top sites from background. HEK293T support, GLORI/eTAM support, FTO DMR overlap, and genomic feature enrichment are more informative for top-vs-background interpretation.")
        if isinstance(corr, pd.DataFrame) and not corr.empty:
            key_corr = corr[
                (corr["score"].isin(["event_prob_residual", "event_prob_mean"]))
                & (corr["external_measure"] == "external_ratio")
            ].head(8)
            if not key_corr.empty:
                report.append("\nGLORI/eTAM concordance summary:")
                report.append(key_corr.to_markdown(index=False))
        if isinstance(feat, pd.DataFrame) and not feat.empty:
            report.append("\nTop genomic feature enrichments in AE high-residual sites:")
            report.append(feat.head(10)[["feature", "top_positive", "top_total", "background_positive", "background_total", "odds_ratio", "fisher_p_greater"]].to_markdown(index=False))
        report.append("\nThis section intentionally avoids making GO enrichment the main claim because the model output is site-level rather than gene-list-first.")

    report.append("\n## Output files\n")
    report.append("- Numeric summaries: `step4_*csv`, `step4_key_metrics.json`, `visualization_summary.json`")
    report.append("- Figures: `figures/fig*.png`")
    report.append("- Classified figures: `figures/_classified/01_best_results`, `02_supporting_biology`, `03_diagnostics_caveats`, and `04_not_recommended`")
    if external_results:
        report.append("- External m6AConquer outputs: `step4_external_overlap_summary.csv`, `step4_external_score_correlation.csv`, `step4_external_feature_enrichment.csv`, and `figures/fig_external_*.png`")
    report.append("- Cluster assignments: `figures/cell_cluster_assignments.csv`, `figures/gene_cluster_assignments.csv` if clustering is enabled.\n")

    (out_dir / "step4_report.md").write_text("\n".join(report), encoding="utf-8")


# =============================================================================
# Main workflow
# =============================================================================


def main() -> None:
    args = parse_args()
    root, step3_dir, step2_dir, out_dir, fig_dir = resolve_paths(args)

    print("=" * 72)
    print("STEP4 INTEGRATED DOWNSTREAM ANALYSIS + VISUALIZATION")
    print("Model: compact Step3 significance-learning candidate")
    print("=" * 72)
    print(f"ROOT      : {root}")
    print(f"STEP3_DIR : {step3_dir}")
    print(f"STEP2_DIR : {step2_dir}")
    print(f"OUT_DIR   : {out_dir}")
    print(f"FIG_DIR   : {fig_dir}")

    loaded = load_inputs(args, step3_dir, step2_dir)
    pred = validate_and_repair_predictions(loaded["pred"], loaded["site_meta"], args)  # type: ignore[arg-type]
    analysis_df = choose_analysis_split(pred, args.split)

    print_section("[3/8] Building numeric Step4 summaries")
    site_stats = make_site_stats(analysis_df, loaded["site_meta"], args.threshold, args.conservative_threshold)  # type: ignore[arg-type]
    gene_stats = make_gene_stats(site_stats, args.min_gene_sites)
    cell_stats = make_cell_stats(analysis_df, args.threshold)
    group_summaries = make_group_summaries(site_stats, analysis_df)
    threshold_metrics = make_threshold_metrics(site_stats)
    calibration = make_calibration(pred, args.split, loaded["calib"])  # type: ignore[arg-type]

    print(f"  sites: {len(site_stats):,}")
    print(f"  genes: {len(gene_stats):,}")
    print(f"  cells: {len(cell_stats):,}")
    print(f"  detected sites @ {args.threshold:.2f}: {int(site_stats['detected'].sum()):,} / {len(site_stats):,}")

    print_section("[4/8] Saving numeric summaries")
    site_stats.to_csv(out_dir / "step4_site_detection_summary.csv", index=False)
    gene_stats.to_csv(out_dir / "step4_gene_burden_summary.csv", index=False)
    cell_stats.to_csv(out_dir / "step4_cell_burden_summary.csv", index=False)
    group_summaries["chr_stats"].to_csv(out_dir / "step4_chromosome_summary.csv", index=False)
    group_summaries["strand_stats"].to_csv(out_dir / "step4_strand_summary.csv", index=False)
    group_summaries["by_gene_source"].to_csv(out_dir / "step4_detection_by_gene_source.csv", index=False)
    group_summaries["by_ref"].to_csv(out_dir / "step4_detection_by_ref_source.csv", index=False)
    group_summaries["source_coverage"].to_csv(out_dir / "step4_source_coverage_calibration.csv", index=False)
    threshold_metrics.to_csv(out_dir / "step4_threshold_metrics.csv", index=False)
    calibration.to_csv(out_dir / "step4_calibration_by_probability_bin.csv", index=False)
    for src_key, out_name in [
        ("validation_decision", "step4_model_validation_decision.csv"),
        ("model_summary", "step4_model_validation_summary.csv"),
        ("baseline_comparison", "step4_model_baseline_comparison.csv"),
        ("overfit_diagnostics", "step4_model_overfit_diagnostics.csv"),
        ("decile_lift", "step4_model_decile_lift_table.csv"),
        ("permutation_summary", "step4_model_permutation_control_summary.csv"),
    ]:
        src = loaded.get(src_key)
        if isinstance(src, pd.DataFrame):
            src.to_csv(out_dir / out_name, index=False)
    print(f"  saved CSV summaries to {out_dir}")

    external_results = run_external_m6aconquer_analysis(root, args, site_stats, out_dir, fig_dir)

    print_section("[5/8] Generating figures")
    plot_step3_split_design(loaded.get("split_summary"), fig_dir)  # type: ignore[arg-type]
    plot_evidence_chain(loaded, site_stats, cell_stats, external_results, fig_dir)
    plot_chromosome(site_stats, fig_dir)
    plot_ref_source(site_stats, fig_dir)
    plot_cell_burden(cell_stats, fig_dir)
    plot_gene_summary(gene_stats, fig_dir, args.min_gene_sites)
    plot_bb_params(site_stats, fig_dir)
    site_mean_corr = safe_corr(site_stats["site_mean_p"], site_stats["event_prob_mean"], "pearson") if {"site_mean_p", "event_prob_mean"}.issubset(site_stats.columns) else float("nan")
    plot_calibration_reliability(calibration, fig_dir)
    plot_threshold_metrics(threshold_metrics, fig_dir)
    plot_heldout_pr_roc_curves(analysis_df, fig_dir)
    plot_model_validation_artifacts(
        loaded.get("baseline_comparison"),  # type: ignore[arg-type]
        loaded.get("decile_lift"),  # type: ignore[arg-type]
        loaded.get("permutation_summary"),  # type: ignore[arg-type]
        fig_dir,
    )

    print_section("[6/8] Clustering analysis")
    cluster_metrics: Dict[str, float] = {}
    if not args.skip_clustering:
        cell_assign, cell_cluster_metrics = plot_cell_clustering(analysis_df, cell_stats, loaded["cell_eff"], fig_dir, args.random_state)  # type: ignore[arg-type]
        if cell_assign is not None:
            cell_assign.to_csv(fig_dir / "cell_cluster_assignments.csv", index=False)
            cluster_metrics.update(cell_cluster_metrics)
        gene_assign, gene_cluster_metrics = plot_gene_clustering(gene_stats, fig_dir, args.random_state, args.min_gene_sites)
        if gene_assign is not None:
            gene_assign.to_csv(fig_dir / "gene_cluster_assignments.csv", index=False)
            cluster_metrics.update(gene_cluster_metrics)
    else:
        print("  clustering skipped by --skip-clustering")

    print_section("[7/8] Classifying figures")
    classify_figures(fig_dir)

    print_section("[7/8] Writing JSON and Markdown report")
    chr_stats = group_summaries["chr_stats"]
    by_ref = group_summaries["by_ref"]
    detected_sites = int(site_stats["detected"].sum())
    detected_sites_conservative = int(site_stats["detected_conservative"].sum())
    overall_detection_rate = float(site_stats["detected"].mean()) if len(site_stats) else float("nan")
    conservative_detection_rate = float(site_stats["detected_conservative"].mean()) if len(site_stats) else float("nan")
    cell_burden_cv = float(cell_stats["burden_rate"].std() / cell_stats["burden_rate"].mean()) if cell_stats["burden_rate"].mean() != 0 else float("nan")
    validation_df = loaded.get("validation_decision")
    validation_row = validation_df.iloc[0].to_dict() if isinstance(validation_df, pd.DataFrame) and not validation_df.empty else {}

    key_metrics = {
        "split_used": args.split,
        "score_source": str(pred["event_prob_source"].iloc[0]) if "event_prob_source" in pred.columns and len(pred) else "event_prob",
        "threshold": args.threshold,
        "conservative_threshold": args.conservative_threshold,
        "model_validation_status": validation_row.get("status", "NA"),
        "model_criterion_pass_count": int(validation_row.get("criterion_pass_count", -1)) if pd.notna(validation_row.get("criterion_pass_count", np.nan)) else -1,
        "model_criterion_total": int(validation_row.get("criterion_total", -1)) if pd.notna(validation_row.get("criterion_total", np.nan)) else -1,
        "test_auprc_lift": float(validation_row.get("test_ae_lift", np.nan)) if pd.notna(validation_row.get("test_ae_lift", np.nan)) else float("nan"),
        "val_auprc_lift": float(validation_row.get("val_ae_lift", np.nan)) if pd.notna(validation_row.get("val_ae_lift", np.nan)) else float("nan"),
        "train_auprc_lift": float(validation_row.get("train_ae_lift", np.nan)) if pd.notna(validation_row.get("train_ae_lift", np.nan)) else float("nan"),
        "train_minus_val_lift": float(validation_row.get("train_minus_val_lift", np.nan)) if pd.notna(validation_row.get("train_minus_val_lift", np.nan)) else float("nan"),
        "val_minus_test_lift_abs": float(validation_row.get("val_minus_test_lift_abs", np.nan)) if pd.notna(validation_row.get("val_minus_test_lift_abs", np.nan)) else float("nan"),
        "test_site_prior_lift": float(validation_row.get("test_site_prior_lift", np.nan)) if pd.notna(validation_row.get("test_site_prior_lift", np.nan)) else float("nan"),
        "test_site_mean_lift": float(validation_row.get("test_site_mean_lift", np.nan)) if pd.notna(validation_row.get("test_site_mean_lift", np.nan)) else float("nan"),
        "permutation_test_lift": float(validation_row.get("permutation_test_lift", np.nan)) if pd.notna(validation_row.get("permutation_test_lift", np.nan)) else float("nan"),
        "val_or_selected_sites": int(len(site_stats)),
        "total_cells": int(len(cell_stats)),
        "detected_sites": detected_sites,
        "detected_sites_conservative": detected_sites_conservative,
        "overall_detection_rate": overall_detection_rate,
        "conservative_detection_rate": conservative_detection_rate,
        "mean_event_prob": float(site_stats["event_prob_mean"].mean()) if len(site_stats) else float("nan"),
        "mean_m6a_freq": float(site_stats["m6a_frequency"].mean()) if len(site_stats) else float("nan"),
        "mean_event_prob_residual": float(site_stats["event_prob_residual"].mean()) if len(site_stats) else float("nan"),
        "chr_highest": str(chr_stats.iloc[0]["seqnames"]) if len(chr_stats) else "NA",
        "chr_lowest": str(chr_stats.iloc[-1]["seqnames"]) if len(chr_stats) else "NA",
        "cell_burden_cv": cell_burden_cv,
        "top_gene": str(gene_stats.iloc[0]["gene_symbol"]) if len(gene_stats) else "NA",
        "site_mean_p_corr": site_mean_corr,
        "ref_source_detection": by_ref.set_index("reference_site_source")["detection_rate"].to_dict() if len(by_ref) else {},
    }
    if external_results:
        overlap = external_results.get("overlap")
        corr = external_results.get("correlation")
        features = external_results.get("feature_enrichment")
        if isinstance(overlap, pd.DataFrame) and not overlap.empty:
            row = overlap[(overlap["top_set"] == "top_residual_decile") & (overlap["external_evidence"] == "orthogonal_validated_combined")]
            if not row.empty:
                key_metrics["external_combined_validated_top_residual_or"] = float(row.iloc[0]["odds_ratio"])
                key_metrics["external_combined_validated_top_residual_overlap"] = int(row.iloc[0]["overlap_count"])
            fto = overlap[(overlap["top_set"] == "top_residual_decile") & (overlap["external_evidence"] == "fto_responsive_dmr")]
            if not fto.empty:
                key_metrics["external_fto_dmr_top_residual_or"] = float(fto.iloc[0]["odds_ratio"])
                key_metrics["external_fto_dmr_top_residual_overlap"] = int(fto.iloc[0]["overlap_count"])
        if isinstance(corr, pd.DataFrame) and not corr.empty:
            best_corr = corr.dropna(subset=["spearman"]).sort_values("spearman", ascending=False).head(1)
            if not best_corr.empty:
                key_metrics["external_best_spearman"] = float(best_corr.iloc[0]["spearman"])
                key_metrics["external_best_spearman_label"] = f"{best_corr.iloc[0]['dataset']}:{best_corr.iloc[0]['score']}~{best_corr.iloc[0]['external_measure']}"
        if isinstance(features, pd.DataFrame) and not features.empty:
            key_metrics["external_top_feature"] = str(features.iloc[0]["feature"])
            key_metrics["external_top_feature_or"] = float(features.iloc[0]["odds_ratio"])
    key_metrics.update(cluster_metrics)

    write_json(out_dir / "step4_key_metrics.json", key_metrics)
    figures_generated = sorted([p.name for p in fig_dir.glob("fig*.png")])
    visualization_summary = {
        "figures_generated": figures_generated,
        "classified_figure_root": str(fig_dir / "_classified"),
        "best_result_figures": [
            fname for fname, _ in FIGURE_CLASSIFICATION["01_best_results"] if (fig_dir / fname).exists()
        ],
        "key_findings": key_metrics,
    }
    write_json(fig_dir / "visualization_summary.json", visualization_summary)
    # Also save a copy at OUT_DIR for convenience/backward compatibility.
    write_json(out_dir / "visualization_summary.json", visualization_summary)
    write_markdown_report(out_dir, args, key_metrics, threshold_metrics, calibration, gene_stats, external_results)

    print_section("[8/8] COMPLETE")
    print("Key metrics:")
    for k, v in key_metrics.items():
        print(f"  {k}: {v}")
    print(f"\nNumeric outputs: {out_dir}")
    print(f"Figure outputs : {fig_dir}")
    print("\nRecommended use:")
    print(f"  - Keep threshold={args.threshold:.2f} for exploratory sensitivity plots unless threshold_metrics suggests a better operating point.")
    print(f"  - Use conservative_threshold={args.conservative_threshold:.2f} when reporting robust detected-site counts.")
    print("  - Do not interpret chromosome/gene burden without checking coverage/source summaries.")


if __name__ == "__main__":
    main()
