"""The CATE error-anatomy plot for a finished benchmark run.

    uv run benchmarks/plots.py --path data/synt_eval/<name>/bench

The sole figure, ``cate_iso_nrmse.png``, places each method by its robust
level error (median squared normalised ATE bias) and heterogeneity recovery
(median Shape R²), with point colour showing median nPEHE. Its analytic
contours are single-episode CATE NRMSE reference contours. It is deliberately
CATE-only: the identity used to construct the contours is not an identity for
the median summaries.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.metrics import TASKS, Task
from benchmarks.plot_frames import BenchmarkRun
from benchmarks.plot_style import abbreviate, save_figure, style_for, use_theme


X_GRID_MIN = 0.0
X_GRID_MAX = 3.0
Y_GRID_MIN = -1.0
Y_GRID_MAX = 1.02
CENSOR_PAD = 0.05
VIEW_PAD = 0.05


@dataclass
class Options:
    dpi: int = 200


def _cate_summary(run: BenchmarkRun) -> pd.DataFrame:
    """Aggregate the CATE quantities with the table's robust median convention.

    The x-axis is the median of the per-episode squared normalised ATE bias;
    this preserves the level-error magnitude without cancellation between
    positive and negative bias. Medians are intentionally used here to match
    the terminal tables and avoid a few near-homogeneous episodes dominating
    the unbounded-below Shape R² mean.
    """
    cate: Task = TASKS["cate"]
    frame = run.frame(cate).dropna(subset=["npehe", "ate_bias_norm", "shape_r2"])
    rows = {}
    for method, group in frame.groupby("method", sort=False):
        rows[method] = {
            "ate_bias_norm_sq": float((group["ate_bias_norm"] ** 2).median()),
            "shape_r2": float(group["shape_r2"].median()),
            "npehe": float(group["npehe"].median()),
            "n_episodes": len(group),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _display_coordinate(value: float, low: float, high: float) -> float:
    """Keep an out-of-grid point visible just beyond the relevant boundary."""
    if value < low:
        return low - CENSOR_PAD
    if value > high:
        return high + CENSOR_PAD
    return value


def plot_cate_iso_nrmse(run: BenchmarkRun, opts: Options) -> None:
    """Draw median squared normalised ATE bias × median Shape R²."""
    if "cate" not in {task.name for task in run.tasks}:
        print("  · cate_iso_nrmse: CATE results are absent — skipped")
        return
    summary = _cate_summary(run)
    if summary.empty:
        print("  · cate_iso_nrmse: no finite CATE metrics — skipped")
        return

    x = summary["ate_bias_norm_sq"].to_numpy()
    y = summary["shape_r2"].to_numpy()
    gx, gy = np.meshgrid(
        np.linspace(X_GRID_MIN, X_GRID_MAX, 400),
        np.linspace(Y_GRID_MIN, 1.0, 400),
    )
    cate_nrmse = np.sqrt(np.maximum(gx + 1.0 - gy, 0.0))

    upper = max(1.0, float(summary["npehe"].max()) * 1.08)
    levels = np.linspace(0.0, upper, 9)
    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    contours = ax.contour(
        gx, gy, cate_nrmse, levels=levels[1:], colors="0.65", linewidths=0.7
    )
    ax.clabel(contours, fmt="%.2f", fontsize=7)

    colour_norm = Normalize(vmin=0.0, vmax=upper)
    colour_mappable = None

    for method, row in summary.iterrows():
        style = style_for(method)
        x_display = _display_coordinate(row.ate_bias_norm_sq, X_GRID_MIN, X_GRID_MAX)
        y_display = _display_coordinate(row.shape_r2, Y_GRID_MIN, Y_GRID_MAX)
        point = ax.scatter(
            x_display,
            y_display,
            c=[row.npehe],
            cmap="RdYlGn_r",
            norm=colour_norm,
            marker=style.get("marker", "o"),
            s=style.get("s", 110),
            edgecolors="white",
            linewidths=0.9,
            zorder=3,
        )
        if colour_mappable is None:
            colour_mappable = point
        ax.annotate(
            abbreviate(method),
            (x_display, y_display),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
            color="0.25",
        )

    ax.axvline(X_GRID_MAX, color="0.45", linewidth=0.8, linestyle=":", zorder=1)
    ax.axhline(Y_GRID_MIN, color="0.45", linewidth=0.8, linestyle=":", zorder=1)
    ax.set(
        xlim=(X_GRID_MIN, X_GRID_MAX + CENSOR_PAD + VIEW_PAD),
        ylim=(Y_GRID_MIN - CENSOR_PAD - VIEW_PAD, Y_GRID_MAX + VIEW_PAD),
    )
    ax.set_xlabel("median (normalised ATE bias)²")
    ax.set_ylabel("median Shape R²")
    if colour_mappable is not None:
        fig.colorbar(colour_mappable, ax=ax, shrink=0.82, label="median nPEHE")
    save_figure(
        fig,
        run.plots_dir / "cate_iso_nrmse.png",
        dpi=opts.dpi,
        close=False,
    )
    save_figure(fig, run.plots_dir / "cate_iso_nrmse.pdf")


def resolve_exp_dir(exp: str) -> Path:
    """Resolve an experiment name to its benchmark output directory."""
    candidates = [
        Path("data/synt_eval") / exp / "bench",
        Path("data/prior_eval") / exp / "bench",
        Path("data/lit_eval") / exp / "bench",
        Path("synthetic") / exp / "bench",
        Path("results") / exp,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise SystemExit(
        f"No experiment directory found for '{exp}' "
        f"(tried: {', '.join(str(candidate) for candidate in candidates)})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the CATE iso-NRMSE plot.")
    parser.add_argument("--exp", default=None, help="Experiment name.")
    parser.add_argument("--path", default=None, help="Benchmark output directory.")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    exp_dir = (
        Path(args.path) if args.path else resolve_exp_dir(args.exp or "first_test")
    )
    run = BenchmarkRun.load(exp_dir)
    use_theme()
    print(
        "Figures → "
        f"{run.plots_dir / 'cate_iso_nrmse.png'} and "
        f"{run.plots_dir / 'cate_iso_nrmse.pdf'}"
    )
    plot_cate_iso_nrmse(run, Options(dpi=args.dpi))


if __name__ == "__main__":
    main()
