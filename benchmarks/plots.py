"""Benchmark figures for a finished benchmark run.

    uv run benchmarks/plots.py --path data/synt_eval/<name>/bench

The entry point produces two CATE-only figures. ``cate_iso_nrmse.png`` places
each method by its robust level error (median squared normalised ATE bias) and
heterogeneity recovery (median Shape R²), with point colour showing median
nPEHE. ``cate_speed_rank.png`` shows the accuracy/speed trade-off: median
benchmark time per episode on a log x-axis, against the median paired rank on
CATE nPEHE with 25th–75th percentile whiskers. Reference rows are excluded
from the rank plot, as they are from every competitor comparison.
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
from matplotlib.ticker import MaxNLocator

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


def _cate_speed_rank_summary(run: BenchmarkRun) -> pd.DataFrame:
    """Summarize runtime and paired CATE nPEHE ranks for each competitor.

    Ranks come from :meth:`BenchmarkRun.ranks`, so they are computed on the
    complete episode block and never include reference rows. Runtime is the
    recorded fit-plus-predict wall time for the same block. Non-positive times
    are omitted because they cannot be represented on the log-scaled x-axis;
    this also makes old prediction caches with missing timing arrays fail
    closed instead of plotting a misleading zero-speed point.
    """
    cate: Task = TASKS["cate"]
    ranks = run.ranks(cate)
    if ranks.empty:
        return pd.DataFrame(
            columns=[
                "median_time_s",
                "median_rank",
                "rank_p25",
                "rank_p75",
                "n_episodes",
            ]
        )

    times = run.competitor_frame(cate).pivot_table(
        index="episode", columns="method", values="time", aggfunc="first"
    )
    times = times.reindex(index=ranks.index, columns=ranks.columns)

    rows = {}
    for method in ranks.columns:
        method_times = times[method]
        valid = method_times.notna() & np.isfinite(method_times) & (method_times > 0)
        if not valid.any():
            continue
        method_ranks = ranks.loc[valid, method]
        method_times = method_times.loc[valid]
        rows[method] = {
            "median_time_s": float(method_times.median()),
            "median_rank": float(method_ranks.median()),
            "rank_p25": float(method_ranks.quantile(0.25)),
            "rank_p75": float(method_ranks.quantile(0.75)),
            "n_episodes": int(len(method_ranks)),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


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


def plot_cate_speed_rank(run: BenchmarkRun, opts: Options) -> None:
    """Draw runtime against the robust paired rank on CATE nPEHE.

    The x-axis is the median fit-plus-predict wall time per benchmark episode,
    in seconds and on a log scale. The y-axis is inverted so rank 1 (best) is
    at the top, with whiskers spanning the per-episode 25th–75th percentile.
    """
    if "cate" not in {task.name for task in run.tasks}:
        print("  · cate_speed_rank: CATE results are absent — skipped")
        return
    summary = _cate_speed_rank_summary(run)
    if summary.empty:
        print("  · cate_speed_rank: no finite CATE ranks and timings — skipped")
        return

    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    label_offsets = ((6, 8), (6, -17), (-6, 8), (-6, -17))

    for i, (method, row) in enumerate(summary.iterrows()):
        style = style_for(method)
        color = style.get("color", "gray")
        x = float(row["median_time_s"])
        y = float(row["median_rank"])
        yerr = np.array(
            [
                [max(0.0, y - float(row["rank_p25"]))],
                [max(0.0, float(row["rank_p75"]) - y)],
            ]
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor=color,
            elinewidth=2.2,
            alpha=0.85,
            capsize=0,
            zorder=2,
        )
        ax.scatter(
            x,
            y,
            color=color,
            marker=style.get("marker", "o"),
            s=style.get("s", 110),
            edgecolors="white",
            linewidths=0.9,
            zorder=3,
        )
        dx, dy = label_offsets[i % len(label_offsets)]
        ax.annotate(
            abbreviate(method),
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="center",
            fontsize=7,
            color="0.25",
        )

    ax.set_xscale("log")
    lower_rank = max(0.5, float(summary["rank_p25"].min()) - 0.5)
    upper_rank = float(summary["rank_p75"].max()) + 0.5
    ax.set_ylim(upper_rank, lower_rank)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Median time per benchmark episode (seconds, log scale)")
    ax.set_ylabel("Median rank on CATE nPEHE (25th–75th percentile)")
    save_figure(
        fig,
        run.plots_dir / "cate_speed_rank.png",
        dpi=opts.dpi,
        close=False,
    )
    save_figure(fig, run.plots_dir / "cate_speed_rank.pdf")


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
    parser = argparse.ArgumentParser(description="Draw benchmark figures.")
    parser.add_argument("--exp", default=None, help="Experiment name.")
    parser.add_argument("--path", default=None, help="Benchmark output directory.")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    exp_dir = (
        Path(args.path) if args.path else resolve_exp_dir(args.exp or "first_test")
    )
    run = BenchmarkRun.load(exp_dir)
    use_theme()
    plot_cate_iso_nrmse(run, Options(dpi=args.dpi))
    plot_cate_speed_rank(run, Options(dpi=args.dpi))


if __name__ == "__main__":
    main()
