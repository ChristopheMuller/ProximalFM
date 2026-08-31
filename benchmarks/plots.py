"""Benchmark figures for a finished benchmark run.

    uv run benchmarks/plots.py --path data/synt_eval/<name>/bench

The entry point produces three CATE figures, one ITE figure, and one ATE figure.
``cate_iso_nrmse.png`` places
each method by its robust level error (median squared normalised ATE bias) and
heterogeneity recovery (median Shape R²), with point colour showing median
nPEHE. ``cate_speed_rank.png`` shows the accuracy/speed trade-off: median
benchmark time per episode on a log x-axis, against the median paired rank on
CATE nPEHE with 25th–75th percentile whiskers. Reference rows are excluded
from the rank plot, as they are from every competitor comparison.
``cate_npehe_boxplot.png`` shows the distribution of per-dataset CATE nPEHE
values for every method on one shared axis.
``ite_npehe_boxplot.png`` shows the corresponding per-dataset ITE nPEHE
distribution.
``ate_scatter.png`` compares each method's episode-level estimated ATE with
the corresponding true ATE from the ``ate`` task, with one shared-axis panel
per method.
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
from benchmarks.plot_style import (
    abbreviate,
    draw_anchors,
    save_figure,
    style_for,
    use_theme,
)


# The readable region is derived from the observed method summaries below.
# Shape R² is unbounded below, so retain a robust lower-tail censor and keep a
# small amount of room below the constant-ATE anchor at zero.
X_GRID_MIN = 0.0
Y_GRID_MAX = 1.02
CENSOR_QUANTILE = 0.95
Y_CENSOR_QUANTILE = 0.05
CENSOR_FLOOR = 0.05
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


def _cate_grid_bounds(summary: pd.DataFrame) -> tuple[float, float]:
    """Choose robust censoring bounds from the observed CATE summaries.

    The level coordinate is non-negative, while Shape R² can be arbitrarily
    negative on difficult or nearly homogeneous episodes. The upper 5% of
    level-error summaries and lower 5% of shape summaries are therefore shown
    just outside the plot rather than stretching the whole figure. The
    constants here are only degenerate-data fallbacks; ordinary runs determine
    both bounds from their observed values.
    """
    x_values = summary["ate_bias_norm_sq"].to_numpy(dtype=float)
    y_values = summary["shape_r2"].to_numpy(dtype=float)
    x_values = x_values[np.isfinite(x_values)]
    y_values = y_values[np.isfinite(y_values)]
    if not len(x_values) or not len(y_values):
        raise ValueError("CATE summary has no finite axis values")

    x_quantile = float(np.quantile(x_values, CENSOR_QUANTILE))
    x_max = max(CENSOR_FLOOR, x_quantile * 1.30)

    y_quantile = float(np.quantile(y_values, Y_CENSOR_QUANTILE))
    y_min = max(-4.0, min(-CENSOR_FLOOR, y_quantile - CENSOR_FLOOR))
    return x_max, y_min


def _place_cate_labels(fig, ax, labels: list) -> None:
    """Spread nearby labels apart using display-space collision checks."""
    if not labels:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = []
    candidates = (
        (5, 4),
        (5, -14),
        (5, 22),
        (5, -32),
        (-5, 22),
        (-5, -32),
        (5, 40),
        (5, -50),
        (-5, 40),
        (-5, -50),
    )
    axes_box = ax.get_window_extent(renderer)
    for label in labels:
        chosen = None
        for offset in candidates:
            label.set_position(offset)
            fig.canvas.draw()
            box = label.get_window_extent(renderer).expanded(1.02, 1.05)
            if box.y1 > axes_box.y1 + 2 or box.y0 < axes_box.y0 - 2:
                continue
            if not any(box.overlaps(previous) for previous in occupied):
                chosen = box
                break
        if chosen is None:
            chosen = label.get_window_extent(renderer)
        occupied.append(chosen)


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


def _npehe_boxplot_data(run: BenchmarkRun, task: Task) -> dict[str, np.ndarray]:
    """Return finite, per-dataset nPEHE values grouped by method.

    Unlike paired ranks, this distribution is method-local: a failed episode
    is omitted only from the method that failed. The result therefore retains
    the useful spread of each method across all datasets it completed.
    """
    frame = run.frame(task)
    if "npehe" not in frame.columns:
        return {}

    values: dict[str, np.ndarray] = {}
    for method, group in frame.groupby("method", sort=False):
        npehe = group["npehe"].to_numpy(dtype=float)
        npehe = npehe[np.isfinite(npehe)]
        if len(npehe):
            values[method] = npehe
    return values


def _cate_npehe_boxplot_data(run: BenchmarkRun) -> dict[str, np.ndarray]:
    """Return finite, per-dataset CATE nPEHE values grouped by method."""
    return _npehe_boxplot_data(run, TASKS["cate"])


def _ite_npehe_boxplot_data(run: BenchmarkRun) -> dict[str, np.ndarray]:
    """Return finite, per-dataset ITE nPEHE values grouped by method."""
    return _npehe_boxplot_data(run, TASKS["ite"])


def _plot_npehe_boxplot(
    run: BenchmarkRun, task_name: str, task_label: str, opts: Options
) -> None:
    """Draw one horizontal nPEHE boxplot per method for a task."""
    task = TASKS[task_name]
    plot_name = f"{task_name}_npehe_boxplot"
    if task_name not in {task.name for task in run.tasks}:
        print(f"  · {plot_name}: {task_label} results are absent — skipped")
        return

    data = _npehe_boxplot_data(run, task)
    if not data:
        print(f"  · {plot_name}: no finite {task_label} nPEHE values — skipped")
        return

    # Put ranked competitors first, followed by references in the same stable
    # order used by the other task figures. The final clause keeps this robust
    # to any non-standard rows in an externally produced results file.
    methods = [method for method in run.method_order(task) if method in data]
    methods += [method for method in run.references(task) if method in data]
    methods += [method for method in data if method not in methods]
    labels = [abbreviate(method) for method in methods]

    fig_height = max(4.8, 0.38 * len(methods) + 1.6)
    fig, ax = plt.subplots(figsize=(10.0, fig_height))
    positions = np.arange(1, len(methods) + 1)
    artists = ax.boxplot(
        [data[method] for method in methods],
        positions=positions,
        vert=False,
        patch_artist=True,
        widths=0.62,
        showfliers=True,
        medianprops={"color": "#222222", "linewidth": 1.8},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
        flierprops={
            "marker": ".",
            "markersize": 3.5,
            "markerfacecolor": "#555555",
            "markeredgecolor": "none",
            "alpha": 0.45,
        },
    )
    for method, box in zip(methods, artists["boxes"]):
        color = style_for(method).get("color", "gray")
        box.set_facecolor(color)
        box.set_edgecolor(color)
        box.set_alpha(0.72)

    draw_anchors(ax, "npehe", axis="x")
    ax.set(
        yticks=positions,
        yticklabels=labels,
        xlabel=f"{task_label} nPEHE per dataset",
        ylabel="Method",
        title=f"{task_label} nPEHE across datasets",
    )
    ax.margins(x=0.04)
    fig.tight_layout()
    save_figure(
        fig,
        run.plots_dir / f"{plot_name}.png",
        dpi=opts.dpi,
        close=False,
    )
    save_figure(fig, run.plots_dir / f"{plot_name}.pdf")


def plot_cate_npehe_boxplot(run: BenchmarkRun, opts: Options) -> None:
    """Draw one horizontal CATE nPEHE boxplot per method.

    Each observation is one dataset/episode-level nPEHE. Reference rows are
    retained as anchors, while failed or non-finite observations are omitted
    only from their own method's box.
    """
    _plot_npehe_boxplot(run, "cate", "CATE", opts)


def plot_ite_npehe_boxplot(run: BenchmarkRun, opts: Options) -> None:
    """Draw one horizontal ITE nPEHE boxplot per method."""
    _plot_npehe_boxplot(run, "ite", "ITE", opts)


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
    x_grid_max, y_grid_min = _cate_grid_bounds(summary)
    x_censored = x > x_grid_max
    y_censored = y < y_grid_min
    gx, gy = np.meshgrid(
        np.linspace(X_GRID_MIN, x_grid_max, 400),
        np.linspace(y_grid_min, 1.0, 400),
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
    labels = []

    for method, row in summary.iterrows():
        style = style_for(method)
        x_display = _display_coordinate(row.ate_bias_norm_sq, X_GRID_MIN, x_grid_max)
        y_display = _display_coordinate(row.shape_r2, y_grid_min, Y_GRID_MAX)
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
        labels.append(
            ax.annotate(
                abbreviate(method),
                (x_display, y_display),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
                color="0.25",
            )
        )

    # Only reserve and mark an out-of-scope strip when at least one observed
    # method was actually censored. Otherwise the axes end at the readable
    # region, so an empty strip is not mistaken for data space.
    x_view_max = x_grid_max + CENSOR_PAD + VIEW_PAD if x_censored.any() else x_grid_max
    y_view_min = y_grid_min - CENSOR_PAD - VIEW_PAD if y_censored.any() else y_grid_min
    if x_censored.any():
        ax.axvline(x_grid_max, color="0.45", linewidth=0.8, linestyle=":", zorder=1)
    if y_censored.any():
        ax.axhline(y_grid_min, color="0.45", linewidth=0.8, linestyle=":", zorder=1)
    ax.set(
        xlim=(X_GRID_MIN, x_view_max),
        ylim=(y_view_min, Y_GRID_MAX + VIEW_PAD),
    )
    ax.set_xlabel("median (normalised ATE bias)²")
    ax.set_ylabel("median Shape R²")
    if colour_mappable is not None:
        fig.colorbar(colour_mappable, ax=ax, shrink=0.82, label="median nPEHE")
    _place_cate_labels(fig, ax, labels)
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


def plot_ate_scatter(run: BenchmarkRun, opts: Options) -> None:
    """Draw one true-versus-estimated ATE panel per method.

    The ``ate`` task is the level-only view of the benchmark. Its rows already
    contain ``ate_true`` (the episode target mean) and ``ate_pred`` (the mean
    predicted effect), so this figure does not need to reconstruct an ATE from
    the row-level predictions. Reference rows are retained as useful anchors;
    methods with no finite estimate contribute no panel. All panels share the
    same axis limits so their calibration and spread can be compared directly.
    """
    if "ate" not in {task.name for task in run.tasks}:
        print("  · ate_scatter: ATE results are absent — skipped")
        return

    frame = run.frame(TASKS["ate"])
    required = {"method", "ate_true", "ate_pred"}
    if not required.issubset(frame.columns):
        print("  · ate_scatter: ATE columns are absent — skipped")
        return
    frame = frame.dropna(subset=["ate_true", "ate_pred"])
    frame = frame[
        np.isfinite(frame["ate_true"].to_numpy())
        & np.isfinite(frame["ate_pred"].to_numpy())
    ]
    if frame.empty:
        print("  · ate_scatter: no finite ATE estimates — skipped")
        return

    true_values = frame["ate_true"].to_numpy()
    pred_values = frame["ate_pred"].to_numpy()
    low = float(min(true_values.min(), pred_values.min()))
    high = float(max(true_values.max(), pred_values.max()))
    span = high - low
    pad = 0.05 * span if span > 0 else max(0.05 * abs(low), 0.05)
    limits = (low - pad, high + pad)

    groups = list(frame.groupby("method", sort=False))
    n_methods = len(groups)
    n_columns = min(4, n_methods)
    n_rows = int(np.ceil(n_methods / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.6 * n_columns, 3.5 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()
    for ax in axes[n_methods:]:
        fig.delaxes(ax)

    for ax, (method, group) in zip(axes[:n_methods], groups):
        style = style_for(method)
        ax.plot(
            limits,
            limits,
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            zorder=1,
        )
        ax.scatter(
            group["ate_true"],
            group["ate_pred"],
            color=style.get("color", "gray"),
            marker=style.get("marker", "o"),
            s=style.get("s", 110),
            edgecolors="white",
            linewidths=0.7,
            alpha=0.78,
            zorder=2,
        )
        ax.set(xlim=limits, ylim=limits, title=method)
        ax.set_aspect("equal", adjustable="box")

    fig.supxlabel("True ATE")
    fig.supylabel("Estimated ATE")
    fig.suptitle("Estimated versus true ATE by method", y=0.995)
    fig.tight_layout(rect=(0.04, 0.04, 1.0, 0.95))
    save_figure(
        fig,
        run.plots_dir / "ate_scatter.png",
        dpi=opts.dpi,
        close=False,
    )
    save_figure(fig, run.plots_dir / "ate_scatter.pdf")


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
    plot_cate_npehe_boxplot(run, Options(dpi=args.dpi))
    plot_ite_npehe_boxplot(run, Options(dpi=args.dpi))
    plot_ate_scatter(run, Options(dpi=args.dpi))


if __name__ == "__main__":
    main()
