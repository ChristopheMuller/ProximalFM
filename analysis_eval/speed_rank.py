"""Low/high-confounding runtime versus paired nPEHE-rank figure.

Run from the repository root::

    uv run python -m analysis_eval.speed_rank
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .design import (
    COLORS,
    RELEVANT_METHODS,
    apply_style,
    method_style,
    ordered_methods,
    speed_rank_figsize,
)
from .plotting import place_point_labels, plt, save_figure
from .semisynt import (
    LEVELS,
    complete_learning_curve_methods,
    load_cate_results,
    summarize_speed_rank,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX_ROOT = REPO_ROOT / "data" / "semisynt_eval" / "semisynt_val_v1"
DEFAULT_OUT_DIR = Path(__file__).with_name("results")
OUTPUT_NAME = "semisynt_speed_rank"
MUTED_METHOD_COLOR = COLORS["light_grey"]
MUTED_METHOD_ALPHA = 0.55


def make_speed_rank_figure(summary: pd.DataFrame):
    """Draw one paired speed/rank panel per confounding level."""
    required = {
        "confounding",
        "method",
        "rank_median",
        "rank_q25",
        "rank_q75",
        "time_mean",
        "time_post_tune_mean",
        "n_ctx",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"speed/rank figure requires columns {sorted(missing)}")
    if summary.empty:
        raise ValueError("speed/rank figure received an empty summary")
    if summary["n_ctx"].nunique() != 1:
        raise ValueError("speed/rank figure requires exactly one context size")
    if set(summary["confounding"]) != set(LEVELS):
        raise ValueError(
            f"speed/rank figure requires confounding levels {LEVELS}, "
            f"got {sorted(summary['confounding'].unique())}"
        )

    methods = ordered_methods(set(summary["method"]))
    apply_style()
    fig, axes = plt.subplots(
        1, 2, figsize=speed_rank_figsize(), sharex=True, sharey=True
    )
    panel_labels = []
    for ax, confounding in zip(axes, LEVELS):
        panel = summary[summary["confounding"].eq(confounding)]
        panel = panel.set_index("method").loc[methods].reset_index()
        labels = []
        for row in panel.itertuples():
            style = method_style(row.method)
            highlighted = row.method in RELEVANT_METHODS
            color = style.color if highlighted else MUTED_METHOD_COLOR
            alpha = style.alpha if highlighted else MUTED_METHOD_ALPHA
            zorder = style.zorder if highlighted else 1
            lower = row.rank_median - row.rank_q25
            upper = row.rank_q75 - row.rank_median
            ax.errorbar(
                row.time_mean,
                row.rank_median,
                yerr=np.array([[lower], [upper]]),
                color=color,
                marker=style.marker,
                markersize=max(style.markersize, 5.0),
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.55,
                linestyle="none",
                linewidth=1.15 if highlighted else 0.8,
                capsize=2.5,
                zorder=zorder + 1,
                alpha=alpha,
            )

            post = row.time_post_tune_mean
            if np.isfinite(post) and post < row.time_mean * (1.0 - 1e-3):
                ax.plot(
                    [post, row.time_mean],
                    [row.rank_median, row.rank_median],
                    color=color,
                    linestyle=":",
                    linewidth=0.9 if highlighted else 0.7,
                    alpha=alpha,
                    zorder=zorder,
                )
                ax.plot(
                    post,
                    row.rank_median,
                    color=color,
                    marker=style.marker,
                    markersize=max(style.markersize, 5.0),
                    markerfacecolor="white",
                    markeredgewidth=1.1,
                    linestyle="none",
                    alpha=alpha,
                    zorder=zorder + 2,
                )

            if highlighted:
                labels.append(
                    ax.annotate(
                        style.label,
                        (row.time_mean, row.rank_median),
                        xytext=(5, 0),
                        textcoords="offset points",
                        color=style.color,
                        fontsize=6.8,
                        va="center",
                        clip_on=False,
                    )
                )

        ax.set_xscale("log")
        ax.set_title(f"{confounding.capitalize()} confounding")
        ax.set_xlabel("Mean runtime (seconds per episode, log scale)")
        panel_labels.append(labels)

    context_size = int(summary["n_ctx"].iloc[0])
    axes[0].set_ylim(len(methods) + 0.75, 0.25)
    axes[0].set_yticks(range(1, len(methods) + 1))
    fig.supylabel("CATE nPEHE rank (median and IQR)")
    # fig.suptitle(f"Speed–accuracy trade-off at context size {context_size:,}")
    for ax, labels in zip(axes, panel_labels):
        place_point_labels(fig, ax, labels)

    handles = [
        Line2D(
            [],
            [],
            color="black",
            marker="o",
            markerfacecolor="black",
            linestyle="none",
            label="End-to-end runtime",
        )
    ]
    has_distinct_post_time = (
        np.isfinite(summary["time_post_tune_mean"])
        & summary["time_post_tune_mean"].lt(summary["time_mean"] * (1.0 - 1e-3))
    ).any()
    if has_distinct_post_time:
        handles.append(
            Line2D(
                [],
                [],
                color="black",
                marker="o",
                markerfacecolor="white",
                linestyle=":",
                label="Post-selection runtime",
            )
        )
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        columnspacing=1.5,
    )
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=DEFAULT_MATRIX_ROOT,
        help=f"Eight-case matrix root (default: {DEFAULT_MATRIX_ROOT})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--name", default=OUTPUT_NAME, help="Output filename stem")
    parser.add_argument(
        "--context-size",
        type=int,
        help="Context size to plot (default: largest available)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional method subset; names must match results.csv exactly",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_cate_results(args.matrix_root)
    complete, incomplete = complete_learning_curve_methods(frame)
    selected = args.methods if args.methods is not None else complete
    methods = ordered_methods(set(selected))
    summary = summarize_speed_rank(frame, methods, context_size=args.context_size)
    fig = make_speed_rank_figure(summary)
    paths = save_figure(fig, args.out_dir, args.name)

    block_counts = ", ".join(
        f"{confounding}={int(group['n_blocks'].iloc[0])}"
        for confounding, group in summary.groupby("confounding", sort=False)
    )
    print(
        f"Ranked {summary['method'].nunique()} methods separately by confounding "
        f"({block_counts} paired blocks) at n_ctx={int(summary['n_ctx'].iloc[0])}."
    )
    if args.methods is None and incomplete:
        print(
            "Excluded methods without complete factorial/context coverage: "
            + ", ".join(sorted(incomplete))
        )
    missing_post = sorted(
        set(
            summary.loc[~np.isfinite(summary["time_post_tune_mean"]), "method"].tolist()
        )
    )
    if missing_post:
        print(
            "Post-selection timing unavailable for: "
            + ", ".join(missing_post)
            + ". Re-run their benchmark caches with --force to add it."
        )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
