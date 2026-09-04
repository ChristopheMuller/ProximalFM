"""Eight-cell context-size learning curves for the semisynthetic benchmark.

Run from the repository root::

    uv run python -m analysis_eval.learning_curves
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from .design import (
    COLORS,
    apply_style,
    learning_curve_figsize,
    method_style,
    ordered_methods,
)
from .plotting import plt, save_figure
from .semisynt import (
    complete_learning_curve_methods,
    load_cate_results,
    summarize_learning_curves,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX_ROOT = REPO_ROOT / "data" / "semisynt_eval" / "semisynt_val_v1"
DEFAULT_OUT_DIR = Path(__file__).with_name("results")
OUTPUT_NAME = "semisynt_learning_curves"
DEFAULT_Y_MAX = 1.2
CONF_PROXY_ORDER = (
    ("low", "low"),
    ("low", "high"),
    ("high", "low"),
    ("high", "high"),
)


def _validate_summary(summary: pd.DataFrame, methods: Sequence[str]) -> list[int]:
    contexts = sorted(int(value) for value in summary["n_ctx"].unique())
    if len(contexts) < 2:
        raise ValueError("learning curves require at least two context sizes")
    expected = {
        (family, confounding, proxy, method, n_ctx)
        for family in ("linear", "nonlinear")
        for confounding, proxy in CONF_PROXY_ORDER
        for method in methods
        for n_ctx in contexts
    }
    observed = set(
        summary[["family", "confounding", "proxy", "method", "n_ctx"]].itertuples(
            index=False, name=None
        )
    )
    missing = expected - observed
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"learning-curve grid is incomplete ({len(missing)} cells missing), "
            f"e.g. {sample}"
        )
    return contexts


def make_learning_curve_figure(
    summary: pd.DataFrame,
    *,
    methods: Sequence[str] | None = None,
    y_max: float = DEFAULT_Y_MAX,
    confidence_interval: bool = False,
    interval: str | None = None,
):
    """Create the 2x4 factorial learning-curve figure.

    Rows distinguish linear/nonlinear mechanisms; columns cross confounding
    and proxy strength. Curves are medians over episodes and the optional faint
    bands are interquartile ranges, not statistical confidence intervals.

    ``interval`` is retained as a backwards-compatible alias for
    ``confidence_interval``. Its ``"iqr"`` value enables the bands and its
    ``"none"`` value disables them.
    """
    if interval is not None:
        if interval not in {"iqr", "none"}:
            raise ValueError("interval must be 'iqr' or 'none'")
        confidence_interval = interval == "iqr"
    if y_max <= 0:
        raise ValueError("y_max must be positive")
    available = set(summary["method"])
    methods = ordered_methods(available) if methods is None else list(methods)
    for method in methods:
        method_style(method)
    contexts = _validate_summary(summary, methods)

    apply_style()
    fig, axes = plt.subplots(
        2, 4, figsize=learning_curve_figsize(), sharex=True, sharey=True
    )
    for row, family in enumerate(("linear", "nonlinear")):
        for col, (confounding, proxy) in enumerate(CONF_PROXY_ORDER):
            ax = axes[row, col]
            cell = summary[
                summary["family"].eq(family)
                & summary["confounding"].eq(confounding)
                & summary["proxy"].eq(proxy)
            ]
            for method in methods:
                curve = cell[cell["method"].eq(method)].sort_values("n_ctx")
                style = method_style(method)
                x = curve["n_ctx"].to_numpy(dtype=float)
                if confidence_interval:
                    ax.fill_between(
                        x,
                        curve["q25"].to_numpy(dtype=float),
                        curve["q75"].to_numpy(dtype=float),
                        color=style.color,
                        alpha=0.045,
                        linewidth=0,
                        zorder=1,
                    )
                ax.plot(x, curve["median"].to_numpy(dtype=float), **style.line_kwargs())

            ax.axhline(
                1.0,
                color=COLORS["grey"],
                linestyle=":",
                linewidth=0.8,
                zorder=0,
            )
            ax.set_xscale("log", base=2)
            ax.set_xticks(contexts)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.minorticks_off()
            ax.set_title(
                f"{confounding.capitalize()} confounding\n"
                f"{proxy.capitalize()} proxy reliability"
            )
            ax.set_ylim(0, y_max)
            if col == 0:
                ax.set_ylabel(f"{family.capitalize()}\nCATE nPEHE")
            if row == 1:
                ax.set_xlabel("Context size")

    handles = [
        Line2D(
            [],
            [],
            label=method_style(method).label,
            **method_style(method).line_kwargs(),
        )
        for method in methods
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(5, len(handles)),
        columnspacing=1.2,
        handlelength=2.6,
    )
    fig.suptitle("Semisynthetic CATE accuracy as context grows")
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
        "--methods",
        nargs="+",
        help="Optional method subset; names must match results.csv exactly",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=DEFAULT_Y_MAX,
        help=f"Y-axis maximum (default: {DEFAULT_Y_MAX})",
    )
    parser.add_argument(
        "--confidence-interval",
        action="store_true",
        default=False,
        help="Draw the episode interquartile bands",
    )
    parser.add_argument(
        "--interval",
        choices=("iqr", "none"),
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_cate_results(args.matrix_root)
    complete, incomplete = complete_learning_curve_methods(frame)
    selected = args.methods if args.methods is not None else complete
    summary = summarize_learning_curves(frame, selected)
    methods = ordered_methods(set(summary["method"]))
    if args.interval is not None and args.confidence_interval:
        raise ValueError("use only one of --confidence-interval and --interval")
    fig = make_learning_curve_figure(
        summary,
        methods=methods,
        y_max=args.y_max,
        confidence_interval=args.confidence_interval,
        interval=args.interval,
    )
    paths = save_figure(fig, args.out_dir, args.name)
    print(
        f"Loaded {frame['case'].nunique()} cases, "
        f"{frame['episode'].nunique()} episode IDs, {len(methods)} methods, "
        f"and {frame['n_ctx'].nunique()} context sizes."
    )
    if args.methods is None and incomplete:
        print(
            "Excluded methods without complete factorial/context coverage: "
            + ", ".join(sorted(incomplete))
        )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
