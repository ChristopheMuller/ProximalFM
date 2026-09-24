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
    relevant_methods,
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
DEFAULT_Y_MAX = 2.0
SINGLE_CASE_FIGSIZE = (4.8, 4.8)
CONFIDENCE_INTERVAL_CHOICES = ("false", "iqr", "std", "ste")
CONF_PROXY_ORDER = (
    ("low", "low"),
    ("low", "high"),
    ("high", "low"),
    ("high", "high"),
)


def _curve_line_kwargs(method: str) -> dict[str, object]:
    """Keep ProximalFM prominent without changing shared method styles."""
    kwargs = method_style(method).line_kwargs()
    if method != "ProximalFM":
        kwargs["linewidth"] *= 0.8
    return kwargs


def draw_learning_curve_panel(
    ax,
    cell: pd.DataFrame,
    *,
    methods: Sequence[str],
    contexts: Sequence[int],
    family: str,
    confounding: str,
    proxy: str,
    y_max: float = DEFAULT_Y_MAX,
    confidence_interval: str = "false",
    show_title: bool = True,
) -> None:
    """Draw one learning-curve cell on an existing axis."""
    for method in methods:
        # Reindex explicitly so a failed fit leaves an honest break instead of
        # connecting the successful points on either side. A method that
        # failed throughout this panel simply contributes no visible curve;
        # it remains in the shared legend and in every panel where it worked.
        curve = cell[cell["method"].eq(method)].set_index("n_ctx").reindex(contexts)
        style = method_style(method)
        x = curve.index.to_numpy(dtype=float)
        if confidence_interval == "iqr":
            lower = curve["q25"].to_numpy(dtype=float)
            upper = curve["q75"].to_numpy(dtype=float)
        elif confidence_interval in {"std", "ste"}:
            spread = curve[confidence_interval].to_numpy(dtype=float)
            center = curve["median"].to_numpy(dtype=float)
            lower = center - spread
            upper = center + spread
        else:
            lower = upper = None
        if lower is not None:
            ax.fill_between(
                x,
                lower,
                upper,
                color=style.color,
                alpha=0.045,
                linewidth=0,
                zorder=1,
            )
        ax.plot(x, curve["median"].to_numpy(dtype=float), **_curve_line_kwargs(method))

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
    if show_title:
        ax.set_title(
            f"{confounding.capitalize()} confounding\n"
            f"{proxy.capitalize()} proxy reliability"
        )
    ax.set_ylim(0, y_max)
    ax.set_ylabel(f"{family.capitalize()}\nCATE nPEHE")
    ax.set_xlabel("Context size")


def _validate_summary(
    summary: pd.DataFrame,
    methods: Sequence[str],
    *,
    case: str | None = None,
) -> list[int]:
    """Validate the selected learning-curve cells and return context sizes."""
    selected = summary if case is None else summary[summary["case"].eq(case)]
    if selected.empty:
        available = sorted(str(value) for value in summary["case"].unique())
        raise ValueError(
            f"unknown case {case!r}; available cases are: {', '.join(available)}"
        )

    contexts = sorted(int(value) for value in selected["n_ctx"].unique())
    if len(contexts) < 2:
        raise ValueError("learning curves require at least two context sizes")

    if case is not None:
        cells = set(
            selected[["family", "confounding", "proxy"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if len(cells) != 1:
            raise ValueError(f"case {case!r} maps to multiple factorial cells")

    keys = ["family", "confounding", "proxy", "method", "n_ctx"]
    duplicates = selected[selected["method"].isin(methods)].duplicated(keys, keep=False)
    if duplicates.any():
        sample = selected.loc[duplicates, keys].head().to_dict("records")
        raise ValueError(
            "learning-curve summary has duplicate method points, " f"e.g. {sample}"
        )
    return contexts


def make_learning_curve_figure(
    summary: pd.DataFrame,
    *,
    methods: Sequence[str] | None = None,
    case: str | None = None,
    y_max: float = DEFAULT_Y_MAX,
    confidence_interval: str | bool = False,
    interval: str | None = None,
):
    """Create the factorial learning-curve figure, or one selected case.

    Rows distinguish linear/nonlinear mechanisms; columns cross confounding
    and proxy strength. Curves are medians over episodes. The optional bands
    can show the interquartile range (``"iqr"``), standard deviation
    (``"std"``), or standard error (``"ste"``) across episodes.

    When ``case`` is provided, it must be a matrix case name such as
    ``"linear_conf_low_proxy_low"`` and only that cell is rendered.

    Missing method/context points are rendered as gaps. This keeps successful
    results visible when a competitor fails in one cell without implying an
    interpolated result for the failed fit.

    ``interval`` is retained as a backwards-compatible alias for
    ``confidence_interval``. Its ``"iqr"`` value selects the IQR bands and its
    ``"none"`` value selects ``"false"``.
    """
    if interval is not None:
        if interval not in {"iqr", "none"}:
            raise ValueError("interval must be 'iqr' or 'none'")
        confidence_interval = "iqr" if interval == "iqr" else "false"
    elif confidence_interval is True:
        # Keep the old programmatic boolean API working as an IQR alias.
        confidence_interval = "iqr"
    elif confidence_interval is False:
        confidence_interval = "false"
    if confidence_interval not in CONFIDENCE_INTERVAL_CHOICES:
        raise ValueError(
            f"confidence_interval must be one of {CONFIDENCE_INTERVAL_CHOICES}"
        )
    if y_max <= 0:
        raise ValueError("y_max must be positive")
    plot_summary = summary if case is None else summary[summary["case"].eq(case)]
    if case is not None and plot_summary.empty:
        available = sorted(str(value) for value in summary["case"].unique())
        raise ValueError(
            f"unknown case {case!r}; available cases are: {', '.join(available)}"
        )
    available = set(plot_summary["method"])
    methods = relevant_methods(available) if methods is None else list(methods)
    absent = set(methods) - available
    if absent:
        raise ValueError(f"requested methods absent from summary: {sorted(absent)}")
    for method in methods:
        method_style(method)
    contexts = _validate_summary(summary, methods, case=case)

    apply_style()
    if case is None:
        panel_specs = [
            (row, col, family, confounding, proxy)
            for row, family in enumerate(("linear", "nonlinear"))
            for col, (confounding, proxy) in enumerate(CONF_PROXY_ORDER)
        ]
        fig, axes = plt.subplots(
            2, 4, figsize=learning_curve_figsize(), sharex=True, sharey=True
        )
        axes = list(axes.flat)
    else:
        factors = plot_summary[["family", "confounding", "proxy"]].drop_duplicates()
        family, confounding, proxy = next(factors.itertuples(index=False, name=None))
        panel_specs = [(0, 0, family, confounding, proxy)]
        fig, axes = plt.subplots(figsize=SINGLE_CASE_FIGSIZE)
        axes = [axes]

    for ax, (row, col, family, confounding, proxy) in zip(axes, panel_specs):
        cell = plot_summary[
            plot_summary["family"].eq(family)
            & plot_summary["confounding"].eq(confounding)
            & plot_summary["proxy"].eq(proxy)
        ]
        draw_learning_curve_panel(
            ax,
            cell,
            methods=methods,
            contexts=contexts,
            family=family,
            confounding=confounding,
            proxy=proxy,
            y_max=y_max,
            confidence_interval=confidence_interval,
            # Each column has the same confounding/proxy combination in both
            # rows, so repeating these titles below adds height but no context.
            show_title=case is not None or row == 0,
        )
        if case is None and col != 0:
            ax.set_ylabel("")
        if case is None and row != 1:
            ax.set_xlabel("")

    handles = [
        Line2D(
            [],
            [],
            label=method_style(method).label,
            **_curve_line_kwargs(method),
        )
        for method in methods
    ]
    # The full matrix has a wide canvas, but a single panel often contains
    # long method labels. Wrap that legend into two columns so it stays inside
    # the saved figure instead of extending past the left and right edges.
    # Four columns give the twelve-method curated set three balanced rows.
    # Five columns left a visually awkward final row with only two entries.
    legend_ncol = min(2, len(handles)) if case is not None else min(4, len(handles))
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=legend_ncol,
        columnspacing=1.2,
        handlelength=2.6,
    )
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=DEFAULT_MATRIX_ROOT,
        help=f"Matrix build root (default: {DEFAULT_MATRIX_ROOT})",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Dimension in a per-dimension matrix (default: 5)",
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
        help=(
            "Method subset; defaults to design.RELEVANT_METHODS. "
            "Names must match results.csv exactly"
        ),
    )
    parser.add_argument(
        "--case",
        "--config",
        dest="case",
        metavar="NAME",
        help=(
            "Plot one matrix case as a single subplot, e.g. "
            "linear_conf_low_proxy_low (alias: --config)"
        ),
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=DEFAULT_Y_MAX,
        help=f"Y-axis maximum (default: {DEFAULT_Y_MAX})",
    )
    parser.add_argument(
        "--confidence-interval",
        type=str.lower,
        choices=CONFIDENCE_INTERVAL_CHOICES,
        default=False,
        help="Interval band: false, iqr, std, or ste (default: false)",
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
    frame = load_cate_results(
        args.matrix_root, selected_case=args.case, dimension=args.dim
    )
    _, incomplete = complete_learning_curve_methods(frame)
    selected = (
        args.methods
        if args.methods is not None
        else relevant_methods(frame["method"].unique())
    )
    summary = summarize_learning_curves(frame, selected)
    methods = ordered_methods(set(summary["method"]))
    if args.interval is not None and args.confidence_interval is not False:
        raise ValueError("use only one of --confidence-interval and --interval")
    fig = make_learning_curve_figure(
        summary,
        methods=methods,
        case=args.case,
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
    incomplete_selected = [method for method in methods if method in incomplete]
    if incomplete_selected:
        print(
            "Plotted available points with gaps for methods without complete "
            "factorial/context coverage: " + ", ".join(incomplete_selected)
        )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
