"""Level/heterogeneity error trajectories as context grows.

Run from the repository root::

    uv run python -m analysis_eval.error_decomposition
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
    CONTEXT_MARKER_AREAS,
    ISO_NPEHE_LEVELS,
    apply_style,
    decomposition_figsize,
    method_style,
    ordered_methods,
    relevant_methods,
)
from .learning_curves import CONF_PROXY_ORDER, DEFAULT_MATRIX_ROOT, SINGLE_CASE_FIGSIZE
from .plotting import plt, save_figure
from .semisynt import (
    complete_learning_curve_methods,
    load_cate_results,
    summarize_error_decomposition,
)

DEFAULT_OUT_DIR = Path(__file__).with_name("results")
OUTPUT_NAME = "semisynt_error_decomposition"


def _validate_summary(
    summary: pd.DataFrame,
    methods: Sequence[str],
    *,
    case: str | None = None,
) -> list[int]:
    """Validate the selected decomposition cells and return context sizes."""
    selected = summary if case is None else summary[summary["case"].eq(case)]
    if selected.empty:
        available = sorted(str(value) for value in summary["case"].unique())
        raise ValueError(
            f"unknown case {case!r}; available cases are: {', '.join(available)}"
        )

    contexts = sorted(int(value) for value in selected["n_ctx"].unique())
    if len(contexts) < 2:
        raise ValueError("decomposition trajectories require two context sizes")

    if case is None:
        cells = {
            (family, confounding, proxy)
            for family in ("linear", "nonlinear")
            for confounding, proxy in CONF_PROXY_ORDER
        }
    else:
        cells = set(
            selected[["family", "confounding", "proxy"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if len(cells) != 1:
            raise ValueError(f"case {case!r} maps to multiple factorial cells")

    expected = {
        (family, confounding, proxy, method, n_ctx)
        for family, confounding, proxy in cells
        for method in methods
        for n_ctx in contexts
    }
    observed = set(
        selected[["family", "confounding", "proxy", "method", "n_ctx"]].itertuples(
            index=False, name=None
        )
    )
    missing = expected - observed
    if missing:
        raise ValueError(
            f"decomposition grid is incomplete ({len(missing)} points missing), "
            f"e.g. {sorted(missing)[:5]}"
        )
    return contexts


def _axis_limits(summary: pd.DataFrame) -> tuple[float, float]:
    """Panel limits with fixed fractional headroom and an origin anchor."""
    x_max = float(summary["level_error"].max())
    y_max = float(summary["shape_error"].max())
    if not np.isfinite(x_max) or not np.isfinite(y_max):
        raise ValueError("decomposition contains non-finite coordinates")
    return max(0.025, x_max * 1.14), max(0.25, y_max * 1.10)


def _draw_iso_npehe(ax, x_max: float, y_max: float) -> None:
    """Draw contours of ``sqrt(level + shape)`` behind the trajectories."""
    x = np.linspace(0.0, x_max, 160)
    y = np.linspace(0.0, y_max, 220)
    xx, yy = np.meshgrid(x, y)
    score = np.sqrt(xx + yy)
    levels = [level for level in ISO_NPEHE_LEVELS if level**2 < x_max + y_max]
    if not levels:
        return
    contours = ax.contour(
        xx,
        yy,
        score,
        levels=levels,
        colors=COLORS["light_grey"],
        linewidths=0.55,
        linestyles="dotted",
        zorder=0,
    )
    ax.clabel(
        contours,
        inline=True,
        fontsize=6.2,
        fmt=lambda value: f"{value:g}",
        colors=COLORS["grey"],
    )


def _validate_limits(limits: Sequence[float], name: str) -> tuple[float, float]:
    if len(limits) != 2:
        raise ValueError(f"{name} must contain a lower and upper bound")
    lower, upper = (float(value) for value in limits)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"{name} must contain finite bounds")
    if upper <= lower:
        raise ValueError(f"{name} upper bound must be greater than its lower bound")
    return lower, upper


def _clip_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return the part of a segment inside the plotting rectangle."""
    x0, y0 = start
    dx, dy = end[0] - x0, end[1] - y0
    t_start, t_end = 0.0, 1.0
    # Liang--Barsky clipping, expressed as p*t <= q for each edge.
    for p, q in (
        (-dx, x0 - xlim[0]),
        (dx, xlim[1] - x0),
        (-dy, y0 - ylim[0]),
        (dy, ylim[1] - y0),
    ):
        if p == 0.0:
            if q < 0.0:
                return None
            continue
        t = q / p
        if p < 0.0:
            if t > t_end:
                return None
            t_start = max(t_start, t)
        else:
            if t < t_start:
                return None
            t_end = min(t_end, t)
    if t_start > t_end:
        return None
    return (
        (x0 + t_start * dx, y0 + t_start * dy),
        (x0 + t_end * dx, y0 + t_end * dy),
    )


def _draw_trajectory(
    ax,
    curve: pd.DataFrame,
    method: str,
    contexts: list[int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    style = method_style(method)
    curve = curve.set_index("n_ctx").loc[contexts].reset_index()
    x = curve["level_error"].to_numpy(dtype=float)
    y = curve["shape_error"].to_numpy(dtype=float)

    # Keep the line data inside the fixed rectangle as well as clipping it at
    # render time. NaN separators prevent a path that leaves the grid from
    # being joined back to it when a later node happens to re-enter.
    line_x: list[float] = []
    line_y: list[float] = []
    for i in range(len(x) - 1):
        clipped = _clip_segment((x[i], y[i]), (x[i + 1], y[i + 1]), xlim, ylim)
        if clipped is None:
            continue
        if line_x:
            line_x.append(np.nan)
            line_y.append(np.nan)
        (x_start, y_start), (x_end, y_end) = clipped
        line_x.extend((x_start, x_end))
        line_y.extend((y_start, y_end))
        if (x_start, y_start) != (x_end, y_end):
            ax.annotate(
                "",
                xy=(x_end, y_end),
                xytext=(x_start, y_start),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": style.color,
                    "linewidth": 0.65,
                    "alpha": 0.75,
                    "mutation_scale": 6.0,
                    "shrinkA": 5.0,
                    "shrinkB": 5.0,
                },
                zorder=style.zorder + 0.1,
            )
    if line_x:
        ax.plot(
            line_x,
            line_y,
            color=style.color,
            linestyle=style.linestyle,
            linewidth=style.linewidth,
            alpha=style.alpha,
            zorder=style.zorder,
        )

    sizes = np.interp(
        np.arange(len(contexts)),
        [0, max(len(contexts) - 1, 1)],
        [CONTEXT_MARKER_AREAS[0], CONTEXT_MARKER_AREAS[-1]],
    )
    for i, (xi, yi) in enumerate(zip(x, y)):
        if not (xlim[0] <= xi <= xlim[1] and ylim[0] <= yi <= ylim[1]):
            continue
        ax.scatter(
            xi,
            yi,
            s=float(sizes[i]),
            marker=style.marker,
            color=style.color,
            edgecolor="white",
            linewidth=0.35,
            alpha=style.alpha,
            zorder=style.zorder + 0.2,
        )


def draw_error_decomposition_panel(
    ax,
    cell: pd.DataFrame,
    *,
    methods: Sequence[str],
    contexts: list[int],
    family: str,
    confounding: str,
    proxy: str,
    xlim: Sequence[float] | None = None,
    ylim: Sequence[float] | None = None,
) -> None:
    """Draw one error-decomposition cell on an existing axis."""
    data_x_max, data_y_max = _axis_limits(cell)
    x_limits = _validate_limits(xlim, "xlim") if xlim is not None else (0.0, data_x_max)
    y_limits = _validate_limits(ylim, "ylim") if ylim is not None else (0.0, data_y_max)
    _draw_iso_npehe(ax, x_limits[1], y_limits[1])
    for method in methods:
        _draw_trajectory(
            ax,
            cell[cell["method"].eq(method)],
            method,
            contexts,
            x_limits,
            y_limits,
        )
    ax.scatter(0, 0, marker="+", s=30, color=COLORS["black"], zorder=8)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_title(
        f"{confounding.capitalize()} confounding\n"
        f"{proxy.capitalize()} proxy reliability"
    )
    ax.set_ylabel(f"{family.capitalize()}\nHeterogeneity error")
    ax.set_xlabel(r"Level error: normalized ATE bias$^2$")


def make_error_decomposition_figure(
    summary: pd.DataFrame,
    *,
    methods: Sequence[str] | None = None,
    case: str | None = None,
    aggregation: str = "median",
    xlim: Sequence[float] | None = None,
    ylim: Sequence[float] | None = None,
):
    """Create the factorial decomposition figure, or one selected case.

    By default, each panel derives its own x and y limits from the selected
    methods. ``xlim`` or ``ylim`` can independently impose a common bound for
    that dimension. Trajectory segments are clipped to the resulting bounds,
    so a node outside the grid makes its line stop at the boundary.

    When ``case`` is provided, it must be a matrix case name such as
    ``"linear_conf_low_proxy_low"`` and only that cell is rendered.
    """
    plot_summary = summary if case is None else summary[summary["case"].eq(case)]
    if case is not None and plot_summary.empty:
        available_cases = sorted(str(value) for value in summary["case"].unique())
        raise ValueError(
            f"unknown case {case!r}; available cases are: "
            f"{', '.join(available_cases)}"
        )
    available = set(plot_summary["method"])
    methods = relevant_methods(available) if methods is None else list(methods)
    methods = ordered_methods(methods)
    contexts = _validate_summary(summary, methods, case=case)
    fixed_xlim = _validate_limits(xlim, "xlim") if xlim is not None else None
    fixed_ylim = _validate_limits(ylim, "ylim") if ylim is not None else None
    apply_style()
    if case is None:
        panel_specs = [
            (row, col, family, confounding, proxy)
            for row, family in enumerate(("linear", "nonlinear"))
            for col, (confounding, proxy) in enumerate(CONF_PROXY_ORDER)
        ]
        fig, axes = plt.subplots(2, 4, figsize=decomposition_figsize())
        axes = list(axes.flat)
    else:
        factors = plot_summary[["family", "confounding", "proxy"]].drop_duplicates()
        family, confounding, proxy = next(factors.itertuples(index=False, name=None))
        panel_specs = [(0, 0, family, confounding, proxy)]
        fig, axis = plt.subplots(figsize=SINGLE_CASE_FIGSIZE)
        axes = [axis]

    for ax, (row, col, family, confounding, proxy) in zip(axes, panel_specs):
        cell = plot_summary[
            plot_summary["family"].eq(family)
            & plot_summary["confounding"].eq(confounding)
            & plot_summary["proxy"].eq(proxy)
            & plot_summary["method"].isin(methods)
        ]
        draw_error_decomposition_panel(
            ax,
            cell,
            methods=methods,
            contexts=contexts,
            family=family,
            confounding=confounding,
            proxy=proxy,
            xlim=fixed_xlim,
            ylim=fixed_ylim,
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
            **method_style(method).line_kwargs(),
        )
        for method in methods
    ]
    legend_ncol = min(2, len(handles)) if case is not None else min(4, len(handles))
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=legend_ncol,
        columnspacing=0.9,
        handlelength=2.1,
    )
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Dimension in a per-dimension matrix (default: 5)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--name", default=OUTPUT_NAME)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
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
        "--aggregation",
        choices=("median", "mean"),
        default="median",
        help="Robust component medians, or means preserving the exact identity",
    )
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="Common level-error limits (default: data-dependent per panel)",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="Common heterogeneity-error limits (default: data-dependent per panel)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_cate_results(
        args.matrix_root, selected_case=args.case, dimension=args.dim
    )
    complete, incomplete = complete_learning_curve_methods(frame)
    selected = args.methods if args.methods is not None else relevant_methods(complete)
    summary = summarize_error_decomposition(
        frame, selected, aggregation=args.aggregation
    )
    fig = make_error_decomposition_figure(
        summary,
        methods=selected,
        case=args.case,
        aggregation=args.aggregation,
        xlim=args.xlim,
        ylim=args.ylim,
    )
    paths = save_figure(fig, args.out_dir, args.name)
    print(
        f"Rendered {len(selected)} methods over {frame['case'].nunique()} cases "
        f"and {frame['n_ctx'].nunique()} context sizes ({args.aggregation})."
    )
    if args.aggregation == "median":
        print(
            "Coordinates are separate component medians; contour values are the "
            "resulting robust decomposed score, not median nPEHE."
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
