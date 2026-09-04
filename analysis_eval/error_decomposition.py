"""Eight-cell level/heterogeneity error trajectories as context grows.

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
)
from .learning_curves import CONF_PROXY_ORDER, DEFAULT_MATRIX_ROOT
from .plotting import plt, save_figure
from .semisynt import load_cate_results, summarize_error_decomposition

DEFAULT_OUT_DIR = Path(__file__).with_name("results")
OUTPUT_NAME = "semisynt_error_decomposition"
DEFAULT_METHODS = (
    "T-OLS",
    "X-RF",
    "NMMR-V-fixed-Wcond",
    "CausalPFN",
    "ProximalFM",
)


def _validate_summary(summary: pd.DataFrame, methods: Sequence[str]) -> list[int]:
    contexts = sorted(int(value) for value in summary["n_ctx"].unique())
    if len(contexts) < 2:
        raise ValueError("decomposition trajectories require two context sizes")
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
        raise ValueError(
            f"decomposition grid is incomplete ({len(missing)} points missing), "
            f"e.g. {sorted(missing)[:5]}"
        )
    return contexts


def _axis_limits(summary: pd.DataFrame) -> tuple[float, float]:
    """Shared limits with fixed fractional headroom and an origin anchor."""
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


def make_error_decomposition_figure(
    summary: pd.DataFrame,
    *,
    methods: Sequence[str] = DEFAULT_METHODS,
    aggregation: str = "median",
    xlim: Sequence[float] | None = None,
    ylim: Sequence[float] | None = None,
):
    """Create eight shared-axis decomposition trajectory panels.

    ``xlim`` and ``ylim`` optionally fix the shared grid as ``(min, max)``.
    Trajectory segments are clipped to those bounds, so a node outside the
    grid makes its line stop at the corresponding boundary.
    """
    methods = ordered_methods(set(methods))
    contexts = _validate_summary(summary, methods)
    data_x_max, data_y_max = _axis_limits(summary[summary["method"].isin(methods)])
    x_limits = _validate_limits(xlim, "xlim") if xlim is not None else (0.0, data_x_max)
    y_limits = _validate_limits(ylim, "ylim") if ylim is not None else (0.0, data_y_max)
    apply_style()
    fig, axes = plt.subplots(
        2, 4, figsize=decomposition_figsize(), sharex=True, sharey=True
    )
    for row, family in enumerate(("linear", "nonlinear")):
        for col, (confounding, proxy) in enumerate(CONF_PROXY_ORDER):
            ax = axes[row, col]
            _draw_iso_npehe(ax, x_limits[1], y_limits[1])
            cell = summary[
                summary["family"].eq(family)
                & summary["confounding"].eq(confounding)
                & summary["proxy"].eq(proxy)
            ]
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
            if col == 0:
                ax.set_ylabel(f"{family.capitalize()}\nHeterogeneity error")
            if row == 1:
                ax.set_xlabel(r"Level error: normalized ATE bias$^2$")

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
        ncol=len(handles),
        columnspacing=1.35,
        handlelength=2.5,
    )
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--name", default=OUTPUT_NAME)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Method subset; defaults to five representative competitors",
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
        help="Fixed level-error axis limits (default: data-dependent)",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="Fixed heterogeneity-error axis limits (default: data-dependent)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_cate_results(args.matrix_root)
    summary = summarize_error_decomposition(
        frame, args.methods, aggregation=args.aggregation
    )
    fig = make_error_decomposition_figure(
        summary,
        methods=args.methods,
        aggregation=args.aggregation,
        xlim=args.xlim,
        ylim=args.ylim,
    )
    paths = save_figure(fig, args.out_dir, args.name)
    print(
        f"Rendered {len(args.methods)} methods over {frame['case'].nunique()} cases "
        f"and {frame['n_ctx'].nunique()} context sizes ({args.aggregation})."
    )
    if args.aggregation == "median":
        print(
            "Coordinates are separate component medians; contour values are the "
            "resulting robust decomposed score, not median nPEHE."
        )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
