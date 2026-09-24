"""Plot accuracy rank against sequential-benchmark runtime.

Run from the repository root::

    uv run python -m analysis_eval.speed_rank

Accuracy ranks come from the completed semi-synthetic matrix. Runtime always
comes from the dedicated sequential speed benchmark, never a matrix cache.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .design import COLORS, apply_style
from .plotting import place_point_labels, plt, save_figure
from matplotlib.lines import Line2D
from .semisynt import CELL_ORDER, load_cate_results
from .speed_context import (
    DEFAULT_BENCH_DIR,
    _display_style,
    _ordered_speed_methods,
    load_speed_results,
    summarize_speed,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX_ROOT = REPO_ROOT / "data" / "semisynt_eval" / "semisynt_val_v1"
DEFAULT_OUT_DIR = Path(__file__).with_name("results") / "speed"
OUTPUT_NAME = "semisynt_speed_rank"
SCOPE_CHOICES = ("overall", "case", "pair", "linearity", "confounding", "proxy")
SPEED_RANK_ALIASES = {
    "ProximalFM-CPU": "ProximalFM",
    "ProximalFM-MPS": "ProximalFM",
}
# Measured speed points that receive their own colour and direct label.
# Case and linearity panels use the corresponding family. Panels that mix
# families (overall, confounding, proxy) use the union of both lists.
# Other methods still enter the rank calculation and appear in grey.
HIGHLIGHTED_METHODS = {
    "linear": (
        "T-OLS",
        "P-Kernel-TabICL",
        "P-Kernel-OLS",
        "X-OLS",
        "T-OLS-WZ-kNN-16",
        "NMMR-V-fixed-Wcond",
        "KPV-X-Wcond",
        "PMMR-X-Wcond",
        "CausalPFN-WZ-kNN-16",
        "ProximalFM-CPU",
        "ProximalFM-MPS",
        "S-OLS",
    ),
    "nonlinear": (
        "T-OLS",
        "P-Kernel-TabICL",
        "P-Kernel-RF",
        "X-OLS",
        "NMMR-V-fixed-Wcond",
        "KPV-X-Wcond",
        "PMMR-X-Wcond",
        "CausalPFN-WZ-kNN-16",
        "ProximalFM-CPU",
        "ProximalFM-MPS",
        "S-OLS",
    ),
}
MUTED_METHOD_COLOR = COLORS["light_grey"]
MUTED_METHOD_ALPHA = 0.55
HARDWARE_LINESTYLE = "-"


def _highlighted_methods_for_panel(panel_name: str) -> set[str]:
    """Choose a family-specific list, or their union for mixed panels."""
    family = panel_name.split(" ", 1)[0].lower()
    if family in HIGHLIGHTED_METHODS:
        return set(HIGHLIGHTED_METHODS[family])
    return set().union(*(set(methods) for methods in HIGHLIGHTED_METHODS.values()))


def _case_label(case: str) -> str:
    family, _, confounding, _, proxy = case.split("_")
    return f"{family.capitalize()} · {confounding} conf. · {proxy} proxy info"


def _rank_panels(
    frame: pd.DataFrame,
    *,
    scope: str,
    case: str | None,
    cases: Sequence[str] | None,
) -> list[tuple[str, list[str]]]:
    """Resolve one or two labelled rank-comparison panels."""
    available = set(frame["case"].unique())
    ordered_cases = [
        f"{family}_conf_{confounding}_proxy_{proxy}"
        for family, confounding, proxy in CELL_ORDER
    ]
    if scope == "overall":
        return [("Overall", ordered_cases)]
    if scope == "case":
        if case is None:
            raise ValueError("--scope case requires --case CASE")
        if case not in available:
            raise ValueError(f"case {case!r} is absent; available={sorted(available)}")
        return [(_case_label(case), [case])]
    if scope == "pair":
        if cases is None or len(cases) != 2:
            raise ValueError("--scope pair requires --cases CASE_A CASE_B")
        if len(set(cases)) != 2:
            raise ValueError("--cases must name two distinct configurations")
        absent = set(cases) - available
        if absent:
            raise ValueError(f"cases absent from matrix: {sorted(absent)}")
        return [(_case_label(value), [value]) for value in cases]
    if case is not None or cases is not None:
        raise ValueError(f"--case/--cases are invalid for --scope {scope}")
    factor = {"linearity": "family", "confounding": "confounding", "proxy": "proxy"}[
        scope
    ]
    levels = ("linear", "nonlinear") if factor == "family" else ("low", "high")
    suffix = {
        "family": "mechanisms",
        "confounding": "confounding",
        "proxy": "proxy information",
    }[factor]
    return [
        (
            f"{level.capitalize()} {suffix}",
            [
                value
                for value in ordered_cases
                if value in set(frame.loc[frame[factor].eq(level), "case"])
            ],
        )
        for level in levels
    ]


def _select_rank_methods(
    frame: pd.DataFrame,
    speed: pd.DataFrame,
    *,
    context_size: int,
    panels: Sequence[tuple[str, list[str]]],
    requested: Sequence[str] | None,
) -> list[str]:
    """Find methods with accuracy and measured sequential speed in every panel."""
    speed_methods = set(speed.loc[speed["n_ctx"].eq(context_size), "method"])
    rank_to_speed = {
        SPEED_RANK_ALIASES.get(speed_method, speed_method)
        for speed_method in speed_methods
    }
    candidates = set(frame["method"]) & rank_to_speed
    if requested is not None:
        requested = list(requested)
        absent = set(requested) - candidates
        if absent:
            raise ValueError(
                "requested methods lack either a rank or sequential timing at "
                f"n_ctx={context_size}: {sorted(absent)}"
            )
        candidates = set(requested)

    work = frame[frame["n_ctx"].eq(context_size) & frame["method"].isin(candidates)]
    required_cases = {case for _, panel_cases in panels for case in panel_cases}
    complete = []
    for method in candidates:
        observed = set(work.loc[work["method"].eq(method), "case"])
        if required_cases <= observed:
            complete.append(method)
    if not complete:
        raise ValueError("no methods have complete rank coverage and sequential timing")
    return _ordered_speed_methods(set(complete))


def summarize_sequential_speed_rank(
    frame: pd.DataFrame,
    speed: pd.DataFrame,
    methods: Sequence[str],
    *,
    context_size: int,
    panels: Sequence[tuple[str, list[str]]],
) -> pd.DataFrame:
    """Rank methods per episode, then attach independent sequential timings."""
    speed_at_context = speed[speed["n_ctx"].eq(context_size)].copy()
    speed_at_context["rank_method"] = speed_at_context["method"].map(
        lambda value: SPEED_RANK_ALIASES.get(value, value)
    )
    speed_at_context = speed_at_context[speed_at_context["rank_method"].isin(methods)]
    rows = []
    work = frame[frame["n_ctx"].eq(context_size) & frame["method"].isin(methods)]
    for panel, panel_cases in panels:
        scores = work[work["case"].isin(panel_cases)].pivot(
            index=["case", "episode"], columns="method", values="npehe"
        )
        scores = scores.reindex(columns=methods).dropna(axis=0, how="any")
        if scores.empty:
            raise ValueError(f"no complete episode blocks remain for panel {panel!r}")
        ranks = scores.rank(axis=1, method="average", ascending=True)
        for speed_row in speed_at_context.itertuples():
            rank_values = ranks[speed_row.rank_method]
            rows.append(
                {
                    "panel": panel,
                    "method": speed_row.method,
                    "rank_method": speed_row.rank_method,
                    "n_ctx": context_size,
                    "rank_median": float(rank_values.median()),
                    "rank_q25": float(rank_values.quantile(0.25)),
                    "rank_q75": float(rank_values.quantile(0.75)),
                    "time_median": float(speed_row.median),
                    "time_q25": float(speed_row.q25),
                    "time_q75": float(speed_row.q75),
                    "time_post_median": float(speed_row.post_median),
                    "n_blocks": len(scores),
                    "n_speed_episodes": int(speed_row.n),
                }
            )
    return pd.DataFrame(rows)


def make_speed_rank_figure(summary: pd.DataFrame):
    """Draw one or two independent speed/rank panels."""
    required = {
        "panel",
        "method",
        "rank_method",
        "rank_median",
        "rank_q25",
        "rank_q75",
        "time_median",
        "time_post_median",
        "n_ctx",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"speed/rank figure requires columns {sorted(missing)}")
    panels = list(summary["panel"].drop_duplicates())
    if not 1 <= len(panels) <= 2:
        raise ValueError("speed/rank figure requires one or two panels")

    rank_count = summary["rank_method"].nunique()
    apply_style()
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(6.07, 3.5) if len(panels) == 1 else (9.4, 3.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()
    panel_labels = []
    any_highlighted_hardware_pair = False
    for ax, panel_name in zip(axes, panels):
        panel = summary[summary["panel"].eq(panel_name)]
        highlighted_methods = _highlighted_methods_for_panel(panel_name)
        labels = []
        hardware = panel.set_index("method")
        has_proximal_hardware_pair = {"ProximalFM-CPU", "ProximalFM-MPS"} <= (
            set(hardware.index) & highlighted_methods
        )
        any_highlighted_hardware_pair |= has_proximal_hardware_pair
        if has_proximal_hardware_pair:
            cpu = hardware.loc["ProximalFM-CPU"]
            mps = hardware.loc["ProximalFM-MPS"]
            ax.plot(
                [mps.time_median, cpu.time_median],
                [mps.rank_median, cpu.rank_median],
                color=COLORS["vermillion"],
                linestyle=HARDWARE_LINESTYLE,
                linewidth=1.35,
                zorder=4,
            )
            ax.plot(
                mps.time_median,
                mps.rank_median,
                color=COLORS["vermillion"],
                marker="o",
                markersize=5.0,
                markerfacecolor=COLORS["vermillion"],
                markeredgewidth=1.2,
                linestyle="none",
                zorder=6,
            )
        for row in panel.itertuples():
            if has_proximal_hardware_pair and row.method == "ProximalFM-MPS":
                continue
            label, color, marker, _, _, markersize, zorder = _display_style(row.method)
            highlighted = row.method in highlighted_methods
            color = color if highlighted else MUTED_METHOD_COLOR
            alpha = 0.98 if highlighted else MUTED_METHOD_ALPHA
            lower = row.rank_median - row.rank_q25
            upper = row.rank_q75 - row.rank_median
            ax.errorbar(
                row.time_median,
                row.rank_median,
                yerr=np.array([[lower], [upper]]),
                color=color,
                marker=marker,
                markersize=max(markersize, 5.0),
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.55,
                linestyle="none",
                linewidth=1.15 if highlighted else 0.8,
                capsize=2.5,
                zorder=zorder + 1 if highlighted else 2,
                alpha=alpha,
            )
            post = row.time_post_median
            if np.isfinite(post) and post > 0 and post < row.time_median * (1.0 - 1e-3):
                ax.plot(
                    [post, row.time_median],
                    [row.rank_median, row.rank_median],
                    color=color,
                    linestyle=":",
                    linewidth=0.9 if highlighted else 0.7,
                    alpha=alpha,
                    zorder=zorder if highlighted else 1,
                )
                ax.plot(
                    post,
                    row.rank_median,
                    color=color,
                    marker=marker,
                    markersize=max(markersize, 5.0),
                    markerfacecolor="white",
                    markeredgewidth=1.1,
                    linestyle="none",
                    alpha=alpha,
                    zorder=zorder + 2 if highlighted else 3,
                )
            if highlighted:
                is_proximal_hardware_label = (
                    has_proximal_hardware_pair and row.method == "ProximalFM-CPU"
                )
                annotation = ax.annotate(
                    "ProximalFM" if is_proximal_hardware_label else label,
                    (
                        (mps.time_median, mps.rank_median)
                        if is_proximal_hardware_label
                        else (row.time_median, row.rank_median)
                    ),
                    xytext=(0, -5) if is_proximal_hardware_label else (5, 0),
                    textcoords="offset points",
                    color=color,
                    fontsize=6.8,
                    ha="center" if is_proximal_hardware_label else "left",
                    va="top" if is_proximal_hardware_label else "center",
                    clip_on=False,
                )
                if not is_proximal_hardware_label:
                    labels.append(annotation)
        ax.set_xscale("log")
        ax.set_title(panel_name)
        ax.set_xlabel("Median runtime (seconds)")
        panel_labels.append(labels)

    axes[0].set_ylim(rank_count + 0.75, 0.25)
    axes[0].set_yticks(range(1, rank_count + 1, 2))
    fig.supylabel("CATE nPEHE rank (median and IQR)")
    for ax, labels in zip(axes, panel_labels):
        place_point_labels(fig, ax, labels)
    legend_handles = [
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
    has_post_selection = (
        np.isfinite(summary["time_post_median"])
        & summary["time_post_median"].gt(0)
        & summary["time_post_median"].lt(summary["time_median"] * (1.0 - 1e-3))
    ).any()
    if has_post_selection:
        legend_handles.append(
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
    if any_highlighted_hardware_pair:
        legend_handles.append(
            Line2D(
                [],
                [],
                color=COLORS["vermillion"],
                linestyle=HARDWARE_LINESTYLE,
                label="ProximalFM: MPS ↔ CPU",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=len(legend_handles),
        columnspacing=1.35,
    )
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--speed-dir", type=Path, default=DEFAULT_BENCH_DIR)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--name", default=OUTPUT_NAME)
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--scope", choices=SCOPE_CHOICES, default="overall")
    parser.add_argument("--case", help="One case for --scope case")
    parser.add_argument(
        "--cases",
        nargs=2,
        metavar=("CASE_A", "CASE_B"),
        help="Two cases for --scope pair",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_cate_results(args.matrix_root, dimension=args.dim)
    speed = summarize_speed(load_speed_results(args.speed_dir))
    contexts = sorted(set(frame["n_ctx"]) & set(speed["n_ctx"]))
    if not contexts:
        raise ValueError(
            "accuracy matrix and sequential benchmark have no context size in common"
        )
    context_size = contexts[-1] if args.context_size is None else args.context_size
    if context_size not in contexts:
        raise ValueError(f"context size {context_size} unavailable; common={contexts}")
    panels = _rank_panels(frame, scope=args.scope, case=args.case, cases=args.cases)
    methods = _select_rank_methods(
        frame, speed, context_size=context_size, panels=panels, requested=args.methods
    )
    summary = summarize_sequential_speed_rank(
        frame, speed, methods, context_size=context_size, panels=panels
    )
    figure = make_speed_rank_figure(summary)
    paths = save_figure(figure, args.out_dir, args.name)
    block_counts = ", ".join(
        f"{panel}={int(group['n_blocks'].iloc[0])}"
        for panel, group in summary.groupby("panel", sort=False)
    )
    print(
        f"Ranked {len(methods)} accuracy methods ({summary['method'].nunique()} speed points) "
        f"at n_ctx={context_size:,}; "
        f"sequential speed uses {int(summary['n_speed_episodes'].iloc[0])} episodes "
        f"({block_counts} rank blocks)."
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
