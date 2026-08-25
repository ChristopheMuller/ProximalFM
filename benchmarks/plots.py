"""plots.py — every figure the benchmark produces.

    uv run benchmarks/plots.py --path data/synt_eval/<name>/bench
    uv run benchmarks/plots.py --exp <name> --only rank_profile iso_npehe
    uv run benchmarks/plots.py --list

Figures are **generated per task from the registry**, not per hardcoded metric:
:data:`benchmarks.metrics.TASKS` says what each task scores, which methods it
admits, what it ranks on and which metrics it shows, and every generic figure
below reads those four things rather than naming ``npehe`` itself. Adding a
metric to a task therefore adds it to the figures without touching this file,
and the ``ate`` task is not a special case — it is a task with level-only
metrics and its own rank key.

Two rules the figures obey, enforced once in
:mod:`benchmarks.plot_frames` (read its docstring before adding a figure):
oracles annotate, they never compete; and every paired statistic (rank,
critical difference) runs on a **complete block** of episodes, with the dropped
count printed.

What is here
────────────
*Per task* (written as ``results/plots/<name>_<task>.png``):

``rank_profile``         how often each method took rank 1, 2, 3, …
``cd_diagram``           Friedman + Nemenyi critical-difference diagram.
``paired_delta``         per-episode difference against one baseline method.
                         Episode difficulty cancels, so this separates methods
                         the summary metrics pool together.
``decomposition``        nPEHE² split into its level and shape halves — an exact
                         identity, see below.
``iso_npehe``            the same identity as a plane: shape R² against ATE
                         RMSE, over analytic iso-nPEHE contours.
``excess_over_oracle``   each method's error ÷ the oracle's error on the *same*
                         episode, which removes episode difficulty.
``cost_accuracy``        accuracy against wall-clock, with the Pareto front.
``strata``               accuracy against episode difficulty axes (proxy
                         strength, confounding, overlap, noise…).
``ate_scatter``          predicted against true ATE, one panel per method.

*Per run*: ``cross_task_ranks`` (rank on ATE vs CATE vs ITE, one line per
method) and ``episode_fit`` (row-level predictions on an easy, a median and a
hard episode).

The identity everything leans on
────────────────────────────────
For one episode, with ``b`` the normalised ATE bias and ``s`` the shape R²::

    nPEHE² = b² + (1 − s)

exactly — level error plus shape error. It is why ``decomposition`` and
``iso_npehe`` are the same picture twice, and it fixes how they aggregate:
components are averaged as *squares* (``ATE RMSE`` is already
``sqrt(mean(b²))``, and the shape term as ``1 − mean(s)``), so the total is
``sqrt(mean(nPEHE²))`` and the identity survives aggregation. Taking medians of
each part instead would put the point off its own contour.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.metrics import Task
from benchmarks.plot_frames import (
    ORACLE_ATE,
    ORACLE_CATE,
    BenchmarkRun,
    select_methods,
    terciles,
)
from benchmarks.plot_style import (
    REFERENCE_STYLE,
    abbreviate,
    color_for,
    draw_anchors,
    save_figure,
    style_for,
    use_theme,
)

warnings.filterwarnings("ignore", category=FutureWarning)

ALWAYS_SHOW = ("ProximalFM",)

DECOMPOSITION_TRIM = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Options:
    """Everything the figures take from the command line."""

    top: int = 8
    baseline: Optional[str] = None
    episodes: int = 3
    meta: Optional[Path] = None
    dpi: int = 200


@dataclass(frozen=True)
class Figure:
    name: str
    scope: str  # "task" — once per task — or "run" — once per run
    fn: Callable
    doc: str


FIGURES: dict[str, Figure] = {}


def figure(name: str, scope: str = "task"):
    def wrap(fn):
        FIGURES[name] = Figure(
            name, scope, fn, (fn.__doc__ or "").strip().split("\n")[0]
        )
        return fn

    return wrap


def out_path(run: BenchmarkRun, name: str, task: Optional[Task] = None) -> Path:
    stem = name if task is None else f"{name}_{task.name}"
    return run.plots_dir / f"{stem}.png"


# ─────────────────────────────────────────────────────────────────────────────
# Shared drawing helpers
# ─────────────────────────────────────────────────────────────────────────────


def _method_legend(ax, methods: Sequence[str], *, references: Sequence[str] = ()):
    handles = [
        Line2D(
            [],
            [],
            color=color_for(m),
            marker=style_for(m).get("marker", "o"),
            linestyle=style_for(m).get("linestyle", "--" if m in references else "-"),
            markersize=style_for(m).get("markersize", 6),
            label=abbreviate(m),
        )
        for m in methods
    ]
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=8,
        ncol=1 + len(handles) // 18,
    )


def _finish(fig, header_inches: float = 1.0) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.subplots_adjust(top=1 - header_inches / fig.get_figheight())


def _decomposition(
    run: BenchmarkRun, task: Task, trim: float = DECOMPOSITION_TRIM
) -> Optional[pd.DataFrame]:
    f = run.frame(task)
    keys = {m.key for m in task.metrics} & set(f.columns)
    if not {"ate_bias_norm", "shape_r2", "npehe"} <= keys:
        return None
    f = f.dropna(subset=["npehe", "shape_r2", "ate_bias_norm"])
    rows = {}
    for method, g in f.groupby("method"):
        n2 = g["npehe"] ** 2
        if trim > 0 and len(g) > 20:
            g = g[n2 <= n2.quantile(1.0 - trim)]
        rows[method] = {
            "level": float((g["ate_bias_norm"] ** 2).mean()),
            "shape": float(1.0 - g["shape_r2"].mean()),
            "total": float((g["npehe"] ** 2).mean()),
            "n": len(g),
            "is_reference": bool(g["is_reference"].any()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


@figure("rank_profile")
def plot_rank_profile(run: BenchmarkRun, task: Task, opts: Options) -> None:
    """How often each method took rank 1, 2, 3, … on the ranking metric."""
    ranks = run.ranks(task)
    if ranks.empty or ranks.shape[1] < 2:
        return
    order = [m for m in run.method_order(task) if m in ranks.columns]
    n_methods = ranks.shape[1]
    counts = pd.DataFrame(
        {r: [(ranks[m] == r).mean() for m in order] for r in range(1, n_methods + 1)},
        index=order,
    )

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(order) + 2.2))
    cmap = plt.get_cmap("RdYlGn_r")
    left = np.zeros(len(order))
    for r in counts.columns:
        vals = counts[r].to_numpy()
        ax.barh(
            order,
            vals,
            left=left,
            color=cmap((r - 1) / max(n_methods - 1, 1)),
            edgecolor="white",
            linewidth=0.4,
        )
        left += vals
    mean_rank = run.mean_rank(task)
    for i, m in enumerate(order):
        ax.text(1.01, i, f"{mean_rank[m]:.2f}", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(
        f"share of episodes at each rank on {task.metric(task.rank_by).label}"
    )
    ax.set_yticks(range(len(order)), [abbreviate(m) for m in order])
    ax.set_title(
        f"{task.title}\nrank profile over {ranks.shape[0]} complete episodes "
        "(green = rank 1; the number at right is the mean rank)",
        fontsize=11,
    )
    fig.tight_layout()
    save_figure(fig, out_path(run, "rank_profile", task), dpi=opts.dpi)


@figure("cd_diagram")
def plot_cd_diagram(run: BenchmarkRun, task: Task, opts: Options) -> None:
    """Friedman + Nemenyi critical-difference diagram over per-episode ranks."""
    try:
        from autorank import autorank, plot_stats
    except ImportError:
        print("  ! autorank not installed; skipping cd_diagram")
        return

    wide, dropped = run.pivot(task, task.rank_by)
    if wide.shape[1] < 3 or wide.shape[0] < 5:
        print(f"  ! cd_diagram({task.name}): need ≥3 methods and ≥5 paired episodes")
        return

    # autorank runs normality/variance tests for its report even on the forced
    # nonparametric path; a constant-metric method makes those warn spuriously.
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        result = autorank(
            wide,
            alpha=0.05,
            order="ascending",
            force_mode="nonparametric",
            verbose=False,
        )

    fig, ax = plt.subplots(figsize=(11, max(2.5, 0.4 * wide.shape[1])))
    plot_stats(result, allow_insignificant=True, ax=ax)
    fig.suptitle(
        f"{task.title}\nCritical difference on "
        f"{task.metric(task.rank_by).label} (left = better) · Friedman + Nemenyi, "
        f"α=0.05, CD={result.cd:.3f} · {wide.shape[0]} complete episodes "
        f"({dropped} dropped)",
        fontsize=11,
        y=1.06,
    )
    save_figure(fig, out_path(run, "cd_diagram", task), dpi=opts.dpi)


@figure("paired_delta")
def plot_paired_delta(run: BenchmarkRun, task: Task, opts: Options) -> None:
    """Per-episode difference against one baseline — difficulty cancels."""
    wide, dropped = run.pivot(task, task.rank_by)
    if wide.empty or wide.shape[1] < 2:
        return
    order = [m for m in run.method_order(task) if m in wide.columns]
    baseline = opts.baseline or _default_baseline(order)
    if baseline not in wide.columns:
        print(f"  ! paired_delta({task.name}): baseline {baseline!r} not scored here")
        return

    others = [m for m in order if m != baseline]
    delta = wide[others].sub(wide[baseline], axis=0)
    med = delta.median().sort_values()
    ordered = list(med.index)

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(ordered) + 2.6))
    ax.boxplot(
        [delta[m].to_numpy() for m in ordered],
        vert=False,
        showfliers=False,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(linewidth=0.8),
        medianprops=dict(color="black", linewidth=1.4),
    )
    for patch, m in zip(ax.patches, ordered):
        patch.set_facecolor(color_for(m))
        patch.set_alpha(0.55)
    ax.axvline(0, color="black", linewidth=1.2)
    for i, m in enumerate(ordered, start=1):
        win = float((delta[m] < 0).mean())
        ax.text(
            ax.get_xlim()[1],
            i,
            f"  beats baseline {win:.0%}",
            va="center",
            fontsize=7,
            color="#333333",
        )
    ax.set_yticks(
        range(1, len(ordered) + 1), [abbreviate(m) for m in ordered], fontsize=8
    )
    ax.invert_yaxis()  # best at the top, as everywhere else
    ax.set_xlabel(
        f"Δ {task.metric(task.rank_by).label} vs {baseline} "
        "(same episode; negative = better than the baseline)"
    )
    ax.set_title(
        f"{task.title}\npaired differences against {baseline} · "
        f"{wide.shape[0]} complete episodes ({dropped} dropped)",
        fontsize=11,
    )
    fig.tight_layout()
    save_figure(fig, out_path(run, "paired_delta", task), dpi=opts.dpi)


def _default_baseline(order: Sequence[str]) -> str:
    """The method everything else is compared against: ProximalFM if it ran,
    else whichever competitor ranked best."""
    for pin in ALWAYS_SHOW:
        for m in order:
            if m.startswith(pin):
                return m
    return order[0]


# ─────────────────────────────────────────────────────────────────────────────
# Per-task figures — error anatomy
# ─────────────────────────────────────────────────────────────────────────────


@figure("decomposition")
def plot_decomposition(run: BenchmarkRun, task: Task, opts: Options) -> None:
    """nPEHE² split into its level and shape halves (an exact identity)."""
    dec = _decomposition(run, task)
    if dec is None or dec.empty:
        return
    order = [m for m in run.method_order(task) if m in dec.index]
    order += [m for m in run.references(task) if m in dec.index]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(order) + 2.4))
    y = np.arange(len(order))
    level = dec.loc[order, "level"].to_numpy()
    shape = dec.loc[order, "shape"].to_numpy()
    ax.barh(y, level, color="#ff7f0e", label="level:  ATE RMSE²  =  mean(b²)")
    ax.barh(y, shape, left=level, color="#1f77b4", label="shape:  1 − mean(Shape R²)")
    for i, m in enumerate(order):
        ax.text(
            level[i] + max(shape[i], 0) + 0.02,
            i,
            f"nPEHE(rms) = {np.sqrt(max(dec.loc[m, 'total'], 0)):.2f}",
            va="center",
            fontsize=7,
        )
    ax.axvline(1.0, color="#d62728", linestyle=":", linewidth=1.2)
    ax.invert_yaxis()
    ax.set_yticks(y, [abbreviate(m) for m in order], fontsize=8)
    upper = float(np.nanquantile(level + shape, 0.9)) * 1.6 + 0.4
    ax.set_xlim(0, max(1.4, upper))
    ax.set_xlabel(
        "mean nPEHE²  =  level + shape   (dotted line: the constant-ATE baseline, 1.0)"
    )
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    ax.set_title(
        f"{task.title}\nwhere the error comes from · a bar past the axis is clipped, "
        f"read its label\nworst {DECOMPOSITION_TRIM:.0%} of episodes trimmed per "
        "method — the identity is linear in episodes, so it survives the trim;\n"
        "the summary tables are untrimmed and report nPEHE as a median, so these "
        "totals are larger and do not match them",
        fontsize=10,
    )
    _finish(fig, 1.1)
    save_figure(fig, out_path(run, "decomposition", task), dpi=opts.dpi)


@figure("iso_npehe")
def plot_iso_npehe(run: BenchmarkRun, task: Task, opts: Options) -> None:
    """Shape R² against ATE RMSE, on analytic iso-nPEHE contours."""
    dec = _decomposition(run, task)
    if dec is None or dec.empty:
        return
    x = np.sqrt(dec["level"].to_numpy())  # ATE RMSE = sqrt(mean(b²))
    y = 1.0 - dec["shape"].to_numpy()  # mean Shape R²

    # A method that recovers *no* heterogeneity has an unbounded negative shape
    # R², and one such method would flatten the whole readable region into a
    # sliver. The frame stops at −4 and anything outside it is drawn on the
    # edge, hollow, with its true coordinates written beside it.
    x_max = max(0.6, float(np.nanquantile(x, 0.95)) * 1.3)
    y_min = max(-4.0, min(-0.1, float(np.nanmin(y)) - 0.15))
    gx, gy = np.meshgrid(np.linspace(0, x_max, 400), np.linspace(y_min, 1.0, 400))
    npehe = np.sqrt(np.clip(gx**2 + (1 - gy), 0, None))

    fig, ax = plt.subplots(figsize=(8.2, 7))
    levels = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    filled = ax.contourf(
        gx, gy, npehe, levels=levels, cmap="RdYlGn_r", alpha=0.35, extend="both"
    )
    cs = ax.contour(gx, gy, npehe, levels=levels, colors="gray", linewidths=0.7)
    ax.clabel(cs, fmt="%.2f", fontsize=7)
    fig.colorbar(
        filled,
        ax=ax,
        shrink=0.8,
        label=f"nPEHE (rms over episodes, worst {DECOMPOSITION_TRIM:.0%} trimmed)",
    )

    for m in dec.index:
        style = style_for(m)
        mx, my = float(np.sqrt(dec.loc[m, "level"])), float(1.0 - dec.loc[m, "shape"])
        px, py = min(mx, x_max), max(my, y_min)
        outside = (px, py) != (mx, my)
        ax.scatter(
            px,
            py,
            color="none" if outside else style.get("color"),
            marker=style.get("marker", "o"),
            s=style.get("s", 130),
            edgecolors=style.get("color") if outside else "white",
            linewidths=1.4 if outside else 0.9,
            zorder=5,
            label=abbreviate(m),
        )
        if outside:
            ax.annotate(
                f"({mx:.2f}, {my:.2f})",
                (px, py),
                textcoords="offset points",
                xytext=(6, 8 if py <= y_min else -10),
                fontsize=6.5,
                color=style.get("color"),
            )
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, 1.02)
    ax.set_xlabel(
        f"ATE RMSE, worst {DECOMPOSITION_TRIM:.0%} trimmed  =  sqrt(mean(b²)),  "
        "b = ATE bias ÷ SD(τ)   — level error"
    )
    ax.set_ylabel("mean Shape R²   — heterogeneity recovered")
    ax.set_title(
        f"{task.title}\nnPEHE² = b² + (1 − Shape R²): the contours are exact, "
        "not fitted.\nUp-and-left is better; position *along* a contour says "
        "whether the error is level or shape.\n"
        f"Both axes are means over the episodes left after trimming each "
        f"method's worst {DECOMPOSITION_TRIM:.0%};\nthe summary tables report an "
        "untrimmed pooled rms (ATE RMSE) and a median (nPEHE), so neither number "
        "matches them.",
        fontsize=10,
    )
    ax.legend(bbox_to_anchor=(1.32, 1), loc="upper left", frameon=False, fontsize=8)
    # No tight_layout: it is incompatible with a figure that already has a
    # colorbar, and `bbox_inches="tight"` at save time does the same job.
    save_figure(fig, out_path(run, "iso_npehe", task), dpi=opts.dpi)


@figure("excess_over_oracle")
def plot_excess_over_oracle(run: BenchmarkRun, task: Task, opts: Options) -> None:
    """Error ÷ the oracle's error on the same episode — difficulty removed."""
    if ORACLE_CATE not in run.references(task):
        return
    both, dropped = run.pivot(task, task.rank_by, include_references=True)
    if both.empty or ORACLE_CATE not in both.columns:
        return
    floor = both[ORACLE_CATE]
    if float(floor.median()) < 0.05:
        # On the CATE task the oracle *is* the target, so it scores ~0 and the
        # ratio is just the metric again with a noisier denominator. The figure
        # only says something where the oracle marks a real floor (the ITE task).
        print(
            f"  · excess_over_oracle({task.name}): oracle error ≈ 0 here, so the "
            "ratio adds nothing — skipped"
        )
        return
    order = [m for m in run.method_order(task) if m in both.columns]
    ratio = both[order].div(floor, axis=0)
    med = ratio.median().sort_values()
    ordered = list(med.index)

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(ordered) + 2.4))
    ax.boxplot(
        [ratio[m].to_numpy() for m in ordered],
        vert=False,
        showfliers=False,
        widths=0.6,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.4),
    )
    for patch, m in zip(ax.patches, ordered):
        patch.set_facecolor(color_for(m))
        patch.set_alpha(0.55)
    ax.axvline(1.0, color="black", linewidth=1.2)
    ax.set_yticks(
        range(1, len(ordered) + 1), [abbreviate(m) for m in ordered], fontsize=8
    )
    ax.invert_yaxis()  # best at the top, as everywhere else
    ax.set_xlabel(
        f"{task.metric(task.rank_by).label} ÷ {ORACLE_CATE}'s on the same episode "
        "(1.0 = at the floor)"
    )
    ax.set_title(
        f"{task.title}\nexcess over the irreducible floor · "
        f"{both.shape[0]} complete episodes ({dropped} dropped)",
        fontsize=11,
    )
    fig.tight_layout()
    save_figure(fig, out_path(run, "excess_over_oracle", task), dpi=opts.dpi)


@figure("cost_accuracy")
def plot_cost_accuracy(run: BenchmarkRun, task: Task, opts: Options) -> None:
    """Accuracy against wall-clock time, with the Pareto front."""
    order = run.method_order(task)
    acc = run.summary_stat(task, task.rank_by, "median")
    time = run.mean_time(task)
    pts = pd.DataFrame({"acc": acc, "time": time}).loc[
        [m for m in order if m in acc.index]
    ]
    pts = pts.dropna()
    if pts.empty:
        return
    pts["time"] = pts["time"].clip(lower=1e-3)

    front = [
        m
        for m in pts.index
        if not (
            (pts["time"] < pts.loc[m, "time"]) & (pts["acc"] < pts.loc[m, "acc"])
        ).any()
    ]
    front = list(pts.loc[front].sort_values("time").index)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(
        pts.loc[front, "time"],
        pts.loc[front, "acc"],
        color="#999999",
        linestyle="--",
        linewidth=1.2,
        zorder=1,
        label="Pareto front",
    )
    for m in pts.index:
        style = style_for(m)
        ax.scatter(
            pts.loc[m, "time"],
            pts.loc[m, "acc"],
            color=style.get("color"),
            marker=style.get("marker", "o"),
            s=style.get("s", 130),
            edgecolors="white",
            linewidths=0.9,
            zorder=5,
        )
        ax.annotate(
            abbreviate(m, 18),
            (pts.loc[m, "time"], pts.loc[m, "acc"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
        )
    draw_anchors(ax, task.rank_by, axis="y", label=False)
    ax.set_xscale("log")
    ax.set_xlabel("mean fit + predict time per episode (s)")
    ax.set_ylabel(f"median {task.metric(task.rank_by).label}")
    ax.set_title(
        f"{task.title}\ncost vs accuracy — down-and-left is better", fontsize=11
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_path(run, "cost_accuracy", task), dpi=opts.dpi)


@figure("ate_scatter")
def plot_ate_scatter(run: BenchmarkRun, task: Task, opts: Options) -> None:
    """Predicted against true episode ATE, one panel per method."""
    f = run.frame(task).dropna(subset=["ate_true", "ate_pred"])
    if f.empty:
        return
    methods = run.method_order(task) + run.references(task)
    methods = [m for m in methods if (f["method"] == m).any()]
    ncol = min(4, len(methods))
    nrow = int(np.ceil(len(methods) / ncol))
    lo = float(np.nanquantile(f["ate_true"], 0.01))
    hi = float(np.nanquantile(f["ate_true"], 0.99))
    pad = 0.15 * (hi - lo + 1e-9)
    lim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.1 * ncol, 3.1 * nrow),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for ax, m in zip(axes.ravel(), methods):
        g = f[f["method"] == m]
        ax.scatter(
            g["ate_true"],
            g["ate_pred"],
            s=8,
            alpha=0.5,
            color=color_for(m),
            edgecolors="none",
        )
        ax.plot(lim, lim, ls="--", c="black", alpha=0.6, linewidth=1)
        ax.set_title(abbreviate(m, 20), fontsize=9)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
    for ax in axes.ravel()[len(methods) :]:
        ax.axis("off")
    fig.supxlabel("true episode ATE (standardized units)")
    fig.supylabel("predicted episode ATE")
    fig.suptitle(f"{task.title}\nATE level, episode by episode", fontsize=11)
    _finish(fig, 0.9)
    save_figure(fig, out_path(run, "ate_scatter", task), dpi=opts.dpi)


@figure("strata")
def plot_strata(run: BenchmarkRun, task: Task, opts: Options) -> None:
    """Accuracy against episode difficulty axes (proxy strength, overlap, …)."""
    try:
        cov = run.covariates(opts.meta)
    except FileNotFoundError as exc:
        print(f"  ! strata({task.name}): {exc}")
        return
    wide, _ = run.pivot(task, task.rank_by, complete=False)
    if wide.empty:
        return
    shown = select_methods(run.method_order(task), opts.top, always=ALWAYS_SHOW)

    axes_cols, cuts = [], {}
    for col in cov.columns:
        cut = terciles(cov[col])
        if cut.notna().any():
            axes_cols.append(col)
            cuts[col] = cut
    if not axes_cols:
        print(f"  ! strata({task.name}): no covariate varies enough to cut")
        return

    ncol = min(3, len(axes_cols))
    nrow = int(np.ceil(len(axes_cols) / ncol))
    fig, axs = plt.subplots(
        nrow, ncol, figsize=(4.4 * ncol, 3.4 * nrow), squeeze=False, sharey=True
    )
    levels = ["Low", "Mid", "High"]
    for ax, col in zip(axs.ravel(), axes_cols):
        cut = cuts[col].reindex(wide.index)
        for m in shown:
            if m not in wide.columns:
                continue
            med = [wide.loc[cut == lv, m].median() for lv in levels]
            style = style_for(m)
            kw = dict(
                color=style.get("color", "gray"),
                marker=style.get("marker", "o"),
                linewidth=1.5,
                markersize=style.get("markersize", 5),
                linestyle=style.get("linestyle", "-"),
            )
            ax.plot(levels, med, **kw)
        counts = [int((cut == lv).sum()) for lv in levels]
        ax.set_title(f"{col}  (n = {', '.join(map(str, counts))})", fontsize=9)
        draw_anchors(ax, task.rank_by, axis="y", label=False)
    for ax in axs.ravel()[len(axes_cols) :]:
        ax.axis("off")
    axs[0][0].set_ylabel(f"median {task.metric(task.rank_by).label}")
    _method_legend(axs[0][-1], [m for m in shown if m in wide.columns])
    fig.suptitle(
        f"{task.title}\nstratified by episode property (terciles) · compare a cut "
        "to its complement, never to the pool\n"
        "R²-based axes are linear, in-sample summaries — they order episodes, "
        "they do not measure a structural quantity",
        fontsize=10,
    )
    _finish(fig, 1.3)
    save_figure(fig, out_path(run, "strata", task), dpi=opts.dpi)


@figure("cross_task_ranks", scope="run")
def plot_cross_task_ranks(run: BenchmarkRun, opts: Options) -> None:
    """Where each method places on ATE vs CATE vs ITE."""
    tasks = run.tasks
    if len(tasks) < 2:
        return
    series = {}
    for t in tasks:
        mr = run.mean_rank(t)
        if not mr.empty:
            series[t.name] = (mr - 1) / max(len(mr) - 1, 1)
    if len(series) < 2:
        return
    names = list(series)
    methods = sorted({m for s in series.values() for m in s.index})

    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    for m in methods:
        xs = [i for i, n in enumerate(names) if m in series[n].index]
        ys = [series[n][m] for n in names if m in series[n].index]
        style = style_for(m)
        kw = dict(
            color=style.get("color", "gray"),
            marker=style.get("marker", "o"),
            markersize=style.get("markersize", 6),
            linewidth=1.6,
            alpha=0.9,
            linestyle=style.get("linestyle", "-"),
        )
        ax.plot(xs, ys, **kw)
        ax.annotate(
            abbreviate(m, 18),
            (xs[-1], ys[-1]),
            textcoords="offset points",
            xytext=(7, 0),
            fontsize=7,
            va="center",
        )
    ax.set_xticks(range(len(names)), [n.upper() for n in names])
    ax.set_ylim(-0.05, 1.05)
    ax.invert_yaxis()
    ax.set_ylabel("normalised mean rank  (0 = best of its field, 1 = worst)")
    ax.set_title(
        "Consistency across the three tasks\n"
        "the fields differ — the ATE and ITE tables admit methods the CATE table "
        "refuses — so ranks are normalised.\n"
        "A method placed high on ATE and low on CATE gets the level right and the "
        "shape wrong; a single point is a method only one table admits.",
        fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, out_path(run, "cross_task_ranks"), dpi=opts.dpi)


@figure("episode_fit", scope="run")
def plot_episode_fit(run: BenchmarkRun, opts: Options) -> None:
    """Row-level predictions on an easy, a median and a hard episode."""
    task = run.primary
    kind, target = run.target_for(task)
    if target is None:
        return
    difficulty = run.difficulty(task).dropna()
    if difficulty.empty:
        return

    picks = _pick_episodes(difficulty, opts.episodes)
    shown = select_methods(run.method_order(task), opts.top, always=ALWAYS_SHOW)
    preds = {m: run.preds(m) for m in shown}
    preds = {m: p for m, p in preds.items() if p is not None}
    if not preds:
        print("  ! episode_fit: no preds/*.npz found")
        return
    ite = run.targets().get("ite")
    truth = "τ*(x)" if kind == "cate" else "realized Y1−Y0"

    for label, ep in picks.items():
        _episode_scatter(run, task, ep, label, target, preds, truth, opts)
        _episode_profile(run, task, ep, label, target, ite, preds, truth, opts)


def _pick_episodes(difficulty: pd.Series, n: int) -> dict[str, int]:
    """Episodes spanning the difficulty range, chosen by the *field's* median
    error — not by index, which would pick whatever happened to be first."""
    ranked = difficulty.sort_values()
    if n <= 1:
        return {"median": int(ranked.index[len(ranked) // 2])}
    qs = np.linspace(0, 1, n)
    names = (
        ["easiest", "median", "hardest"]
        if n == 3
        else [f"q{int(q * 100):02d}" for q in qs]
    )
    idx = np.clip((qs * (len(ranked) - 1)).round().astype(int), 0, len(ranked) - 1)
    picks: dict[str, int] = {}
    for i, j in enumerate(idx):
        ep = int(ranked.index[j])
        # A short run has fewer episodes than quantiles asked for; drawing the
        # same episode three times under three labels helps nobody.
        if ep not in picks.values():
            picks[names[i]] = ep
    return picks


def _episode_scatter(run, task, ep, label, target, preds, truth, opts) -> None:
    methods = [m for m in preds if not np.all(np.isnan(preds[m][ep]))]
    if not methods:
        return
    t = target[ep]
    ncol = min(4, len(methods))
    nrow = int(np.ceil(len(methods) / ncol))
    allv = np.concatenate([t] + [preds[m][ep] for m in methods])
    lo, hi = float(np.nanquantile(allv, 0.01)), float(np.nanquantile(allv, 0.99))
    pad = 0.12 * (hi - lo + 1e-9)
    lim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.0 * ncol, 3.0 * nrow),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for ax, m in zip(axes.ravel(), methods):
        p = preds[m][ep]
        ax.scatter(t, p, s=16, alpha=0.7, color=color_for(m), edgecolors="none")
        ax.plot(lim, lim, ls="--", c="black", alpha=0.6, linewidth=1)
        ax.set_title(abbreviate(m, 20), fontsize=9)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
    for ax in axes.ravel()[len(methods) :]:
        ax.axis("off")
    fig.supxlabel(f"true {truth}")
    fig.supylabel("predicted effect")
    fig.suptitle(
        f"Episode {ep} ({label} of {run.n_episodes(task)} by the field's median error)"
        f" · SD(true effect) = {t.std():.3f}\n"
        "a slope below 1 is shrinkage; a vertical offset is ATE bias · a small SD "
        "is why a 'hardest' episode is often a nearly homogeneous one",
        fontsize=10,
    )
    _finish(fig, 1.0)
    save_figure(
        fig, out_path(run, f"episode_scatter_ep{ep}_{label}", task), dpi=opts.dpi
    )


def _episode_profile(run, task, ep, label, target, ite, preds, truth, opts) -> None:
    t = target[ep]
    order = np.argsort(t)
    x = np.arange(len(t))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    if ite is not None and ite.shape == target.shape and not np.allclose(ite, target):
        ax.scatter(
            x,
            ite[ep][order],
            s=10,
            color="#cccccc",
            zorder=0,
            label="realized Y1−Y0 (the noise τ(x) must not chase)",
        )
    ax.plot(x, t[order], color="black", linewidth=2.2, zorder=5, label=f"true {truth}")
    for m, p in preds.items():
        if np.all(np.isnan(p[ep])):
            continue
        style = style_for(m)
        kw = dict(
            color=style.get("color", "gray"),
            linewidth=1.2,
            alpha=0.9,
            label=m,
            linestyle=style.get("linestyle", "-"),
        )
        ax.plot(x, p[ep][order], **kw)
    ax.set_xlabel("query rows, sorted by the true effect")
    ax.set_ylabel("treatment effect (standardized units)")
    ax.set_title(
        f"Episode {ep} ({label}) — effect profile · SD(true effect) = {t.std():.3f}\n"
        "a flat line is a method predicting the ATE; a shifted copy of the black "
        "curve is pure level bias\n"
        "every metric here divides by that SD, so a nearly flat black curve is why "
        "an episode reads as hard",
        fontsize=10,
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=7)
    fig.tight_layout()
    save_figure(
        fig, out_path(run, f"episode_profile_ep{ep}_{label}", task), dpi=opts.dpi
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def resolve_exp_dir(exp: str) -> Path:
    """Resolve an experiment name to its benchmark output directory.

    Two layouts exist:
      * a build's own bench dir -> data/synt_eval/<exp>/bench (config_val.yaml)
      * an explicit output_dir  -> results/<exp>
    The build layout wins if both are present; `synthetic/<exp>/bench` is
    still accepted for builds made before the default moved under `data/`.
    """
    candidates = [
        Path("data/synt_eval") / exp / "bench",
        Path("data/prior_eval") / exp / "bench",
        Path("data/lit_eval") / exp / "bench",
        Path("synthetic") / exp / "bench",
        Path("results") / exp,
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise SystemExit(
        f"No experiment directory found for '{exp}' "
        f"(tried: {', '.join(str(c) for c in candidates)}). "
        "Run benchmarks/run.py first, or pass --path explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark figures — one set per task, from results.csv."
    )
    parser.add_argument(
        "--exp", default=None, help="Experiment name (see resolve_exp_dir)."
    )
    parser.add_argument("--path", default=None, help="Benchmark output directory.")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Which tasks to draw (default: every task the run scored).",
    )
    parser.add_argument(
        "--target",
        default=None,
        choices=["cate", "ite", "ate"],
        help="Deprecated alias for a single --tasks value.",
    )
    parser.add_argument(
        "--only", nargs="*", default=None, help="Draw only these figures."
    )
    parser.add_argument("--skip", nargs="*", default=(), help="Skip these figures.")
    parser.add_argument(
        "--list", action="store_true", help="List the figures and exit."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="How many methods the one-line-per-method figures keep (best first).",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Method the paired-difference figure compares against "
        "(default: ProximalFM if it ran, else the best-ranked method).",
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Row-level example episodes."
    )
    parser.add_argument(
        "--meta",
        default=None,
        help="A frozen set's root (holding frozen_meta.pt) — adds its sampled "
        "hyperparameters as extra stratification axes.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    if args.list:
        width = max(len(n) for n in FIGURES)
        for name, fig in FIGURES.items():
            print(f"  {name:<{width}}  [{fig.scope}]  {fig.doc}")
        return

    exp_dir = (
        Path(args.path) if args.path else resolve_exp_dir(args.exp or "first_test")
    )
    run = BenchmarkRun.load(exp_dir)
    opts = Options(
        top=args.top,
        baseline=args.baseline,
        episodes=args.episodes,
        meta=Path(args.meta) if args.meta else None,
        dpi=args.dpi,
    )

    wanted = list(args.tasks or ([args.target] if args.target else [])) or None
    tasks = [t for t in run.tasks if wanted is None or t.name in wanted]
    if not tasks:
        raise SystemExit(
            f"No scored task matches {wanted}; this run has: "
            f"{', '.join(t.name for t in run.tasks)}."
        )

    unknown = (set(args.only or ()) | set(args.skip)) - set(FIGURES)
    if unknown:
        raise SystemExit(
            f"Unknown figure(s): {', '.join(sorted(unknown))}. Try --list."
        )
    selected = [
        f
        for f in FIGURES.values()
        if (args.only is None or f.name in args.only) and f.name not in args.skip
    ]

    use_theme()
    print(f"Figures → {run.plots_dir}")
    for fig in selected:
        if fig.scope == "run":
            _guarded(fig, fig.fn, run, opts)
            continue
        for task in tasks:
            _guarded(fig, fig.fn, run, task, opts, task_name=task.name)


def _guarded(fig: Figure, fn, *args, task_name: str | None = None) -> None:
    """Run one figure; a failure names itself and the rest still get drawn."""
    where = fig.name if task_name is None else f"{fig.name}({task_name})"
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001 — a broken figure must not stop the sheet
        print(f"  ! {where} failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
