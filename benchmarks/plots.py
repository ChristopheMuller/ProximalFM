"""plots.py — every figure the benchmark produces.

    uv run benchmarks/plots.py --path data/synt_eval/<name>/bench
    uv run benchmarks/plots.py --exp <name> --only scorecard iso_npehe
    uv run benchmarks/plots.py --list

Figures are **generated per task from the registry**, not per hardcoded metric:
:data:`benchmarks.metrics.TASKS` says what each task scores, which methods it
admits, what it ranks on and which metrics it shows, and every generic figure
below reads those four things rather than naming ``npehe`` itself. Adding a
metric to a task therefore adds it to the ECDF grid and the scorecard without
touching this file, and the ``ate`` task is not a special case — it is a task
with level-only metrics and its own rank key.

Two rules the figures obey, enforced once in
:mod:`benchmarks.plot_frames` (read its docstring before adding a figure):
oracles annotate, they never compete; and every paired statistic (rank,
win-rate, performance profile, critical difference) runs on a **complete
block** of episodes, with the dropped count printed.

What is here
────────────
*Per task* (written as ``results/plots/<name>_<task>.png``):

``ecdf``                 per-metric ECDFs; the missing mass above each curve is
                         that method's failure/degenerate share.
``rank_profile``         how often each method took rank 1, 2, 3, …
``cd_diagram``           Friedman + Nemenyi critical-difference diagram.
``performance_profile``  Dolan–Moré: share of episodes within a factor τ of the
                         per-episode best. Peak performance vs robustness.
``head2head``            pairwise win-rate matrix — what the CD diagram is a
                         one-dimensional summary of.
``paired_delta``         per-episode difference against one baseline method.
                         Episode difficulty cancels, so this separates methods
                         the marginal ECDFs show as overlapping.
``decomposition``        nPEHE² split into its level and shape halves — an exact
                         identity, see below.
``iso_npehe``            the same identity as a plane: shape R² against ATE
                         RMSE, over analytic iso-nPEHE contours.
``excess_over_oracle``   each method's error ÷ the oracle's error on the *same*
                         episode, which removes episode difficulty.
``cost_accuracy``        accuracy against wall-clock, with the Pareto front.
``strata``               accuracy against episode difficulty axes (proxy
                         strength, confounding, overlap, noise…).
``robustness``           failure rate and share of episodes lost to the trivial
                         predictor.
``bias_caterpillar``     per-episode signed ATE bias, sorted — separates
                         systematic bias from spread, which ``ATE RMSE`` pools.
``ate_scatter``          predicted against true ATE, one panel per method.
``scorecard``            the summary sheet: one row per method, six aligned
                         panels. Start here.

*Per run*: ``cross_task_ranks`` (rank on ATE vs CATE vs ITE, one line per
method), ``task_compression`` (why the ITE table is insensitive),
``target_noise`` (is the Monte-Carlo target good enough to rank on), and
``episode_fit`` (row-level predictions on an easy, a median and a hard episode).

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

# Run as a script (`uv run benchmarks/plots.py`), so the repo root has to be on
# the path before the `benchmarks.` imports resolve — same as run.py.
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
    METRIC_ANCHORS,
    REFERENCE_STYLE,
    abbreviate,
    color_for,
    draw_anchors,
    save_figure,
    style_for,
    use_theme,
)

warnings.filterwarnings("ignore", category=FutureWarning)

#: Methods pinned into every figure that has to thin the field, whatever their
#: rank — the point of the benchmark is where *this* model lands.
ALWAYS_SHOW = ("ProximalFM",)

#: |ATE bias| ÷ SD(τ) above which a level-only estimate is called a failure.
#: The row-level tasks get an exact per-episode reference instead (the
#: Oracle-ATE row), so this constant is only ever used by the ``ate`` task,
#: whose oracle is unbiased by construction and therefore no threshold at all.
ATE_FAILURE_THRESHOLD = 0.5

#: Share of each method's worst episodes dropped before the nPEHE²
#: decomposition is averaged. See :func:`_decomposition` — the identity survives
#: it, the axis does not survive without it.
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


def _direction_hint(direction: Optional[str]) -> str:
    """The axis-label parenthesis, from the same ``direction`` the tables bold
    on — including ``"zero"``, which is neither lower nor higher."""
    return {
        "min": "lower is better",
        "max": "higher is better",
        "zero": "closer to zero is better",
    }.get(direction or "", "informational")


def _finish(fig, header_inches: float = 1.0) -> None:
    """Lay the panels out, then reserve absolute room above them for the title.

    ``tight_layout`` reserves a *fraction*, so on a tall multi-row figure a
    three-line suptitle lands inside the first row of panels. Reserving inches
    keeps the header the same size whatever the figure grows to.
    """
    with warnings.catch_warnings():
        # Legends are deliberately placed outside the axes; `bbox_inches="tight"`
        # at save time is what makes room for them, not tight_layout.
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.subplots_adjust(top=1 - header_inches / fig.get_figheight())


def _robust_limits(values: np.ndarray, key: str) -> tuple[float, float]:
    """Axis limits showing the body of a distribution, not its worst outlier.

    nPEHE has an unbounded right tail — a handful of near-homogeneous episodes
    reach the hundreds — so the 90th percentile sets the frame and the tail runs
    off it. The metric's own anchors are always kept in view, because a figure
    that crops ``nPEHE = 1`` has thrown away its reference point.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 1.0)
    if key in ("shape_r2", "cate_r2"):
        return (max(-2.0, float(np.quantile(v, 0.05))), 1.05)
    if key == "ate_bias_norm":
        m = float(np.quantile(np.abs(v), 0.95))
        return (-m * 1.15, m * 1.15)
    anchors = [a for a, _, _ in METRIC_ANCHORS.get(key, []) if a > 0]
    hi = max([float(np.quantile(v, 0.90))] + [a * 1.3 for a in anchors])
    return (0.0, hi * 1.05 + 1e-9)


def _ecdf(values: np.ndarray, n_total: int) -> tuple[np.ndarray, np.ndarray]:
    """ECDF whose ceiling is the *scored* share, so the gap below 1 is visible.

    An episode a method failed on, or one the metric is undefined on, is not
    dropped from the denominator — it would otherwise reward a method for
    scoring only the episodes it found easy.
    """
    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    return v, np.arange(1, v.size + 1) / max(n_total, 1)


def _decomposition(
    run: BenchmarkRun, task: Task, trim: float = DECOMPOSITION_TRIM
) -> Optional[pd.DataFrame]:
    """Per-method ``level`` / ``shape`` halves of mean nPEHE², or None.

    Aggregated as squares so the two halves still add to ``mean(nPEHE²)`` — see
    the module docstring. Returns None for a task without both halves.

    The tail is trimmed because ``mean(nPEHE²)`` is otherwise a plot of the
    worst three episodes: a near-homogeneous one divides by a Var(τ) close to
    the threshold and contributes thousands. Trimming keeps the identity
    *exact*, because a mean over a subset is still linear in episodes and all
    three components are averaged over the **same** subset — which is precisely
    why the trim happens here, once, rather than per figure.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Per-task figures — distributions
# ─────────────────────────────────────────────────────────────────────────────


@figure("ecdf")
def plot_ecdf(run: BenchmarkRun, task: Task, opts: Options) -> None:
    """Per-metric ECDFs, one panel per metric the task shows."""
    f = run.frame(task)
    # An older results.csv may predate a metric the task now declares; the
    # figure shows what the file has rather than failing on the whole grid.
    metrics = [m for m in task.metrics if m.show and m.key in f.columns]
    if not metrics:
        return
    order = run.method_order(task)
    refs = run.references(task)
    n_ep = run.n_episodes(task)

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(5.2 * len(metrics), 4.4), squeeze=False
    )
    for ax, m in zip(axes[0], metrics):
        pooled = f[m.key].to_numpy(dtype=float)
        for method in order + refs:
            v = f.loc[f["method"] == method, m.key]
            xs, ys = _ecdf(v.to_numpy(dtype=float), n_ep)
            if xs.size == 0:
                continue
            style = style_for(method)
            kw = dict(
                color=style.get("color", "gray"),
                linewidth=1.5,
                linestyle=style.get("linestyle", "-"),
            )
            if method in refs:
                kw.update(REFERENCE_STYLE)
            ax.step(xs, ys, where="post", **kw)
        draw_anchors(ax, m.key, axis="x", label=False)
        ax.set_xlim(*_robust_limits(pooled, m.key))
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(f"{m.label}  ({_direction_hint(m.direction)})")
        ax.set_ylabel("share of episodes ≤ x")
        ax.set_title(m.label)

    _method_legend(axes[0][-1], order + refs, references=refs)
    fig.suptitle(
        f"{task.title}\n{n_ep} episodes · a curve ending below 1.0 means that "
        "method was unscored on the rest (failed or homogeneous)",
        fontsize=11,
    )
    _finish(fig, 1.0)
    save_figure(fig, out_path(run, "ecdf", task), dpi=opts.dpi)


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


@figure("performance_profile")
def plot_performance_profile(run: BenchmarkRun, task: Task, opts: Options) -> None:
    """Dolan–Moré: share of episodes within a factor τ of the per-episode best."""
    wide, dropped = run.pivot(task, task.rank_by)
    if wide.empty or wide.shape[1] < 2:
        return
    best = wide.min(axis=1)
    wide = wide[best > 0]
    best = best[best > 0]
    if wide.empty:
        return
    ratio = wide.div(best, axis=0)
    taus = np.geomspace(1.0, float(np.nanquantile(ratio.to_numpy(), 0.99)) * 1.2, 300)

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in run.method_order(task):
        if m not in ratio.columns:
            continue
        style = style_for(m)
        kw = dict(
            color=style.get("color", "gray"),
            linewidth=1.6,
            linestyle=style.get("linestyle", "-"),
        )
        y = (ratio[m].to_numpy()[:, None] <= taus[None, :]).mean(axis=0)
        ax.step(taus, y, where="post", **kw)
    ax.set_xscale("log")
    ax.set_xlim(1, taus[-1])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("τ — factor of the per-episode best")
    ax.set_ylabel("share of episodes within τ of the best")
    ax.set_title(
        f"{task.title}\nperformance profile on {task.metric(task.rank_by).label} · "
        f"{wide.shape[0]} complete episodes ({dropped} dropped)\n"
        "height at τ=1 is how often a method *is* the best; the right tail is "
        "how badly it fails when it is not",
        fontsize=10,
    )
    _method_legend(ax, run.method_order(task))
    fig.tight_layout()
    save_figure(fig, out_path(run, "performance_profile", task), dpi=opts.dpi)


@figure("head2head")
def plot_head2head(run: BenchmarkRun, task: Task, opts: Options) -> None:
    """Pairwise win-rate matrix over the same episodes."""
    wide, dropped = run.pivot(task, task.rank_by)
    if wide.empty or wide.shape[1] < 2:
        return
    order = [m for m in run.method_order(task) if m in wide.columns]
    v = wide[order].to_numpy()
    wins = (v[:, :, None] < v[:, None, :]).mean(axis=0)
    ties = (v[:, :, None] == v[:, None, :]).mean(axis=0)
    rate = wins + 0.5 * ties
    np.fill_diagonal(rate, np.nan)

    n = len(order)
    fig, ax = plt.subplots(figsize=(0.55 * n + 4, 0.55 * n + 3.4))
    im = ax.imshow(rate, cmap="RdBu", vmin=0, vmax=1)
    ax.set_xticks(range(n), [abbreviate(m, 18) for m in order], rotation=90, fontsize=8)
    ax.set_yticks(range(n), [abbreviate(m, 18) for m in order], fontsize=8)
    if n <= 16:
        for i in range(n):
            for j in range(n):
                if i != j:
                    ax.text(
                        j,
                        i,
                        f"{rate[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if 0.25 < rate[i, j] < 0.75 else "white",
                    )
    fig.colorbar(im, ax=ax, shrink=0.7, label="P(row beats column)")
    ax.set_title(
        f"{task.title}\nhead-to-head win rate on {task.metric(task.rank_by).label} · "
        f"{wide.shape[0]} complete episodes ({dropped} dropped)",
        fontsize=10,
    )
    # No tight_layout: it is incompatible with a figure that already has a
    # colorbar, and `bbox_inches="tight"` at save time does the same job.
    save_figure(fig, out_path(run, "head2head", task), dpi=opts.dpi)


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


@figure("robustness")
def plot_robustness(run: BenchmarkRun, task: Task, opts: Options) -> None:
    """Failure rate and share of episodes lost to the trivial predictor."""
    order = run.method_order(task)
    fail = run.failure_rate(task).reindex(order)

    both, _ = run.pivot(task, task.rank_by, complete=False, include_references=True)
    if ORACLE_ATE in both.columns:
        threshold = both[ORACLE_ATE]
        label = f"worse than {ORACLE_ATE} on the same episode"
    else:
        threshold = pd.Series(ATE_FAILURE_THRESHOLD, index=both.index)
        label = f"{task.metric(task.rank_by).label} > {ATE_FAILURE_THRESHOLD}"
    lost = pd.Series(
        {m: float(both[m].gt(threshold).mean()) for m in order if m in both.columns}
    ).reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.4 * len(order) + 2.6), sharey=True)
    for ax, series, title in (
        (axes[0], fail, "failure rate (all-NaN predictions)"),
        (axes[1], lost, label),
    ):
        ax.barh(
            order,
            series.to_numpy(dtype=float),
            color=[color_for(m) for m in order],
            alpha=0.8,
        )
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("share of episodes")
    axes[0].invert_yaxis()
    axes[0].set_yticks(range(len(order)), [abbreviate(m) for m in order])
    fig.suptitle(
        f"{task.title}\nrobustness — the two ways a good median hides a bad method",
        fontsize=11,
    )
    _finish(fig, 1.0)
    save_figure(fig, out_path(run, "robustness", task), dpi=opts.dpi)


@figure("bias_caterpillar")
def plot_bias_caterpillar(run: BenchmarkRun, task: Task, opts: Options) -> None:
    """Per-episode signed ATE bias, sorted — bias against spread."""
    if "ate_bias_norm" not in {m.key for m in task.metrics}:
        return
    wide, _ = run.pivot(task, "ate_bias_norm", complete=False, include_references=True)
    if wide.empty:
        return
    order = run.method_order(task)
    refs = run.references(task)
    shown = select_methods(order, opts.top, always=ALWAYS_SHOW) + refs

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for m in shown:
        if m not in wide.columns:
            continue
        v = np.sort(wide[m].dropna().to_numpy())
        if v.size == 0:
            continue
        q = np.linspace(0, 1, v.size)
        style = style_for(m)
        kw = dict(
            color=style.get("color", "gray"),
            linewidth=1.5,
            linestyle=style.get("linestyle", "-"),
        )
        if m in refs:
            kw.update(REFERENCE_STYLE)
        ax.plot(q, v, **kw)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(*_robust_limits(wide.to_numpy(dtype=float).ravel(), "ate_bias_norm"))
    ax.set_xlabel("episode quantile (episodes sorted per method)")
    ax.set_ylabel("normalised ATE bias  (÷ SD(τ))")
    ax.set_title(
        f"{task.title}\nsorted per-episode ATE bias · a curve offset from zero is "
        "systematic bias; a steep curve is variance.\n"
        "ATE RMSE pools the two — this separates them.",
        fontsize=10,
    )
    _method_legend(ax, [m for m in shown if m in wide.columns], references=refs)
    fig.tight_layout()
    save_figure(fig, out_path(run, "bias_caterpillar", task), dpi=opts.dpi)


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


# ─────────────────────────────────────────────────────────────────────────────
# The summary sheet
# ─────────────────────────────────────────────────────────────────────────────


@figure("scorecard")
def plot_scorecard(run: BenchmarkRun, task: Task, opts: Options) -> None:
    """One row per method: ranks, error distribution, anatomy, cost — aligned."""
    order = run.method_order(task)
    if not order:
        return
    ranks = run.ranks(task)
    dec = _decomposition(run, task)
    f = run.frame(task)
    n_ep = run.n_episodes(task)
    key = task.rank_by
    if key not in run.results.columns:
        return
    metric = task.metric(key)
    fail = run.failure_rate(task)
    time = run.mean_time(task)

    # Cross-task position, normalised so tasks with different method counts are
    # comparable: 0 = best of its field, 1 = worst.
    cross = {}
    for other in run.tasks:
        mr = run.mean_rank(other)
        if mr.empty:
            continue
        n = len(mr)
        cross[other.name] = (mr - 1) / max(n - 1, 1)

    widths = [1.6, 1.5, 2.4, 2.0, 1.1, 0.9]
    fig, axs = plt.subplots(
        len(order),
        6,
        figsize=(sum(widths) * 1.55, 0.42 * len(order) + 2.2),
        gridspec_kw={"width_ratios": widths, "hspace": 0.0, "wspace": 0.25},
        squeeze=False,
    )
    pooled = f[key].to_numpy(dtype=float)
    x_lo, x_hi = _robust_limits(pooled, key)
    dec_hi = (
        float(np.nanquantile((dec["level"] + dec["shape"]).to_numpy(), 0.9)) * 1.4 + 0.3
        if dec is not None
        else 1.0
    )
    t_lo = max(float(time.min()), 1e-3) if len(time) else 1e-3
    t_hi = max(float(time.max()), t_lo * 10)

    for i, m in enumerate(order):
        row = axs[i]
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)

        # 0 — name and mean rank
        row[0].text(
            0.0,
            0.5,
            abbreviate(m, 22),
            va="center",
            ha="left",
            fontsize=8,
            color=color_for(m),
            fontweight="bold",
        )
        if m in ranks.columns:
            row[0].text(
                1.0, 0.5, f"{ranks[m].mean():.2f}", va="center", ha="right", fontsize=7
            )
        row[0].set_xlim(0, 1)

        # 1 — rank profile
        if m in ranks.columns:
            cmap = plt.get_cmap("RdYlGn_r")
            left = 0.0
            n_m = ranks.shape[1]
            for r in range(1, n_m + 1):
                w = float((ranks[m] == r).mean())
                row[1].barh(
                    [0], [w], left=[left], color=cmap((r - 1) / max(n_m - 1, 1))
                )
                left += w
        row[1].set_xlim(0, 1)
        row[1].set_ylim(-0.5, 0.5)

        # 2 — the metric's ECDF, against the pooled one
        xs, ys = _ecdf(f.loc[f["method"] == m, key].to_numpy(dtype=float), n_ep)
        px, py = _ecdf(pooled, n_ep * max(len(order), 1))
        row[2].step(px, py, where="post", color="#cccccc", linewidth=1.0)
        if xs.size:
            row[2].step(xs, ys, where="post", color=color_for(m), linewidth=1.4)
        draw_anchors(row[2], key, axis="x", label=False)
        row[2].set_xlim(x_lo, x_hi)
        row[2].set_ylim(0, 1.05)

        # 3 — the level/shape decomposition
        if dec is not None and m in dec.index:
            row[3].barh([0], [dec.loc[m, "level"]], color="#ff7f0e")
            row[3].barh(
                [0], [dec.loc[m, "shape"]], left=[dec.loc[m, "level"]], color="#1f77b4"
            )
            row[3].axvline(1.0, color="#d62728", linestyle=":", linewidth=1.0)
        row[3].set_xlim(0, dec_hi)
        row[3].set_ylim(-0.5, 0.5)

        # 4 — position on each task (0 = best of its field)
        for j, (tname, series) in enumerate(cross.items()):
            if m in series.index:
                row[4].scatter(
                    series[m],
                    -j,
                    color=color_for(m),
                    s=28,
                    edgecolors="white",
                    linewidths=0.6,
                )
            row[4].plot([0, 1], [-j, -j], color="#e5e5e5", linewidth=0.8, zorder=0)
        row[4].set_xlim(-0.05, 1.05)
        row[4].set_ylim(-len(cross) + 0.5, 0.5)

        # 5 — cost and failures
        row[5].set_xscale("log")  # before the ticks: it installs its own
        if m in time.index and np.isfinite(time[m]):
            row[5].scatter([max(time[m], t_lo)], [0], color=color_for(m), s=26)
        row[5].set_xlim(t_lo * 0.7, t_hi * 1.4)
        row[5].set_ylim(-0.5, 0.5)
        row[5].set_xticks([])
        row[5].set_xticks([], minor=True)
        rate = float(fail.get(m, 0.0))
        if rate > 0:
            row[5].text(
                1.02,
                0.5,
                f"{rate:.0%} fail",
                transform=row[5].transAxes,
                va="center",
                fontsize=6.5,
                color="#d62728",
            )

    headers = [
        "method / mean rank",
        "rank profile (green = 1st)",
        f"{metric.label} ECDF (grey = field)",
        "nPEHE² = level + shape" if dec is not None else "—",
        "position per task\n" + " / ".join(cross),
        "time (log)",
    ]
    for ax, title in zip(axs[0], headers):
        ax.set_title(title, fontsize=7.5, pad=10)
    # Axis ticks only on the bottom row, so the columns read as shared scales.
    axs[-1][2].set_xticks([x_lo, x_hi], [f"{x_lo:.2g}", f"{x_hi:.2g}"], fontsize=6)
    axs[-1][3].set_xticks([0, dec_hi], ["0", f"{dec_hi:.2g}"], fontsize=6)
    axs[-1][4].set_xticks([0, 1], ["best", "worst"], fontsize=6)
    axs[-1][5].set_xticks([t_lo, t_hi], [f"{t_lo:.2g}s", f"{t_hi:.2g}s"], fontsize=6)
    axs[-1][5].set_xticks([], minor=True)

    fig.suptitle(
        f"{task.title}\n{n_ep} episodes · rows ordered by mean rank · every panel "
        "is a distribution over episodes, not a point estimate",
        fontsize=11,
    )
    # Not tight_layout: the rows are deliberately glued together (hspace=0) and
    # it would prise them apart again.
    fig.subplots_adjust(top=1 - 1.0 / fig.get_figheight(), bottom=0.06, right=0.93)
    save_figure(fig, out_path(run, "scorecard", task), dpi=opts.dpi)


# ─────────────────────────────────────────────────────────────────────────────
# Per-run figures
# ─────────────────────────────────────────────────────────────────────────────


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


@figure("task_compression", scope="run")
def plot_task_compression(run: BenchmarkRun, opts: Options) -> None:
    """Why the ITE table is insensitive: the same methods, both targets."""
    names = {t.name: t for t in run.tasks}
    if not {"cate", "ite"} <= set(names):
        return
    cate = run.summary_stat(names["cate"], "npehe", "median")
    ite = run.summary_stat(names["ite"], "npehe", "median")
    shared = [m for m in run.method_order(names["cate"]) if m in ite.index]
    if not shared:
        return

    fig, ax = plt.subplots(figsize=(7, 6.4))
    for m in shared:
        style = style_for(m)
        ax.scatter(
            cate[m],
            ite[m],
            color=style.get("color"),
            marker=style.get("marker", "o"),
            s=style.get("s", 120),
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
        ax.annotate(
            abbreviate(m, 16),
            (cate[m], ite[m]),
            textcoords="offset points",
            xytext=(6, 3),
            fontsize=7,
        )
    lim = max(float(np.nanmax(cate[shared])), float(np.nanmax(ite[shared]))) * 1.15
    ax.plot([0, lim], [0, lim], ls="--", color="black", alpha=0.5, linewidth=1)
    ax.axhline(1.0, color="#d62728", linestyle=":", linewidth=1.2)
    ax.axvline(1.0, color="#d62728", linestyle=":", linewidth=1.2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("median nPEHE on the CATE target  τ*(x)")
    ax.set_ylabel("median nPEHE on the realized ITE")
    ax.set_title(
        "The same methods against both targets\n"
        "the ITE axis is compressed onto ≈1 by E[Var(Y1−Y0 | X)] — which is why "
        "the benchmark leads with CATE",
        fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, out_path(run, "task_compression"), dpi=opts.dpi)


@figure("target_noise", scope="run")
def plot_target_noise(run: BenchmarkRun, opts: Options) -> None:
    """How much of the scored target's variance is Monte-Carlo noise."""
    s2 = run.results.groupby("episode")["s2_over_var"].median().dropna()
    if s2.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(s2.to_numpy(), bins=40, color="#1f77b4", alpha=0.8)
    ax.axvline(
        0.05,
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        label="0.05 — the run warns above this",
    )
    ax.axvline(
        float(s2.median()),
        color="black",
        linestyle="-",
        linewidth=1.2,
        label=f"median {float(s2.median()):.4f}",
    )
    k = run.data.get("cate_K_eff") if (run.dir / "data.pt").exists() else None
    k_txt = f" · K_eff = {int(np.asarray(k).min())}" if k is not None else ""
    ax.set_xlabel("s²_over_var — Monte-Carlo noise share of Var(τ*)")
    ax.set_ylabel("episodes")
    ax.set_title(
        f"Is the target good enough to rank on?{k_txt}\n"
        "never subtracted: a large share means every method scores worse than it "
        "is, the target's own noise having landed in the residual",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_path(run, "target_noise"), dpi=opts.dpi)


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
