"""metrics.py — scoring the benchmark, one declarative object per task.

All datasets are standardised upstream (``benchmarks/data.py``) using
context-only statistics, so every episode is already on the same outcome scale
before any method runs. The metrics below are *additionally* scale-free w.r.t.
the treatment effect itself (everything is expressed relative to SD(τ)), so the
numbers are directly comparable across episodes and can be aggregated.

Three tasks, two targets
────────────────────────
There are two per-row ground truths on the query rows, and three tables — the
third is a different *view* of the first, not a third target:

``cate``  τ*(x) = E[Y1 − Y0 | X = x], estimated by the generator's Monte-Carlo
          average ``MU1 − MU0`` over ``K_eff`` replicates. **The primary
          task**: it is the estimand ProximalFM is trained on, and the best any
          method seeing only X can attain.
``ite``   the realized draw ``Y1 − Y0``, which carries that row's own ``U`` and
          noise. No method conditioning on X alone can beat
          ``E[Var(Y1 − Y0 | X)]`` here, so this is where methods that use the
          query-row proxies — τ(x, w) — are allowed their real advantage, read
          against the ``Oracle-CATE(X)`` reference row that marks the floor.
``ate``   the level only, scored against the CATE target (the ITE one when the
          set has no replicates). The one table where a method that returns a
          single scalar per episode says anything.

Scoring a τ(x) estimator against ``ite`` is not wrong, just insensitive: the
noise floor is often ~95% of Var(ITE), which compresses every method onto
nPEHE ≈ 1. That is why the benchmark leads with the ``cate`` task whenever the
set carries a Monte-Carlo target.

How it is put together
──────────────────────
Four layers, each doing one thing:

1. **Primitives** — :func:`mse`, :func:`rmse`, :func:`nrmse`, :func:`r2`,
   :func:`shape_r2`, :func:`bias`, :func:`norm_bias`. Plain numpy on two 1-D
   arrays. No NaN handling (done once, upstream of them) and no bookkeeping.
2. **:class:`Metric`** — a primitive plus everything the rest of the pipeline
   needs to know about it: its column name, how it is aggregated over episodes,
   which direction is better, how to colour it.
3. **:class:`Task`** — the single source of truth for one table: which target it
   scores, which estimands it may rank, its metrics, its rank key, its titles.
   :data:`TASKS` holds the three.
4. **:class:`TaskReport`** — one task's per-episode rows, and the three things
   anyone wants from them: :meth:`~TaskReport.summary`,
   :meth:`~TaskReport.rich_table`, :meth:`~TaskReport.latex`.

Adding a metric is one :class:`Metric` in one :class:`Task`; both renderers pick
it up, because there is no second column spec anywhere.

Note that a per-method object is deliberately *not* part of this: ``Mean Rank``
is a comparison across methods within an episode, so the natural unit is the
task, and per-method numbers are a ``groupby`` on the frame the CSV needs anyway.

``s2_over_var`` (``cate`` target only) is ``mean(S2)/Var(τ*)``: how much of the
scored target's variance is Monte-Carlo sampling noise in the target *itself*.
It is not corrected for — the metrics score ``MU1 − MU0`` as given — so a large
value means the set was built with too small a ``--mc_cate``, and every method on
that set looks worse than it is, the target's own noise having landed in the
residual. Roughly 1e-4 at K=4000; 7e-2 at K=8.

Design note
───────────
Everything here is numpy and pandas. No torch, no model, nothing from
``proximalfm`` — which is why the point metrics are *duplicated* rather than
shared with ``train/_eval.py::CausalEvaluator``. The two are formula-identical
and ``tests/test_bench_metrics.py`` pins that they agree; keep the population
variance (``ddof=0``) on both sides if you touch either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from benchmarks.estimands import canonical_estimand, parse_estimand

# Episodes where the scored target's variance is below this threshold are
# treated as "constant treatment effect" and excluded from the scale-free
# metrics (everything whose denominator is Var(τ)). Kept equal to
# CausalEvaluator's homogeneous_var_threshold so the trainer's synt_point/*
# curves and this table drop the same episodes.
TARGET_VAR_THRESHOLD = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 1. Metric primitives
# ─────────────────────────────────────────────────────────────────────────────
# Two 1-D arrays in, one float out. Population variance (ddof=0) throughout, to
# match CausalEvaluator's `unbiased=False`. Callers guarantee no NaNs and, for
# the scale-free ones, that Var(target) > 0.


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error."""
    return float(((pred - target) ** 2).mean())


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error — in the outcome's own units, so *not*
    comparable across episodes. Kept as the building block of :func:`nrmse`."""
    return float(np.sqrt(mse(pred, target)))


def nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    """RMSE ÷ SD(target) — this is nPEHE.

    1.0 is the do-nothing baseline: predicting the episode's constant true ATE
    scores exactly 1, so anything below is doing real work.
    """
    return rmse(pred, target) / float(np.sqrt(target.var()))


def r2(pred: np.ndarray, target: np.ndarray) -> float:
    """``1 − MSE/Var(target)`` — the same accuracy as :func:`nrmse`, on the
    familiar R² scale. Identically ``1 − nrmse²``."""
    return float(1.0 - mse(pred, target) / target.var())


def shape_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """``1 − Var(resid)/Var(target)`` — heterogeneity recovery with the level
    error removed. The most discriminative metric in proximal CATE; the
    constant-ATE baseline sits at exactly 0."""
    return float(1.0 - (pred - target).var() / target.var())


def bias(pred: np.ndarray, target: np.ndarray) -> float:
    """``mean(pred) − mean(target)``, in the outcome's units."""
    return float(pred.mean() - target.mean())


def norm_bias(pred: np.ndarray, target: np.ndarray) -> float:
    """Signed ATE bias ÷ SD(target), so systematic over/under-estimation is
    visible and the episodes can be pooled."""
    return bias(pred, target) / float(np.sqrt(target.var()))


def abs_norm_bias(pred: np.ndarray, target: np.ndarray) -> float:
    """``|norm_bias|`` — the per-episode piece of the pooled ATE RMSE."""
    return abs(norm_bias(pred, target))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Metric and column descriptors
# ─────────────────────────────────────────────────────────────────────────────


def _style_npehe(v: float) -> str:
    if pd.isna(v):
        return ""
    return "green" if v < 0.75 else "yellow" if v < 1.0 else "red"


def _style_r2(v: float) -> str:
    if pd.isna(v):
        return ""
    return "green" if v > 0.5 else "yellow" if v > 0.0 else "red"


def _style_bias(v: float) -> str:
    if pd.isna(v):
        return ""
    a = abs(v)
    return "green" if a < 0.1 else "yellow" if a < 0.3 else "red"


def _style_failure(v: float) -> str:
    if pd.isna(v) or v == 0:
        return ""
    return "yellow" if v < 0.05 else "red"


@dataclass(frozen=True)
class Column:
    """A rendered column: everything both renderers need and nothing else.

    ``direction`` says which values are good, and is read by *both* the console
    colouring and the LaTeX top-3 bolding: ``"min"``, ``"max"``, ``"zero"``
    (closest to zero) or ``None`` (informational — never bolded).

    ``short`` is the console heading when the full ``label`` would cost too much
    terminal width — the only thing the two renderers are allowed to disagree
    about, because one of them has ~100 columns to spend and the other does not.
    ``label`` remains the CSV and LaTeX spelling.
    """

    label: str
    fmt: str = "{:.3f}"
    direction: Optional[str] = None
    style: Optional[Callable[[float], str]] = None
    short: Optional[str] = None

    @property
    def heading(self) -> str:
        """The console heading."""
        return self.short or self.label


@dataclass(frozen=True)
class Metric(Column):
    """A primitive plus how it is stored, aggregated and shown.

    ``agg`` is the one knob that answers "pooled, or a distribution over
    episodes?":

    ``"median"``      the median over episodes, plus ``<label> p25`` / ``p75``
                      companions. The default, and the same convention as the
                      trainer's ``synt_point/*`` keys — the R²-style metrics are
                      unbounded below, so a handful of near-homogeneous episodes
                      can drag a *mean* arbitrarily negative.
    ``"pooled_rms"``  ``sqrt(mean(v²))`` over episodes: one number pooling both
                      bias and episode-to-episode spread. This is ``ATE RMSE``.
    ``"mean"``        the plain mean.

    ``scale_free`` metrics divide by Var(τ) and are therefore left NaN on a
    homogeneous episode. ``show=False`` keeps a metric in the CSV but out of both
    rendered tables.
    """

    key: str = ""
    fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
    agg: str = "median"
    scale_free: bool = True
    show: bool = True


# The per-method aggregates. Not metrics — they are computed from the
# bookkeeping columns, not from (pred, target) — but they render identically.
RANK_COLUMN = Column("Mean Rank", "{:.2f}", "min", short="Rank")
FAILURE_COLUMN = Column("Failure Rate", "{:.0%}", "min", _style_failure, short="Fail")
TIME_COLUMN = Column("Time (s)", "{:.2f}", None)
ESTIMAND_COLUMN = Column("Estimand", "{}", None)


NPEHE = Metric(
    label="nPEHE",
    direction="min",
    style=_style_npehe,
    key="npehe",
    fn=nrmse,
)
CATE_R2 = Metric(
    label="CATE R²",
    direction="max",
    style=_style_r2,
    key="cate_r2",
    fn=r2,
    # 1 − nPEHE²: on screen it costs a column and says nothing new. It stays in
    # summary_*.csv for anyone who prefers the R² scale.
    show=False,
)
SHAPE_R2 = Metric(
    label="Shape R²",
    direction="max",
    style=_style_r2,
    key="shape_r2",
    fn=shape_r2,
)
ATE_BIAS = Metric(
    label="ATE Bias",
    fmt="{:+.3f}",
    direction="zero",
    style=_style_bias,
    key="ate_bias_norm",
    fn=norm_bias,
)
PEHE = Metric(
    label="PEHE",
    direction="min",
    style=_style_bias,
    key="pehe",
    fn=rmse,
    # The one metric here that does NOT divide by Var(tau), which is the whole
    # reason it exists: on a homogeneous set (`d_X = 0` makes tau*(x) constant,
    # so Var(tau) is exactly 0) every scale-free metric is NaN and the tables
    # come out empty. Absolute accuracy against a constant target is still a
    # real quantity, and it is the only one that sees a method inventing
    # heterogeneity that is not there -- rmse^2 = bias^2 + Var(predictions),
    # where the level metrics see only the first term. Measured on a d_X = 0
    # set, four of twelve methods had an RMSE 3-5x their |ATE bias|.
    #
    # `rmse`'s own docstring warns it is in the outcome's units and so not
    # comparable across episodes; that warning does not bite here, because
    # `benchmarks/data.py` has already divided every episode's outcome by its
    # context-only y std. The number is in standardized-y units throughout.
    scale_free=False,
)
ATE_RMSE = Metric(
    label="ATE RMSE",
    direction="min",
    style=_style_bias,
    key="ate_abs_bias_norm",
    fn=abs_norm_bias,
    agg="pooled_rms",
)

# The level/shape decomposition, in the order the tables read it.
EFFECT_METRICS = (NPEHE, PEHE, CATE_R2, SHAPE_R2, ATE_BIAS, ATE_RMSE)
# The `ate` task stays level-only on purpose. It is the one table that admits
# proxy-conditioning methods, and it admits them *because* a level metric is
# meaningful for a `cate(x,w)` output; a row-level PEHE against tau*(x) is not,
# it just penalizes them for answering the question they were asked. That is
# the same reasoning that keeps them out of the `cate` table.
LEVEL_METRICS = (ATE_BIAS, ATE_RMSE)


# ─────────────────────────────────────────────────────────────────────────────
# 3. The tasks
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Task:
    """One evaluated task — one table, one set of admissible methods.

    ``scores_against`` names the per-row targets this task will use, best first:
    the ``ate`` task prefers the Monte-Carlo CATE and falls back to the realized
    ITE when the set has no replicates.

    Admissibility is two checks and nothing more — is the output a per-row effect
    or a scalar (``kinds``), and does it read the proxies of the row it is
    predicting (``allow_proxy_conditioning``):

    * ``cate`` (the headline) shows only per-row methods that condition on X
      alone. A ``cate(x,w)`` output estimates E[Y1−Y0 | X, W], a *different*
      quantity, and ranking it against τ*(x) would be a category error — and
      the same goes for ``cate(x,u)``, an ``is_oracle: true`` competitor
      handed the true confounder.
    * ``ite`` admits every per-row method, since that is where conditioning on
      the proxies is a genuine advantage.
    * ``ate`` admits everyone: its columns are level-only, which is meaningful
      for a τ(x) method too.

    Admissibility is a fairness rule for *competitors* only. The reference rows
    bypass it entirely (``run.score_everything``): they are never ranked, so all
    three oracles anchor all three tables — including the ones their own tag
    would exclude, such as ``Oracle-ITE`` (``cate(x,w,u)``) on the CATE track,
    which is exactly where it says what the realized noise costs.
    """

    name: str
    scores_against: tuple[str, ...]
    metrics: tuple[Metric, ...]
    rank_by: str
    kinds: tuple[str, ...]
    allow_proxy_conditioning: bool
    title: str
    caption: str
    latex_label: str

    def admits(self, tag: str) -> bool:
        """Whether an output tagged ``tag`` belongs in this task's table."""
        kind, inputs = parse_estimand(tag)
        if kind not in self.kinds:
            return False
        return self.allow_proxy_conditioning or not ({"w", "z", "u"} & set(inputs))

    def target_kind(self, available: Sequence[str]) -> Optional[str]:
        """Which of the available per-row targets this task scores, or None."""
        return next((k for k in self.scores_against if k in available), None)

    def metric(self, key: str) -> Metric:
        for m in self.metrics:
            if m.key == key:
                return m
        raise KeyError(f"task {self.name!r} has no metric {key!r}")

    def columns(self) -> list[Column]:
        """Every rendered column, in table order. The one spec both renderers
        read — there is deliberately no second copy."""
        return [
            ESTIMAND_COLUMN,
            RANK_COLUMN,
            *[m for m in self.metrics if m.show],
            FAILURE_COLUMN,
            TIME_COLUMN,
        ]


TASKS: dict[str, Task] = {
    "cate": Task(
        name="cate",
        scores_against=("cate",),
        metrics=EFFECT_METRICS,
        rank_by="npehe",
        kinds=("cate",),
        allow_proxy_conditioning=False,
        title="CATE  —  scored against tau*(x) = E[Y1-Y0 | X]",
        caption=(
            "Benchmark summary on the CATE target "
            "$\\tau^*(x) = E[Y_1 - Y_0 \\mid X]$."
        ),
        latex_label="tab:bench_cate",
    ),
    "ite": Task(
        name="ite",
        scores_against=("ite",),
        metrics=EFFECT_METRICS,
        rank_by="npehe",
        kinds=("cate",),
        allow_proxy_conditioning=True,
        title="ITE  —  scored against the realized Y1-Y0",
        caption=(
            "Benchmark summary on the realized ITE $Y_1 - Y_0$. Methods whose "
            "estimand names $w$ or $z$ condition on the query row's own proxies."
        ),
        latex_label="tab:bench_ite",
    ),
    "ate": Task(
        name="ate",
        # Not a third target: the level view of the CATE rows, or of the ITE
        # ones when the set carries no Monte-Carlo target.
        scores_against=("cate", "ite"),
        metrics=LEVEL_METRICS,
        rank_by="ate_abs_bias_norm",
        kinds=("cate", "ate"),
        allow_proxy_conditioning=True,
        title="ATE  —  level only",
        caption="Benchmark summary at the ATE level.",
        latex_label="tab:bench_ate",
    ),
}


def primary_task(available: Sequence[str]) -> str:
    """The task the live table and the default plots speak in."""
    return "cate" if "cate" in available else "ite"


# ─────────────────────────────────────────────────────────────────────────────
# 4a. Scoring
# ─────────────────────────────────────────────────────────────────────────────

# Bookkeeping columns every row carries, whatever the task.
BOOKKEEPING_COLUMNS = (
    "method",
    "episode",
    "target",
    "estimand",
    "is_reference",
    "ate_true",
    "ate_pred",
    "ate_bias",
    "target_var",
    "s2_over_var",
    "homogeneous",
    "failed",
    "n_test",
    "time",
)


def score_episode(
    pred: np.ndarray,
    target: np.ndarray,
    task: Task,
    *,
    s2: Optional[np.ndarray] = None,
    target_var_threshold: float = TARGET_VAR_THRESHOLD,
) -> dict:
    """Score one episode's predictions against one per-row target.

    The three things that are easy to get subtly wrong are done exactly once
    here, before any metric runs: the all-NaN failure flag, the partial-NaN
    fallback to the episode's mean prediction, and the homogeneity gate that
    leaves every scale-free metric NaN when Var(τ) ≈ 0.

    ``s2`` is the per-row sampling variance of a Monte-Carlo target; it is
    *reported* (as ``s2_over_var``) and never subtracted.
    """
    failed = bool(np.all(np.isnan(pred)))

    ate_true = float(target.mean())
    target_var = float(target.var())  # population variance (ddof=0)
    homogeneous = target_var < target_var_threshold

    row = {
        "ate_true": ate_true,
        "ate_pred": float("nan"),
        "ate_bias": float("nan"),
        "target_var": target_var,
        "s2_over_var": (
            float(np.nanmean(s2) / target_var)
            if s2 is not None and not homogeneous
            else float("nan")
        ),
        "homogeneous": homogeneous,
        "failed": failed,
        "n_test": len(target),
    }
    row.update({m.key: float("nan") for m in task.metrics})

    if failed:
        return row

    ate_pred = float(np.nanmean(pred))
    row["ate_pred"] = ate_pred
    row["ate_bias"] = ate_pred - ate_true

    # Any partial NaNs are scored against the episode's mean prediction.
    pred = np.where(np.isnan(pred), ate_pred, pred)

    for m in task.metrics:
        if m.scale_free and homogeneous:
            continue  # denominator is Var(τ) ≈ 0 — undefined, stays NaN
        row[m.key] = m.fn(pred, target)
    return row


def score_method(
    preds: np.ndarray,
    target: np.ndarray,
    times: np.ndarray,
    method: str,
    task: Task,
    *,
    estimand: str = "cate(x)",
    is_reference: bool = False,
    s2: Optional[np.ndarray] = None,
    target_var_threshold: float = TARGET_VAR_THRESHOLD,
) -> pd.DataFrame:
    """Score one method for one task, over every episode.

    The estimand is stored canonically, so a legacy tag from an old config lands
    in the frame as the tag every consumer of the frame now expects.
    """
    estimand = canonical_estimand(estimand)
    rows = [
        {
            "method": method,
            "episode": b,
            "target": task.name,
            "estimand": estimand,
            "is_reference": is_reference,
            "time": float(times[b]),
            **score_episode(
                np.asarray(preds[b], dtype=float),
                np.asarray(target[b], dtype=float),
                task,
                s2=None if s2 is None else np.asarray(s2[b], dtype=float),
                target_var_threshold=target_var_threshold,
            ),
        }
        for b in range(len(preds))
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Aggregation and rendering
# ─────────────────────────────────────────────────────────────────────────────


def _aggregate(values: pd.Series, agg: str) -> dict[str, float]:
    """One metric's per-episode values → its summary column(s), keyed by the
    suffix appended to the metric's label."""
    if agg == "median":
        return {
            "": values.median(),
            " p25": values.quantile(0.25),
            " p75": values.quantile(0.75),
        }
    if agg == "pooled_rms":
        return {"": float(np.sqrt((values**2).mean()))}
    if agg == "mean":
        return {"": values.mean()}
    raise ValueError(f"unknown aggregation {agg!r}")


@dataclass
class TaskReport:
    """One task's per-episode rows, and the three views anyone wants of them."""

    task: Task
    frame: pd.DataFrame
    _summary: Optional[pd.DataFrame] = field(default=None, repr=False)

    @classmethod
    def from_results(cls, results: pd.DataFrame, task: Task) -> "TaskReport":
        """Pick this task's rows out of the long frame ``run.py`` writes."""
        rows = results[results["target"] == task.name]
        return cls(task, rows.reset_index(drop=True))

    @classmethod
    def from_summary_csv(cls, path, task: Task) -> "TaskReport":
        """Reopen an already-aggregated ``summary_<task>.csv``.

        For the LaTeX front-end, which renders a finished run rather than
        rescoring one — so the per-episode frame is not available, and only
        :meth:`latex` (and :meth:`summary`) work on the result.
        """
        summary = pd.read_csv(path, index_col="method")
        return cls(task, pd.DataFrame(), _summary=summary)

    @property
    def empty(self) -> bool:
        return self.frame.empty

    def summary(self) -> pd.DataFrame:
        """One row per method: the metrics aggregated as each one declares,
        plus the rank, the failure rate and the timing.

        Rows are ordered by the mean rank (best head-to-head first). Reference
        rows (the oracles) are shown but never ranked: an oracle is not a
        competitor, and letting it take rank 1 would shift every real method.
        """
        if self._summary is not None:
            return self._summary

        df = self.frame.reset_index(drop=True)
        if df.empty:
            self._summary = df
            return df

        # Failed and homogeneous episodes have a NaN rank key and are simply
        # excluded from both the rank and its mean.
        rankable = df[~df["is_reference"]]
        df["method_rank"] = rankable.groupby("episode")[self.task.rank_by].rank(
            method="min", ascending=True
        )

        def _agg(g: pd.DataFrame) -> pd.Series:
            out = {
                ESTIMAND_COLUMN.label: g["estimand"].iloc[0],
                # An anchor for reading the table, not a competitor: unranked
                # here, and excluded from the LaTeX best-of bolding.
                "Reference": bool(g["is_reference"].any()),
                RANK_COLUMN.label: g["method_rank"].mean(),
            }
            for m in self.task.metrics:
                for suffix, value in _aggregate(g[m.key], m.agg).items():
                    out[m.label + suffix] = value
            out[FAILURE_COLUMN.label] = g["failed"].mean()
            out[TIME_COLUMN.label] = g["time"].mean()
            return pd.Series(out)

        summary = df.groupby("method", sort=False).apply(_agg)
        # Push all-NaN ranks to the end, which is where the unranked reference
        # rows land.
        self._summary = summary.sort_values(
            RANK_COLUMN.label, ascending=True, na_position="last"
        )
        return self._summary

    # ── Rendering ────────────────────────────────────────────────────────────

    def rich_table(self, title: Optional[str] = None):
        """The console table. Columns come from ``Task.columns()``."""
        from rich.table import Table
        from rich import box

        summary = self.summary()
        tbl = Table(
            title=title or self.task.title, box=box.SIMPLE_HEAD, show_footer=False
        )
        tbl.add_column("Method", style="bold cyan", no_wrap=True)
        # Wide enough for the longest tag ("cate(x,w,z,u)"): rich shrinks
        # flexible columns on a narrow terminal, and a truncated "cate…" says
        # nothing.
        tbl.add_column("Est.", justify="center", no_wrap=True, min_width=13)
        columns = [c for c in self.task.columns() if c is not ESTIMAND_COLUMN]
        for col in columns:
            tbl.add_column(col.heading, justify="right")

        for method, row in summary.iterrows():
            cells = []
            for col in columns:
                v = row[col.label]
                if pd.isna(v):
                    cells.append("-")
                    continue
                s = col.fmt.format(v)
                style = col.style(v) if col.style else ""
                cells.append(f"[{style}]{s}[/{style}]" if style else s)
            tbl.add_row(str(method), str(row[ESTIMAND_COLUMN.label]), *cells)
        return tbl

    def latex(self, methods: Optional[Sequence[str]] = None) -> str:
        """A LaTeX table with the best three *competing* values bolded per
        column, direction taken from the same ``Column.direction`` the console
        colours read.

        ``methods`` restricts the rows; reference rows are anchors for reading
        the table, so a filter never silently drops them.
        """
        df = self.summary()
        reference = (
            df["Reference"].astype(bool)
            if "Reference" in df.columns
            else pd.Series(False, index=df.index)
        )
        if methods:
            keep = [m for m in methods if m in df.index]
            keep += [m for m in df.index[reference] if m not in keep]
            df, reference = df.loc[keep], reference.loc[keep]

        columns = [c for c in self.task.columns() if c.label in df.columns]
        rendered = pd.DataFrame(
            {
                c.label: _format_column(df[c.label], c, ranked=~reference)
                for c in columns
            },
            index=df.index,
        )
        rendered.index.name = None
        rendered.columns = [c.replace("R²", "$R^2$") for c in rendered.columns]
        # The cells are pre-formatted strings, so pandas would left-align every
        # column; the numbers want to line up on the right.
        column_format = "l" + "".join(
            "l" if c is ESTIMAND_COLUMN else "r" for c in columns
        )
        return rendered.style.to_latex(
            caption=self.task.caption,
            label=self.task.latex_label,
            hrules=True,
            column_format=column_format,
        ).replace("\\begin{table}\n", "\\begin{table}[htbp]\n\\centering\n")


def _format_column(series: pd.Series, column: Column, ranked: pd.Series) -> list[str]:
    """One column's cells as LaTeX, its best three *competing* rows bolded.

    ``ranked`` restricts which rows may win the bold: the oracle rows would
    otherwise take every slot on the metrics they define. The winners are picked
    by row, not by value, so a reference row that happens to tie a winner is not
    bolded along with it.
    """
    pool = series[ranked].dropna()
    if column.direction == "max":
        best = set(pool.nlargest(3).index)
    elif column.direction == "min":
        best = set(pool.nsmallest(3).index)
    elif column.direction == "zero":
        best = set(pool.abs().nsmallest(3).index)
    else:  # informational — never bolded
        best = set()

    cells = []
    for method, val in series.items():
        if pd.isna(val):
            cells.append("-")
        elif isinstance(val, str):  # the Estimand tag
            cells.append(f"\\texttt{{{val}}}")
        else:
            # The shared format strings are written for the console, where `%`
            # is just a character; in LaTeX it starts a comment and would
            # swallow the rest of the row.
            s = column.fmt.format(val).replace("%", "\\%")
            cells.append(f"\\textbf{{{s}}}" if method in best else s)
    return cells
