"""
metrics.py — Evaluation metrics for CATE benchmarks.

All datasets are standardised upstream (run._generate_data) using
context-only statistics, so every episode is already on the same outcome scale
before any method runs.  The metrics below are *additionally* scale-free w.r.t.
the treatment effect itself (everything is expressed relative to SD(τ)), so the
numbers are directly comparable across episodes and can be aggregated.

Which τ?  Two, and the distinction is the point
───────────────────────────────────────────────
Every function here is agnostic about *which* per-row target it is handed; the
caller says so with ``target_kind`` and the value is carried into the ``target``
column so both can live in one frame.  The two, on the same query rows:

``cate``  τ*(x) = E[Y1 − Y0 | X = x], estimated by the prior's Monte-Carlo
          average ``MU1 − MU0`` over ``K_eff`` replicates.  **The primary
          track**: it is the estimand ProximalFM is trained on, and the one a
          τ(x) method can actually attain.
``ite``   the realized draw ``Y1 − Y0``, which carries that row's own ``U`` and
          noise.  No method conditioning on X alone can beat
          ``E[Var(Y1 − Y0 | X)]`` here, so this track is where methods that use
          the query-row proxies — τ(x, w) — are allowed their real advantage,
          read against the ``Oracle-CATE(X)`` reference row that marks the floor.

Scoring a τ(x) estimator against ``ite`` is not wrong, just insensitive: the
noise floor is often ~95% of Var(ITE), which compresses every method onto
nPEHE ≈ 1.  That is why the benchmark defaults to the ``cate`` track whenever
the set carries a Monte-Carlo target.

``s2_over_var`` (``cate`` track only) is ``mean(S2)/Var(τ*)``: how much of the
scored target's variance is Monte-Carlo sampling noise in the target *itself*.
It is not corrected for — the metrics score ``MU1 − MU0`` as given — so a large
value means the set was built with too small a ``--mc_cate``, and every method on
that set looks worse than it is, the target's own noise having landed in the
residual.  Roughly 1e-4 at K=4000; 7e-2 at K=8.

The summary table keeps these numbers per method:

1. nPEHE  (median over episodes)
       npehe = √( mean((τ̂ − τ)²) / Var(τ) )
   Overall CATE accuracy, scale-free.  Reference point: the "predict the
   constant ATE" baseline sits at exactly 1.0, so anything < 1 is doing real
   work.  Var(τ)→0 episodes would blow this up, so (a) homogeneous episodes
   (Var(τ) < TARGET_VAR_THRESHOLD) are excluded and (b) we report the MEDIAN,
   which is robust to the remaining small-Var(τ) tail.

1b. CATE R²  (median over episodes)
       cate_r2 = 1 − mean((τ̂ − τ)²) / Var(τ)  == 1 − npehe²
   The same overall accuracy as nPEHE, expressed on the familiar R² scale
   (1 = perfect, 0 = constant-ATE baseline).

2. Shape R²  (median over episodes)
       shape_r2 = 1 − Var(τ̂ − τ) / Var(τ)
   Pure heterogeneity recovery, with the ATE-level error removed.  The most
   discriminative metric in proximal CATE.  Constant-ATE baseline sits at
   exactly 0.  Its full distribution (boxplot) matters more than the median;
   the per-episode value is kept for plotting.

3. ATE bias  (signed, normalised, median over episodes)
       ate_bias_norm = (mean(τ̂) − mean(τ)) / SD(τ)
   The level component of the same decomposition.  Signed, so systematic
   over/under-estimation is visible.

3b. ATE RMSE  (normalised, over episodes)
       ate_rmse = √( mean( ((mean(τ̂) − mean(τ)) / SD(τ))² ) )
   The unsigned magnitude of the ATE-level error, pooling both bias and
   episode-to-episode spread of the ATE estimate.

4. Mean rank  (by nPEHE)
   Per episode, methods are ranked by nPEHE (1 = best); averaged over episodes.
   Robust head-to-head ordering that ignores metric magnitudes.  The summary
   table is ordered by this column (best mean rank first).

5. Failure rate
   Fraction of episodes where the method produced no usable prediction (all
   NaN).  Failed episodes do not contribute to the rank.

6. Time (s)  — mean wall-clock fit+predict per episode.

Decomposition note: npehe² = 1 − cate_r2, and shape_r2 = cate_r2 + bias²/Var(τ),
so (1) and (2) together with (3) describe the standard level/shape split.

Design note
───────────
All functions operate on plain numpy arrays and return pandas DataFrames or
Series.  No torch; only r2_score-free numpy.  This keeps the metrics layer
independent of the modelling stack.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

# Episodes where the scored target's variance is below this threshold are
# treated as "constant treatment effect" and excluded from the scale-free
# aggregations (nPEHE, shape R², normalised ATE bias) whose denominator is
# Var(τ).  Kept equal to CausalEvaluator's homogeneous_var_threshold so the
# trainer's synt_point/* curves and this table drop the same episodes.
TARGET_VAR_THRESHOLD = 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# The estimand tag
# ─────────────────────────────────────────────────────────────────────────────
# A method output is labelled by what it answers, spelled out in full:
#
#   cate(x)        per-row effect conditioning on X alone — the primary target
#   cate(x,w)      ... and on the query row's own W: a strictly finer conditional
#   cate(x,w,z)
#   cate(x,u)      an oracle handed the true confounder
#   ate(x)         one scalar per episode
#
# The *kind* says whether the output varies per row (``cate``) or is a constant
# (``ate``). The parenthesis lists the variables the prediction reads at
# **predict** time — using a variable only to fit a nuisance does not count,
# which is why ProximalPLearner is ``cate(x)`` despite being fitted on both
# proxies. That is exactly the property the tables need to route on.
ESTIMAND_KINDS = ("cate", "ate")
# Canonical order, so the same conditioning set always prints the same string.
ESTIMAND_INPUTS = ("x", "w", "z", "u")

# The pre-spelled-out tags. Old configs (`estimand: x|xw|ate`), old results.csv
# files and old preds caches keep meaning exactly what they did.
LEGACY_ESTIMANDS = {"x": "cate(x)", "xw": "cate(x,w)", "ate": "ate(x)"}

TRACKS = ("cate", "ite", "ate")


def format_estimand(kind: str, inputs) -> str:
    """``("cate", ["w", "x"]) -> "cate(x,w)"`` — the canonical spelling."""
    if kind not in ESTIMAND_KINDS:
        raise ValueError(f"estimand kind must be one of {ESTIMAND_KINDS}, got {kind!r}")
    seen = set(inputs)
    unknown = seen - set(ESTIMAND_INPUTS)
    if unknown:
        raise ValueError(
            f"unknown estimand input(s) {sorted(unknown)}; "
            f"expected a subset of {ESTIMAND_INPUTS}"
        )
    return f"{kind}({','.join(i for i in ESTIMAND_INPUTS if i in seen)})"


def parse_estimand(tag: str) -> tuple[str, tuple[str, ...]]:
    """``"cate(x,w)" -> ("cate", ("x", "w"))``, inputs in canonical order.

    Accepts the legacy tags too, so a config or a results.csv written before the
    inputs were spelled out still parses.
    """
    tag = str(tag).strip()
    tag = LEGACY_ESTIMANDS.get(tag, tag)
    if not tag.endswith(")") or "(" not in tag:
        raise ValueError(
            f"estimand must look like 'cate(x,w)' or be one of "
            f"{sorted(LEGACY_ESTIMANDS)}, got {tag!r}"
        )
    kind, _, rest = tag[:-1].partition("(")
    inputs = {p.strip() for p in rest.split(",") if p.strip()}
    format_estimand(kind, inputs)  # validates both halves; raises on garbage
    return kind, tuple(i for i in ESTIMAND_INPUTS if i in inputs)


def canonical_estimand(tag: str) -> str:
    """Validate a tag and return its canonical spelling (legacy tags included)."""
    return format_estimand(*parse_estimand(tag))


def estimand_in_track(tag: str, track: str) -> bool:
    """Whether the ``track`` table can rank an output tagged ``tag``.

    Two checks, and nothing more: is it a per-row effect or a scalar, and does
    it read the proxies of the row it is predicting?

    ``cate`` (the headline) shows only per-row methods that condition on X
    alone — a ``cate(x,w)`` output estimates E[Y1-Y0 | X, W], a different
    quantity, and ranking it against tau*(x) would be a category error. The
    ``u`` of an oracle row is deliberately *not* disqualifying: those rows are
    in the table as upper bounds, which is the whole point of computing them.
    ``ite`` admits every per-row method, since that is where conditioning on the
    proxies is a genuine advantage. ``ate`` shows everyone: its columns are
    level-only, which is meaningful for a tau(x) method too, and it is the one
    table where a constant-prediction method says anything.
    """
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}; expected one of {list(TRACKS)}")
    kind, inputs = parse_estimand(tag)
    if track == "ate":
        return True
    if kind != "cate":
        return False
    return track == "ite" or not ({"w", "z"} & set(inputs))


# Small epsilon added to denominators to guard against zero-SD episodes.
_EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode metrics container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EpisodeMetrics:
    method: str
    episode: int

    # ── What was scored, and by whom ─────────────────────────────────────────
    target: str  # "cate" (MU1−MU0) or "ite" (Y1−Y0)
    estimand: str  # what this output answers, e.g. "cate(x)", "cate(x,w)", "ate(x)"
    is_reference: bool  # an oracle row: shown, but never ranked against

    # ── ATE level ────────────────────────────────────────────────────────────
    ate_true: float
    ate_pred: float
    ate_bias: float
    ate_bias_norm: Optional[float]  # signed, ÷ SD(τ)

    # ── CATE ─────────────────────────────────────────────────────────────────
    npehe: Optional[float]  # √(mean(resid²) / Var(τ))
    cate_r2: Optional[float]  # 1 − mean(resid²) / Var(τ)  == 1 − npehe²
    shape_r2: Optional[float]  # 1 − Var(resid) / Var(τ)

    # ── Bookkeeping ──────────────────────────────────────────────────────────
    target_var: float  # Var(τ) of the scored target — every denominator
    s2_over_var: Optional[float]  # MC noise share of the target ("cate" only)
    homogeneous: bool
    failed: bool
    n_test: int
    time: float


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_episode_metrics(
    method: str,
    episode: int,
    pred: np.ndarray,
    target: np.ndarray,
    time: float,
    target_var_threshold: float = TARGET_VAR_THRESHOLD,
    *,
    target_kind: str = "cate",
    estimand: str = "cate(x)",
    is_reference: bool = False,
    s2: Optional[np.ndarray] = None,
) -> EpisodeMetrics:
    """Score one episode's predictions against one per-row target.

    ``target`` is whichever ground truth the caller is scoring — the MC CATE or
    the realized ITE — and ``target_kind`` records which, so the two can be
    aggregated separately out of one frame. ``s2`` is the per-row sampling
    variance of a Monte-Carlo target; it is *reported* (as ``s2_over_var``) and
    never subtracted.
    """
    # Store the canonical spelling, so a legacy tag from an old config lands in
    # the frame as the tag every consumer of the frame now expects.
    estimand = canonical_estimand(estimand)
    failed = bool(np.all(np.isnan(pred)))

    ate_true = float(target.mean())
    target_var = float(target.var())  # population variance (ddof=0)
    sd_tau = float(np.sqrt(target_var))
    homogeneous = target_var < target_var_threshold

    s2_over_var = None
    if s2 is not None and not homogeneous:
        s2_over_var = float(np.nanmean(s2) / target_var)

    if failed:
        ate_pred = float("nan")
        ate_bias = float("nan")
        ate_bias_norm = None
        npehe = cate_r2 = shape_r2 = None
    else:
        ate_pred = float(np.nanmean(pred))
        ate_bias = ate_pred - ate_true

        # Replace any partial NaNs with the episode mean prediction.
        pred_clean = np.where(np.isnan(pred), ate_pred, pred)
        residuals = pred_clean - target

        if homogeneous:
            # Var(τ) ≈ 0 → scale-free quantities are undefined; report None.
            ate_bias_norm = None
            npehe = cate_r2 = shape_r2 = None
        else:
            ate_bias_norm = ate_bias / sd_tau
            npehe = float(np.sqrt((residuals**2).mean() / target_var))
            cate_r2 = float(1.0 - (residuals**2).mean() / target_var)
            shape_r2 = float(1.0 - residuals.var() / target_var)

    return EpisodeMetrics(
        method=method,
        episode=episode,
        target=target_kind,
        estimand=estimand,
        is_reference=is_reference,
        ate_true=ate_true,
        ate_pred=ate_pred,
        ate_bias=ate_bias,
        ate_bias_norm=ate_bias_norm,
        npehe=npehe,
        cate_r2=cate_r2,
        shape_r2=shape_r2,
        target_var=target_var,
        s2_over_var=s2_over_var,
        homogeneous=homogeneous,
        failed=failed,
        n_test=len(target),
        time=time,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_all_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    method: str,
    target_var_threshold: float = TARGET_VAR_THRESHOLD,
    *,
    target_kind: str = "cate",
    estimand: str = "cate(x)",
    is_reference: bool = False,
    s2: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Score one method against one target, over every episode."""
    rows = [
        asdict(
            compute_episode_metrics(
                method,
                b,
                preds[b],
                targets[b],
                times[b],
                target_var_threshold,
                target_kind=target_kind,
                estimand=estimand,
                is_reference=is_reference,
                s2=None if s2 is None else s2[b],
            )
        )
        for b in range(len(preds))
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Summary aggregation
# ─────────────────────────────────────────────────────────────────────────────


def select_track(all_results: pd.DataFrame, track: str) -> pd.DataFrame:
    """The rows of the per-episode frame that the ``track`` table shows.

    ``cate``/``ite`` select on the scored target *and* on which estimands can be
    ranked fairly against it (:func:`estimand_in_track`). ``ate`` is not a third
    scoring pass: it reuses the CATE-target rows (the ITE ones when the set has
    no CATE target) and simply shows every method, because the ATE-level columns
    are meaningful for all of them — it is where a method that only ever returns
    a scalar belongs.
    """
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}; expected one of {list(TRACKS)}")

    df = all_results
    if track == "ate":
        source = "cate" if (df["target"] == "cate").any() else "ite"
    else:
        source = track
    df = df[df["target"] == source]
    # Reset: the frame is a concat of one block per (method, target), so its
    # index repeats — and the rank assignment below aligns on it.
    keep = df["estimand"].map(lambda tag: estimand_in_track(tag, track))
    return df[keep].reset_index(drop=True)


def build_summary_table(
    all_results: pd.DataFrame, track: Optional[str] = None
) -> pd.DataFrame:
    """Collapse the per-episode frame into one row per method.

    Columns: nPEHE, CATE R², Shape R², ATE Bias, ATE RMSE, Mean Rank,
    Failure Rate, Time (s), Estimand.  Rows are ordered by the mean nPEHE rank
    (by ATE RMSE on the ``ate`` track, where nPEHE is not the question).

    ``track`` filters via :func:`select_track`; pass ``None`` for a frame that is
    already a single track's rows.
    """
    df = (
        select_track(all_results, track)
        if track is not None
        else all_results.reset_index(drop=True)
    )

    # Per-episode rank by nPEHE (1 = best). Failed/homogeneous episodes have
    # NaN npehe and are simply excluded from both the rank and its mean.
    # Reference rows (the oracles) are shown but never ranked: an oracle is not
    # a competitor, and letting it take rank 1 would shift every real method.
    rank_key = "ate_rmse_ep" if track == "ate" else "npehe"
    if track == "ate":
        df["ate_rmse_ep"] = df["ate_bias_norm"].abs()
    rankable = df[~df["is_reference"]]
    df["method_rank"] = rankable.groupby("episode")[rank_key].rank(
        method="min", ascending=True
    )

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "Estimand": g["estimand"].iloc[0],
                # An anchor for reading the table, not a competitor: unranked
                # here, and excluded from the LaTeX best-of bolding.
                "Reference": bool(g["is_reference"].any()),
                "nPEHE": g["npehe"].median(),
                "CATE R²": g["cate_r2"].median(),
                "Shape R²": g["shape_r2"].median(),
                "ATE Bias": g["ate_bias_norm"].median(),
                "ATE RMSE": float(np.sqrt((g["ate_bias_norm"] ** 2).mean())),
                "Mean Rank": g["method_rank"].mean(),
                "Failure Rate": g["failed"].mean(),
                "Time (s)": g["time"].mean(),
            }
        )

    summary = df.groupby("method", sort=False).apply(_agg)

    # Order by the mean rank (best head-to-head first); push all-NaN to the end,
    # which is where the unranked reference rows land.
    summary = summary.sort_values("Mean Rank", ascending=True, na_position="last")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Rich-formatted display
# ─────────────────────────────────────────────────────────────────────────────


def _style_npehe(v: float) -> str:
    if pd.isna(v):
        return ""
    if v < 0.75:
        return "green"
    if v < 1.0:
        return "yellow"
    return "red"


def _style_cate_r2(v: float) -> str:
    if pd.isna(v):
        return ""
    if v > 0.5:
        return "green"
    if v > 0.0:
        return "yellow"
    return "red"


def _style_shape_r2(v: float) -> str:
    if pd.isna(v):
        return ""
    if v > 0.5:
        return "green"
    if v > 0.0:
        return "yellow"
    return "red"


def _style_bias(v: float) -> str:
    if pd.isna(v):
        return ""
    a = abs(v)
    if a < 0.1:
        return "green"
    if a < 0.3:
        return "yellow"
    return "red"


def _style_failure(v: float) -> str:
    if pd.isna(v) or v == 0:
        return ""
    if v < 0.05:
        return "yellow"
    return "red"


# (column label, summary key, formatter, styler). "CATE R²" is deliberately not
# here: it is 1 − nPEHE², so on screen it costs a column and says nothing new.
# It stays in summary_*.csv for anyone who prefers the R² scale.
_RICH_COLUMNS = {
    "Mean Rank": ("Mean Rank", "{:.2f}", lambda v: ""),
    "nPEHE": ("nPEHE", "{:.3f}", _style_npehe),
    "Shape R²": ("Shape R²", "{:.3f}", _style_shape_r2),
    "ATE Bias": ("ATE Bias", "{:+.3f}", _style_bias),
    "ATE RMSE": ("ATE RMSE", "{:.3f}", _style_bias),
    "Fail %": ("Failure Rate", "{:.0%}", _style_failure),
    "Time (s)": ("Time (s)", "{:.2f}", lambda v: ""),
}

DEFAULT_RICH_COLUMNS = tuple(_RICH_COLUMNS)
# The ATE track ranks on the level only; its nPEHE/Shape columns would just
# repeat the CATE track for every method that has one.
ATE_RICH_COLUMNS = ("Mean Rank", "ATE Bias", "ATE RMSE", "Fail %", "Time (s)")


def make_summary_rich_table(
    summary: pd.DataFrame,
    title: str = "Benchmark Results",
    columns: tuple = DEFAULT_RICH_COLUMNS,
) -> "Table":
    from rich.table import Table
    from rich import box

    tbl = Table(title=title, box=box.SIMPLE_HEAD, show_footer=False)
    tbl.add_column("Method", style="bold cyan", no_wrap=True)
    # Wide enough for the longest tag ("cate(x,w,z,u)"): rich shrinks flexible
    # columns on a narrow terminal, and a truncated "cate…" says nothing.
    tbl.add_column("Est.", justify="center", no_wrap=True, min_width=13)
    for label in columns:
        tbl.add_column(label, justify="right")

    def styled(val: float, style: str, fmt: str = "{:.3f}") -> str:
        if pd.isna(val):
            return "-"
        s = fmt.format(val)
        return f"[{style}]{s}[/{style}]" if style else s

    for method, row in summary.iterrows():
        cells = []
        for label in columns:
            key, fmt, style_fn = _RICH_COLUMNS[label]
            cells.append(styled(row[key], style_fn(row[key]), fmt))
        tbl.add_row(str(method), str(row.get("Estimand", "")), *cells)

    return tbl
