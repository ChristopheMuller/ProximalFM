"""
bench_metrics.py — Evaluation metrics for CATE benchmarks.

All datasets are standardised upstream (bench_run._generate_data) using
context-only statistics, so every episode is already on the same outcome scale
before any method runs.  The metrics below are *additionally* scale-free w.r.t.
the treatment effect itself (everything is expressed relative to SD(τ)), so the
numbers are directly comparable across episodes and can be aggregated.

The summary table keeps these numbers per method:

1. nPEHE  (median over episodes)
       npehe = √( mean((τ̂ − τ)²) / Var(τ) )
   Overall CATE accuracy, scale-free.  Reference point: the "predict the
   constant ATE" baseline sits at exactly 1.0, so anything < 1 is doing real
   work.  Var(τ)→0 episodes would blow this up, so (a) homogeneous episodes
   (Var(τ) < ITE_VAR_THRESHOLD) are excluded and (b) we report the MEDIAN,
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

# Episodes where the true ITE variance is below this threshold are treated as
# "constant treatment effect" and excluded from the scale-free aggregations
# (nPEHE, shape R², normalised ATE bias) whose denominator is Var(τ).
ITE_VAR_THRESHOLD = 1e-6

# Small epsilon added to denominators to guard against zero-SD episodes.
_EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode metrics container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EpisodeMetrics:
    method: str
    episode: int

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
    ite_var: float
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
    true_ite: np.ndarray,
    time: float,
    ite_var_threshold: float = ITE_VAR_THRESHOLD,
) -> EpisodeMetrics:
    failed = bool(np.all(np.isnan(pred)))

    ate_true = float(true_ite.mean())
    ite_var = float(true_ite.var())  # population variance (ddof=0)
    sd_tau = float(np.sqrt(ite_var))
    homogeneous = ite_var < ite_var_threshold

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
        residuals = pred_clean - true_ite

        if homogeneous:
            # Var(τ) ≈ 0 → scale-free quantities are undefined; report None.
            ate_bias_norm = None
            npehe = cate_r2 = shape_r2 = None
        else:
            ate_bias_norm = ate_bias / sd_tau
            npehe = float(np.sqrt((residuals**2).mean() / ite_var))
            cate_r2 = float(1.0 - (residuals**2).mean() / ite_var)
            shape_r2 = float(1.0 - residuals.var() / ite_var)

    return EpisodeMetrics(
        method=method,
        episode=episode,
        ate_true=ate_true,
        ate_pred=ate_pred,
        ate_bias=ate_bias,
        ate_bias_norm=ate_bias_norm,
        npehe=npehe,
        cate_r2=cate_r2,
        shape_r2=shape_r2,
        ite_var=ite_var,
        homogeneous=homogeneous,
        failed=failed,
        n_test=len(true_ite),
        time=time,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_all_metrics(
    preds: np.ndarray,
    true_ites: np.ndarray,
    times: np.ndarray,
    method: str,
    ite_var_threshold: float = ITE_VAR_THRESHOLD,
) -> pd.DataFrame:
    rows = [
        asdict(
            compute_episode_metrics(
                method, b, preds[b], true_ites[b], times[b], ite_var_threshold
            )
        )
        for b in range(len(preds))
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Summary aggregation
# ─────────────────────────────────────────────────────────────────────────────


def build_summary_table(all_results: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-episode frame into one row per method.

    Columns: nPEHE, CATE R², Shape R², ATE Bias, ATE RMSE, Mean Rank,
    Failure Rate, Time (s).  Rows are ordered by the mean nPEHE rank.
    """
    df = all_results.copy()

    # Per-episode rank by nPEHE (1 = best). Failed/homogeneous episodes have
    # NaN npehe and are simply excluded from both the rank and its mean.
    df["npehe_rank"] = df.groupby("episode")["npehe"].rank(method="min", ascending=True)

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "nPEHE": g["npehe"].median(),
                "CATE R²": g["cate_r2"].median(),
                "Shape R²": g["shape_r2"].median(),
                "ATE Bias": g["ate_bias_norm"].median(),
                "ATE RMSE": float(np.sqrt((g["ate_bias_norm"] ** 2).mean())),
                "Mean Rank": g["npehe_rank"].mean(),
                "Failure Rate": g["failed"].mean(),
                "Time (s)": g["time"].mean(),
            }
        )

    summary = df.groupby("method", sort=False).apply(_agg)

    # Order by the mean nPEHE rank (best head-to-head first); push all-NaN to
    # the end.
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


def make_summary_rich_table(
    summary: pd.DataFrame,
    title: str = "Benchmark Results",
) -> "Table":
    from rich.table import Table
    from rich import box

    tbl = Table(title=title, box=box.SIMPLE_HEAD, show_footer=False)
    tbl.add_column("Method", style="bold cyan", no_wrap=True)
    tbl.add_column("Mean Rank", justify="right")
    tbl.add_column("nPEHE", justify="right")
    tbl.add_column("CATE R²", justify="right")
    tbl.add_column("Shape R²", justify="right")
    tbl.add_column("ATE Bias", justify="right")
    tbl.add_column("ATE RMSE", justify="right")
    tbl.add_column("Fail %", justify="right")
    tbl.add_column("Time (s)", justify="right")

    def styled(val: float, style: str, fmt: str = "{:.3f}") -> str:
        if pd.isna(val):
            return "-"
        s = fmt.format(val)
        return f"[{style}]{s}[/{style}]" if style else s

    for method, row in summary.iterrows():
        tbl.add_row(
            str(method),
            styled(row["Mean Rank"], "", "{:.2f}"),
            styled(row["nPEHE"], _style_npehe(row["nPEHE"])),
            styled(row["CATE R²"], _style_cate_r2(row["CATE R²"])),
            styled(row["Shape R²"], _style_shape_r2(row["Shape R²"])),
            styled(row["ATE Bias"], _style_bias(row["ATE Bias"]), "{:+.3f}"),
            styled(row["ATE RMSE"], _style_bias(row["ATE RMSE"])),
            styled(row["Failure Rate"], _style_failure(row["Failure Rate"]), "{:.0%}"),
            styled(row["Time (s)"], "", "{:.2f}"),
        )

    return tbl
