"""
bench_metrics.py — Evaluation metrics for CATE benchmarks.

Two distinct estimation targets
────────────────────────────────
ATE  (Average Treatment Effect)
    A scalar per episode: mean(τ̂) vs mean(τ).
    Every estimator produces an ATE, even a constant-effect model.
    Summary stats: Bias, MAE, RMSE, Std.

CATE (Conditional Average Treatment Effect)
    A function τ̂(x) evaluated at n_test query points per episode.
    Summarised by within-episode R².
    Only meaningful when the DGP admits *heterogeneous* effects.
    Guard: episodes where Var(τ) < ITE_VAR_THRESHOLD are labelled
    'homogeneous' and excluded from R² averaging, with a usage fraction
    reported so you know how many episodes contributed.

Design note
───────────
All functions operate on plain numpy arrays and return pandas DataFrames or
Series.  No torch, no sklearn imports except r2_score.  This keeps the
metrics layer independent of the modelling stack.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Episodes where the true ITE variance is below this threshold are treated as
# "constant treatment effect" and excluded from CATE R² aggregation.
ITE_VAR_THRESHOLD = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode metrics container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EpisodeMetrics:
    """Metrics for one (method, episode) pair.

    Using a dataclass (rather than a plain dict) gives IDE auto-complete and
    makes it easy to convert to a DataFrame row via asdict().
    """

    method: str
    episode: int

    # ATE
    ate_true: float
    ate_pred: float
    ate_error: float  # signed: pred - true

    # CATE
    cate_r2: Optional[float]  # None when homogeneous
    homogeneous: bool  # True when Var(τ) < threshold
    ite_var: float  # actual variance (useful for debugging)

    n_test: int


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_episode_metrics(
    method: str,
    episode: int,
    pred: np.ndarray,
    true_ite: np.ndarray,
    ite_var_threshold: float = ITE_VAR_THRESHOLD,
) -> EpisodeMetrics:
    """Compute all metrics for one (method, episode) pair.

    Args:
        method    : display name of the estimator.
        episode   : episode index (0-based).
        pred      : predicted ITEs, shape (n_test,).  May contain NaN.
        true_ite  : ground-truth ITEs, shape (n_test,).
        ite_var_threshold : episodes with Var(τ) below this skip CATE R².

    Returns:
        EpisodeMetrics dataclass instance.
    """
    ate_pred = float(np.nanmean(pred))
    ate_true = float(true_ite.mean())
    ite_var = float(true_ite.var())

    # An episode is "homogeneous" when the true treatment effect does not
    # vary across units.  In that case R² is undefined (division by zero in
    # the denominator) and any estimator that predicts a constant will achieve
    # R²=1 trivially — so we exclude these episodes from CATE evaluation.
    homogeneous = ite_var < ite_var_threshold
    all_nan = np.all(np.isnan(pred))

    if homogeneous or all_nan:
        cate_r2 = None
    else:
        # Replace NaN predictions with the episode mean to avoid r2_score
        # crashing, while still penalising partial failures.
        pred_clean = np.where(np.isnan(pred), ate_pred, pred)
        cate_r2 = float(r2_score(true_ite, pred_clean))

    return EpisodeMetrics(
        method=method,
        episode=episode,
        ate_true=ate_true,
        ate_pred=ate_pred,
        ate_error=ate_pred - ate_true,
        cate_r2=cate_r2,
        homogeneous=homogeneous,
        ite_var=ite_var,
        n_test=len(true_ite),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_all_metrics(
    preds: np.ndarray,
    true_ites: np.ndarray,
    method: str,
    ite_var_threshold: float = ITE_VAR_THRESHOLD,
) -> pd.DataFrame:
    """Compute per-episode metrics for all episodes of one method.

    Args:
        preds      : shape (n_batch, n_test).
        true_ites  : shape (n_batch, n_test).
        method     : display name.

    Returns:
        DataFrame with one row per episode, columns = EpisodeMetrics fields.
    """
    rows = [
        asdict(
            compute_episode_metrics(
                method, b, preds[b], true_ites[b], ite_var_threshold
            )
        )
        for b in range(len(preds))
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Summary aggregation
# ─────────────────────────────────────────────────────────────────────────────


def _summarise_ate(df: pd.DataFrame) -> pd.Series:
    """ATE error summary for one method's episodes."""
    e = df["ate_error"]
    return pd.Series(
        {
            "Bias": round(e.mean(), 4),
            "Std": round(e.std(), 4),
            "MAE": round(e.abs().mean(), 4),
            "MedAE": round(e.abs().median(), 4),
            "RMSE": round(np.sqrt((e**2).mean()), 4),
            "RMedSE": round(np.sqrt(np.median(e**2)), 4),
        }
    )


def _summarise_cate(df: pd.DataFrame) -> pd.Series:
    """CATE R² summary, excluding homogeneous-effect episodes.

    Reports 'Fraction used' so the reader knows how informative the R² is.
    If all episodes are homogeneous (no ITE variance), all R² entries are NaN
    and a clear warning is embedded in the summary.
    """
    valid = df.loc[~df["homogeneous"], "cate_r2"].dropna()
    n_total = len(df)
    n_valid = len(valid)
    frac_used = n_valid / n_total if n_total > 0 else 0.0

    if n_valid == 0:
        # Warn clearly rather than silently returning NaN
        return pd.Series(
            {
                "Mean R²": float("nan"),
                "Median R²": float("nan"),
                "Std R²": float("nan"),
                "Episodes used": 0,
                "Fraction used": 0.0,
                "⚠ note": "All episodes homogeneous — CATE R² not defined",
            }
        )

    return pd.Series(
        {
            "Mean R²": round(valid.mean(), 4),
            "Median R²": round(valid.median(), 4),
            "Std R²": round(valid.std(), 4),
            "Episodes used": n_valid,
            "Fraction used": round(frac_used, 3),
            "⚠ note": (
                "" if frac_used > 0.9 else f"Only {frac_used:.0%} of episodes used"
            ),
        }
    )


def build_summary_tables(
    all_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ATE and CATE summary tables from concatenated episode results.

    Args:
        all_results : DataFrame returned by concatenating compute_all_metrics()
                      across all methods.  Must contain a 'method' column.

    Returns:
        (ate_summary, cate_summary) — both indexed by method name.
    """
    ate_summary = all_results.groupby("method", sort=False).apply(_summarise_ate)
    cate_summary = all_results.groupby("method", sort=False).apply(_summarise_cate)
    return ate_summary, cate_summary


# ─────────────────────────────────────────────────────────────────────────────
# Rich-formatted display helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_ate_rich_table(
    ate_summary: pd.DataFrame, title: str = "ATE Estimation"
) -> "Table":  # noqa: F821
    """Build a Rich Table for the ATE summary.  Imported lazily to keep metrics
    module usable without rich installed."""
    from rich.table import Table
    from rich import box

    tbl = Table(title=title, box=box.SIMPLE_HEAD, show_footer=False)
    tbl.add_column("Method", style="bold cyan", no_wrap=True)
    for col in ate_summary.columns:
        tbl.add_column(col, justify="right")

    medAE_015 = ate_summary["MedAE"].quantile(0.15)
    medAE_085 = ate_summary["MedAE"].quantile(0.85)

    for method, row in ate_summary.iterrows():
        # Colour the MedAE column: green if |MedAE| < 0.05, yellow < 0.2, red otherwise
        medAE = abs(row["MedAE"])
        medAE_style = (
            "green" if medAE < medAE_015 else ("yellow" if medAE < medAE_085 else "red")
        )
        cells = []
        for i, val in enumerate(row):
            style = medAE_style if ate_summary.columns[i] == "MedAE" else ""
            cells.append(f"[{style}]{val}[/{style}]" if style else str(val))
        tbl.add_row(str(method), *cells)

    return tbl


def make_cate_rich_table(
    cate_summary: pd.DataFrame, title: str = "CATE Estimation (R²)"
) -> "Table":  # noqa: F821
    from rich.table import Table
    from rich import box

    tbl = Table(title=title, box=box.SIMPLE_HEAD, show_footer=False)
    tbl.add_column("Method", style="bold cyan", no_wrap=True)
    for col in cate_summary.columns:
        tbl.add_column(col, justify="right")

    for method, row in cate_summary.iterrows():
        r2 = row.get("Mean R²", float("nan"))
        try:
            r2_style = "green" if r2 > 0.5 else ("yellow" if r2 > 0.0 else "red")
        except (TypeError, ValueError):
            r2_style = ""
        cells = []
        for col_name, val in row.items():
            style = r2_style if "R²" in str(col_name) else ""
            cells.append(f"[{style}]{val}[/{style}]" if style else str(val))
        tbl.add_row(str(method), *cells)

    return tbl
