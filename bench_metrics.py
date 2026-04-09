"""
bench_metrics.py — Evaluation metrics for CATE benchmarks.

Two distinct estimation targets
────────────────────────────────
ATE  (Average Treatment Effect)
    A scalar per episode: mean(τ̂) vs mean(τ).
    Every estimator produces an ATE, even a constant-effect model.
    Summary stats: Bias, MAE, RMSE, Std  — all in raw *and* normalised
    (÷ SD(τ)) units so that high-variance episodes do not dominate.
    Cross-episode R² measures how well predicted ATEs track true ATEs
    across the batch.

CATE (Conditional Average Treatment Effect)
    A function τ̂(x) evaluated at n_test query points per episode.
    Reported with both absolute metrics (MAE, RMSE, normalised by SD(τ))
    and the R² family:
        cate_r2       — standard R² (penalises both level and shape errors)
        cate_r2_shape — R² after removing the ATE bias (pure heterogeneity)
        cate_r2_ate_penalty — bias² / Var(τ), the ATE-level contribution
    Decomposition identity: cate_r2 = cate_r2_shape − cate_r2_ate_penalty
    Guard: episodes where Var(τ) < ITE_VAR_THRESHOLD are labelled
    'homogeneous' and excluded from R²-family averaging.  Absolute metrics
    (MAE, RMSE) are still computed for those episodes.

Design note
───────────
All functions operate on plain numpy arrays and return pandas DataFrames or
Series.  No torch; only r2_score is imported from sklearn.  This keeps the
metrics layer independent of the modelling stack.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Episodes where the true ITE variance is below this threshold are treated as
# "constant treatment effect" and excluded from CATE R²-family aggregation.
ITE_VAR_THRESHOLD = 1e-6

# Small epsilon added to denominators to guard against zero-SD episodes.
_EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode metrics container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EpisodeMetrics:
    """Metrics for one (method, episode) pair.

    Using a dataclass (rather than a plain dict) gives IDE auto-complete and
    makes it easy to convert to a DataFrame row via asdict().

    ATE fields
    ──────────
    ate_error      : signed error in raw units (pred − true).
    ate_error_norm : ate_error ÷ SD(τ).  Scale-free; comparable across
                     episodes with different outcome magnitudes.

    CATE absolute fields  (always computed when pred is not all-NaN)
    ────────────────────
    cate_mae / cate_rmse          : raw units.
    cate_mae_norm / cate_rmse_norm: divided by SD(τ), scale-free.
                                    None for homogeneous episodes.

    CATE R² fields  (None when homogeneous or all-NaN)
    ─────────────
    cate_r2             : standard R² — penalises both level shift and shape.
    cate_r2_shape       : R² after removing the ATE bias term.  Measures
                          pure heterogeneity recovery.
    cate_r2_ate_penalty : bias² / Var(τ).  The fraction of variance explained
                          that is lost purely due to ATE mis-estimation.
                          Identity: cate_r2 = cate_r2_shape − cate_r2_ate_penalty
    """

    method: str
    episode: int

    # ── ATE ──────────────────────────────────────────────────────────────────
    ate_true: float
    ate_pred: float
    ate_error: float        # signed: pred − true
    ate_error_norm: float   # ate_error / (SD(τ) + ε)

    # ── CATE absolute ────────────────────────────────────────────────────────
    cate_mae: Optional[float]        # None when all_nan
    cate_rmse: Optional[float]
    cate_mae_norm: Optional[float]   # None when all_nan or homogeneous
    cate_rmse_norm: Optional[float]

    # ── CATE R² ──────────────────────────────────────────────────────────────
    cate_r2: Optional[float]             # None when homogeneous or all_nan
    cate_r2_shape: Optional[float]       # None when homogeneous or all_nan
    cate_r2_ate_penalty: Optional[float] # None when homogeneous or all_nan

    homogeneous: bool   # True when Var(τ) < threshold
    ite_var: float      # actual variance (useful for debugging)
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
        ite_var_threshold : episodes with Var(τ) below this skip R²-family metrics.

    Returns:
        EpisodeMetrics dataclass instance.
    """
    ate_pred = float(np.nanmean(pred))
    ate_true = float(true_ite.mean())
    ite_var  = float(true_ite.var())
    ite_std  = float(np.sqrt(ite_var))

    homogeneous = ite_var < ite_var_threshold
    all_nan     = bool(np.all(np.isnan(pred)))

    # ── ATE ──────────────────────────────────────────────────────────────────
    ate_error      = ate_pred - ate_true
    ate_error_norm = ate_error / (ite_std + _EPS)

    # ── CATE absolute ────────────────────────────────────────────────────────
    if all_nan:
        cate_mae = cate_rmse = cate_mae_norm = cate_rmse_norm = None
    else:
        pred_clean   = np.where(np.isnan(pred), ate_pred, pred)
        residuals    = pred_clean - true_ite
        cate_mae     = float(np.abs(residuals).mean())
        cate_rmse    = float(np.sqrt((residuals ** 2).mean()))
        if homogeneous:
            # Normalisation is undefined (SD(τ) ≈ 0); keep absolute values only.
            cate_mae_norm  = None
            cate_rmse_norm = None
        else:
            cate_mae_norm  = cate_mae  / (ite_std + _EPS)
            cate_rmse_norm = cate_rmse / (ite_std + _EPS)

    # ── CATE R² family ───────────────────────────────────────────────────────
    # R² is only meaningful when the true effect has non-trivial variance.
    # Skipped entirely when all predictions are NaN.
    if homogeneous or all_nan:
        cate_r2 = cate_r2_shape = cate_r2_ate_penalty = None
    else:
        # pred_clean already computed above
        cate_r2 = float(r2_score(true_ite, pred_clean))

        # Decompose: MSE = Var(τ̂ − τ) + (mean(τ̂) − mean(τ))²
        # R²_shape       = 1 − Var(τ̂ − τ) / Var(τ)
        # R²_ate_penalty = bias² / Var(τ)
        # => R² = R²_shape − R²_ate_penalty
        bias_sq             = ate_error ** 2
        cate_r2_ate_penalty = float(bias_sq / (ite_var + _EPS))
        cate_r2_shape       = float(cate_r2 + cate_r2_ate_penalty)

    return EpisodeMetrics(
        method=method,
        episode=episode,
        ate_true=ate_true,
        ate_pred=ate_pred,
        ate_error=ate_error,
        ate_error_norm=ate_error_norm,
        cate_mae=cate_mae,
        cate_rmse=cate_rmse,
        cate_mae_norm=cate_mae_norm,
        cate_rmse_norm=cate_rmse_norm,
        cate_r2=cate_r2,
        cate_r2_shape=cate_r2_shape,
        cate_r2_ate_penalty=cate_r2_ate_penalty,
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
    """ATE error summary for one method's episodes.

    Raw errors (Bias, MAE, RMSE …) are supplemented by normalised equivalents
    (NormBias, NormMAE, NormRMSE) computed from ate_error_norm, which divides
    each episode's error by SD(τ) so that high-variance episodes do not dominate.

    ATE R² measures how well the predicted ATEs track the true ATEs *across*
    the episode batch (treating each episode as one observation).  A method
    that always predicts the right direction/ranking of ATEs scores well here
    even if its absolute errors are large.
    """
    e      = df["ate_error"]
    e_norm = df["ate_error_norm"]

    # Cross-episode ATE R²: does τ̂_mean correlate with τ_mean across episodes?
    try:
        ate_r2 = round(float(r2_score(df["ate_true"], df["ate_pred"])), 4)
    except Exception:
        ate_r2 = float("nan")

    return pd.Series(
        {
            # ── Raw ──────────────────────────────────────────────────────────
            "Bias":    round(e.mean(), 4),
            "Std":     round(e.std(), 4),
            "MAE":     round(e.abs().mean(), 4),
            "MedAE":   round(e.abs().median(), 4),
            "RMSE":    round(np.sqrt((e ** 2).mean()), 4),
            "RMedSE":  round(np.sqrt(np.median(e ** 2)), 4),
            # ── Normalised (÷ SD(τ)) ─────────────────────────────────────────
            "NormBias":  round(e_norm.mean(), 4),
            "NormMAE":   round(e_norm.abs().mean(), 4),
            "NormRMSE":  round(np.sqrt((e_norm ** 2).mean()), 4),
            # ── Cross-episode R² ─────────────────────────────────────────────
            "ATE R²":  ate_r2,
        }
    )


def _summarise_cate(df: pd.DataFrame) -> pd.Series:
    """CATE summary: absolute metrics + R² family, excluding homogeneous episodes.

    Absolute metrics (MAE, RMSE) are averaged over episodes where predictions
    exist, regardless of homogeneity.  The normalised variants (÷ SD(τ)) and
    all R²-family metrics are restricted to heterogeneous episodes.

    Reports 'Episodes used' and 'Fraction used' so the reader knows how
    informative the R²-family stats are.
    """
    heterogeneous = df.loc[~df["homogeneous"]]
    n_total = len(df)
    n_valid = len(heterogeneous)
    frac_used = n_valid / n_total if n_total > 0 else 0.0

    # ── Absolute CATE metrics (all non-all-NaN episodes) ─────────────────────
    mae_vals  = df["cate_mae"].dropna()
    rmse_vals = df["cate_rmse"].dropna()
    mean_mae  = round(mae_vals.mean(),  4) if len(mae_vals)  > 0 else float("nan")
    mean_rmse = round(rmse_vals.mean(), 4) if len(rmse_vals) > 0 else float("nan")

    # Normalised absolute (heterogeneous episodes only)
    mae_n_vals  = heterogeneous["cate_mae_norm"].dropna()
    rmse_n_vals = heterogeneous["cate_rmse_norm"].dropna()
    mean_mae_n  = round(mae_n_vals.mean(),  4) if len(mae_n_vals)  > 0 else float("nan")
    mean_rmse_n = round(rmse_n_vals.mean(), 4) if len(rmse_n_vals) > 0 else float("nan")

    if n_valid == 0:
        return pd.Series(
            {
                "MAE":           mean_mae,
                "RMSE":          mean_rmse,
                "NormMAE":       float("nan"),
                "NormRMSE":      float("nan"),
                "Mean R²":       float("nan"),
                "Median R²":     float("nan"),
                "Mean R²(shape)":   float("nan"),
                "Mean ATE penalty": float("nan"),
                "Episodes used": 0,
                "⚠ note": "All episodes homogeneous — R² not defined",
            }
        )

    r2_vals     = heterogeneous["cate_r2"].dropna()
    r2s_vals    = heterogeneous["cate_r2_shape"].dropna()
    r2pen_vals  = heterogeneous["cate_r2_ate_penalty"].dropna()

    return pd.Series(
        {
            # ── Absolute ─────────────────────────────────────────────────────
            "MAE":      mean_mae,
            "RMSE":     mean_rmse,
            "NormMAE":  mean_mae_n,
            "NormRMSE": mean_rmse_n,
            # ── Standard R² ──────────────────────────────────────────────────
            "Mean R²":   round(r2_vals.mean(),   4),
            "Median R²": round(r2_vals.median(), 4),
            "Std R²":    round(r2_vals.std(),    4),
            # ── R² decomposition ─────────────────────────────────────────────
            # R²(shape): pure heterogeneity recovery, ATE bias removed.
            # ATE penalty: fraction of variance lost due to ATE mis-estimation.
            # Identity: Mean R² ≈ Mean R²(shape) − Mean ATE penalty
            "Mean R²(shape)":    round(r2s_vals.mean(),   4),
            "Median R²(shape)":  round(r2s_vals.median(), 4),
            "Mean ATE penalty":  round(r2pen_vals.mean(), 4),
            # ── Coverage ─────────────────────────────────────────────────────
            "Episodes used": n_valid,
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
    ate_summary  = all_results.groupby("method", sort=False).apply(_summarise_ate)
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

    norm_mae_015 = ate_summary["NormMAE"].quantile(0.15)
    norm_mae_085 = ate_summary["NormMAE"].quantile(0.85)

    for method, row in ate_summary.iterrows():
        cells = []
        for col_name, val in row.items():
            style = ""
            if col_name == "NormMAE":
                v = abs(val)
                style = (
                    "green" if v < norm_mae_015 else
                    ("yellow" if v < norm_mae_085 else "red")
                )
            elif col_name == "ATE R²":
                try:
                    style = (
                        "green" if val > 0.5 else
                        ("yellow" if val > 0.0 else "red")
                    )
                except (TypeError, ValueError):
                    style = ""
            cells.append(f"[{style}]{val}[/{style}]" if style else str(val))
        tbl.add_row(str(method), *cells)

    return tbl


def make_cate_rich_table(
    cate_summary: pd.DataFrame, title: str = "CATE Estimation"
) -> "Table":  # noqa: F821
    from rich.table import Table
    from rich import box

    tbl = Table(title=title, box=box.SIMPLE_HEAD, show_footer=False)
    tbl.add_column("Method", style="bold cyan", no_wrap=True)
    for col in cate_summary.columns:
        tbl.add_column(col, justify="right")

    for method, row in cate_summary.iterrows():
        # Colour anchor: Mean R²(shape) — the most interpretable R² column.
        r2_shape = row.get("Mean R²(shape)", float("nan"))
        try:
            r2_style = (
                "green" if r2_shape > 0.5 else
                ("yellow" if r2_shape > 0.0 else "red")
            )
        except (TypeError, ValueError):
            r2_style = ""

        # NormMAE colouring: relative within the table.
        norm_mae_vals = cate_summary["NormMAE"].dropna()
        nm_015 = norm_mae_vals.quantile(0.15) if len(norm_mae_vals) > 0 else 0
        nm_085 = norm_mae_vals.quantile(0.85) if len(norm_mae_vals) > 0 else 1

        cells = []
        for col_name, val in row.items():
            style = ""
            if "R²" in str(col_name):
                style = r2_style
            elif col_name == "NormMAE":
                try:
                    style = (
                        "green" if val < nm_015 else
                        ("yellow" if val < nm_085 else "red")
                    )
                except (TypeError, ValueError):
                    style = ""
            elif col_name == "Mean ATE penalty":
                # High penalty = ATE bias is hurting R² a lot → flag in yellow/red.
                try:
                    style = (
                        "green" if val < 0.1 else
                        ("yellow" if val < 0.5 else "red")
                    )
                except (TypeError, ValueError):
                    style = ""
            cells.append(f"[{style}]{val}[/{style}]" if style else str(val))
        tbl.add_row(str(method), *cells)

    return tbl
