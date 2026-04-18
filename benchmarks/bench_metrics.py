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
    cate_medae: Optional[float]      # <-- NEW
    cate_rmedse: Optional[float]     # <-- NEW
    
    cate_mae_norm: Optional[float]   # None when all_nan or homogeneous
    cate_rmse_norm: Optional[float]
    cate_medae_norm: Optional[float] # <-- NEW
    cate_rmedse_norm: Optional[float]# <-- NEW
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
# ── CATE absolute ────────────────────────────────────────────────────────
    if all_nan:
        cate_mae = cate_rmse = cate_medae = cate_rmedse = None
        cate_mae_norm = cate_rmse_norm = cate_medae_norm = cate_rmedse_norm = None
    else:
        pred_clean   = np.where(np.isnan(pred), ate_pred, pred)
        residuals    = pred_clean - true_ite
        
        cate_mae     = float(np.abs(residuals).mean())
        cate_rmse    = float(np.sqrt((residuals ** 2).mean()))
        cate_medae   = float(np.median(np.abs(residuals)))          # <-- NEW
        cate_rmedse  = float(np.sqrt(np.median(residuals ** 2)))    # <-- NEW
        
        if homogeneous:
            cate_mae_norm = cate_rmse_norm = cate_medae_norm = cate_rmedse_norm = None
        else:
            cate_mae_norm    = cate_mae  / (ite_std + _EPS)
            cate_rmse_norm   = cate_rmse / (ite_std + _EPS)
            cate_medae_norm  = cate_medae / (ite_std + _EPS)        # <-- NEW
            cate_rmedse_norm = cate_rmedse / (ite_std + _EPS)       # <-- NEW
    
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
        cate_medae=cate_medae,              # <-- NEW
        cate_rmedse=cate_rmedse,            # <-- NEW
        cate_mae_norm=cate_mae_norm,
        cate_rmse_norm=cate_rmse_norm,
        cate_medae_norm=cate_medae_norm,    # <-- NEW
        cate_rmedse_norm=cate_rmedse_norm,  # <-- NEW    
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
            "NormMedAE": round(e_norm.abs().median(), 4),
            "NormRMSE":  round(np.sqrt((e_norm ** 2).mean()), 4),
            "NormRMedSE":round(np.sqrt(np.median(e_norm ** 2)), 4),
            # ── Cross-episode R² ─────────────────────────────────────────────
            "ATE R²":  ate_r2,
        }
    )


def _summarise_cate(df: pd.DataFrame) -> pd.Series:
    heterogeneous = df.loc[~df["homogeneous"]]
    n_total = len(df)
    n_valid = len(heterogeneous)
    frac_used = n_valid / n_total if n_total > 0 else 0.0

    # ── Absolute CATE metrics (all non-all-NaN episodes) ─────────────────────
    mae_vals    = df["cate_mae"].dropna()
    rmse_vals   = df["cate_rmse"].dropna()
    medae_vals  = df["cate_medae"].dropna()
    rmedse_vals = df["cate_rmedse"].dropna()
    
    mean_mae    = round(mae_vals.mean(),  4) if len(mae_vals)  > 0 else float("nan")
    mean_rmse   = round(rmse_vals.mean(), 4) if len(rmse_vals) > 0 else float("nan")
    mean_medae  = round(medae_vals.mean(), 4) if len(medae_vals) > 0 else float("nan")
    mean_rmedse = round(rmedse_vals.mean(), 4) if len(rmedse_vals) > 0 else float("nan")

    # Normalised absolute (heterogeneous episodes only)
    mae_n_vals    = heterogeneous["cate_mae_norm"].dropna()
    rmse_n_vals   = heterogeneous["cate_rmse_norm"].dropna()
    medae_n_vals  = heterogeneous["cate_medae_norm"].dropna()
    rmedse_n_vals = heterogeneous["cate_rmedse_norm"].dropna()
    
    mean_mae_n    = round(mae_n_vals.mean(),  4) if len(mae_n_vals)  > 0 else float("nan")
    mean_rmse_n   = round(rmse_n_vals.mean(), 4) if len(rmse_n_vals) > 0 else float("nan")
    mean_medae_n  = round(medae_n_vals.mean(), 4) if len(medae_n_vals) > 0 else float("nan")
    mean_rmedse_n = round(rmedse_n_vals.mean(), 4) if len(rmedse_n_vals) > 0 else float("nan")

    if n_valid == 0:
        return pd.Series(
            {
                "RMSE":          mean_rmse,
                "RMedSE":        mean_rmedse,
                "NormRMSE":      float("nan"),
                "NormRMedSE":    float("nan"),
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
            "RMSE":       mean_rmse,
            "RMedSE":     mean_rmedse,
            "NormRMSE":   mean_rmse_n,
            "NormRMedSE": mean_rmedse_n,
            # ── Standard R² ──────────────────────────────────────────────────
            "Mean R²":   round(r2_vals.mean(),   4),
            "Median R²": round(r2_vals.median(), 4),
            "Std R²":    round(r2_vals.std(),    4),
            # ── R² decomposition ─────────────────────────────────────────────
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

def _get_column_styles(series: pd.Series, col_name: str) -> dict:
    """Returns a dict mapping dataframe index to a rich color string.
    Dynamically infers whether higher/lower/closer to 0 is better based on the col name.
    """
    styles = {}
    
    if col_name in ["⚠ note", "Episodes used"]:
        return {k: "" for k in series.index}
        
    # Convert to numeric to safely compute quantiles
    valid = pd.to_numeric(series, errors='coerce').dropna()
    if len(valid) == 0:
        return {k: "" for k in series.index}

    # 1. R² Metrics: Higher is better (Fixed absolute thresholds)
    if "R²" in col_name and "Std" not in col_name:
        for k, v in series.items():
            if pd.isna(v) or isinstance(v, str): styles[k] = ""
            elif v > 0.5: styles[k] = "green"
            elif v > 0.0: styles[k] = "yellow"
            else: styles[k] = "red"
        return styles
        
    # 2. Penalty Metric: Lower is better (Fixed absolute thresholds)
    if "penalty" in col_name.lower():
        for k, v in series.items():
            if pd.isna(v) or isinstance(v, str): styles[k] = ""
            elif v < 0.1: styles[k] = "green"
            elif v < 0.5: styles[k] = "yellow"
            else: styles[k] = "red"
        return styles

    # 3. Bias Metrics: Closer to 0 is better (Relative quantiles)
    if "Bias" in col_name:
        abs_valid = valid.abs()
        q_green = abs_valid.quantile(0.33)
        q_yellow = abs_valid.quantile(0.66)
        for k, v in series.items():
            if pd.isna(v) or isinstance(v, str): styles[k] = ""
            else:
                v_abs = abs(v)
                if v_abs <= q_green: styles[k] = "green"
                elif v_abs <= q_yellow: styles[k] = "yellow"
                else: styles[k] = "red"
        return styles

    # 4. Error Metrics (MAE, RMSE, Std, etc.): Lower is better (Relative quantiles)
    q_green = valid.quantile(0.33)
    q_yellow = valid.quantile(0.66)
    for k, v in series.items():
        if pd.isna(v) or isinstance(v, str): styles[k] = ""
        elif v <= q_green: styles[k] = "green"
        elif v <= q_yellow: styles[k] = "yellow"
        else: styles[k] = "red"
        
    return styles


def make_ate_rich_table(
    ate_summary: pd.DataFrame, title: str = "ATE Estimation"
) -> "Table":  # noqa: F821
    """Build a Rich Table for the ATE summary."""
    from rich.table import Table
    from rich import box

    tbl = Table(title=title, box=box.SIMPLE_HEAD, show_footer=False)
    tbl.add_column("Method", style="bold cyan", no_wrap=True)
    
    # Add columns right-aligned
    for col in ate_summary.columns:
        tbl.add_column(col, justify="right")

    # Pre-calculate column styles universally
    col_styles = {
        col: _get_column_styles(ate_summary[col], col)
        for col in ate_summary.columns
    }

    for method, row in ate_summary.iterrows():
        cells = []
        for col_name, val in row.items():
            style = col_styles[col_name].get(method, "")
            cells.append(f"[{style}]{val}[/{style}]" if style else str(val))
        tbl.add_row(str(method), *cells)

    return tbl


def make_cate_rich_table(
    cate_summary: pd.DataFrame, title: str = "CATE Estimation"
) -> "Table":  # noqa: F821
    """Build a Rich Table for the CATE summary."""
    from rich.table import Table
    from rich import box

    tbl = Table(title=title, box=box.SIMPLE_HEAD, show_footer=False)
    tbl.add_column("Method", style="bold cyan", no_wrap=True)
    
    # Add columns right-aligned
    for col in cate_summary.columns:
        tbl.add_column(col, justify="right")

    # Pre-calculate column styles universally
    col_styles = {
        col: _get_column_styles(cate_summary[col], col)
        for col in cate_summary.columns
    }

    for method, row in cate_summary.iterrows():
        cells = []
        for col_name, val in row.items():
            style = col_styles[col_name].get(method, "")
            cells.append(f"[{style}]{val}[/{style}]" if style else str(val))
        tbl.add_row(str(method), *cells)

    return tbl