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
    method: str
    episode: int

    # ── ATE ──────────────────────────────────────────────────────────────────
    ate_true: float
    ate_pred: float
    ate_error: float

    # ── CATE absolute ────────────────────────────────────────────────────────
    cate_mae: Optional[float]
    cate_rmse: Optional[float]
    cate_medae: Optional[float]
    cate_rmedse: Optional[float]

    # ── CATE R² ──────────────────────────────────────────────────────────────
    cate_r2: Optional[float]
    cate_r2_shape: Optional[float]
    cate_r2_ate_penalty: Optional[float]

    homogeneous: bool
    ite_var: float
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
    ite_var = float(true_ite.var())
    ite_std = float(np.sqrt(ite_var))

    homogeneous = ite_var < ite_var_threshold
    all_nan = bool(np.all(np.isnan(pred)))

    # ── ATE ──────────────────────────────────────────────────────────────────
    ate_error = ate_pred - ate_true

    # ── CATE absolute ────────────────────────────────────────────────────────
    if all_nan:
        cate_mae = cate_rmse = cate_medae = cate_rmedse = None
    else:
        pred_clean = np.where(np.isnan(pred), ate_pred, pred)
        residuals = pred_clean - true_ite

        cate_mae = float(np.abs(residuals).mean())
        cate_rmse = float(np.sqrt((residuals**2).mean()))
        cate_medae = float(np.median(np.abs(residuals)))
        cate_rmedse = float(np.sqrt(np.median(residuals**2)))

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
        bias_sq = ate_error**2
        cate_r2_ate_penalty = float(bias_sq / (ite_var + _EPS))
        cate_r2_shape = float(cate_r2 + cate_r2_ate_penalty)

    return EpisodeMetrics(
        method=method,
        episode=episode,
        ate_true=ate_true,
        ate_pred=ate_pred,
        ate_error=ate_error,
        cate_mae=cate_mae,
        cate_rmse=cate_rmse,
        cate_medae=cate_medae,  # <-- NEW
        cate_rmedse=cate_rmedse,  # <-- NEW
        cate_r2=cate_r2,
        cate_r2_shape=cate_r2_shape,
        cate_r2_ate_penalty=cate_r2_ate_penalty,
        homogeneous=homogeneous,
        ite_var=ite_var,
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
    """Compute per-episode metrics for all episodes of one method.

    Args:
        preds      : shape (n_batch, n_test).
        true_ites  : shape (n_batch, n_test).
        times      : shape (n_batch,). Execution time per episode.
        method     : display name.

    Returns:
        DataFrame with one row per episode, columns = EpisodeMetrics fields.
    """
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
    e = df["ate_error"]

    # Cross-episode ATE R²: does τ̂_mean correlate with τ_mean across episodes?
    try:
        ate_r2 = round(float(r2_score(df["ate_true"], df["ate_pred"])), 4)
    except Exception:
        ate_r2 = float("nan")

    return pd.Series(
        {
            # ── Raw ──────────────────────────────────────────────────────────
            "Bias": round(e.mean(), 4),
            "Std": round(e.std(), 4),
            "MAE": round(e.abs().mean(), 4),
            "MedAE": round(e.abs().median(), 4),
            "RMSE": round(np.sqrt((e**2).mean()), 4),
            "RMedSE": round(np.sqrt(np.median(e**2)), 4),
            # ── Cross-episode R² ─────────────────────────────────────────────
            "ATE R²": ate_r2,
            # ── Time ─────────────────────────────────────────────────────────
            "Time (s)": round(df["time"].mean(), 4),
        }
    )


def _summarise_cate(df: pd.DataFrame) -> pd.Series:
    heterogeneous = df.loc[~df["homogeneous"]]
    n_total = len(df)
    n_valid = len(heterogeneous)
    frac_used = n_valid / n_total if n_total > 0 else 0.0

    # ── Absolute CATE metrics (all non-all-NaN episodes) ─────────────────────
    mae_vals = df["cate_mae"].dropna()
    rmse_vals = df["cate_rmse"].dropna()
    medae_vals = df["cate_medae"].dropna()
    rmedse_vals = df["cate_rmedse"].dropna()

    mean_mae = round(mae_vals.mean(), 4) if len(mae_vals) > 0 else float("nan")
    mean_rmse = round(rmse_vals.mean(), 4) if len(rmse_vals) > 0 else float("nan")
    mean_medae = round(medae_vals.mean(), 4) if len(medae_vals) > 0 else float("nan")
    mean_rmedse = round(rmedse_vals.mean(), 4) if len(rmedse_vals) > 0 else float("nan")

    if n_valid == 0:
        return pd.Series(
            {
                "MAE": mean_mae,
                "MedAE": mean_medae,
                "RMSE": mean_rmse,
                "RMedSE": mean_rmedse,
                "Mean R²": float("nan"),
                "Median R²": float("nan"),
                "Mean R²(shape)": float("nan"),
                "Mean ATE penalty": float("nan"),
                "Episodes used": 0,
                "Time (s)": round(df["time"].mean(), 4),
                "⚠ note": "All episodes homogeneous — R² not defined",
            }
        )

    r2_vals = heterogeneous["cate_r2"].dropna()
    r2s_vals = heterogeneous["cate_r2_shape"].dropna()
    r2pen_vals = heterogeneous["cate_r2_ate_penalty"].dropna()

    return pd.Series(
        {
            # ── Absolute ─────────────────────────────────────────────────────
            "MAE": mean_mae,
            "MedAE": mean_medae,
            "RMSE": mean_rmse,
            "RMedSE": mean_rmedse,
            # ── Standard R² ──────────────────────────────────────────────────
            "Mean R²": round(r2_vals.mean(), 4),
            "Median R²": round(r2_vals.median(), 4),
            "Std R²": round(r2_vals.std(), 4),
            # ── R² decomposition ─────────────────────────────────────────────
            "Mean R²(shape)": round(r2s_vals.mean(), 4),
            "Median R²(shape)": round(r2s_vals.median(), 4),
            "Mean ATE penalty": round(r2pen_vals.mean(), 4),
            # ── Coverage ─────────────────────────────────────────────────────
            "Episodes used": n_valid,
            # ── Time ─────────────────────────────────────────────────────────
            "Time (s)": round(df["time"].mean(), 4),
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


def _get_column_styles(series: pd.Series, col_name: str) -> dict:
    """Returns a dict mapping dataframe index to a rich color string.
    Dynamically infers whether higher/lower/closer to 0 is better based on the col name.
    """
    styles = {}

    if col_name in ["⚠ note", "Episodes used"]:
        return {k: "" for k in series.index}

    # Convert to numeric to safely compute quantiles
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if len(valid) == 0:
        return {k: "" for k in series.index}

    # 1. R² Metrics: Higher is better (Fixed absolute thresholds)
    if "R²" in col_name and "Std" not in col_name:
        for k, v in series.items():
            if pd.isna(v) or isinstance(v, str):
                styles[k] = ""
            elif v > 0.5:
                styles[k] = "green"
            elif v > 0.0:
                styles[k] = "yellow"
            else:
                styles[k] = "red"
        return styles

    if "Time" in col_name:
        for k, v in series.items():
            styles[k] = ""
        return styles

    # 2. Penalty Metric: Lower is better (Fixed absolute thresholds)
    if "penalty" in col_name.lower():
        for k, v in series.items():
            if pd.isna(v) or isinstance(v, str):
                styles[k] = ""
            elif v < 0.1:
                styles[k] = "green"
            elif v < 0.5:
                styles[k] = "yellow"
            else:
                styles[k] = "red"
        return styles

    # 3. Bias Metrics: Closer to 0 is better (Relative quantiles)
    if "Bias" in col_name:
        abs_valid = valid.abs()
        q_green = abs_valid.quantile(0.33)
        q_yellow = abs_valid.quantile(0.66)
        for k, v in series.items():
            if pd.isna(v) or isinstance(v, str):
                styles[k] = ""
            else:
                v_abs = abs(v)
                if v_abs <= q_green:
                    styles[k] = "green"
                elif v_abs <= q_yellow:
                    styles[k] = "yellow"
                else:
                    styles[k] = "red"
        return styles

    # 4. Error Metrics (MAE, RMSE, Std, etc.): Lower is better (Relative quantiles)
    q_green = valid.quantile(0.33)
    q_yellow = valid.quantile(0.66)
    for k, v in series.items():
        if pd.isna(v) or isinstance(v, str):
            styles[k] = ""
        elif v <= q_green:
            styles[k] = "green"
        elif v <= q_yellow:
            styles[k] = "yellow"
        else:
            styles[k] = "red"

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
        col: _get_column_styles(ate_summary[col], col) for col in ate_summary.columns
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
        col: _get_column_styles(cate_summary[col], col) for col in cate_summary.columns
    }

    for method, row in cate_summary.iterrows():
        cells = []
        for col_name, val in row.items():
            style = col_styles[col_name].get(method, "")
            cells.append(f"[{style}]{val}[/{style}]" if style else str(val))
        tbl.add_row(str(method), *cells)

    return tbl


def make_combined_rich_table(
    ate_summary: pd.DataFrame,
    cate_summary: pd.DataFrame,
    title: str = "Benchmark Results",
) -> "Table":
    from rich.table import Table
    from rich import box

    tbl = Table(title=title, box=box.SIMPLE_HEAD, show_footer=False)
    tbl.add_column("Method", style="bold cyan", no_wrap=True)

    # ── ATE Columns ──
    tbl.add_column("ATE R²", justify="right")
    tbl.add_column("ATE RMSE", justify="right")

    # ── CATE Columns ──
    tbl.add_column("CATE R²", justify="right")
    tbl.add_column("CATE R²(shape)", justify="right")
    tbl.add_column("CATE RMSE", justify="right")

    tbl.add_column("Time (s)", justify="right")

    # Pre-calculate column styles using safely accessed Series
    ate_styles = {
        "ATE R²": _get_column_styles(
            ate_summary.get("ATE R²", pd.Series(dtype=float)), "ATE R²"
        ),
        "RMSE": _get_column_styles(
            ate_summary.get("RMSE", pd.Series(dtype=float)), "RMSE"
        ),
        "Time (s)": _get_column_styles(
            ate_summary.get("Time (s)", pd.Series(dtype=float)), "Time (s)"
        ),
    }

    cate_styles = {
        "Mean R²": _get_column_styles(
            cate_summary.get("Mean R²", pd.Series(dtype=float)), "Mean R²"
        ),
        "Mean R²(shape)": _get_column_styles(
            cate_summary.get("Mean R²(shape)", pd.Series(dtype=float)), "Mean R²(shape)"
        ),
        "RMSE": _get_column_styles(
            cate_summary.get("RMSE", pd.Series(dtype=float)), "RMSE"
        ),
    }

    for method in ate_summary.index:
        ate_row = ate_summary.loc[method]
        cate_row = (
            cate_summary.loc[method]
            if method in cate_summary.index
            else pd.Series(dtype=float)
        )

        def get_styled(row, col_key, styles, format_str="{:.4f}"):
            val = row.get(col_key, float("nan"))
            if pd.isna(val):
                return "-"
            style = styles.get(col_key, {}).get(method, "")
            formatted_val = format_str.format(val)
            return f"[{style}]{formatted_val}[/{style}]" if style else formatted_val

        # Align exactly with the 6 metric columns added above
        cells = [
            get_styled(ate_row, "ATE R²", ate_styles),
            get_styled(ate_row, "RMSE", ate_styles),
            get_styled(cate_row, "Mean R²", cate_styles),
            get_styled(cate_row, "Mean R²(shape)", cate_styles),
            get_styled(cate_row, "RMSE", cate_styles),
            get_styled(ate_row, "Time (s)", ate_styles, "{:.2f}"),
        ]

        tbl.add_row(str(method), *cells)

    return tbl
