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
    cate_rmse: Optional[float]
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
    if np.all(np.isnan(pred)):
        ate_pred = np.nan
    else:
        ate_pred = float(np.nanmean(pred))
    ate_true = float(true_ite.mean())
    ite_var = float(true_ite.var())

    homogeneous = ite_var < ite_var_threshold
    all_nan = bool(np.all(np.isnan(pred)))

    ate_error = ate_pred - ate_true

    if all_nan:
        cate_rmse = cate_rmedse = None
    else:
        pred_clean = np.where(np.isnan(pred), ate_pred, pred)
        residuals = pred_clean - true_ite

        cate_rmse = float(np.sqrt((residuals**2).mean()))
        cate_rmedse = float(np.sqrt(np.median(residuals**2)))

    if homogeneous or all_nan:
        cate_r2 = cate_r2_shape = cate_r2_ate_penalty = None
    else:
        cate_r2 = float(r2_score(true_ite, pred_clean))
        bias_sq = ate_error**2
        cate_r2_ate_penalty = float(bias_sq / (ite_var + _EPS))
        cate_r2_shape = float(cate_r2 + cate_r2_ate_penalty)

    return EpisodeMetrics(
        method=method,
        episode=episode,
        ate_true=ate_true,
        ate_pred=ate_pred,
        ate_error=ate_error,
        cate_rmse=cate_rmse,
        cate_rmedse=cate_rmedse,
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
    e = df["ate_error"]

    try:
        ate_r2 = round(float(r2_score(df["ate_true"], df["ate_pred"])), 4)
    except Exception:
        ate_r2 = float("nan")

    return pd.Series(
        {
            "Bias": round(e.mean(), 4),
            "RMSE": round(np.sqrt((e**2).mean()), 4),
            "RMedSE": round(np.sqrt(np.median(e**2)), 4),
            "ATE R²": ate_r2,
            "Median Regret": round(df["ate_sq_regret"].median(), 4)
            if "ate_sq_regret" in df
            else float("nan"),
            "Time (s)": round(df["time"].mean(), 4),
        }
    )


def _summarise_cate(df: pd.DataFrame) -> pd.Series:
    heterogeneous = df.loc[~df["homogeneous"]]
    n_valid = len(heterogeneous)

    rmse_vals = df["cate_rmse"].dropna()
    rmedse_vals = df["cate_rmedse"].dropna()

    mean_rmse = round(rmse_vals.mean(), 4) if len(rmse_vals) > 0 else float("nan")
    mean_rmedse = round(rmedse_vals.mean(), 4) if len(rmedse_vals) > 0 else float("nan")

    med_rmse_regret = (
        round(df["cate_rmse_regret"].median(), 4)
        if "cate_rmse_regret" in df
        else float("nan")
    )
    med_rmedse_regret = (
        round(df["cate_rmedse_regret"].median(), 4)
        if "cate_rmedse_regret" in df
        else float("nan")
    )

    if n_valid == 0:
        return pd.Series(
            {
                "RMSE": mean_rmse,
                "RMedSE": mean_rmedse,
                "RMSE Regret": med_rmse_regret,
                "RMedSE Regret": med_rmedse_regret,
                "Median R²": float("nan"),
                "Median R²(shape)": float("nan"),
                "Mean ATE penalty": float("nan"),
            }
        )

    r2_vals = heterogeneous["cate_r2"].dropna()
    r2s_vals = heterogeneous["cate_r2_shape"].dropna()
    r2pen_vals = heterogeneous["cate_r2_ate_penalty"].dropna()

    return pd.Series(
        {
            "RMSE": mean_rmse,
            "RMedSE": mean_rmedse,
            "RMSE Regret": med_rmse_regret,
            "RMedSE Regret": med_rmedse_regret,
            "Median R²": round(r2_vals.median(), 4),
            "Median R²(shape)": round(r2s_vals.median(), 4),
            "Mean ATE penalty": round(r2pen_vals.mean(), 4),
        }
    )


def build_summary_tables(
    all_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 1. Compute base errors needed for regret ratios
    all_results["ate_sq_error"] = all_results["ate_error"] ** 2

    # 2. Find the minimum error per episode across all methods (The empirical "Oracle")
    min_ate_sq = all_results.groupby("episode")["ate_sq_error"].transform("min")
    min_cate_rmse = all_results.groupby("episode")["cate_rmse"].transform("min")
    min_cate_rmedse = all_results.groupby("episode")["cate_rmedse"].transform("min")

    # 3. Calculate the Regret Ratios (Method Error / Best Error)
    all_results["ate_sq_regret"] = all_results["ate_sq_error"] / (min_ate_sq + _EPS)
    all_results["cate_rmse_regret"] = all_results["cate_rmse"] / (min_cate_rmse + _EPS)
    all_results["cate_rmedse_regret"] = all_results["cate_rmedse"] / (
        min_cate_rmedse + _EPS
    )

    ate_summary = all_results.groupby("method", sort=False).apply(_summarise_ate)
    cate_summary = all_results.groupby("method", sort=False).apply(_summarise_cate)
    return ate_summary, cate_summary


# ─────────────────────────────────────────────────────────────────────────────
# Rich-formatted display helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_column_styles(series: pd.Series, col_name: str) -> dict:
    styles = {}

    valid = pd.to_numeric(series, errors="coerce").dropna()
    if len(valid) == 0:
        return {k: "" for k in series.index}

    # Regret Metrics: Closer to 1.0 is better
    if "Regret" in col_name:
        for k, v in series.items():
            if pd.isna(v) or isinstance(v, str):
                styles[k] = ""
            elif v <= 1.25:  # Within 25% of the oracle
                styles[k] = "green"
            elif v <= 2.5:
                styles[k] = "yellow"
            else:
                styles[k] = "red"
        return styles

    # R² Metrics: Higher is better
    if "R²" in col_name:
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
        return {k: "" for k in series.index}

    # Error Metrics (RMSE, Bias, Penalty): Lower is better (Relative quantiles)
    abs_valid = valid.abs() if "Bias" in col_name else valid
    q_green = abs_valid.quantile(0.33)
    q_yellow = abs_valid.quantile(0.66)

    for k, v in series.items():
        if pd.isna(v) or isinstance(v, str):
            styles[k] = ""
        else:
            val_to_check = abs(v) if "Bias" in col_name else v
            if val_to_check <= q_green:
                styles[k] = "green"
            elif val_to_check <= q_yellow:
                styles[k] = "yellow"
            else:
                styles[k] = "red"

    return styles


def make_combined_rich_table(
    all_results: pd.DataFrame,
    ate_summary: pd.DataFrame,
    cate_summary: pd.DataFrame,
    title: str = "Benchmark Results",
) -> "Table":
    from rich.table import Table
    from rich import box

    # 1. SORTING: Determine method ordering strictly by mean CATE RMSE (ascending)
    cate_rmse_means = (
        all_results.groupby("method")["cate_rmse"].mean().sort_values(ascending=True)
    )

    tbl = Table(title=title, box=box.SIMPLE_HEAD, show_footer=False)
    tbl.add_column("Method", style="bold cyan", no_wrap=True)

    # ── ATE Columns ── (ATE Rank column removed)
    tbl.add_column("ATE RMSE", justify="right")
    tbl.add_column("ATE Regret", justify="right")

    # ── CATE Columns ──
    tbl.add_column("CATE R²", justify="right")
    tbl.add_column("CATE R²(shape)", justify="right")
    tbl.add_column("CATE RMSE", justify="right")
    tbl.add_column("CATE Regret", justify="right")

    tbl.add_column("Time (s)", justify="right")

    # Get styles for metrics
    ate_styles = {
        "RMSE": _get_column_styles(
            ate_summary.get("RMSE", pd.Series(dtype=float)), "RMSE"
        ),
        "Median Regret": _get_column_styles(
            ate_summary.get("Median Regret", pd.Series(dtype=float)), "Median Regret"
        ),
        "Time (s)": _get_column_styles(
            ate_summary.get("Time (s)", pd.Series(dtype=float)), "Time (s)"
        ),
    }

    cate_styles = {
        "Median R²": _get_column_styles(
            cate_summary.get("Median R²", pd.Series(dtype=float)), "Median R²"
        ),
        "Median R²(shape)": _get_column_styles(
            cate_summary.get("Median R²(shape)", pd.Series(dtype=float)),
            "Median R²(shape)",
        ),
        "RMSE": _get_column_styles(
            cate_summary.get("RMSE", pd.Series(dtype=float)), "RMSE"
        ),
        "RMSE Regret": _get_column_styles(
            cate_summary.get("RMSE Regret", pd.Series(dtype=float)), "RMSE Regret"
        ),
    }

    # 2. Iterate over the sorted index (ordered by CATE RMSE ascending)
    for method in cate_rmse_means.index:
        ate_row = (
            ate_summary.loc[method]
            if method in ate_summary.index
            else pd.Series(dtype=float)
        )
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

        # Stacking cells dynamically matching our updated column structure
        cells = [
            get_styled(ate_row, "RMSE", ate_styles),
            get_styled(ate_row, "Median Regret", ate_styles, "{:.2f}x"),
            get_styled(cate_row, "Median R²", cate_styles),
            get_styled(cate_row, "Median R²(shape)", cate_styles),
            get_styled(cate_row, "RMSE", cate_styles),
            get_styled(cate_row, "RMSE Regret", cate_styles, "{:.2f}x"),
            get_styled(ate_row, "Time (s)", ate_styles, "{:.2f}"),
        ]

        tbl.add_row(str(method), *cells)

    return tbl
