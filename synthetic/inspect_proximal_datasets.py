"""
inspect_proximal_datasets.py

Inspection / verification tool for the realistic proximal data-generating
families in `realistic_proximal_mechanisms.py`.

For every mechanism (aggregated over its K randomized datasets) it verifies and
reports:

  1. TENSOR SHAPES & COMPLETENESS
       - dense/active dims of U, X, W, Z; seq_len; train/test split
       - per-dataset completeness  min(d_Z, d_W) >= d_U

  2. MARGINALS
       - per-variable mean / std / skew / excess-kurtosis / min / max
       - fraction of (near-)discrete columns  (realism signal)
       - marginal-density grid figure

  3. UNMEASURED CONFOUNDING  (the headline diagnostics; uses oracle U/Y0/Y1)
       - true ATE  vs  naive contrast
       - bias of an adjustment-set ladder:
             naive  ->  adjust(X)  ->  adjust(X,W)  ->  proximal(Z,W)  ->  adjust(X,U) oracle
         The gap |ATE_adj(X) - ATE_true| is the confounding LEFT after backdoor on
         the measured covariates; it is large iff U matters.
       - U -> A strength : AUC gain of the propensity model from adding U beyond X
       - U -> Y strength : partial R^2 of U in the outcome given (X, A) and (X, A, W)
       - model-free check: partial corr(A, Y | X) vs partial corr(A, Y | X, U)

  4. PROXY VALIDITY  (negative-control conditions of the proximal model)
       - NC.1   partial corr(Z, Y | A, U, X) ~ 0
       - NC.2   partial corr(W, A | U, X) ~ 0   and   partial corr(W, Z | U, X) ~ 0
       - U-relevance : partial corr(Z, U | X) and partial corr(W, U | X)  (want large)
       - completeness rank : smallest singular value of Cov(Z, W | X)

  5. OVERLAP / POSITIVITY
       - quantiles of the true propensity A_probs; fraction in [0.1, 0.9];
         overlap between the propensity densities of the treated and controls

Run:
    python inspect_proximal_datasets.py
Outputs a console report, a summary CSV, and several PNG figures.
"""
#%%
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from realistic_proximal_mechanisms import (
    RealisticGeneratorConfig,
    RealisticProximalGenerator,
    N_MECHANISMS,
)


# ----------------------------------------------------------------------------
# Small numerical helpers
# ----------------------------------------------------------------------------


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _ridge_fit(Xd: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Ridge regression with intercept. Returns coefficients [intercept, ...]."""
    n = Xd.shape[0]
    Xb = np.concatenate([np.ones((n, 1)), Xd], axis=1)
    p = Xb.shape[1]
    A = Xb.T @ Xb + lam * np.eye(p)
    A[0, 0] -= lam  # do not penalize intercept
    beta = np.linalg.solve(A, Xb.T @ y)
    return beta


def _ridge_predict(beta: np.ndarray, Xd: np.ndarray) -> np.ndarray:
    n = Xd.shape[0]
    Xb = np.concatenate([np.ones((n, 1)), Xd], axis=1)
    return Xb @ beta


def _residualize(target: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Residual of `target` after linear regression on `controls`."""
    if controls.shape[1] == 0:
        return target - target.mean(axis=0, keepdims=True)
    beta = _ridge_fit(controls, target)
    return target - _ridge_predict(beta, controls)


def _flex_residualize(
    target: np.ndarray, controls: np.ndarray, seed: int = 0
) -> np.ndarray:
    """Cross-fitted nonparametric residual of `target` given `controls`.

    Uses 2-fold cross-fitted gradient boosting so that the residual reflects the
    GENUINE conditional dependence structure (including interactions and
    nonlinear links), not leftover curvature that a linear fit would miss.
    This is what makes the NC / relevance / U->Y diagnostics faithful under the
    nonlinear DGPs.
    """
    target = target.reshape(target.shape[0], -1)
    if controls.shape[1] == 0:
        return target - target.mean(axis=0, keepdims=True)
    n = target.shape[0]
    rng = np.random.RandomState(seed)
    fold = rng.randint(0, 2, size=n)
    out = np.zeros_like(target)
    params = dict(max_iter=80, max_depth=3, learning_rate=0.12, random_state=seed)
    for j in range(target.shape[1]):
        for f in (0, 1):
            tr, te = fold != f, fold == f
            if tr.sum() < 16 or te.sum() < 4:
                out[te, j] = target[te, j] - target[tr, j].mean() if tr.sum() else 0.0
                continue
            m = HistGradientBoostingRegressor(**params).fit(controls[tr], target[tr, j])
            out[te, j] = target[te, j] - m.predict(controls[te])
    return out


def partial_corr_flex(
    a: np.ndarray, b: np.ndarray, controls: np.ndarray, seed: int = 0
) -> float:
    """Average |partial correlation| between a and b given controls, cross-fitted."""
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    ra = _flex_residualize(a, controls, seed)
    rb = _flex_residualize(b, controls, seed + 1)
    vals = []
    for i in range(ra.shape[1]):
        for j in range(rb.shape[1]):
            sa, sb = ra[:, i], rb[:, j]
            ds = sa.std() * sb.std()
            if ds < 1e-10:
                continue
            vals.append(abs(float(np.mean((sa - sa.mean()) * (sb - sb.mean())) / ds)))
    return float(np.mean(vals)) if vals else 0.0


def partial_r2_flex(
    target: np.ndarray, extra: np.ndarray, base: np.ndarray, seed: int = 0
) -> float:
    """Cross-fitted partial R^2 of `extra` for `target` over `base` controls.

    Captures interaction / nonlinear confounding that a linear partial R^2 misses
    (e.g. confounding that lives only in an X*U interaction).
    """
    y = target.reshape(-1, 1)
    res_base = _flex_residualize(y, base, seed)
    rss_base = float((res_base**2).sum())
    full = np.concatenate([base, extra], axis=1) if base.shape[1] else extra
    res_full = _flex_residualize(y, full, seed)
    rss_full = float((res_full**2).sum())
    if rss_base < 1e-12:
        return 0.0
    return float(np.clip((rss_base - rss_full) / rss_base, 0.0, 1.0))


def auc_lift_flex(
    a: np.ndarray, base: np.ndarray, extra: np.ndarray, seed: int = 0
) -> float:
    """Cross-fitted AUC lift for predicting A from `extra` beyond `base` (flexible)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    a = a.astype(int).ravel()
    if a.min() == a.max():
        return np.nan
    n = len(a)
    rng = np.random.RandomState(seed)
    fold = rng.randint(0, 2, size=n)
    params = dict(max_iter=80, max_depth=3, learning_rate=0.12, random_state=seed)

    def oof_auc(feat):
        pred = np.zeros(n)
        for f in (0, 1):
            tr, te = fold != f, fold == f
            if len(np.unique(a[tr])) < 2:
                return np.nan
            m = HistGradientBoostingClassifier(**params).fit(feat[tr], a[tr])
            pred[te] = m.predict_proba(feat[te])[:, 1]
        try:
            return roc_auc_score(a, pred)
        except Exception:
            return np.nan

    full = np.concatenate([base, extra], axis=1) if base.shape[1] else extra
    return float(oof_auc(full) - oof_auc(base))


def partial_corr(a: np.ndarray, b: np.ndarray, controls: np.ndarray) -> float:
    """Average absolute partial correlation between columns of a and b given controls."""
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    ra = _residualize(a, controls)
    rb = _residualize(b, controls)
    vals = []
    for i in range(ra.shape[1]):
        for j in range(rb.shape[1]):
            sa, sb = ra[:, i], rb[:, j]
            ds = sa.std() * sb.std()
            if ds < 1e-10:
                continue
            vals.append(abs(float(np.mean((sa - sa.mean()) * (sb - sb.mean())) / ds)))
    return float(np.mean(vals)) if vals else 0.0


def partial_r2(target: np.ndarray, extra: np.ndarray, base: np.ndarray) -> float:
    """Partial R^2 of `extra` for `target`, over the base controls.

    = (RSS_base - RSS_full) / RSS_base, clipped to [0, 1].
    """
    y = target.reshape(-1, 1)
    res_base = _residualize(y, base)
    rss_base = float((res_base**2).sum())
    full = np.concatenate([base, extra], axis=1) if base.shape[1] else extra
    res_full = _residualize(y, full)
    rss_full = float((res_full**2).sum())
    if rss_base < 1e-12:
        return 0.0
    return float(np.clip((rss_base - rss_full) / rss_base, 0.0, 1.0))


def _hgb_ate(S: np.ndarray, a: np.ndarray, y: np.ndarray) -> float:
    """T-learner ATE: separate fast gradient-boosted regressors per arm, g-computed."""
    a = a.astype(int).ravel()
    i0, i1 = a == 0, a == 1
    if i0.sum() < 8 or i1.sum() < 8:
        return np.nan
    common = dict(max_iter=120, max_depth=3, learning_rate=0.1, random_state=0)
    m0 = HistGradientBoostingRegressor(**common).fit(S[i0], y[i0])
    m1 = HistGradientBoostingRegressor(**common).fit(S[i1], y[i1])
    return float((m1.predict(S) - m0.predict(S)).mean())


def proximal_linear_ate(
    X: np.ndarray, Z: np.ndarray, W: np.ndarray, a: np.ndarray, y: np.ndarray
) -> float:
    """Linear two-stage proximal estimator (outcome-bridge / P2SLS).

    Stage 1: W_hat = E[W | A, Z, X]   (one ridge per W column)
    Stage 2: regress Y on [A, X, W_hat]; coefficient on A is the proximal ATE.
    This is the linear special case of the proximal g-formula in the paper.
    """
    a = a.reshape(-1, 1).astype(float)
    stage1_design = np.concatenate([a, Z, X], axis=1)
    W_hat = np.zeros_like(W)
    for j in range(W.shape[1]):
        beta = _ridge_fit(stage1_design, W[:, j : j + 1])
        W_hat[:, j : j + 1] = _ridge_predict(beta, stage1_design)
    stage2_design = np.concatenate([a, X, W_hat], axis=1)
    beta2 = _ridge_fit(stage2_design, y.reshape(-1, 1))
    return float(beta2[1, 0])  # [intercept, A, X..., W_hat...]


def overlap_coefficient(p0: np.ndarray, p1: np.ndarray, bins: int = 40) -> float:
    """Histogram-intersection overlap of two propensity distributions in [0, 1]."""
    edges = np.linspace(0, 1, bins + 1)
    h0, _ = np.histogram(p0, bins=edges, density=True)
    h1, _ = np.histogram(p1, bins=edges, density=True)
    w = edges[1] - edges[0]
    return float(np.sum(np.minimum(h0, h1)) * w)


def frac_discrete(col: np.ndarray, max_unique_ratio: float = 0.05) -> bool:
    u = np.unique(np.round(col, 6))
    return len(u) <= max(2, int(max_unique_ratio * len(col)))


# ----------------------------------------------------------------------------
# Per-mechanism inspection
# ----------------------------------------------------------------------------


def inspect_mechanism(gen: RealisticProximalGenerator, mech_id: int) -> Dict:
    """Run all diagnostics for one mechanism, aggregated over its K datasets."""
    batch = gen.get_batch(mech_id)
    (
        U,
        X,
        W,
        Z,
        A,
        A_probs,
        Y_obs,
        Y0,
        Y1,
        d_U,
        d_X,
        d_W,
        d_Z,
        seq_lens,
        train_sizes,
    ) = batch

    K = X.shape[0]
    T = X.shape[1]

    per_ds = []  # list of dicts, one per dataset
    completeness_ok = []
    marg_records: List[Dict] = []

    for k in range(K):
        du, dx, dw, dz = int(d_U[k]), int(d_X[k]), int(d_W[k]), int(d_Z[k])
        completeness_ok.append(min(dz, dw) >= du)

        Uk = _to_np(U[k, :, :du])
        Xk = _to_np(X[k, :, :dx])
        Wk = _to_np(W[k, :, :dw])
        Zk = _to_np(Z[k, :, :dz])
        ak = _to_np(A[k, :, 0])
        apk = _to_np(A_probs[k, :, 0])
        yk = _to_np(Y_obs[k, :, 0])
        y0k = _to_np(Y0[k, :, 0])
        y1k = _to_np(Y1[k, :, 0])

        # --- marginals (per active column) ---
        for vname, arr in [("U", Uk), ("X", Xk), ("W", Wk), ("Z", Zk)]:
            for c in range(arr.shape[1]):
                col = arr[:, c]
                marg_records.append(
                    dict(
                        mech=mech_id,
                        var=vname,
                        dataset=k,
                        mean=float(col.mean()),
                        std=float(col.std()),
                        skew=float(sps.skew(col)),
                        kurt=float(sps.kurtosis(col)),
                        cmin=float(col.min()),
                        cmax=float(col.max()),
                        discrete=frac_discrete(col),
                    )
                )

        # --- causal ground truth ---
        true_ate = float((y1k - y0k).mean())
        naive = float(yk[ak == 1].mean() - yk[ak == 0].mean())

        # adjustment-set ladder (T-learner g-computation)
        ate_X = _hgb_ate(Xk, ak, yk)
        ate_XW = _hgb_ate(np.concatenate([Xk, Wk], axis=1), ak, yk)
        ate_XU = _hgb_ate(np.concatenate([Xk, Uk], axis=1), ak, yk)  # oracle backdoor
        ate_prox = proximal_linear_ate(Xk, Zk, Wk, ak, yk)

        # --- confounding strength (cross-fitted, nonparametric) ---
        # U -> A : AUC lift of propensity from adding U beyond X
        u_to_a = auc_lift_flex(ak, Xk, Uk, seed=mech_id * 100 + k)

        # U -> Y : flexible partial R^2 of U given (X, A), and residual given
        # (X, A, W). Cross-fitted GBMs catch interaction/nonlinear confounding
        # (e.g. M11's X*U term) that a linear partial R^2 cannot see.
        XA = np.concatenate([Xk, ak.reshape(-1, 1)], axis=1)
        u_to_y_XA = partial_r2_flex(yk, Uk, XA, seed=mech_id * 100 + k)
        XAW = np.concatenate([Xk, ak.reshape(-1, 1), Wk], axis=1)
        u_to_y_after_W = partial_r2_flex(yk, Uk, XAW, seed=mech_id * 100 + k)

        # model-free confounding signal
        pc_AY_X = partial_corr_flex(
            ak.reshape(-1, 1), yk.reshape(-1, 1), Xk, seed=mech_id * 100 + k
        )
        pc_AY_XU = partial_corr_flex(
            ak.reshape(-1, 1),
            yk.reshape(-1, 1),
            np.concatenate([Xk, Uk], 1),
            seed=mech_id * 100 + k,
        )

        # --- proxy validity (NC.1, NC.2, relevance), all cross-fitted ---
        # Flexible conditioning is essential here: linear residualisation leaves
        # nonlinear U,X curvature in the residuals and makes NC.1 look violated
        # on exp-link / nonlinear-outcome mechanisms when it is not.
        AUX = np.concatenate([ak.reshape(-1, 1), Uk, Xk], axis=1)
        UX = np.concatenate([Uk, Xk], axis=1)
        nc1 = partial_corr_flex(
            Zk, yk.reshape(-1, 1), AUX, seed=mech_id * 100 + k
        )  # ~0 wanted
        nc2_WA = partial_corr_flex(
            Wk, ak.reshape(-1, 1), UX, seed=mech_id * 100 + k
        )  # ~0 wanted
        nc2_WZ = partial_corr_flex(Wk, Zk, UX, seed=mech_id * 100 + k)  # ~0 wanted
        relevance_Z = partial_corr_flex(
            Zk, Uk, Xk, seed=mech_id * 100 + k
        )  # large wanted
        relevance_W = partial_corr_flex(
            Wk, Uk, Xk, seed=mech_id * 100 + k
        )  # large wanted

        # completeness rank: smallest singular value of Cov(Z, W | X)
        Zr, Wr = _residualize(Zk, Xk), _residualize(Wk, Xk)
        cov_zw = (Zr.T @ Wr) / len(Zr)
        svals = np.linalg.svd(cov_zw, compute_uv=False)
        min_sv = float(svals[min(du, len(svals)) - 1]) if len(svals) else 0.0

        # --- overlap ---
        frac_overlap = float(np.mean((apk > 0.1) & (apk < 0.9)))
        ov_coef = overlap_coefficient(apk[ak == 0], apk[ak == 1])

        per_ds.append(
            dict(
                du=du,
                dx=dx,
                dw=dw,
                dz=dz,
                true_ate=true_ate,
                naive=naive,
                ate_X=ate_X,
                ate_XW=ate_XW,
                ate_XU=ate_XU,
                ate_prox=ate_prox,
                bias_naive=naive - true_ate,
                bias_X=ate_X - true_ate,
                bias_XW=ate_XW - true_ate,
                bias_XU=ate_XU - true_ate,
                bias_prox=ate_prox - true_ate,
                u_to_a=u_to_a,
                u_to_y_XA=u_to_y_XA,
                u_to_y_after_W=u_to_y_after_W,
                pc_AY_X=pc_AY_X,
                pc_AY_XU=pc_AY_XU,
                nc1=nc1,
                nc2_WA=nc2_WA,
                nc2_WZ=nc2_WZ,
                relevance_Z=relevance_Z,
                relevance_W=relevance_W,
                min_sv=min_sv,
                frac_overlap=frac_overlap,
                overlap_coef=ov_coef,
                a_rate=float(ak.mean()),
            )
        )

    df = pd.DataFrame(per_ds)

    def agg(col):
        return float(np.nanmean(df[col].values))

    summary = dict(
        mech=mech_id,
        name=RealisticProximalGenerator.mechanism_name(mech_id),
        K=K,
        T=T,
        completeness=float(np.mean(completeness_ok)),
        du=f"{df.du.min()}-{df.du.max()}",
        dz=f"{df.dz.min()}-{df.dz.max()}",
        dw=f"{df.dw.min()}-{df.dw.max()}",
        true_ate=agg("true_ate"),
        # confounding (mean absolute bias across datasets)
        absbias_naive=float(np.nanmean(np.abs(df.bias_naive))),
        absbias_X=float(np.nanmean(np.abs(df.bias_X))),
        absbias_XW=float(np.nanmean(np.abs(df.bias_XW))),
        absbias_prox=float(np.nanmean(np.abs(df.bias_prox))),
        absbias_XU=float(np.nanmean(np.abs(df.bias_XU))),
        u_to_a=agg("u_to_a"),
        u_to_y_XA=agg("u_to_y_XA"),
        u_to_y_after_W=agg("u_to_y_after_W"),
        pc_AY_X=agg("pc_AY_X"),
        pc_AY_XU=agg("pc_AY_XU"),
        # proxy validity
        nc1=agg("nc1"),
        nc2_WA=agg("nc2_WA"),
        nc2_WZ=agg("nc2_WZ"),
        relevance_Z=agg("relevance_Z"),
        relevance_W=agg("relevance_W"),
        min_sv=agg("min_sv"),
        # overlap
        frac_overlap=agg("frac_overlap"),
        overlap_coef=agg("overlap_coef"),
        a_rate=agg("a_rate"),
    )
    return dict(
        summary=summary, per_ds=df, marginals=pd.DataFrame(marg_records), batch=batch
    )


# ----------------------------------------------------------------------------
# Console reporting
# ----------------------------------------------------------------------------


def print_report(s: Dict) -> None:
    name = s["name"]
    story = RealisticProximalGenerator.mechanism_story(s["mech"])
    print("\n" + "=" * 92)
    print(f"  M{s['mech']:02d}  {name}    (K={s['K']} datasets, T={s['T']} rows each)")
    print("  " + story)
    print("-" * 92)
    print(
        f"  SHAPES/COMPLETENESS   d_U={s['du']}  d_Z={s['dz']}  d_W={s['dw']}   "
        f"min(d_Z,d_W)>=d_U in {s['completeness']*100:.0f}% of datasets"
    )
    print(f"  TRUE ATE              {s['true_ate']:+.3f}")
    print("  UNMEASURED CONFOUNDING  (mean |bias| of ATE across datasets)")
    print(f"      naive .............. {s['absbias_naive']:.3f}")
    print(
        f"      adjust(X) .......... {s['absbias_X']:.3f}   <- confounding LEFT after measured backdoor"
    )
    print(f"      adjust(X,W) ........ {s['absbias_XW']:.3f}   (W is a proxy, not U)")
    print(
        f"      proximal(Z,W) ...... {s['absbias_prox']:.3f}   <- linear proxy bridge"
    )
    print(f"      adjust(X,U) oracle . {s['absbias_XU']:.3f}   (should be ~0)")
    print("  CONFOUNDING STRENGTH  (cross-fitted, nonparametric)")
    print(
        f"      U->A (AUC lift) {s['u_to_a']:+.3f}    U->Y partialR2|X,A {s['u_to_y_XA']:.3f}"
        f"  -> still {s['u_to_y_after_W']:.3f} after W"
    )
    print(
        f"      corr(A,Y|X) {s['pc_AY_X']:.3f}  ->  corr(A,Y|X,U) {s['pc_AY_XU']:.3f}"
    )
    print("  PROXY VALIDITY  (cross-fitted partial corr)")
    print(
        f"      NC.1  corr(Z,Y|A,U,X)={s['nc1']:.3f}~0   "
        f"NC.2 corr(W,A|U,X)={s['nc2_WA']:.3f}~0  corr(W,Z|U,X)={s['nc2_WZ']:.3f}~0"
    )
    print(
        f"      relevance corr(Z,U|X)={s['relevance_Z']:.3f}  corr(W,U|X)={s['relevance_W']:.3f}"
        f"  (large)   completeness min-sv(Cov(Z,W|X))={s['min_sv']:.3f}"
    )
    print("  OVERLAP")
    print(
        f"      P(0.1<e<0.9)={s['frac_overlap']:.2f}   density overlap={s['overlap_coef']:.2f}"
        f"   treated rate={s['a_rate']:.2f}"
    )


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def fig_marginals(results: List[Dict], out: Path) -> None:
    """One row per mechanism; columns = U, X, Z, W, A_probs, Y_obs (dataset 0)."""
    vars_ = ["U", "X", "Z", "W", "A_probs", "Y_obs"]
    M = len(results)
    fig, axes = plt.subplots(M, 6, figsize=(20, 2.4 * M))
    for r, res in enumerate(results):
        (U, X, W, Z, A, A_probs, Y_obs, *_rest) = res["batch"]
        du, dx, dw, dz = res["per_ds"].iloc[0][["du", "dx", "dw", "dz"]].astype(int)
        cols = {
            "U": _to_np(U[0, :, :du]),
            "X": _to_np(X[0, :, :dx]),
            "Z": _to_np(Z[0, :, :dz]),
            "W": _to_np(W[0, :, :dw]),
            "A_probs": _to_np(A_probs[0, :, :1]),
            "Y_obs": _to_np(Y_obs[0, :, :1]),
        }
        for c, v in enumerate(vars_):
            ax = axes[r, c] if M > 1 else axes[c]
            data = cols[v]
            if v == "A_probs":
                ax.hist(data.ravel(), bins=np.linspace(0, 1, 21), color="#444")
                ax.set_xlim(0, 1)
            else:
                for j in range(min(data.shape[1], 6)):
                    ax.hist(data[:, j], bins=40, alpha=0.45, histtype="stepfilled")
            if r == 0:
                ax.set_title(v, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"M{res['summary']['mech']:02d}", fontweight="bold")
            ax.set_yticks([])
    fig.suptitle("Marginal distributions per mechanism (dataset 0)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_confounding_ladder(summ: pd.DataFrame, out: Path) -> None:
    labels = [f"M{m}" for m in summ.mech]
    x = np.arange(len(labels))
    ladders = [
        ("naive", "absbias_naive", "#2c7bb6"),
        ("adjust(X)", "absbias_X", "#fdae61"),
        ("adjust(X,W)", "absbias_XW", "#d7a1c6"),
        ("proximal(Z,W)", "absbias_prox", "#7b3294"),
        ("oracle(X,U)", "absbias_XU", "#2ca02c"),
    ]
    w = 0.16
    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (lab, col, color) in enumerate(ladders):
        ax.bar(x + (i - 2) * w, summ[col].values, w, label=lab, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean |ATE bias|")
    ax.set_title(
        "Confounding ladder: bias remaining under each adjustment set "
        "(measured backdoor fails, proxies/oracle recover)",
        fontweight="bold",
    )
    ax.legend(ncol=5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_confounding_strength(summ: pd.DataFrame, out: Path) -> None:
    labels = [f"M{m}" for m in summ.mech]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].bar(x, summ.u_to_a.values, color="#2c7bb6")
    axes[0].set_title(
        "U -> A : propensity AUC lift from U (beyond X)", fontweight="bold"
    )
    axes[1].bar(
        x - 0.2, summ.u_to_y_XA.values, 0.4, label="given (X,A)", color="#d7191c"
    )
    axes[1].bar(
        x + 0.2, summ.u_to_y_after_W.values, 0.4, label="given (X,A,W)", color="#fdae61"
    )
    axes[1].set_title(
        "U -> Y : partial R^2 of U (and residual after W)", fontweight="bold"
    )
    axes[1].legend()
    axes[2].bar(
        x - 0.2, summ.relevance_Z.values, 0.4, label="corr(Z,U|X)", color="#1b9e77"
    )
    axes[2].bar(
        x + 0.2, summ.relevance_W.values, 0.4, label="corr(W,U|X)", color="#7570b3"
    )
    axes[2].set_title(
        "Proxy U-relevance (needed for identification)", fontweight="bold"
    )
    axes[2].legend()
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_proxy_validity(summ: pd.DataFrame, out: Path) -> None:
    labels = [f"M{m}" for m in summ.mech]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (col, lab, color) in enumerate(
        [
            ("nc1", "NC.1 corr(Z,Y|A,U,X)", "#2c7bb6"),
            ("nc2_WA", "NC.2 corr(W,A|U,X)", "#fdae61"),
            ("nc2_WZ", "NC.2 corr(W,Z|U,X)", "#d7191c"),
        ]
    ):
        ax.bar(x + (i - 1) * 0.25, summ[col].values, 0.25, label=lab, color=color)
    ax.axhline(0.1, color="k", ls="--", alpha=0.6, label="~0 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(
        "Negative-control conditions (all should sit near 0)", fontweight="bold"
    )
    ax.set_ylabel("avg |partial correlation|")
    ax.legend(ncol=4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_overlap(results: List[Dict], out: Path) -> None:
    M = len(results)
    ncol = 4
    nrow = int(np.ceil(M / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3 * nrow))
    axes = np.array(axes).ravel()
    for i, res in enumerate(results):
        (_, _, _, _, A, A_probs, *_rest) = res["batch"]
        ap = _to_np(A_probs[0, :, 0])
        a = _to_np(A[0, :, 0])
        ax = axes[i]
        ax.hist(
            ap[a == 0],
            bins=np.linspace(0, 1, 30),
            alpha=0.5,
            label="A=0",
            color="#2c7bb6",
        )
        ax.hist(
            ap[a == 1],
            bins=np.linspace(0, 1, 30),
            alpha=0.5,
            label="A=1",
            color="#d7191c",
        )
        ax.set_title(
            f"M{res['summary']['mech']:02d} {res['summary']['name'][:20]}", fontsize=9
        )
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7)
    for j in range(M, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Propensity overlap by treatment group (dataset 0)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--K", type=int, default=5, help="datasets per mechanism")
    # ap.add_argument("--seq_len", type=int, default=800)
    # ap.add_argument("--seed", type=int, default=123)
    # ap.add_argument("--outdir", type=str, default="inspection_out")
    # args = ap.parse_args()

    args = argparse.Namespace(K=5, seq_len=800, seed=123, outdir="inspection_out")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = RealisticGeneratorConfig(
        save_dir=str(out / "gen"),
        K=args.K,
        seq_len=args.seq_len,
        train_size=0.5,
        min_dim_U=1,
        max_dim_U=3,
        min_dim_X=2,
        max_dim_X=8,
        min_dim_W=1,
        max_dim_W=8,
        min_dim_Z=1,
        max_dim_Z=8,
        seed=args.seed,
    )
    gen = RealisticProximalGenerator(cfg)

    print(
        f"Inspecting {N_MECHANISMS} mechanisms x K={args.K} datasets "
        f"x T={args.seq_len} rows ..."
    )

    results, summaries = [], []
    for m in range(1, N_MECHANISMS + 1):
        res = inspect_mechanism(gen, m)
        print_report(res["summary"])
        results.append(res)
        summaries.append(res["summary"])

    summ = pd.DataFrame(summaries)
    csv_path = out / "summary.csv"
    summ.to_csv(csv_path, index=False)

    # figures
    fig_marginals(results, out / "marginals.png")
    fig_confounding_ladder(summ, out / "confounding_ladder.png")
    fig_confounding_strength(summ, out / "confounding_strength.png")
    fig_proxy_validity(summ, out / "proxy_validity.png")
    fig_overlap(results, out / "overlap.png")

    # compact cross-mechanism table
    print("\n" + "=" * 92)
    print("  CROSS-MECHANISM SUMMARY")
    print("=" * 92)
    show = summ[
        [
            "mech",
            "name",
            "true_ate",
            "absbias_naive",
            "absbias_X",
            "absbias_prox",
            "absbias_XU",
            "u_to_a",
            "u_to_y_XA",
            "nc1",
            "relevance_W",
            "frac_overlap",
        ]
    ].copy()
    show.columns = [
        "M",
        "name",
        "ATE",
        "|b|naive",
        "|b|X",
        "|b|prox",
        "|b|oracle",
        "U->A",
        "U->Y",
        "NC1",
        "relW",
        "overlap",
    ]
    with pd.option_context(
        "display.width",
        200,
        "display.max_columns",
        20,
        "display.float_format",
        lambda v: f"{v:.3f}",
    ):
        print(show.to_string(index=False))

    print(f"\nSaved: {csv_path}")
    for f in [
        "marginals.png",
        "confounding_ladder.png",
        "confounding_strength.png",
        "proxy_validity.png",
        "overlap.png",
    ]:
        print(f"Saved: {out / f}")


if __name__ == "__main__":
    main()

# %%
