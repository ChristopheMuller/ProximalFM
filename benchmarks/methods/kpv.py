"""
KPV — Kernel Proxy Variable (two-stage kernel regression)
==========================================================
Non-parametric proximal causal estimator that learns a bridge function in two
regularised kernel regression stages.

Reference: Mastouri et al. "Proximal Causal Learning with Kernels" (ICML 2021)

Pipeline:
  Stage 1 (data split D1):
      Γ = (K_ZAX + λ₁ n₁ I)⁻¹ K_ZAX(D1, D2)
      Bridges outcome proxy W to the (A, Z, X) space.

  Stage 2 (data split D2):
      Σ = (Γᵀ K_W Γ) ⊙ K_A(D2, D2)
      α = Γ ⊙ [(Σ + λ₂ n₂ I)⁻¹ Y₂]     (n₁ × n₂)

  Prediction of E[Y | do(a)]:
      ŷ(a) = w̄ᵀ α k_A(A₂, a)
      where w̄ = mean_row(K_W(W1, W1))

  CATE = ŷ(1) − ŷ(0)  (for binary treatment)

All computation is pure numpy/scipy — no JAX dependency.

Tuning entry point: KPVConfig — every knob lives there with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import solve

from benchmarks.methods.kernels import ColumnWiseGaussianKernel, as_2d, BinaryKernel


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class KPVConfig:
    # ── Data split ────────────────────────────────────────────────────────────
    # Fraction of training data used in Stage 1 (remainder goes to Stage 2).
    split_ratio: float = 0.5

    # ── Kernel ────────────────────────────────────────────────────────────────
    # Multiplicative rescaling applied to the median-heuristic bandwidth σ.
    # Values < 1 → narrower kernel (more local); > 1 → wider (smoother).
    # Used directly when tune_kernel_scale is False; otherwise it is the centre
    # (1.0 = plain median heuristic) that the search grid is spread around.
    kernel_scale: float = 1.0

    # ── Kernel length-scale search ────────────────────────────────────────────
    # When True, search a geometric grid of scale multipliers around the median
    # heuristic and keep the one minimising the Stage-2 LOOCV loss (λ₁, λ₂ are
    # retuned inside each candidate). The Stage-2 loss is used because its target
    # (Y) is fixed across scales, so scores are comparable — unlike the Stage-1
    # loss, whose K_WW target itself changes with the W bandwidth.
    tune_kernel_scale: bool = False
    kernel_scale_min: float = 0.5
    kernel_scale_max: float = 2.0
    n_kernel_scale_search: int = 5

    # ── Stage 1 regularisation ────────────────────────────────────────────────
    # Set lam1 to a fixed value to skip cross-validation.
    # Leave as None to tune automatically over the log-grid [lam1_min, lam1_max].
    lam1: Optional[float] = None
    lam1_min: float = 1e-5
    lam1_max: float = 1e1
    n_lam1_search: int = 10

    # ── Stage 2 regularisation ────────────────────────────────────────────────
    # Same convention as Stage 1.
    lam2: Optional[float] = None
    lam2_min: float = 1e-5
    lam2_max: float = 1e1
    n_lam2_search: int = 10

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 1234


# ─────────────────────────────────────────────────────────────────────────────
# LOOCV helpers
# ─────────────────────────────────────────────────────────────────────────────


def _loocv_stage1(K_ZAX: np.ndarray, K_W: np.ndarray, lam1: float) -> float:
    """
    Leave-one-out cross-validation score for Stage 1 regularisation.

    Measures how well the kernel embedding predicts the outcome proxy W
    via the influence matrix H = I − K_ZAX (K_ZAX + λ₁ n I)⁻¹.

    Via the standard ridge identity I − K (K + cI)⁻¹ = c (K + cI)⁻¹ with
    c = λ₁ n, this simplifies to H = λ₁ n (K_ZAX + λ₁ n I)⁻¹, avoiding the
    extra matrix multiply (and matching the form used in _loocv_stage2).

    Lower is better.
    """
    n = K_ZAX.shape[0]
    H = lam1 * n * solve(K_ZAX + lam1 * n * np.eye(n), np.eye(n), assume_a="pos")
    diag_H = np.diag(H)
    # Avoid division by zero in degenerate cases.
    diag_H = np.where(np.abs(diag_H) < 1e-12, 1e-12, diag_H)
    tH_inv = np.diag(1.0 / diag_H)
    return float(np.trace(tH_inv @ H @ K_W @ H @ tH_inv))


def _loocv_stage2(Sigma: np.ndarray, Y: np.ndarray, lam2: float) -> float:
    """
    Leave-one-out cross-validation score for Stage 2 regularisation.

    Equivalent to the original cal_loocv_alpha but uses the identity
    D_t @ M_Γ = Σ (derived in the paper), simplifying to:
        H = I − Σ (Σ + λ₂ n I)⁻¹ = λ₂ n (Σ + λ₂ n I)⁻¹

    Lower is better.
    """
    n = Sigma.shape[0]
    Sigma_reg = Sigma + lam2 * n * np.eye(n)
    H = lam2 * n * solve(Sigma_reg, np.eye(n), assume_a="pos")
    diag_H = np.diag(H)
    diag_H = np.where(np.abs(diag_H) < 1e-12, 1e-12, diag_H)
    tH_inv = np.diag(1.0 / diag_H)
    return float(np.linalg.norm(tH_inv @ H @ Y))


# ─────────────────────────────────────────────────────────────────────────────
# Estimator
# ─────────────────────────────────────────────────────────────────────────────


class KPVEstimator:
    """
    Sklearn-style CATE estimator implementing the Kernel Proxy Variable method.

    Usage::

        cfg = KPVConfig(split_ratio=0.5, lam1=1e-3, lam2=1e-3)
        est = KPVEstimator(cfg)
        est.fit(X_train, A_train, Y_train, W=W_train, Z=Z_train)
        cate = est.predict(X_test)   # shape (n_test,)

    The ``cfg`` dataclass is the single place to tune all hyperparameters.
    """

    def __init__(self, cfg: KPVConfig | None = None) -> None:
        self.cfg = KPVConfig() if cfg is None else cfg

        self._alpha: np.ndarray | None = None  # (n1, n2)

        self._A2: np.ndarray | None = None  # (n2, d_A)
        self._W1: np.ndarray | None = None  # (n1, d_W)
        self._X2: np.ndarray | None = None  # (n2, d_X), if X exists

        self._W_ref: np.ndarray | None = None  # usually full training W
        self._X_ref: np.ndarray | None = None  # usually full training X
        self._K_W1_ref: np.ndarray | None = None  # K_W(W1, W_ref)

        self._marginal_xw_mean: np.ndarray | None = None  # (n1, n2), or (n1, 1) if no X
        self._w_marginal_mean: np.ndarray | None = None  # (n1,)

        self._k_A: ColumnWiseGaussianKernel | None = None
        self._k_W: ColumnWiseGaussianKernel | None = None
        self._k_X: ColumnWiseGaussianKernel | None = None

        self._has_x: bool = False

        self.lam1_: float | None = None
        self.lam2_: float | None = None
        self.kernel_scale_: float | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        *,
        W: np.ndarray,
        Z: np.ndarray,
    ) -> "KPVEstimator":
        """
        Fit the two-stage KPV model.

        Args:
            X: Covariates,                shape (n, d_X).
            A: Treatment (binary/cont.),  shape (n,) or (n, d_A).
            Y: Observed outcome,          shape (n,).
            W: Outcome-inducing proxy,    shape (n, d_W).
            Z: Treatment-inducing proxy,  shape (n, d_Z).

        Returns:
            self (for method chaining).
        """
        cfg = self.cfg
        np.random.seed(cfg.seed)

        A = as_2d(A)
        Y = np.asarray(Y, dtype=float).reshape(-1)
        W = as_2d(W)
        Z = as_2d(Z)
        X = as_2d(X)

        # ── Split ──────────────────────────────────────────────────────────
        n = len(Y)
        n1 = int(n * cfg.split_ratio)
        idx = np.random.permutation(n)
        i1, i2 = idx[:n1], idx[n1:]

        A1, A2 = A[i1], A[i2]
        W1, W2 = W[i1], W[i2]
        Z1, Z2 = Z[i1], Z[i2]
        X1, X2 = X[i1], X[i2]
        Y2 = Y[i2]
        n1, n2 = len(i1), len(i2)
        has_x = X1.shape[1] > 0

        # ── Select the kernel length-scale ─────────────────────────────────
        # Either the single fixed scale, or a geometric grid around the median
        # heuristic (see KPVConfig). Each candidate refits both stages (with
        # their own λ tuning) and is ranked by the Stage-2 LOOCV loss.
        if cfg.tune_kernel_scale:
            scales = np.logspace(
                np.log10(cfg.kernel_scale_min),
                np.log10(cfg.kernel_scale_max),
                cfg.n_kernel_scale_search,
            )
        else:
            scales = [cfg.kernel_scale]

        best = None
        for scale in scales:
            res = self._fit_stages(
                float(scale), A1, A2, W1, Z1, Z2, X1, X2, Y2, n1, n2, has_x
            )
            if best is None or res["score"] < best["score"]:
                best = res

        self.kernel_scale_ = best["scale"]
        self.lam1_ = best["lam1"]
        self.lam2_ = best["lam2"]
        self._alpha = best["alpha"]  # (n1, n2)

        k_A, k_W, k_X = best["k_A"], best["k_W"], best["k_X"]

        # Store bridge basis points
        self._A2 = A2
        self._W1 = W1
        self._X2 = X2

        # Store full empirical reference distribution for prediction heads.
        # This is what lets the prediction head choose joint, marginal,
        # independent, or conditional averaging over (X, W).
        self._W_ref = W
        self._X_ref = X
        self._has_x = has_x

        # Store kernels
        self._k_A = k_A
        self._k_W = k_W
        self._k_X = k_X

        # Cache K_W(W1, W_ref): shape (n1, n)
        K_W1_ref = k_W(W1, W)
        self._K_W1_ref = K_W1_ref

        # Marginal W embedding over the full training distribution:
        # w_mean[i] = mean_l k_W(W1_i, W_l)
        self._w_marginal_mean = K_W1_ref.mean(axis=1)  # (n1,)

        if self._has_x:
            K_X2_ref = k_X(X2, X)  # (n2, n)
            self._marginal_xw_mean = K_W1_ref @ K_X2_ref.T / len(Y)  # (n1, n2)
        else:
            self._marginal_xw_mean = self._w_marginal_mean[:, np.newaxis]  # (n1, 1)

        return self

    # ── Two-stage fit for a single kernel scale ─────────────────────────────────

    def _fit_stages(
        self,
        scale: float,
        A1: np.ndarray,
        A2: np.ndarray,
        W1: np.ndarray,
        Z1: np.ndarray,
        Z2: np.ndarray,
        X1: np.ndarray,
        X2: np.ndarray,
        Y2: np.ndarray,
        n1: int,
        n2: int,
        has_x: bool,
    ) -> dict:
        """
        Run both KPV stages at a fixed kernel length-scale multiplier.

        Fits the kernel bandwidths on D1 (rescaled by ``scale``), tunes λ₁/λ₂
        by closed-form LOOCV, and returns the resulting bridge weights together
        with the Stage-2 LOOCV score used to compare scales. Called once per
        candidate scale by ``fit``.
        """
        cfg = self.cfg

        # ── Fit kernel bandwidths on D1 ────────────────────────────────────
        k_A = BinaryKernel()
        k_Z = ColumnWiseGaussianKernel()
        k_W = ColumnWiseGaussianKernel()
        k_X = ColumnWiseGaussianKernel()

        k_A.fit(A1, scale=scale)
        k_Z.fit(Z1, scale=scale)
        k_W.fit(W1, scale=scale)
        if has_x:
            k_X.fit(X1, scale=scale)

        # X-kernel factor: an empty (d_X = 0) covariate set contributes a
        # neutral factor of 1 (empty product), so the bridge collapses to the
        # (A, Z, W)-only model without ever touching the unfitted k_X.
        def kx(a: np.ndarray, b: np.ndarray) -> np.ndarray | float:
            return k_X(a, b) if has_x else 1.0

        # ── Stage 1 ────────────────────────────────────────────────────────
        # K_ZAX(D1, D1) — product kernel over (A, Z, X)
        K_ZAX_11 = k_A(A1, A1) * k_Z(Z1, Z1) * kx(X1, X1)
        K_W1W1 = k_W(W1, W1)

        lam1 = cfg.lam1
        if lam1 is None:
            lam1 = self._tune(
                lambda l: _loocv_stage1(K_ZAX_11, K_W1W1, l),
                cfg.lam1_min,
                cfg.lam1_max,
                cfg.n_lam1_search,
            )

        # Cross-split kernel: (n1, n2) — needed to form Γ
        K_ZAX_12 = k_A(A1, A2) * k_Z(Z1, Z2) * kx(X1, X2)

        # Γ = (K_ZAX_11 + λ₁ n₁ I)⁻¹ K_ZAX_12  → (n1, n2)
        Gamma = solve(
            K_ZAX_11 + lam1 * n1 * np.eye(n1),
            K_ZAX_12,
            assume_a="pos",
        )

        # ── Stage 2 ────────────────────────────────────────────────────────
        # gKg = Γᵀ K_W Γ  → (n2, n2)
        gKg = Gamma.T @ K_W1W1 @ Gamma

        K_A2 = k_A(A2, A2)
        Sigma = gKg * K_A2 * kx(X2, X2)

        lam2 = cfg.lam2
        if lam2 is None:
            lam2 = self._tune(
                lambda l: _loocv_stage2(Sigma, Y2, l),
                cfg.lam2_min,
                cfg.lam2_max,
                cfg.n_lam2_search,
            )

        # Stage-2 LOOCV loss at the chosen λ₂ — the criterion for ranking
        # scales. Comparable across scales because the target Y₂ is fixed.
        score = _loocv_stage2(Sigma, Y2, lam2)

        # v = (Σ + λ₂ n₂ I)⁻¹ Y₂  → (n2,)
        v = solve(Sigma + lam2 * n2 * np.eye(n2), Y2, assume_a="pos")

        # α[i, j] = Γ[i, j] * v[j]  → (n1, n2)
        alpha = Gamma * v[np.newaxis, :]

        return {
            "scale": scale,
            "lam1": lam1,
            "lam2": lam2,
            "alpha": alpha,
            "k_A": k_A,
            "k_W": k_W,
            "k_X": k_X,
            "score": score,
        }

    # ── Prediction heads ──────────────────────────────────────────────────────

    def predict_mean_outcome(
        self,
        a: float | np.ndarray,
        *,
        mode: str = "marginal",
        X: np.ndarray | None = None,
        W: np.ndarray | None = None,
        n: int | None = None,
    ) -> np.ndarray:
        """
        Predict E[Y | do(A=a)] or bridge-based conditional variants.

        Modes:
            "marginal":
                ATE / causal dose-response head.
                Averages h(a, X, W) over the empirical joint training
                distribution of (X, W).

            "paired":
                Bridge contrast/value head at supplied paired observations.
                Requires W, and requires X if the model was fit with X.
                Returns h(a_i, x_i, w_i).

            "independent":
                X-specific head assuming W ⟂ X.
                For each supplied x_i, averages h(a, x_i, W) over the
                marginal training distribution of W.

            "conditional":
                X-specific head using kernel smoothing to approximate
                E_W[h(a, x_i, W) | X=x_i].

        Args:
            a:
                Scalar treatment, treatment grid of shape (n,), or treatment
                matrix of shape (n, d_A).
            mode:
                One of {"marginal", "paired", "independent", "conditional"}.
            X:
                Test covariates, required for paired/independent/conditional
                when the model was fit with non-empty X.
            W:
                Test outcome proxies, required for paired mode.
            n:
                Optional number of outputs when `a` is scalar and the chosen
                mode is constant.

        Returns:
            Predictions, shape depends on mode:
                - marginal with scalar a: shape (1,) unless n or X/W is given
                - marginal with treatment grid: shape (n_grid,)
                - paired/independent/conditional: shape (n_test,)
        """
        self._check_fitted()

        mode = mode.lower()
        if mode in {"ate", "marginal"}:
            pred = self._predict_marginal_mean(a)
            n_out = self._infer_n(X=X, W=W, n=n)
            return self._repeat_if_scalar(pred, n_out)

        if mode == "paired":
            return self._predict_paired_mean(a, X=X, W=W)

        if mode == "independent":
            return self._predict_independent_mean(a, X=X, n=n)

        if mode == "conditional":
            return self._predict_conditional_mean(a, X=X, n=n)

        raise ValueError(
            "Unknown prediction mode. Expected one of "
            "{'marginal', 'paired', 'independent', 'conditional'}."
        )

    def predict(
        self,
        X: np.ndarray | None = None,
        *,
        W: np.ndarray | None = None,
        Z=None,
        mode: str = "marginal",
        a1: float | np.ndarray = 1.0,
        a0: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """
        Predict treatment contrast under the selected prediction head.

        For binary treatment this returns:

            head(a1) - head(a0)

        Modes:
            "marginal":
                ATE contrast. Returned as a constant vector if X/W is supplied.

            "paired":
                h(a1, x_i, w_i) - h(a0, x_i, w_i).
                Requires W, and requires X if the model has X.

            "independent":
                E_{W marginal}[h(a1, x_i, W)] -
                E_{W marginal}[h(a0, x_i, W)].

            "conditional":
                Kernel-smoothed approximation of
                E[h(a1, x_i, W) | X=x_i] -
                E[h(a0, x_i, W) | X=x_i].

        Z is accepted for benchmark/API compatibility but is not used at
        prediction time.
        """
        self._check_fitted()

        n = self._infer_n(X=X, W=W, n=None)
        if n is None:
            n = 1

        y1 = self.predict_mean_outcome(a1, mode=mode, X=X, W=W, n=n)
        y0 = self.predict_mean_outcome(a0, mode=mode, X=X, W=W, n=n)

        return np.asarray(y1 - y0, dtype=float).reshape(-1)

    def predict_ate(
        self,
        a1: float | np.ndarray = 1.0,
        a0: float | np.ndarray = 0.0,
    ) -> float | np.ndarray:
        """
        Convenience wrapper for the marginal ATE / dose-response contrast.
        """
        y1 = self.predict_mean_outcome(a1, mode="marginal")
        y0 = self.predict_mean_outcome(a0, mode="marginal")
        out = y1 - y0
        return float(out.item()) if out.size == 1 else out

    def predict_bridge(
        self,
        treatment: float | np.ndarray,
        *,
        X: np.ndarray | None = None,
        W: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the learned bridge h(a, x, w) at supplied points.

        This is equivalent to predict_mean_outcome(..., mode="paired"), but
        without taking a treatment contrast.
        """
        return self.predict_mean_outcome(treatment, mode="paired", X=X, W=W)

    # ── Individual prediction heads ───────────────────────────────────────────

    def _predict_marginal_mean(self, a: float | np.ndarray) -> np.ndarray:
        """
        Marginal ATE / dose-response head:

            mean_l h(a, X_l, W_l)

        using the empirical joint training distribution of (X, W).
        """
        A_test = self._as_treatment_matrix(a)

        # K_A(A2, a_test): (n2, n_test)
        K_A_test = self._k_A(self._A2, A_test)

        # M[i, j] = E_hat[k_W(W1_i, W) k_X(X2_j, X)]
        # If no X, M has shape (n1, 1) and broadcasts over alpha.
        M = self._marginal_xw_mean  # (n1, n2) or (n1, 1)

        # coef[j] = sum_i alpha[i, j] M[i, j]
        coef_over_A2 = (self._alpha * M).sum(axis=0)  # (n2,)

        return (coef_over_A2 @ K_A_test).ravel()

    def _predict_paired_mean(
        self,
        a: float | np.ndarray,
        *,
        X: np.ndarray | None,
        W: np.ndarray | None,
    ) -> np.ndarray:
        """
        Paired bridge head:

            h(a_i, x_i, w_i)

        If a is scalar, it is repeated for all supplied pairs.
        """
        if W is None:
            raise ValueError("mode='paired' requires W.")

        W = as_2d(W)
        n = W.shape[0]

        X_checked = self._validate_X_for_prediction(X, n=n, required=self._has_x)
        A_test = self._as_treatment_matrix(a, n=n)

        # K_W(W1, W_test): (n1, n)
        K_W_test = self._k_W(self._W1, W)

        # left[i, j] = sum_r k_W(W1_r, W_i) alpha[r, j]
        # shape: (n, n2)
        left = K_W_test.T @ self._alpha

        # K_A(A2, A_test): (n2, n)
        right = self._k_A(self._A2, A_test)

        if self._has_x:
            # K_X(X2, X_test): (n2, n)
            right = right * self._k_X(self._X2, X_checked)

        # pred_i = left_i dot right_i
        return np.einsum("ij,ji->i", left, right).ravel()

    def _predict_independent_mean(
        self,
        a: float | np.ndarray,
        *,
        X: np.ndarray | None,
        n: int | None = None,
    ) -> np.ndarray:
        """
        Independent head:

            E_{W marginal}[h(a, x_i, W)]

        This assumes W is independent of X for the prediction target.
        """
        if self._has_x:
            X_checked = self._validate_X_for_prediction(X, n=None, required=True)
            n_test = X_checked.shape[0]
        else:
            X_checked = None
            n_test = n

        A_test = self._as_treatment_matrix(a, n=n_test)
        n_test = A_test.shape[0]

        # E_W[k_W(W1_i, W)]
        w_mean = self._w_marginal_mean  # (n1,)

        # left[j] = sum_i E_W[k_W(W1_i, W)] alpha[i, j]
        left = w_mean @ self._alpha  # (n2,)

        right = self._k_A(self._A2, A_test)  # (n2, n_test)

        if self._has_x:
            right = right * self._k_X(self._X2, X_checked)  # (n2, n_test)

        return np.sum(left[:, np.newaxis] * right, axis=0).ravel()

    def _predict_conditional_mean(
        self,
        a: float | np.ndarray,
        *,
        X: np.ndarray | None,
        n: int | None = None,
    ) -> np.ndarray:
        """
        Conditional head:

            E_hat[h(a, x_i, W) | X=x_i]

        where the conditional expectation over W is approximated by
        Nadaraya-Watson kernel smoothing over the training pairs (X_ref, W_ref):

            weights_l(x_i) ∝ k_X(X_ref_l, x_i).
        """
        if not self._has_x:
            # With no X, P(W | X) = P(W) by construction of the model input.
            return self._predict_independent_mean(a, X=None, n=n)

        X_checked = self._validate_X_for_prediction(X, n=None, required=True)
        n_test = X_checked.shape[0]
        A_test = self._as_treatment_matrix(a, n=n_test)

        # K_X(X_ref, X_test): (n_ref, n_test)
        K_X_ref_test = self._k_X(self._X_ref, X_checked)

        denom = K_X_ref_test.sum(axis=0, keepdims=True)
        weights = K_X_ref_test / np.maximum(denom, 1e-12)

        # If all weights underflow for a test point, fall back to uniform.
        bad = denom.ravel() <= 1e-12
        if np.any(bad):
            weights[:, bad] = 1.0 / K_X_ref_test.shape[0]

        # Conditional W mean features:
        # W_cond_mean[:, i] = sum_l weight_l(x_i) k_W(W1, W_ref_l)
        # shape: (n1, n_test)
        W_cond_mean = self._K_W1_ref @ weights

        # left[i, j] = sum_r W_cond_mean[r, i] alpha[r, j]
        # shape: (n_test, n2)
        left = W_cond_mean.T @ self._alpha

        right = self._k_A(self._A2, A_test) * self._k_X(self._X2, X_checked)

        return np.einsum("ij,ji->i", left, right).ravel()

    # ── Prediction utilities ──────────────────────────────────────────────────

    def _as_treatment_matrix(
        self,
        a: float | np.ndarray,
        n: int | None = None,
    ) -> np.ndarray:
        """
        Convert treatment input to shape (n_test, d_A).

        Rules:
            scalar a:
                repeated n times if n is given, else shape (1, d_A)

            1D a with scalar treatment:
                interpreted as treatment grid, shape (len(a), 1)

            1D a with vector treatment:
                interpreted as one treatment vector, shape (1, d_A)

            2D a:
                must already have shape (n_test, d_A)
        """
        d_A = self._A2.shape[1]
        arr = np.asarray(a, dtype=float)

        if arr.ndim == 0:
            n_out = 1 if n is None else n
            return np.full((n_out, d_A), float(arr))

        if arr.ndim == 1:
            if d_A == 1:
                out = arr.reshape(-1, 1)
            elif arr.size == d_A:
                out = arr.reshape(1, d_A)
            else:
                raise ValueError(
                    f"Treatment has shape {arr.shape}; expected scalar, "
                    f"(*, 1), or (*, {d_A})."
                )

        elif arr.ndim == 2:
            if arr.shape[1] != d_A:
                raise ValueError(
                    f"Treatment has shape {arr.shape}; expected second "
                    f"dimension {d_A}."
                )
            out = arr

        else:
            raise ValueError(
                f"Treatment has shape {arr.shape}; expected scalar, 1D, or 2D."
            )

        if n is not None:
            if out.shape[0] == 1 and n > 1:
                out = np.repeat(out, n, axis=0)
            elif out.shape[0] != n:
                raise ValueError(
                    f"Treatment has {out.shape[0]} rows, but expected {n}."
                )

        return out

    def _validate_X_for_prediction(
        self,
        X: np.ndarray | None,
        *,
        n: int | None,
        required: bool,
    ) -> np.ndarray | None:
        if not self._has_x:
            return None

        if X is None:
            if required:
                raise ValueError("This prediction mode requires X.")
            return None

        X = as_2d(X)

        if X.shape[1] != self._X2.shape[1]:
            raise ValueError(
                f"X has shape {X.shape}; expected second dimension "
                f"{self._X2.shape[1]}."
            )

        if n is not None and X.shape[0] != n:
            raise ValueError(f"X has {X.shape[0]} rows, but expected {n}.")

        return X

    @staticmethod
    def _infer_n(
        *,
        X: np.ndarray | None,
        W: np.ndarray | None,
        n: int | None,
    ) -> int | None:
        if n is not None:
            return n
        if X is not None:
            return as_2d(X).shape[0]
        if W is not None:
            return as_2d(W).shape[0]
        return None

    @staticmethod
    def _repeat_if_scalar(pred: np.ndarray, n: int | None) -> np.ndarray:
        pred = np.asarray(pred, dtype=float).reshape(-1)
        if n is not None and pred.size == 1:
            return np.full(n, pred.item(), dtype=float)
        return pred

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self._alpha is None:
            raise RuntimeError("Call fit() before predict().")

    @staticmethod
    def _tune(
        criterion,
        low: float,
        high: float,
        n_points: int,
    ) -> float:
        """Grid search over a log-spaced λ grid; returns the best value."""
        candidates = np.logspace(np.log10(low), np.log10(high), n_points)
        scores = {lam: criterion(lam) for lam in candidates}
        return min(scores, key=scores.get)
