"""
PMMR — Proximal Maximum Moment Restriction
==========================================
Single-stage kernel ridge estimator for proximal causal inference. Learns an
outcome bridge function h(A, W) by minimising a kernelised maximum-moment
restriction with two ridge penalties.

Reference: Mastouri et al. "Proximal Causal Learning with Kernels" (ICML 2021)

Let, over the n training points, with product kernels over (A, Z, X) and (A, W, X):
    Φ = K_A ⊙ K_Z ⊙ K_X      (instrument-side feature gram)
    L = K_A ⊙ K_W ⊙ K_X      (bridge-side feature gram)

K_X is shared by both grams; K_Z alone carries an extra widening factor
(kernel_scale_z). See PMMRConfig.

The bridge coefficients solve the regularised normal equations:
    α = (L Φ L + λ₁ n L + λ₂ n I)⁻¹ L Φ Y

Prediction of E[Y | do(a)] marginalises over the *joint* empirical (W, X):
    ŷ(a) = αᵀ [ k_A(A_train, a) ⊙ wx̄ ]
    where wx̄ = mean_row(K_W ⊙ K_X)

The mean is taken over the elementwise product, not of each gram separately —
E_{W,X}[·] under the joint empirical measure is (1/n) Σ_j k_W(w_i,w_j) k_X(x_i,x_j),
which is not mean_row(K_W) ⊙ mean_row(K_X) unless W ⊥ X in sample.

CATE = ŷ(1) - ŷ(0)  (for binary treatment).

All computation is pure numpy/scipy — no JAX dependency.

Tuning entry point: PMMRConfig. Note that PMMR is *not* a fixed-hyperparameter
method: the reference implementation grid-searches λ and selects on a held-out
V-statistic. We do not, so the defaults here are unvalidated starting points
rather than tuned values — see PMMRConfig for the mapping to the paper's λ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import solve, LinAlgError

from benchmarks.methods.kernels import ColumnWiseGaussianKernel, as_2d, BinaryKernel


# Escalation budget for _solve_pd: each retry multiplies the jitter by 100, so
# six tries span ten orders of magnitude above the configured starting value.
_JITTER_MAX_TRIES = 6
_JITTER_ESCALATION = 100.0

# The reference implementation sets the instrument-side bandwidth to
# `median / 0.6` while leaving the bridge side at the plain median heuristic.
# Their kernel is exp(-d²/(2σ²)) and ours is exp(-d²/(2·median²·scale)), so
# their divisor of 0.6 on σ is a scale of 1/0.6² here.
_REFERENCE_Z_WIDENING = 1.0 / 0.6**2  # ≈ 2.778


def _solve_pd(lhs: np.ndarray, rhs: np.ndarray, jitter: float) -> np.ndarray:
    """
    Solve the symmetric positive-definite system ``lhs @ x = rhs`` by Cholesky,
    adding a jitter *relative to the scale of lhs* and escalating it if the
    factorisation fails.

    Why relative. ``lam2 * n * I`` is an absolute floor, so how much
    conditioning it actually buys depends on the magnitude of ``L Φ L``, which
    varies by orders of magnitude across episodes (it grows like n² and shrinks
    as the kernels narrow). Scaling the jitter by ``mean(diag(lhs))`` — the mean
    eigenvalue — makes it a bound on the condition number instead: a jitter of j
    caps cond at roughly 1/j regardless of episode scale.

    Why it matters for low-dimensional data. Φ and L are products of per-column
    Gaussian grams, so with few proxy columns and no covariates they are close
    to rank-deficient and ``L Φ L`` is numerically singular: with lam2 = 0 and
    scalar W, Z, the assembled matrix reaches rcond ≈ 1e-21, well past what
    double precision can resolve. The default lam2 already lifts that in the
    cases measured here, but nothing guarantees it does for every episode, and
    the failure is silent — an ill-conditioned solve returns a plausible-looking
    α whose component in the near-null space is noise.

    Cholesky (``assume_a="pos"``) is used rather than the previous LDL
    (``assume_a="sym"``) because it is about twice as fast and, more usefully,
    *raises* on a matrix that is not positive definite instead of quietly
    returning a solution. That failure is the signal this helper acts on.
    """
    n = lhs.shape[0]
    scale = float(np.trace(lhs)) / n
    if not np.isfinite(scale) or scale <= 0.0:
        raise LinAlgError(
            f"PMMR normal-equation matrix has non-positive/non-finite scale "
            f"(mean diagonal = {scale}); the kernel grams are degenerate."
        )

    eye = np.eye(n)
    current = jitter
    for _ in range(_JITTER_MAX_TRIES):
        try:
            return solve(lhs + current * scale * eye, rhs, assume_a="pos")
        except LinAlgError:
            current *= _JITTER_ESCALATION

    raise LinAlgError(
        f"PMMR normal equations remained non-positive-definite after "
        f"{_JITTER_MAX_TRIES} jitter escalations (up to "
        f"{current / _JITTER_ESCALATION:.1e} * mean(diag)). Raise lam2, widen "
        f"kernel_scale, or check for duplicate/constant proxy columns."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PMMRConfig:
    # ── Kernel ────────────────────────────────────────────────────────────────
    # Global multiplicative rescaling of the median-heuristic bandwidth σ.
    # Values < 1 → narrower kernel (more local); > 1 → wider (smoother).
    kernel_scale: float = 1.0

    # Extra widening applied to the treatment-inducing proxy kernel K_Z alone —
    # the factor that distinguishes Φ from L. Composes with kernel_scale, so
    # Z's bandwidth is kernel_scale * kernel_scale_z and everything else's is
    # kernel_scale. This has to be its own knob rather than a different default
    # on a shared one: the two grams want different bandwidths, and no single
    # scale can express that.
    #
    # Scope note: X enters *both* Φ and L, and we deliberately give it one
    # shared bandwidth rather than widening its Φ-side copy. The reference
    # cannot settle this — its experiments set X = ∅ throughout ("we assume
    # that X is a null set"), so the 0.6 divisor was only ever applied to
    # (A, Z). Widening X on the Φ side would make the objective less sensitive
    # to moment violations that vary with X, which is the wrong trade when X is
    # exactly what CATE conditions on. The competing consideration — a product
    # kernel decays faster the more factors it has, so Φ drifts toward I as d_X
    # grows — is better handled by tuning kernel_scale globally.
    #
    # Why it is not a nuisance parameter. k decides which moment violations the
    # MMR objective can see, and both limits are degenerate:
    #   narrow → Φ → I    → the objective collapses to Σᵢ dᵢ², i.e. plain kernel
    #                       ridge of Y on (A, W, X) — the paper's KernelRidge-W
    #                       baseline. The proximal correction is gone; how much
    #                       bias that costs depends on how good a proxy W is.
    #   wide   → Φ → 11ᵀ  → the objective collapses to (mean d)², a single
    #                       moment, and h is badly under-identified.
    # Measured, n=1500, true ATE 2.0, naive contrast 4.27, lam1=1e-3, over
    # kernel_scale_z = 1e-4 … 1e6:
    #   informative W (proxy noise 0.5; KernelRidge-W = 2.74)
    #       2.45  2.26  2.20  2.17  2.17  2.14  2.30  2.13  3.82
    #   noisy W       (proxy noise 1.5; KernelRidge-W = 3.74)
    #       3.58  3.35  2.89  2.44  2.27  2.17  2.15  2.55  4.09
    # The wide end is the sharper failure — it walks back to the naive contrast.
    # The narrow end degrades gracefully with an informative W and badly with a
    # noisy one. The default sits in a broad flat basin in both cases.
    #
    # The default reproduces the reference implementation, which hardcodes
    # `ak = median(AZ) / 0.6` (`W_kern_divisor`) against a plain median
    # heuristic on the bridge side — a deliberate step away from the Φ ≈ I
    # cliff. Note their Appendix D describes a plain median heuristic on *both*
    # sides; only the released code widens Φ. We follow the code.
    kernel_scale_z: float = _REFERENCE_Z_WIDENING

    # ── Regularisation ────────────────────────────────────────────────────────
    # lam1 is the RKHS-norm penalty on the bridge (the αᵀLα term). It maps onto
    # the paper's λ — which multiplies L in α = (LΦL + λL)⁻¹LΦy with Φ scaled by
    # 1/n² — as  lam1 = λ·n.  So a *fixed* lam1 means λ ∝ 1/n, which decays
    # faster than the n^(-1/2)..n^(-1/4) the convergence theorem asks for; the
    # right reparameterisation is lam1 = c·√n.  The paper grid-searches λ over
    # [4.9e-6, 0.25] (50 points, selected on a held-out V-statistic); lam1 = 1.0
    # sits mid-grid at n≈2000 but is a guess, not a selected value, and visibly
    # over-shrinks h relative to lam1 ≈ 1e-3 on synthetic checks.
    lam1: float = 1.0

    # lam2 is NOT cosmetic jitter — it sets the smallest eigenvalue of the
    # normal-equation matrix, and hence whether the solve means anything. LΦL is
    # badly rank-deficient: measured at n=2000, lam2=1e-4 gives eig_min ≈ 2e-1
    # (cond ≈ 5e6), while lam2=0 gives eig_min ≈ 2e-5 (cond ≈ 5e10). Low-dim
    # W/Z with no X is worse still. Do not lower it without checking rcond.
    # (The reference implementation stabilises differently: jitter added to L
    # and to the Z-gram before the products are formed, not to the assembled
    # left-hand side.)
    lam2: float = 1e-4

    # Relative jitter for the Cholesky solve, in units of mean(diag(lhs)) — see
    # _solve_pd. Unlike lam2 this is not a statistical knob: it is insurance
    # against a numerically singular normal-equation matrix on low-dimensional
    # proxies, and escalates by 100x per retry if the factorisation fails. At
    # 1e-10 it caps the condition number near 1e10 while moving the fitted
    # bridge by ~1e-11 relative on well-conditioned data, i.e. it is inert
    # whenever it is not needed.
    jitter: float = 1e-10

    # No seed: fit() is fully deterministic. Do not add a global
    # np.random.seed() call here — estimators share a process in the episode
    # pool, so it would reseed whatever runs next.


# ─────────────────────────────────────────────────────────────────────────────
# Estimator
# ─────────────────────────────────────────────────────────────────────────────


class PMMREstimator:
    """
    Sklearn-style CATE estimator implementing PMMR.

    Usage::

        cfg = PMMRConfig(lam1=1.0, lam2=1e-4)
        est = PMMREstimator(cfg)
        est.fit(X_train, A_train, Y_train, W=W_train, Z=Z_train)
        cate = est.predict(X_test)   # shape (n_test,)

    The ``cfg`` dataclass is the single place to tune all hyperparameters.
    """

    def __init__(self, cfg: PMMRConfig | None = None) -> None:
        self.cfg = cfg or PMMRConfig()
        # Learnt quantities stored after fit()
        self._alpha: np.ndarray | None = None  # (n,)
        self._wx_mean: np.ndarray | None = (
            None  # (n,) — joint empirical mean embedding of (W, X)
        )
        self._A_train: np.ndarray | None = None  # (n, d_A)
        self._W_train: np.ndarray | None = None  # (n, d_W)
        self._X_train: np.ndarray | None = None  # (n, d_X)
        self._k_A: BinaryKernel | ColumnWiseGaussianKernel | None = None
        self._k_W: ColumnWiseGaussianKernel | None = None
        self._k_X: ColumnWiseGaussianKernel | None = None
        self._K_W_train: np.ndarray | None = None  # (n, n) train-train W gram

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        *,
        W: np.ndarray,
        Z: np.ndarray,
    ) -> "PMMREstimator":
        """
        Fit the PMMR bridge function.

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

        A = as_2d(A)
        Y = np.asarray(Y, dtype=float).reshape(-1)
        W = as_2d(W)
        Z = as_2d(Z)
        X = as_2d(X)
        n = len(Y)

        # ── Fit kernel bandwidths (median heuristic) ───────────────────────
        # Everything uses kernel_scale; the proxy Z alone is widened further by
        # kernel_scale_z. K_A has no bandwidth (indicator) and K_X is shared by
        # both grams, so Z is the only factor that differs between Φ and L.
        k_A = BinaryKernel().fit(A)
        k_Z = ColumnWiseGaussianKernel().fit(
            Z, scale=cfg.kernel_scale * cfg.kernel_scale_z
        )
        k_W = ColumnWiseGaussianKernel().fit(W, scale=cfg.kernel_scale)

        K_A = k_A(A, A)
        K_Z = k_Z(Z, Z)
        K_W = k_W(W, W)

        if X.shape[1] > 0:
            k_X = ColumnWiseGaussianKernel().fit(X, scale=cfg.kernel_scale)
            K_X = k_X(X, X)
        else:
            k_X = None
            K_X = np.ones((n, n))

        # Compute Gram matrices. X is shared: only K_Z carries the extra
        # widening, so the covariate factor is identical on both sides.
        Phi = K_A * K_Z * K_X
        L = K_A * K_W * K_X

        # Solve normal equations
        LPhi = L @ Phi
        lhs = LPhi @ L + cfg.lam1 * n * L + cfg.lam2 * n * np.eye(n)
        rhs = LPhi @ Y
        self._alpha = _solve_pd(lhs, rhs, cfg.jitter)

        # Empirical joint mean embedding of (W, X), used by the marginal ATE path.
        self._wx_mean = (K_W * K_X).mean(axis=1)

        # Persist training state needed by the paired / independent / conditional
        # prediction paths.
        self._A_train = A
        self._W_train = W
        self._X_train = X
        self._k_A = k_A
        self._k_W = k_W
        self._k_X = k_X
        self._K_W_train = K_W

        return self

    def predict(
        self,
        X: np.ndarray,
        *,
        W: np.ndarray | None = None,
        strategy: Literal[
            "marginal", "paired", "independent", "conditional"
        ] = "marginal",
    ) -> np.ndarray:
        """
        Predict a binary treatment contrast.

        strategy="marginal":
            Return the global ATE repeated for each test row.

        strategy="paired":
            Requires W. Return h(1,W_i,X_i) - h(0,W_i,X_i).

        strategy="independent":
            Return mean_j h(1,W_j,X_i) - h(0,W_j,X_i),
            averaging over marginal training W.

        strategy="conditional":
            Return kernel-smoothed approximation to
            E[h(1,W,X_i)-h(0,W,X_i) | X_i].
        """
        self._check_fitted()

        X = as_2d(X)

        if strategy == "marginal":
            return self.predict_mean_outcome(
                1.0, n=X.shape[0]
            ) - self.predict_mean_outcome(0.0, n=X.shape[0])

        if strategy == "paired":
            if W is None:
                raise ValueError("strategy='paired' requires W.")
            return self._predict_effect_paired_w(X, as_2d(W))

        if strategy == "independent":
            return self._predict_effect_independent_w(X)

        if strategy == "conditional":
            return self._predict_effect_conditional_w(X)

        raise ValueError(f"Unknown prediction strategy: {strategy!r}")

    def _predict_effect_paired_w(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        return self.predict_bridge(1.0, W=W, X=X) - self.predict_bridge(0.0, W=W, X=X)

    def predict_bridge(
        self,
        a: float | np.ndarray,
        W: np.ndarray,
        X: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Evaluate h(a, W, X) at paired test points.
        """
        self._check_fitted()

        W = as_2d(W)
        n_test = W.shape[0]

        if np.isscalar(a):
            A_test = np.full((n_test, self._A_train.shape[1]), float(a))
        else:
            A_test = as_2d(a)

        K_A = self._k_A(self._A_train, A_test)
        K_W = self._k_W(self._W_train, W)

        if self._k_X is not None:
            if X is None:
                raise ValueError("X must be provided because the model was fit with X.")
            X = as_2d(X)
            K_X = self._k_X(self._X_train, X)
        else:
            K_X = 1.0

        K_test = K_A * K_W * K_X
        return (self._alpha @ K_test).ravel()

    def predict_mean_outcome(self, a: float | np.ndarray, n: int = 1) -> np.ndarray:
        self._check_fitted()

        if np.isscalar(a):
            a_arr = np.full((n, self._A_train.shape[1]), float(a))
        else:
            a_arr = as_2d(a)

        K_A = self._k_A(self._A_train, a_arr)
        K_test = K_A * self._wx_mean[:, None]

        return (self._alpha @ K_test).ravel()

    def _predict_mean_outcome_independent_w(
        self, a: float, X: np.ndarray
    ) -> np.ndarray:
        self._check_fitted()
        X = as_2d(X)
        n_test = X.shape[0]

        A_test = np.full((n_test, self._A_train.shape[1]), float(a))

        K_A = self._k_A(self._A_train, A_test)

        if self._k_X is not None:
            K_X = self._k_X(self._X_train, X)
        else:
            K_X = np.ones_like(K_A)

        # mean over marginal W_train
        w_mean = self._K_W_train.mean(axis=1)

        K_test = K_A * K_X * w_mean[:, None]
        return (self._alpha @ K_test).ravel()

    def _predict_effect_independent_w(self, X: np.ndarray) -> np.ndarray:
        return self._predict_mean_outcome_independent_w(
            1.0, X
        ) - self._predict_mean_outcome_independent_w(0.0, X)

    def _predict_mean_outcome_conditional_w(
        self,
        a: float,
        X: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:
        self._check_fitted()

        if self._k_X is None:
            # No X was used in training, so conditional and marginal collapse.
            return self.predict_mean_outcome(a, n=X.shape[0])

        X = as_2d(X)
        n_test = X.shape[0]

        A_test = np.full((n_test, self._A_train.shape[1]), float(a))

        # K_X_train_test[j, t] = k_X(X_j, x_t)
        K_X_train_test = self._k_X(self._X_train, X)

        # Kernel weights q_j(x_t) approx P(W_j | X=x_t)
        weights = K_X_train_test / (K_X_train_test.sum(axis=0, keepdims=True) + eps)

        # cond_w_mean[i, t] = sum_j k_W(W_i, W_j) q_j(x_t)
        cond_w_mean = self._K_W_train @ weights

        K_A = self._k_A(self._A_train, A_test)

        # h(a, W, x) also contains k_X(X_i, x)
        K_test = K_A * K_X_train_test * cond_w_mean

        return (self._alpha @ K_test).ravel()

    def _predict_effect_conditional_w(self, X: np.ndarray) -> np.ndarray:
        return self._predict_mean_outcome_conditional_w(
            1.0, X
        ) - self._predict_mean_outcome_conditional_w(0.0, X)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self._alpha is None:
            raise RuntimeError("Call fit() before predict().")
