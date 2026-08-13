"""P-learner with Ghassami-style kernel bridge nuisances.

This module is a compact Python port of the kernel bridge estimator used in
Ghassami et al. style proximal causal learning code and wraps it in the
P-learner pseudo-outcome regression of Sverdrup & Cui (2023).

Arrays convention:
    X: (n, d_x) CATE covariates
    Z: (n, d_z) treatment-inducing proxies
    W: (n, d_w) outcome-inducing proxies
    A: (n,) binary treatment in {0, 1}
    Y: (n,) observed outcome
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def _as_2d(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array; got shape {arr.shape}")
    return arr


def _as_1d(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return arr


def _check_lengths(*arrays: np.ndarray) -> None:
    n = arrays[0].shape[0]
    if any(a.shape[0] != n for a in arrays):
        raise ValueError("X, Z, W, A, and Y must have the same number of rows")


def rbf_kernel(
    X: np.ndarray, Y: Optional[np.ndarray] = None, sigma: float = 1.0
) -> np.ndarray:
    """Gaussian RBF kernel exp(-||x-y||^2 / (2 * sigma)).

    This matches the `rdetools::rbfkernel` parameterization used by the
    reference R implementation (Ghassami et al. style proximal code), namely
    `K = exp(-D / (2 * sigma))` with D the squared Euclidean distance. Here
    `sigma` is the width parameter in that R convention (not an inverse
    length-scale), so `kernel_sigma` values suggested in the R code transfer
    over directly.
    """
    if Y is None:
        Y = X
    d2 = pairwise_distances(X, Y, metric="sqeuclidean")
    return np.exp(-d2 / (2.0 * float(sigma)))


def _pinv(A: np.ndarray, rcond: float = 1e-10, hermitian: bool = False) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if hermitian:
        A = 0.5 * (A + A.T)
    return np.linalg.pinv(A, rcond=rcond, hermitian=hermitian)


def _invsqrt_psd(A: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Moore-Penrose inverse square-root of a symmetric PSD matrix."""
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vmax = max(float(np.max(vals)), 1.0)
    keep = vals > rcond * vmax
    invsqrt_vals = np.zeros_like(vals)
    invsqrt_vals[keep] = 1.0 / np.sqrt(vals[keep])
    return (vecs * invsqrt_vals) @ vecs.T


@dataclass
class BridgeParams:
    """Hyperparameters for the kernel bridge estimator.

    The names mirror the R code: H is the RKHS for h(W, A, X), Q is the RKHS
    for q(Z, A, X). For example, lm_Hh1 penalizes h at A=1, and lm_Qh1
    penalizes the critic used to estimate h at A=1.
    """

    # RBF width in the rdetools convention: K = exp(-||.||^2 / (2 * kernel_sigma)).
    kernel_sigma: float = 1.0

    lm_Hh1: float = 1e-3
    lm_Qh1: float = 1e-3
    lm_Hh0: float = 1e-3
    lm_Qh0: float = 1e-3

    lm_Hq1: float = 1e-3
    lm_Qq1: float = 1e-3
    lm_Hq0: float = 1e-3
    lm_Qq0: float = 1e-3

    # The R snippets use n^0.8 in in-sample evaluation and n^0.7 in cross-fit
    # evaluation. For P-learner cross-fitting, 0.7 is usually the closer port.
    exponent: float = 0.7

    # Full kernel is O(n^3). Nyström is often needed for n in the thousands.
    nystroem: bool = False
    appro_rate: float = 0.05
    n_components: Optional[int] = None

    # In practice, RBF kernels are sensitive to scale; set False for closer
    # literal parity with already-standardized R code.
    scale_kernel: bool = True

    pinv_rcond: float = 1e-10
    random_state: Optional[int] = None


class KernelBridgeNuisance:
    """Ghassami-style kernel estimates of h(w,a,x) and q(z,a,x)."""

    def __init__(self, params: Optional[BridgeParams] = None):
        self.params = params if params is not None else BridgeParams()

    def fit(
        self, X: ArrayLike, Z: ArrayLike, W: ArrayLike, A: ArrayLike, Y: ArrayLike
    ) -> "KernelBridgeNuisance":
        X = _as_2d(X, "X")
        Z = _as_2d(Z, "Z")
        W = _as_2d(W, "W")
        A = _as_1d(A, "A")
        Y = _as_1d(Y, "Y")
        _check_lengths(X, Z, W, A.reshape(-1, 1), Y.reshape(-1, 1))
        if not np.all(np.isin(A, [0.0, 1.0])):
            raise ValueError("A must be binary and encoded as 0/1")

        self.n_train_ = X.shape[0]
        self.dx_ = X.shape[1]
        self.dz_ = Z.shape[1]
        self.dw_ = W.shape[1]

        H_raw = np.column_stack([W, X])
        Q_raw = np.column_stack([Z, X])
        if self.params.scale_kernel:
            self.h_scaler_ = StandardScaler().fit(H_raw)
            self.q_scaler_ = StandardScaler().fit(Q_raw)
            H = self.h_scaler_.transform(H_raw)
            Q = self.q_scaler_.transform(Q_raw)
        else:
            self.h_scaler_ = None
            self.q_scaler_ = None
            H, Q = H_raw, Q_raw

        self.H_train_ = H
        self.Q_train_ = Q

        K_H = rbf_kernel(H, sigma=self.params.kernel_sigma)
        K_Q = rbf_kernel(Q, sigma=self.params.kernel_sigma)

        if self.params.nystroem:
            self._fit_nystroem(K_H, K_Q, A, Y)
        else:
            self._fit_full(K_H, K_Q, A, Y)
        return self

    def _gamma_full(self, K: np.ndarray, lm: float) -> np.ndarray:
        n = K.shape[0]
        reg = lm / (n**self.params.exponent)
        return (
            0.25
            * K
            @ _pinv(K / n + reg * np.eye(n), self.params.pinv_rcond, hermitian=True)
        )

    def _fit_full(
        self, K_H: np.ndarray, K_Q: np.ndarray, A: np.ndarray, Y: np.ndarray
    ) -> None:
        n = len(A)
        coef_scale = n**2 / (n**self.params.exponent)

        def fit_h(a: int, lm_Hh: float, lm_Qh: float) -> np.ndarray:
            d = (A == a).astype(float)
            Gamma = self._gamma_full(K_Q, lm_Qh)
            DGamD = (d[:, None] * Gamma) * d[None, :]
            M = K_H @ DGamD @ K_H + coef_scale * lm_Hh * K_H
            rhs = K_H @ (d * (Gamma @ (Y * d)))
            return _pinv(M, self.params.pinv_rcond) @ rhs

        def fit_q(a: int, lm_Hq: float, lm_Qq: float) -> np.ndarray:
            d = (A == a).astype(float)
            Gamma = self._gamma_full(K_H, lm_Hq)
            DGamD = (d[:, None] * Gamma) * d[None, :]
            M = K_Q @ DGamD @ K_Q + coef_scale * lm_Qq * K_Q
            rhs = K_Q @ (d * (Gamma @ np.ones(n)))
            return _pinv(M, self.params.pinv_rcond) @ rhs

        self.alpha_h1_ = fit_h(1, self.params.lm_Hh1, self.params.lm_Qh1)
        self.alpha_h0_ = fit_h(0, self.params.lm_Hh0, self.params.lm_Qh0)
        self.alpha_q1_ = fit_q(1, self.params.lm_Hq1, self.params.lm_Qq1)
        self.alpha_q0_ = fit_q(0, self.params.lm_Hq0, self.params.lm_Qq0)
        self.mode_ = "full"

    def _fit_nystroem(
        self, K_H: np.ndarray, K_Q: np.ndarray, A: np.ndarray, Y: np.ndarray
    ) -> None:
        n = len(A)
        rng = np.random.default_rng(self.params.random_state)
        r = self.params.n_components
        if r is None:
            r = int(np.ceil(n * self.params.appro_rate))
        r = max(2, min(int(r), n))
        idx = np.sort(rng.choice(n, size=r, replace=False))
        self.nys_idx_ = idx

        invsqrt_H = _invsqrt_psd(K_H[np.ix_(idx, idx)], self.params.pinv_rcond)
        invsqrt_Q = _invsqrt_psd(K_Q[np.ix_(idx, idx)], self.params.pinv_rcond)
        V = K_H[:, idx] @ invsqrt_H  # h feature basis, n x r
        D = K_Q[:, idx] @ invsqrt_Q  # q feature basis, n x r

        self.invsqrt_H_ = invsqrt_H
        self.invsqrt_Q_ = invsqrt_Q
        self.V_train_ = V
        self.D_train_ = D
        coef_scale = n**2 / (n**self.params.exponent)

        def gamma_lowrank(B: np.ndarray, lm: float) -> np.ndarray:
            reg = lm / (n**self.params.exponent)
            return (
                0.25
                * B
                @ _pinv(
                    B.T @ B / n + reg * np.eye(r),
                    self.params.pinv_rcond,
                    hermitian=True,
                )
                @ B.T
            )

        def fit_h(a: int, lm_Hh: float, lm_Qh: float) -> np.ndarray:
            d = (A == a).astype(float)
            Gamma = gamma_lowrank(D, lm_Qh)
            DV = d[:, None] * V
            M = DV.T @ Gamma @ DV + coef_scale * lm_Hh * np.eye(r)
            rhs = V.T @ (d * (Gamma @ (Y * d)))
            return _pinv(M, self.params.pinv_rcond, hermitian=True) @ rhs

        def fit_q(a: int, lm_Hq: float, lm_Qq: float) -> np.ndarray:
            d = (A == a).astype(float)
            Gamma = gamma_lowrank(V, lm_Hq)
            DD = d[:, None] * D
            M = DD.T @ Gamma @ DD + coef_scale * lm_Qq * np.eye(r)
            rhs = D.T @ (d * (Gamma @ np.ones(n)))
            return _pinv(M, self.params.pinv_rcond, hermitian=True) @ rhs

        self.alpha_h1_ = fit_h(1, self.params.lm_Hh1, self.params.lm_Qh1)
        self.alpha_h0_ = fit_h(0, self.params.lm_Hh0, self.params.lm_Qh0)
        self.alpha_q1_ = fit_q(1, self.params.lm_Hq1, self.params.lm_Qq1)
        self.alpha_q0_ = fit_q(0, self.params.lm_Hq0, self.params.lm_Qq0)
        self.mode_ = "nystroem"

    def _transform_H(self, W: ArrayLike, X: ArrayLike) -> np.ndarray:
        W = _as_2d(W, "W")
        X = _as_2d(X, "X")
        _check_lengths(W, X)
        raw = np.column_stack([W, X])
        return self.h_scaler_.transform(raw) if self.h_scaler_ is not None else raw

    def _transform_Q(self, Z: ArrayLike, X: ArrayLike) -> np.ndarray:
        Z = _as_2d(Z, "Z")
        X = _as_2d(X, "X")
        _check_lengths(Z, X)
        raw = np.column_stack([Z, X])
        return self.q_scaler_.transform(raw) if self.q_scaler_ is not None else raw

    def predict_h1(self, W: ArrayLike, X: ArrayLike) -> np.ndarray:
        H = self._transform_H(W, X)
        if self.mode_ == "full":
            K = rbf_kernel(H, self.H_train_, sigma=self.params.kernel_sigma)
            return K @ self.alpha_h1_
        K_idx = rbf_kernel(
            H, self.H_train_[self.nys_idx_], sigma=self.params.kernel_sigma
        )
        return K_idx @ self.invsqrt_H_ @ self.alpha_h1_

    def predict_h0(self, W: ArrayLike, X: ArrayLike) -> np.ndarray:
        H = self._transform_H(W, X)
        if self.mode_ == "full":
            K = rbf_kernel(H, self.H_train_, sigma=self.params.kernel_sigma)
            return K @ self.alpha_h0_
        K_idx = rbf_kernel(
            H, self.H_train_[self.nys_idx_], sigma=self.params.kernel_sigma
        )
        return K_idx @ self.invsqrt_H_ @ self.alpha_h0_

    def predict_h(self, W: ArrayLike, A: ArrayLike, X: ArrayLike) -> np.ndarray:
        A = _as_1d(A, "A")
        return A * self.predict_h1(W, X) + (1.0 - A) * self.predict_h0(W, X)

    def predict_q1(self, Z: ArrayLike, X: ArrayLike) -> np.ndarray:
        Q = self._transform_Q(Z, X)
        if self.mode_ == "full":
            K = rbf_kernel(Q, self.Q_train_, sigma=self.params.kernel_sigma)
            return K @ self.alpha_q1_
        K_idx = rbf_kernel(
            Q, self.Q_train_[self.nys_idx_], sigma=self.params.kernel_sigma
        )
        return K_idx @ self.invsqrt_Q_ @ self.alpha_q1_

    def predict_q0(self, Z: ArrayLike, X: ArrayLike) -> np.ndarray:
        Q = self._transform_Q(Z, X)
        if self.mode_ == "full":
            K = rbf_kernel(Q, self.Q_train_, sigma=self.params.kernel_sigma)
            return K @ self.alpha_q0_
        K_idx = rbf_kernel(
            Q, self.Q_train_[self.nys_idx_], sigma=self.params.kernel_sigma
        )
        return K_idx @ self.invsqrt_Q_ @ self.alpha_q0_

    def predict_q(self, Z: ArrayLike, A: ArrayLike, X: ArrayLike) -> np.ndarray:
        A = _as_1d(A, "A")
        return A * self.predict_q1(Z, X) + (1.0 - A) * self.predict_q0(Z, X)


class PLearner(BaseEstimator, RegressorMixin):
    """P-learner: cross-fit proximal scores, then regress scores on X."""

    def __init__(
        self,
        bridge_params: Optional[BridgeParams] = None,
        final_model: Optional[RegressorMixin] = None,
        n_splits: int = 3,
        random_state: Optional[int] = None,
        q_clip: Optional[Tuple[float, float]] = None,
        refit_full_nuisance: bool = False,
        n_jobs: int = 1,
    ):
        self.bridge_params = (
            bridge_params if bridge_params is not None else BridgeParams()
        )
        self.final_model = final_model
        self.n_splits = n_splits
        self.random_state = random_state
        self.q_clip = q_clip
        self.refit_full_nuisance = refit_full_nuisance
        # Only reaches the default final model below. It used to be a hardcoded -1,
        # which nests "every core" inside whatever process the learner is running in
        # — harmless while `build_method` always supplies a `final_model`, and an
        # oversubscription bomb the day it doesn't.
        self.n_jobs = n_jobs

    def fit(
        self, X: ArrayLike, Z: ArrayLike, W: ArrayLike, A: ArrayLike, Y: ArrayLike
    ) -> "PLearner":
        X = _as_2d(X, "X")
        Z = _as_2d(Z, "Z")
        W = _as_2d(W, "W")
        A = _as_1d(A, "A")
        Y = _as_1d(Y, "Y")
        _check_lengths(X, Z, W, A.reshape(-1, 1), Y.reshape(-1, 1))

        n = X.shape[0]
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        gamma_hat = np.empty(n, dtype=float)
        self.fold_nuisances_ = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            # Different Nyström subsamples by fold, reproducibly.
            params = replace(
                self.bridge_params,
                random_state=None
                if self.random_state is None
                else self.random_state + 1009 * fold,
            )
            nuisance = KernelBridgeNuisance(params).fit(
                X[train_idx], Z[train_idx], W[train_idx], A[train_idx], Y[train_idx]
            )
            self.fold_nuisances_.append(nuisance)

            X_te, Z_te, W_te, A_te, Y_te = (
                X[test_idx],
                Z[test_idx],
                W[test_idx],
                A[test_idx],
                Y[test_idx],
            )
            h_a = nuisance.predict_h(W_te, A_te, X_te)
            h_1 = nuisance.predict_h1(W_te, X_te)
            h_0 = nuisance.predict_h0(W_te, X_te)
            q_a = nuisance.predict_q(Z_te, A_te, X_te)
            if self.q_clip is not None:
                lo, hi = self.q_clip
                q_a = np.clip(q_a, lo, hi)
            sign = np.where(A_te == 1.0, 1.0, -1.0)
            gamma_hat[test_idx] = sign * q_a * (Y_te - h_a) + h_1 - h_0

        self.gamma_ = gamma_hat
        self.ate_ = float(np.mean(gamma_hat))

        if self.final_model is None:
            model = RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=10,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            model = clone(self.final_model)
        self.final_model_ = model.fit(X, gamma_hat)

        if self.refit_full_nuisance:
            self.nuisance_full_ = KernelBridgeNuisance(self.bridge_params).fit(
                X, Z, W, A, Y
            )
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        X = _as_2d(X, "X")
        return self.final_model_.predict(X)
