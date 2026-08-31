"""Proxy-adjusted g-computation with a TabICL outcome regression.

This is deliberately a diagnostic baseline, not a proximal estimator.  TabICL
learns the observational regression ``E[Y | X, W, Z, A]`` and prediction plugs
that regression into a truncated kernel estimate of ``P(W, Z | X=x)``.  The
proximal assumptions do not imply that adjusting for both proxies removes the
latent confounding, so a good result is evidence that ordinary proxy adjustment
is competitive on a dataset, not that the proximal bridge was recovered.

The conditional proxy distribution is represented by the *aligned* context
atoms ``(W_i, Z_i)``.  It is never the Cartesian product of the two blocks.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from benchmarks.methods.base import CATEEstimator

__all__ = ["TabICLGComp"]


def _matrix(name: str, value, *, n: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {array.shape}.")
    if n is not None and len(array) != n:
        raise ValueError(f"{name} has {len(array)} rows; expected {n}.")
    return array


class TabICLGComp(CATEEstimator):
    """TabICL outcome regression followed by proxy marginalisation.

    Args:
        outcome_model: A fresh sklearn-compatible TabICL regressor.
        proxy_top_k: Number of context ``(W_i, Z_i)`` atoms retained per query.
        kernel_scale: Multiplier on the median-heuristic X-kernel bandwidths.
        predict_batch_size: Maximum synthetic rows sent to TabICL per call.
    """

    def __init__(
        self,
        outcome_model,
        *,
        proxy_top_k: int = 32,
        kernel_scale: float = 1.0,
        predict_batch_size: int = 4096,
    ) -> None:
        if int(proxy_top_k) < 1:
            raise ValueError("proxy_top_k must be at least 1.")
        if not np.isfinite(kernel_scale) or float(kernel_scale) <= 0:
            raise ValueError("kernel_scale must be a finite positive number.")
        if int(predict_batch_size) < 1:
            raise ValueError("predict_batch_size must be at least 1.")

        self.outcome_model = outcome_model
        self.proxy_top_k = int(proxy_top_k)
        self.kernel_scale = float(kernel_scale)
        self.predict_batch_size = int(predict_batch_size)

        self._X_context: np.ndarray | None = None
        self._W_context: np.ndarray | None = None
        self._Z_context: np.ndarray | None = None
        self._inv_width: np.ndarray | None = None

    def fit(self, X, A, Y, *, W=None, Z=None) -> "TabICLGComp":
        if W is None or Z is None:
            raise ValueError("TabICLGComp requires both W and Z during fit().")

        X = _matrix("X", X)
        n = len(X)
        if n == 0:
            raise ValueError("TabICLGComp requires at least one context row.")
        W = _matrix("W", W, n=n)
        Z = _matrix("Z", Z, n=n)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)
        if len(A) != n or len(Y) != n:
            raise ValueError(f"A and Y must have {n} rows; got {len(A)} and {len(Y)}.")

        # Keep the exact aligned atoms used by the outcome regression.  Copies
        # prevent a caller mutating the episode arrays after fit().
        self._X_context = np.array(X, copy=True)
        self._W_context = np.array(W, copy=True)
        self._Z_context = np.array(Z, copy=True)

        sigmas = np.empty(X.shape[1], dtype=np.float64)
        for d in range(X.shape[1]):
            column = np.asarray(X[:, [d]], dtype=np.float64)
            distances = cdist(column, column, metric="euclidean")
            sigma = 2.0 * np.median(distances) ** 2 * self.kernel_scale
            sigmas[d] = sigma if np.isfinite(sigma) and sigma > 0 else 1e-3
        self._inv_width = 1.0 / np.sqrt(sigmas)

        features = np.hstack([X, W, Z, A[:, None]])
        self.outcome_model.fit(features, Y)
        return self

    def _topk_atoms(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return context indices and renormalised RBF weights per query."""
        assert self._X_context is not None
        assert self._inv_width is not None

        k = min(self.proxy_top_k, len(self._X_context))
        if X.shape[1] == 0:
            distances = np.zeros((len(X), len(self._X_context)), dtype=np.float64)
        else:
            # hypot.reduce is stable for very large finite query values, unlike
            # forming squared distances before the top-k neighbours are known.
            delta = (
                np.asarray(X, dtype=np.float64)[:, None, :]
                - np.asarray(self._X_context, dtype=np.float64)[None, :, :]
            ) * self._inv_width[None, None, :]
            distances = np.hypot.reduce(delta, axis=2)

        # Stable sorting makes ties (constant X is the important case)
        # deterministic.  Ranking by distance is identical to ranking by the
        # Gaussian log weight -distance**2.
        indices = np.argsort(distances, axis=1, kind="stable")[:, :k]
        selected = np.take_along_axis(distances, indices, axis=1)

        # Compute -(d^2-d_min^2) without first squaring d.  If even the robust
        # distance overflowed, all selected atoms receive uniform weight rather
        # than propagating NaNs.
        minimum = selected[:, :1]
        difference = selected - minimum
        with np.errstate(over="ignore", invalid="ignore"):
            logits = -difference * (selected + minimum)
        logits = np.where(difference == 0, 0.0, logits)

        bad_rows = ~np.isfinite(minimum[:, 0])
        logits = np.nan_to_num(logits, nan=-np.inf, posinf=0.0, neginf=-np.inf)
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        if np.any(bad_rows):
            weights[bad_rows] = 1.0 / k
        return indices, weights

    def _predict_chunks(self, features: np.ndarray) -> np.ndarray:
        result = np.empty(len(features), dtype=np.float64)
        for start in range(0, len(features), self.predict_batch_size):
            stop = min(start + self.predict_batch_size, len(features))
            prediction = np.asarray(self.outcome_model.predict(features[start:stop]))
            result[start:stop] = prediction.reshape(-1)
        return result

    def predict(self, X, *, W=None, Z=None) -> np.ndarray:
        if (
            self._X_context is None
            or self._W_context is None
            or self._Z_context is None
        ):
            raise ValueError("Call fit() before predict().")

        X = _matrix("X", X)
        if X.shape[1] != self._X_context.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} columns; expected {self._X_context.shape[1]}."
            )
        if len(X) == 0:
            return np.empty(0, dtype=np.float64)

        indices, weights = self._topk_atoms(X)
        n_query, k = indices.shape
        flat_indices = indices.reshape(-1)

        X_atoms = np.repeat(X, k, axis=0)
        W_atoms = self._W_context[flat_indices]
        Z_atoms = self._Z_context[flat_indices]
        common = np.hstack([X_atoms, W_atoms, Z_atoms])

        # Both arms use exactly the same aligned proxy atoms and weights.  Query
        # W/Z arguments are intentionally ignored: they have been integrated out.
        dtype = np.result_type(common.dtype, np.float32)
        arm0 = np.hstack([common, np.zeros((len(common), 1), dtype=dtype)])
        arm1 = np.hstack([common, np.ones((len(common), 1), dtype=dtype)])
        potential = self._predict_chunks(np.vstack([arm0, arm1]))
        contrast = potential[len(common) :] - potential[: len(common)]
        return np.sum(contrast.reshape(n_query, k) * weights, axis=1)
