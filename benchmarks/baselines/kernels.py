"""
Shared kernel utilities for the proximal kernel baselines (KPV, PMMR).
=====================================================================
Pure numpy/scipy implementations — no JAX dependency.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


class ColumnWiseGaussianKernel:
    """
    Product of independent Gaussian kernels, one per input dimension.

    Each column gets its own bandwidth set by the median heuristic:
        σ_d = 2 * median(pairwise distances in column d)² * scale

    The product kernel is then  K(x, x') = ∏_d exp(−‖x_d − x'_d‖² / σ_d).

    A degenerate (constant) column would give σ_d = 0; we floor it at a small
    positive value to avoid division-by-zero during fitting.
    """

    def __init__(self) -> None:
        self._sigmas: list[float] = []

    def fit(self, data: np.ndarray, scale: float = 1.0) -> "ColumnWiseGaussianKernel":
        self._sigmas = []
        for d in range(data.shape[1]):
            col = data[:, [d]]
            dists = cdist(col, col, "euclidean")
            sigma = 2.0 * (np.median(dists) ** 2) * scale
            self._sigmas.append(sigma if sigma > 0 else 1e-3)
        return self

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the n×m kernel matrix K(x, y)."""
        if not self._sigmas:
            raise RuntimeError("Call fit() before computing the kernel matrix.")
        K = np.ones((x.shape[0], y.shape[0]))
        for d, sigma in enumerate(self._sigmas):
            sq_dist = cdist(x[:, [d]], y[:, [d]], "sqeuclidean")
            K *= np.exp(-sq_dist / sigma)
        return K


def as_2d(arr: np.ndarray) -> np.ndarray:
    """Coerce an array to shape (n, d): a 1-D vector becomes a column (n, 1)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    return arr


class BinaryKernel:
    """
    Kronecker delta kernel for binary inputs.
    Evaluates to 1 if elements match, and 0 if they do not.
    """

    def __init__(self) -> None:
        pass

    def fit(self, data: np.ndarray, scale: float = 1.0) -> "BinaryKernel":
        data = as_2d(data)
        values = np.unique(data)
        if data.shape[1] != 1:
            raise ValueError(
                "BinaryKernel currently expects scalar binary treatment, shape (n, 1)."
            )
        if not np.all(np.isin(values, [0.0, 1.0])):
            raise ValueError(f"BinaryKernel expects values in {{0, 1}}, got {values}.")
        return self

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the n×m indicator kernel matrix K(x, y)."""
        x = as_2d(x)
        y = as_2d(y)

        # Super clean, vectorized matching logic:
        # (1 * 1) + (0 * 0) = 1  |  (1 * 0) + (0 * 1) = 0
        res = x.dot(y.T) + (1.0 - x).dot(1.0 - y.T)
        return res
