"""base.py — the one interface every benchmark method implements.

Split out of ``__init__`` so an estimator that lives in its own module (rather
than as a thin wrapper in the registry) can subclass it without importing the
package it is part of.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["CATEEstimator", "FeatureBuilder"]


class FeatureBuilder:
    """Centralises feature-stacking so every estimator sees the same matrix.

    Having this as a standalone object (rather than a method on each class)
    guarantees that S_Learner and T_Learner build identical feature vectors
    for a given config — important for fair comparisons.

    Args:
        use_W          : include outcome proxy W columns.
        use_Z          : include treatment proxy Z columns.
        interaction_AX : append element-wise A * X_j for each covariate.
                         Useful in stage-2 of 2SLS to allow heterogeneous
                         treatment effects.
    """

    def __init__(
        self,
        use_W: bool = True,
        use_Z: bool = True,
        interaction_AX: bool = False,
    ):
        self.use_W = use_W
        self.use_Z = use_Z
        self.interaction_AX = interaction_AX

    def build(
        self,
        X: np.ndarray,
        *,
        A: Optional[np.ndarray] = None,
        W: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Stack selected columns into one (n, d) design matrix.

        Args:
            X : covariates,          shape (n, d_x).
            A : treatment indicator, shape (n,) or (n, 1).  Optional.
            W : outcome proxy,       shape (n, d_w).         Optional.
            Z : treatment proxy,     shape (n, d_z).         Optional.

        Returns:
            np.ndarray of shape (n, d_total).
        """
        parts: list[np.ndarray] = [X]

        if self.use_W and W is not None:
            parts.append(W)
        if self.use_Z and Z is not None:
            parts.append(Z)

        if A is not None:
            a = np.asarray(A, dtype=float)
            if a.ndim == 1:
                a = a[:, None]
            parts.append(a)

            if self.interaction_AX:
                parts.append(a * X)

        return np.hstack(parts)


class CATEEstimator:
    def fit(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        *,
        W: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
    ) -> "CATEEstimator":
        raise NotImplementedError

    def predict(
        self,
        X: np.ndarray,
        *,
        W: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return estimated ITE (τ̂) for each row of X, shape (n,)."""
        raise NotImplementedError

    def predict_variants(
        self,
        X: np.ndarray,
        *,
        W: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
    ) -> dict[Optional[str], np.ndarray]:
        """Return one or more named prediction sets from a single fitted model.

        This is the generic "one fit → many outputs" entry point used by the
        benchmark runner. The default implementation yields a single output
        under the key ``None``; the runner maps that to the config's ``name``.

        Multi-output estimators (e.g. NMMR producing paired / independent /
        conditional CATE from one trained network) override this to return
        several arrays keyed by their full output display names, reusing the
        model trained in a single ``fit()`` call — no retraining per variant.
        """
        return {None: self.predict(X, W=W, Z=Z)}
