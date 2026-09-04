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

    def diagnostics(self) -> dict[str, float]:
        """Scalar diagnostics from the most recent ``fit``, keyed by name.

        A side channel alongside predictions: not something scored against a
        target, but something worth tracking per episode about *how* the fit
        behaved — a regularizer hitting a grid edge, a clip binding, a solver
        falling back. Default is empty; an estimator opts in by overriding
        this and populating whatever instance state it needs during ``fit``.

        One fit produces one diagnostics dict, shared by every output of
        :meth:`predict_variants` (the same way one fit shares one wall-clock
        time across variants) — diagnostics describe the fit, not a
        particular output. Values must be finite floats: the runner stores
        one array per key across episodes, the same way it stores ``times``,
        so a key that only some episodes emit is padded with NaN elsewhere.
        """
        return {}

    def tuning_time(self) -> float:
        """Seconds spent selecting hyperparameters during the latest ``fit``.

        The benchmark's total episode time includes construction, tuning, the
        final full-context fit, and prediction. Estimators with an explicit
        search override this method so the runner can additionally report the
        counterfactual post-selection latency ``total - tuning``. Untuned
        estimators correctly default to zero.
        """
        return 0.0
