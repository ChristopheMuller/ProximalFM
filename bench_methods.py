"""
bench_methods.py — CATE estimator classes for the benchmark.

Design principles
─────────────────
1. Single interface  Every estimator exposes only fit() and predict().
                     The run script never needs to know what's inside.

2. Centralised feature construction
                     FeatureBuilder owns all logic for stacking X, W, Z,
                     A, and interactions.  Subclasses call it; they never
                     duplicate np.hstack() logic.

3. Configuration over inheritance
                     Method variants (e.g. "S-OLS with proxies" vs
                     "S-OLS without proxies") are created by passing
                     different kwargs — not by writing new subclasses.
                     build_method() translates a YAML config dict into
                     a concrete estimator instance.

4. Fail loudly       Missing dependencies (tabicl) raise an ImportError
                     with an actionable message, not a silent fallback.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from tabicl import TabICLRegressor
    _TABICL_AVAILABLE = True
except ImportError:
    _TABICL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Feature construction
# ─────────────────────────────────────────────────────────────────────────────

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
                a = a[:, None]          # (n,) → (n, 1)
            parts.append(a)

            if self.interaction_AX:
                parts.append(a * X)     # (n, d_x) — effect heterogeneity terms

        return np.hstack(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Base interface
# ─────────────────────────────────────────────────────────────────────────────

class CATEEstimator:
    """Abstract base class.  Subclasses must implement fit() and predict()."""

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


# ─────────────────────────────────────────────────────────────────────────────
# Meta-learners
# ─────────────────────────────────────────────────────────────────────────────

class S_Learner(CATEEstimator):
    """S-Learner: one model sees (X, [W], [Z], A) and predicts Y.

    CATE estimate: τ̂(x) = f(x, 1) - f(x, 0).

    Args:
        base_model : a fitted sklearn-compatible estimator.
        use_W, use_Z: whether to include proxy variables as features.
        interactions: whether to include A×X interaction terms.
    """

    def __init__(
        self,
        base_model,
        use_W: bool = True,
        use_Z: bool = True,
        interactions: bool = False,
    ):
        self.fb    = FeatureBuilder(use_W=use_W, use_Z=use_Z, interaction_AX=interactions) 
        self.model = base_model

    def fit(self, X, A, Y, *, W=None, Z=None):
        self.model.fit(self.fb.build(X, A=A, W=W, Z=Z), Y)
        return self

    def predict(self, X, *, W=None, Z=None):
        n = X.shape[0]
        y1 = self.model.predict(self.fb.build(X, A=np.ones(n),  W=W, Z=Z))
        y0 = self.model.predict(self.fb.build(X, A=np.zeros(n), W=W, Z=Z))
        return y1 - y0


class T_Learner(CATEEstimator):
    """T-Learner: separate model per treatment arm.

    CATE estimate: τ̂(x) = f₁(x) - f₀(x).

    Args:
        base_model_factory : zero-argument callable that returns a fresh
                             sklearn estimator.  Called twice (one per arm)
                             so the two models are truly independent.
        use_W, use_Z       : whether to include proxy variables as features.
                             Note: A is *not* a feature for T-learner (it
                             defines the sample split, not a column).
    """

    def __init__(
        self,
        base_model_factory: Callable,
        use_W: bool = True,
        use_Z: bool = True,
    ):
        self.fb     = FeatureBuilder(use_W=use_W, use_Z=use_Z)
        self.model0 = base_model_factory()
        self.model1 = base_model_factory()

    def fit(self, X, A, Y, *, W=None, Z=None):
        feat   = self.fb.build(X, W=W, Z=Z)   # A excluded
        mask0  = (A == 0)
        mask1  = (A == 1)
        if mask0.any():
            self.model0.fit(feat[mask0], Y[mask0])
        if mask1.any():
            self.model1.fit(feat[mask1], Y[mask1])
        return self

    def predict(self, X, *, W=None, Z=None):
        feat = self.fb.build(X, W=W, Z=Z)
        return self.model1.predict(feat) - self.model0.predict(feat)


# ─────────────────────────────────────────────────────────────────────────────
# Proximal 2SLS
# ─────────────────────────────────────────────────────────────────────────────

class P2SLS(CATEEstimator):
    """Proximal Two-Stage Least Squares causal estimator.
 
    Notation
    ────────
        Z — treatment-inducing proxy  (instrument; correlated with A, not Y|U,X)
        W — outcome-inducing proxy    (pre-treatment; correlated with U, not A|U,X)
        U — unmeasured confounder
 
    Stage 1  (nuisance / bridge function)
    ──────────────────────────────────────
        Ŵ = α₀ + α_A·A + α_X·X + α_Z·Z  [+ α_AX·(A⊗X)  if interactions]
 
        Why A⊗X in stage 1 when interactions=True?
            Standard 2SLS requires the first stage to include every exogenous
            regressor that appears in the structural (second-stage) equation.
            When interactions=True, A⊗X enters stage 2 as an exogenous column.
            Omitting it from stage 1 causes the stage-1 residuals to correlate
            with A⊗X, violating the orthogonality condition needed for
            consistent estimation of the bridge function coefficients.
 
    Stage 2  (outcome / structural equation)
    ─────────────────────────────────────────
        Y = β₀ + β_A·A + β_X·X + β_W·Ŵ  [+ β_AX·(A⊗X)  if interactions]
 
        Substituting Ŵ removes the confounding path through W.
        β_A is the ATE; β_AX encodes effect heterogeneity in X.
 
    CATE identification and the pre-treatment cancellation
    ───────────────────────────────────────────────────────
        W is measured *before* treatment, so the potential outcome W(a) = W
        for all a — intervening on A does not change W.  This means that in
        the structural equation the term β_W·W is the *same* under A=1 and A=0:
 
            CATE(x, w) = h(w, 1, x) − h(w, 0, x)
                       = (β_A + β_X·x + β_W·w + β_AX·x)
                         − (β_X·x + β_W·w)
                       = β_A + β_AX·x
 
        The β_W·w terms cancel exactly.  The CATE depends only on X, not on W.
 
        Consequence for predict():
            We must use the *same* W (or Ŵ) for both counterfactuals.
            Using W_hat(A=1) for the A=1 branch and W_hat(A=0) for the A=0
            branch introduces a spurious term  β_W·(Ŵ₁ − Ŵ₀)  that has no
            causal interpretation and biases the CATE estimate.
 
    Args:
        interactions : if True, append A×X columns to *both* stage 1 and
                       stage 2 feature matrices.
    """
 
    def __init__(self, interactions: bool = False):
        self.interactions = interactions
        self.stage1 = LinearRegression()
        self.stage2 = LinearRegression()
 
    # ── internal helpers ──────────────────────────────────────────────────
 
    def _s1_features(self, A: np.ndarray, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """[A, X, Z, (A⊗X)] — same column set at fit and predict time.
 
        A⊗X is included whenever interactions=True because every exogenous
        column that appears in stage 2 must appear in stage 1 (2SLS requirement).
        """
        a = A.reshape(-1, 1)
        parts = [a, X, Z]
        if self.interactions:
            parts.append(a * X)
        return np.hstack(parts)
 
    def _s2_features(self, A: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """[A, X, W, (A⊗X)] — W here is either Ŵ (training) or W_fixed (predict)."""
        a = A.reshape(-1, 1)
        parts = [a, X, W]
        if self.interactions:
            parts.append(a * X)
        return np.hstack(parts)
 
    # ── public interface ──────────────────────────────────────────────────
 
    def fit(self, X, A, Y, *, W=None, Z=None):
        if W is None or Z is None:
            raise ValueError(
                "P2SLS requires both W (outcome proxy) and Z (treatment proxy). "
                "Switch to S/T-Learner for proxy-free baselines."
            )
        # Stage 1: project W onto all exogenous variables (A, X, Z, and A⊗X)
        W_hat = self.stage1.fit(self._s1_features(A, X, Z), W).predict(
            self._s1_features(A, X, Z)
        )
        # Stage 2: regress Y on treatment, covariates, and instrumented proxy
        self.stage2.fit(self._s2_features(A, X, W_hat), Y)
        return self
 
    def predict(self, X, *, W=None, Z=None) -> np.ndarray:
        """Estimate CATE as h(W_fixed, 1, X) − h(W_fixed, 0, X).
 
        W_fixed is held constant across both counterfactuals, reflecting the
        fact that W is pre-treatment and does not respond to A.
 
        Priority for W_fixed:
            1. Observed W_q  — use directly; it cancels in the contrast anyway.
            2. Z available   — predict Ŵ from stage 1 using a neutral A=0
                               (any fixed A works since the β_W term cancels).
            3. Neither       — raise, because we cannot form a valid contrast.
 
        Note: because β_W·W_fixed cancels in the difference, the CATE result
        is numerically identical regardless of which W_fixed is chosen.
        The choice matters only for the *level* of Ŷ(a), not the contrast.
        """
        if W is None and Z is None:
            raise ValueError(
                "P2SLS.predict() needs at least W (observed proxy) or Z "
                "(to predict Ŵ).  Both are None."
            )
 
        n = X.shape[0]
 
        if W is not None:
            # Observed pre-treatment proxy — preferred, no stage-1 needed
            W_fixed = W
        else:
            # Predict Ŵ from Z; use A=0 as neutral value (cancels in contrast)
            W_fixed = self.stage1.predict(self._s1_features(np.zeros(n), X, Z))
 
        A1 = np.ones(n)
        A0 = np.zeros(n)
 
        # Same W_fixed for both branches — spurious β_W·(Ŵ₁ − Ŵ₀) term is gone
        y1 = self.stage2.predict(self._s2_features(A1, X, W_fixed))
        y0 = self.stage2.predict(self._s2_features(A0, X, W_fixed))
        return y1 - y0


# ─────────────────────────────────────────────────────────────────────────────
# Base-model registry + factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_base_model(name: str):
    """Return a fresh sklearn estimator by short name.

    Why a factory function rather than a class dict?
        RandomForestRegressor must not share random_state across instances
        or arms — returning a new object each call prevents accidental aliasing.
    """
    if name == "linear":
        return LinearRegression()
    if name == "rf":
        return RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    if name == "tabicl":
        if not _TABICL_AVAILABLE:
            raise ImportError(
                "tabicl is not installed. Run `pip install tabicl` or remove "
                "TabICL methods from bench_config.yaml."
            )
        return TabICLRegressor()
    raise ValueError(
        f"Unknown base model {name!r}. Choices: 'linear', 'rf', 'tabicl'."
    )


def build_method(cfg: dict) -> tuple[str, CATEEstimator]:
    """Instantiate a CATE estimator from a single YAML method-config dict.

    This is the bridge between the config file and the estimator objects.
    The run script calls this; it never imports individual estimator classes.

    Args:
        cfg : one entry from the 'methods' list in bench_config.yaml.

    Returns:
        (display_name, estimator_instance)
    """
    name = cfg["name"]
    cls  = cfg["cls"]

    if cls == "S_Learner":
        method = S_Learner(
            base_model=_make_base_model(cfg.get("base", "linear")),
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
            interactions=cfg.get("interactions", False), 
        )

    elif cls == "T_Learner":
        base_name = cfg.get("base", "linear")
        # Pass a factory (not an instance) so T_Learner can create two
        # independent model objects.  The lambda captures base_name by value.
        method = T_Learner(
            base_model_factory=lambda bn=base_name: _make_base_model(bn),
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
        )

    elif cls == "P2SLS":
        method = P2SLS(interactions=cfg.get("interactions", False))

    else:
        raise ValueError(
            f"Unknown method class {cls!r}. "
            "Valid choices: 'S_Learner', 'T_Learner', 'P2SLS'."
        )

    return name, method