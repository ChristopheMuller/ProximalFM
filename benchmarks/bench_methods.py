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
import warnings

# Add this at the very top of your script
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from tabicl import TabICLRegressor

    _TABICL_AVAILABLE = True
except ImportError:
    _TABICL_AVAILABLE = False

import sys
import os
import torch


# --- DoPFN Setup ---
dopfn_path = os.path.abspath("DoPFN")
if dopfn_path not in sys.path:
    sys.path.append(dopfn_path)
try:
    from DoPFN.scripts.transformer_prediction_interface.base import DoPFNRegressor

    _DOPFN_AVAILABLE = True
except ImportError:
    _DOPFN_AVAILABLE = False

# --- CausalPFN Setup (with FAISS Monkeypatch) ---
try:
    import os
    # Prevent PyTorch/FAISS thread collisions (crucial for benchmark stability)
    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[k] = "1"
    os.environ["KMP_DUPLICATE_OK"] = "True"

    from causalpfn import CATEEstimator as CausalPFN_Model
    _CAUSALPFN_AVAILABLE = True
except ImportError:
    _CAUSALPFN_AVAILABLE = False

from contextlib import contextmanager


@contextmanager
def silent():
    """Context manager to suppress stdout, stderr, and Python warnings."""
    with open(os.devnull, "w") as devnull:
        # Save the original streams
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        # Redirect both stdout and stderr to the black hole
        sys.stdout = devnull
        sys.stderr = devnull

        # Catch and ignore all PyTorch/Python warnings temporarily
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                yield
            finally:
                # Restore the original streams
                sys.stdout = old_stdout
                sys.stderr = old_stderr

def _switch_src_context(target_repo_name: str, competitor_repo_name: str):
    """
    Forces Python to forget the competitor's 'src' folder and strictly 
    bind 'src' to the target repository.
    """
    import importlib
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    target_path = str(PROJECT_ROOT / target_repo_name)
    competitor_path = str(PROJECT_ROOT / competitor_repo_name)

    # 1. Ban the competitor: Remove it completely from the path
    while competitor_path in sys.path:
        sys.path.remove(competitor_path)

    # 2. Elevate the target: Put it at the absolute front
    if target_path in sys.path:
        sys.path.remove(target_path)
    sys.path.insert(0, target_path)

    # 3. Obliterate the cache: Delete anything related to 'src'
    keys_to_delete = [k for k in sys.modules if k == "src" or k.startswith("src.")]
    for k in keys_to_delete:
        del sys.modules[k]

    # 4. Invalidate filesystem cache: Forces Python to look at sys.path[0] again
    importlib.invalidate_caches()


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
                a = a[:, None]
            parts.append(a)

            if self.interaction_AX:
                parts.append(a * X)

        return np.hstack(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Base interface
# ─────────────────────────────────────────────────────────────────────────────


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
        self.fb = FeatureBuilder(use_W=use_W, use_Z=use_Z, interaction_AX=interactions)
        self.model = base_model

    def fit(self, X, A, Y, *, W=None, Z=None):
        self.model.fit(self.fb.build(X, A=A, W=W, Z=Z), Y)
        return self

    def predict(self, X, *, W=None, Z=None):
        n = X.shape[0]
        y1 = self.model.predict(self.fb.build(X, A=np.ones(n), W=W, Z=Z))
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
        self.fb = FeatureBuilder(use_W=use_W, use_Z=use_Z)
        self.model0 = base_model_factory()
        self.model1 = base_model_factory()

    def fit(self, X, A, Y, *, W=None, Z=None):
        feat = self.fb.build(X, W=W, Z=Z)
        mask0 = A == 0
        mask1 = A == 1
        if mask0.any():
            self.model0.fit(feat[mask0], Y[mask0])
        if mask1.any():
            self.model1.fit(feat[mask1], Y[mask1])
        return self

    def predict(self, X, *, W=None, Z=None):
        feat = self.fb.build(X, W=W, Z=Z)
        return self.model1.predict(feat) - self.model0.predict(feat)


class X_Learner(CATEEstimator):
    """X-Learner: crosses information between treatment arms.

    Stage 1: Fit separate models per treatment arm: f₀(x) and f₁(x).
    Stage 2: Impute pseudo-effects using the opposite arm's model:
             D¹ = Y¹ - f₀(X¹)
             D⁰ = f₁(X⁰) - Y⁰
    Stage 3: Fit models to predict D¹ and D⁰: τ₁(x) and τ₀(x).
    Stage 4: Combine using a propensity score g(x) = P(A=1|X):
             τ̂(x) = g(x)τ₀(x) + (1 - g(x))τ₁(x)

    Args:
        base_model_factory : zero-argument callable returning a fresh sklearn
                             estimator for the base outcomes and CATE models.
        use_W, use_Z       : whether to include proxy variables as features.
    """

    def __init__(
        self,
        base_model_factory: Callable,
        use_W: bool = True,
        use_Z: bool = True,
    ):
        self.fb = FeatureBuilder(use_W=use_W, use_Z=use_Z)

        # Stage 1 models (base response)
        self.model0 = base_model_factory()
        self.model1 = base_model_factory()

        # Stage 2 models (CATE estimators)
        self.tau0 = base_model_factory()
        self.tau1 = base_model_factory()

        # Propensity score model for Stage 3 weighting
        # We use a standard Logistic Regression here, but this could be abstracted
        self.prop_model = LogisticRegression(max_iter=1000)

    def fit(self, X, A, Y, *, W=None, Z=None):
        feat = self.fb.build(X, W=W, Z=Z)
        mask0 = A == 0
        mask1 = A == 1

        if not (mask0.any() and mask1.any()):
            raise ValueError("X_Learner requires samples from both treatment arms.")

        # --- Stage 1: Fit base models ---
        self.model0.fit(feat[mask0], Y[mask0])
        self.model1.fit(feat[mask1], Y[mask1])

        # --- Stage 2: Impute pseudo-effects ---
        # For controls, predict counterfactual under treatment
        D0 = self.model1.predict(feat[mask0]) - Y[mask0]

        # For treated, predict counterfactual under control
        D1 = Y[mask1] - self.model0.predict(feat[mask1])

        # --- Stage 3: Fit CATE models on imputed effects ---
        self.tau0.fit(feat[mask0], D0)
        self.tau1.fit(feat[mask1], D1)

        # --- Fit propensity score model for the weighting ---
        self.prop_model.fit(feat, A)

        return self

    def predict(self, X, *, W=None, Z=None):
        feat = self.fb.build(X, W=W, Z=Z)

        pred_tau0 = self.tau0.predict(feat)
        pred_tau1 = self.tau1.predict(feat)

        g = self.prop_model.predict_proba(feat)[:, 1]

        return g * pred_tau0 + (1 - g) * pred_tau1


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


class NMMR_Wrappers(CATEEstimator):
    def __init__(self, model_type: str, estimate_cate: bool, **kwargs):

        _switch_src_context(target_repo_name="NMMR", competitor_repo_name="CausalFM")

        from src.data.ate.data_class import PVTrainDataSet
        from src.models.PMMR.model import PMMRModel
        from src.models.kernelPV.model import KernelPVModel
        self.model_type = model_type
        self.estimate_cate = estimate_cate

        self._PVTrainDataSet = PVTrainDataSet

        # Extract only the kwargs meant for NMMR (ignore framework keys)
        safe_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["name", "cls", "model_type", "use_W", "use_Z"]
        }

        if model_type == "PMMR":
            self.model = PMMRModel(**safe_kwargs)
        elif model_type == "KernelPV":
            self.model = KernelPVModel(**safe_kwargs)
        else:
            raise ValueError(f"Unknown NMMR model type: {model_type}")

    def fit(self, X, A, Y, *, W=None, Z=None):
        if W is None or Z is None:
            raise ValueError("NMMR methods require both W and Z proxies.")

        train_data = self._PVTrainDataSet(
            treatment=np.asarray(A).reshape(-1, 1),
            treatment_proxy=np.asarray(Z),
            outcome_proxy=np.asarray(W),
            outcome=np.asarray(Y),
            backdoor=np.asarray(X),
        )

        self.model.fit(train_data, data_name="bench")
        return self

    def predict(self, X, *, W=None, Z=None):
        n = X.shape[0]

        t0 = np.zeros((n, 1))
        t1 = np.ones((n, 1))

        if self.estimate_cate:
            predsY0 = self.model.predict(treatment=t0, backdoor=np.asarray(X))
            predsY1 = self.model.predict(treatment=t1, backdoor=np.asarray(X))

            return (predsY1 - predsY0).squeeze()

        predsY0 = self.model.predict(treatment=t0)
        predsY1 = self.model.predict(treatment=t1)

        return (predsY1 - predsY0).squeeze()


# Inside bench_methods.py


class NMMR_Estimator(CATEEstimator):
    def __init__(self, **kwargs):
        # Local import to prevent global path collisions
        import sys
        from pathlib import Path

        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        nmmr_path = str(PROJECT_ROOT / "NMMR")

        if nmmr_path not in sys.path:
            sys.path.insert(0, nmmr_path)

        from src.models.NMMR.NMMR_wrapper import NMMRCATEEstimator

        train_params = {
            "network_width": kwargs.get("network_width", 64),
            "network_depth": kwargs.get("network_depth", 3),
            "n_epochs": kwargs.get("n_epochs", 100),
            "batch_size": kwargs.get("batch_size", 256),
            "learning_rate": kwargs.get("learning_rate", 1e-3),
            "loss_name": kwargs.get("loss_name", "U_statistic"),
        }

        self.model = NMMRCATEEstimator(train_params=train_params)

    def fit(self, X, A, Y, *, W=None, Z=None):
        if W is None or Z is None:
            raise ValueError("NMMR requires W and Z proxies.")
        self.model.fit(X, A, Y, W, Z)
        return self

    def predict(self, X, *, W=None, Z=None):
        return self.model.predict(X)


class DFPV_Estimator(CATEEstimator):
    """Wrapper for the Deep Feature Proxy Variable (DFPV) Model."""

    def __init__(self, **kwargs):
        import sys
        from pathlib import Path

        PROJECT_ROOT = Path(__file__).resolve().parent.parent

        # 1. Isolate the environment for DFPV (Assuming it shares the NMMR repo structure)
        dfpv_repo_path = str(PROJECT_ROOT / "NMMR")

        if dfpv_repo_path not in sys.path:
            sys.path.insert(0, dfpv_repo_path)

        # 2. Purge the 'src' cache to prevent module collisions
        keys = [k for k in sys.modules if k == "src" or k.startswith("src.")]
        for k in keys:
            del sys.modules[k]

        # 3. Import classes directly in __init__ after environment is prepped
        from src.data.ate.data_class import PVTrainDataSet
        from src.models.DFPV.model import DFPVModel

        import torch  # Make sure torch is imported here

        torch.set_num_threads(1)

        self._PVTrainDataSet = PVTrainDataSet
        self._DFPVModel = DFPVModel

        self.lam1 = kwargs.get("lam1", 0.1)
        self.lam2 = kwargs.get("lam2", 0.1)
        self.feat_dim = kwargs.get("feat_dim", 16)
        self.hidden_dim = kwargs.get("hidden_dim", 32)
        self.split_ratio = kwargs.get("split_ratio", 0.5)
        self.add_stage1_intercept = kwargs.get("add_stage1_intercept", True)
        self.add_stage2_intercept = kwargs.get("add_stage2_intercept", True)

        self.model = None

    def _build_mlp(self, in_dim, out_dim):
        """Constructs a simple MLP for the DFPV feature extractors."""
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, out_dim),
        )

    def fit(self, X, A, Y, *, W=None, Z=None):
        if W is None or Z is None:
            raise ValueError("DFPV requires both W and Z proxies.")

        n = X.shape[0]

        # Split data into Stage 1 and Stage 2
        indices = np.random.permutation(n)
        split_idx = int(n * self.split_ratio)
        idx1, idx2 = indices[:split_idx], indices[split_idx:]

        A_2d = np.asarray(A).reshape(-1, 1)
        Y_2d = np.asarray(Y).reshape(-1, 1)
        X_2d = np.asarray(X)
        W_2d = np.asarray(W)
        Z_2d = np.asarray(Z)

        # Use the class loaded in __init__
        train1 = self._PVTrainDataSet(
            treatment=A_2d[idx1],
            treatment_proxy=Z_2d[idx1],
            outcome_proxy=W_2d[idx1],
            outcome=Y_2d[idx1],
            backdoor=X_2d[idx1],
        )

        train2 = self._PVTrainDataSet(
            treatment=A_2d[idx2],
            treatment_proxy=Z_2d[idx2],
            outcome_proxy=W_2d[idx2],
            outcome=Y_2d[idx2],
            backdoor=X_2d[idx2],
        )

        dim_A = A_2d.shape[1]
        dim_X = X_2d.shape[1]
        dim_W = W_2d.shape[1]
        dim_Z = Z_2d.shape[1]

        # Instantiate the DFPV model using the class loaded in __init__
        self.model = self._DFPVModel(
            treatment_1st_net=self._build_mlp(dim_A, self.feat_dim),
            treatment_2nd_net=self._build_mlp(dim_A, self.feat_dim),
            treatment_proxy_net=self._build_mlp(dim_Z, self.feat_dim),
            outcome_proxy_net=self._build_mlp(dim_W, self.feat_dim),
            backdoor_1st_net=self._build_mlp(dim_X, self.feat_dim),
            backdoor_2nd_net=self._build_mlp(dim_X, self.feat_dim),
            add_stage1_intercept=self.add_stage1_intercept,
            add_stage2_intercept=self.add_stage2_intercept,
        )

        self.model.fit(train1, train2, self.lam1, self.lam2)
        return self

    def predict(self, X, *, W=None, Z=None):
        if self.model is None:
            raise RuntimeError("Must call fit() before predict().")

        n = X.shape[0]

        A1 = np.ones((n, 1))
        A0 = np.zeros((n, 1))

        y1 = self.model.predict(A1)
        y0 = self.model.predict(A0)

        return (y1 - y0).squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# Foundation Model Wrappers
# ─────────────────────────────────────────────────────────────────────────────

_CAUSALFM_CACHE = {}
_CAUSALPFN_CACHE = {}


class CausalFM_Estimator(CATEEstimator):
    def __init__(self, checkpoint_path: str, use_W: bool = True, use_Z: bool = True):
        
        # Lock in CausalFM, kick out NMMR
        _switch_src_context(target_repo_name="CausalFM", competitor_repo_name="NMMR")

        try:
            from causalfm.models import StandardCATEModel
        except ImportError as e:
            raise ImportError(f"CausalFM import failed even after switch: {e}")

        torch.set_num_threads(1)

        if checkpoint_path not in _CAUSALFM_CACHE:
            with silent():
                _CAUSALFM_CACHE[checkpoint_path] = StandardCATEModel.from_pretrained(
                    checkpoint_path
                )

        self.model = _CAUSALFM_CACHE[checkpoint_path]
        self.fb = FeatureBuilder(use_W=use_W, use_Z=use_Z, interaction_AX=False)

        self.ctx_x = None
        self.ctx_a = None
        self.ctx_y = None

    def fit(self, X, A, Y, *, W=None, Z=None):
        # Build unified covariate matrix (stacking X, W, Z if configured)
        feat = self.fb.build(X, W=W, Z=Z)

        # Store as PyTorch tensors with correct shapes (n, 1) for A and Y
        self.ctx_x = torch.Tensor(feat)
        self.ctx_a = torch.Tensor(A).reshape(-1, 1)
        self.ctx_y = torch.Tensor(Y).reshape(-1, 1)
        return self

    def predict(self, X, *, W=None, Z=None):
        if self.ctx_x is None:
            raise ValueError("Must call fit() before predict().")

        # Build test covariate matrix
        feat = self.fb.build(X, W=W, Z=Z)
        x_test = torch.Tensor(feat)

        with torch.no_grad():
            result = self.model.estimate_cate(
                self.ctx_x, self.ctx_a, self.ctx_y, x_test
            )

        # Convert back to numpy array of shape (n,)
        return result["cate"].cpu().numpy().squeeze()


class DoPFN_Estimator(CATEEstimator):
    """Wrapper for DoPFNRegressor.

    Uses in-context learning by storing the training data via .fit() and
    evaluating counterfactuals during .predict().
    """

    def __init__(
        self,
        use_W: bool = True,
        use_Z: bool = True,
    ):
        if not _DOPFN_AVAILABLE:
            raise ImportError(
                "DoPFN is not installed or cannot be found. "
                "Ensure it is installed and sys.path is correct."
            )

        torch.set_num_threads(1)

        with silent():
            self.model = DoPFNRegressor()

        self.fb = FeatureBuilder(use_W=use_W, use_Z=use_Z, interaction_AX=False)

    def fit(self, X, A, Y, *, W=None, Z=None):
        feat = self.fb.build(X, A=A, W=W, Z=Z)

        with silent():
            self.model.fit(feat, Y)
        return self

    def predict(self, X, *, W=None, Z=None):
        n = X.shape[0]

        # Build counterfactual covariate matrices
        feat1 = self.fb.build(X, A=np.ones(n), W=W, Z=Z)
        feat0 = self.fb.build(X, A=np.zeros(n), W=W, Z=Z)

        with silent():
            with torch.no_grad():
                pred1 = self.model.predict(torch.Tensor(feat1))
                pred0 = self.model.predict(torch.Tensor(feat0))

        # Convert to numpy and calculate CATE (Y1 - Y0)
        if isinstance(pred1, torch.Tensor):
            pred1 = pred1.detach().cpu().numpy()
        if isinstance(pred0, torch.Tensor):
            pred0 = pred0.detach().cpu().numpy()

        return (pred1 - pred0).squeeze()


class CausalPFN_Estimator(CATEEstimator):
    def __init__(self, use_W: bool = True, use_Z: bool = True):
        if not _CAUSALPFN_AVAILABLE:
            raise ImportError("causalpfn is not installed.")

        import torch
        torch.set_num_threads(1)

        if "default" not in _CAUSALPFN_CACHE:
            with silent():
                _CAUSALPFN_CACHE["default"] = CausalPFN_Model(device="cpu", verbose=False)

        self.model = _CAUSALPFN_CACHE["default"]
        self.fb = FeatureBuilder(use_W=use_W, use_Z=use_Z, interaction_AX=False)

    def fit(self, X, A, Y, *, W=None, Z=None):
        feat = self.fb.build(X, W=W, Z=Z)

        X_train = np.ascontiguousarray(feat, dtype=np.float32)
        A_train = np.ascontiguousarray(A.ravel(), dtype=np.float32)
        Y_train = np.ascontiguousarray(Y.ravel(), dtype=np.float32)

        with silent():
            self.model.fit(X_train, A_train, Y_train)
        return self

    def predict(self, X, *, W=None, Z=None):
        feat = self.fb.build(X, W=W, Z=Z)
        X_test = np.ascontiguousarray(feat, dtype=np.float32)

        with silent():
            estimated_cate = self.model.estimate_cate(X_test)

        return np.asarray(estimated_cate).squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# Base-model registry + factory
# ─────────────────────────────────────────────────────────────────────────────


def _make_base_model(name: str):
    """Return a fresh sklearn estimator by short name.

    Why a factory function rather than a class dict?
        RandomForestRegressor must not share random_state across instances
        or arms — returning a new object each call prevents accidental aliasing.
    """
    if name == "tabicl" and not _TABICL_AVAILABLE:
        raise ImportError("tabicl is not installed.")

    factories: dict[str, Callable] = {
        "linear": LinearRegression,
        "rf": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "tabicl": TabICLRegressor if _TABICL_AVAILABLE else lambda: None,
    }

    if name not in factories:
        raise ValueError(
            f"Unknown base model {name!r}. Choices: 'linear', 'rf', 'tabicl'."
        )

    return factories[name]()


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
    cls = cfg["cls"]

    factories: dict[str, Callable[[], CATEEstimator]] = {
        "S_Learner": lambda: S_Learner(
            base_model=_make_base_model(cfg.get("base", "linear")),
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
            interactions=cfg.get("interactions", False),
        ),
        "T_Learner": lambda: T_Learner(
            base_model_factory=lambda bn=cfg.get("base", "linear"): _make_base_model(
                bn
            ),
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
        ),
        "X_Learner": lambda: X_Learner(
            base_model_factory=lambda bn=cfg.get("base", "linear"): _make_base_model(
                bn
            ),
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
        ),
        "P2SLS": lambda: P2SLS(interactions=cfg.get("interactions", False)),
        "CausalFM": lambda: CausalFM_Estimator(
            checkpoint_path=cfg["checkpoint"],
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
        ),
        "DoPFN": lambda: DoPFN_Estimator(
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
        ),
        "CausalPFN": lambda: CausalPFN_Estimator(
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
        ),
        "NMMRWrappers": lambda: NMMR_Wrappers(
            model_type=cfg.get("model_type", "PMMR"),
            estimate_cate=cfg.get("estimate_cate", True),
            **{
                k: v for k, v in cfg.items() if k not in ["model_type", "estimate_cate"]
            },
        ),
        "NMMR": lambda: NMMR_Estimator(
            **{k: v for k, v in cfg.items() if k not in ["name", "cls"]}
        ),
        "DFPV": lambda: DFPV_Estimator(
            **{k: v for k, v in cfg.items() if k not in ["name", "cls"]}
        ),
    }
    if cls not in factories:
        raise ValueError(
            f"Unknown method class {cls!r}. "
            f"Valid choices: {', '.join(map(repr, factories.keys()))}."
        )

    return name, factories[cls]()
