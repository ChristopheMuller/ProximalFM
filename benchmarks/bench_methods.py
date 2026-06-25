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
    """Proximal Two-Stage Least Squares causal estimator for ATE.

    Notation
    ────────
        Z — treatment-inducing proxy  (correlated with A, but not Y | U, X)
        W — outcome-inducing proxy    (pre-treatment; correlated with U, not A | U, X)
        U — unmeasured confounder

    Stage 1  (proxy bridge / nuisance regression)
    ─────────────────────────────────────────────
        W = α₀ + α_A·A + α_X·X + α_Z·Z + ε

        We estimate E[W | A, X, Z] by linear regression and obtain W_hat.

    Stage 2  (outcome / structural equation)
    ────────────────────────────────────────
        Y = β₀ + β_A·A + β_X·X + β_W·W_hat + η

        The coefficient β_A is the ATE under the linear proximal 2SLS model.
    """

    def __init__(self):
        self.stage1 = LinearRegression()
        self.stage2 = LinearRegression()
        self.ate_ = None

    def _s1_features(self, A: np.ndarray, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Return [A, X, Z] for the first stage."""
        a = A.reshape(-1, 1)
        return np.hstack([a, X, Z])

    def _s2_features(self, A: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Return [A, X, W] for the second stage."""
        a = A.reshape(-1, 1)
        W = np.asarray(W)
        if W.ndim == 1:
            W = W.reshape(-1, 1)
        return np.hstack([a, X, W])

    def fit(self, X, A, Y, *, W=None, Z=None):
        if W is None or Z is None:
            raise ValueError(
                "P2SLS requires both W (outcome proxy) and Z (treatment proxy)."
            )

        s1 = self._s1_features(A, X, Z)
        W_hat = self.stage1.fit(s1, W).predict(s1)

        s2 = self._s2_features(A, X, W_hat)
        self.stage2.fit(s2, Y)

        self.ate_ = float(self.stage2.coef_.reshape(-1)[0])
        return self

    def predict(self, X, **kwargs) -> np.ndarray:
        """Return the estimated ATE for every observation."""
        if self.ate_ is None:
            raise ValueError("Call fit() before predict().")

        return np.full(X.shape[0], self.ate_)


from plearner import BridgeParams, PLearner as BasePLearner, RegressorMixin


class ProximalPLearner(CATEEstimator):
    """Wrapper for the Ghassami & Sverdrup style proximal P-Learner.

    Adapts the native cross-fitting and kernel bridge estimation steps
    to match the benchmark's unified API interface.
    """

    def __init__(
        self,
        bridge_params: Optional[BridgeParams] = None,
        final_model: Optional[RegressorMixin] = None,
        n_splits: int = 3,
        random_state: Optional[int] = None,
        q_clip: Optional[tuple[float, float]] = None,
        refit_full_nuisance: bool = False,
    ):
        self.bridge_params = (
            bridge_params if bridge_params is not None else BridgeParams()
        )
        self.final_model = final_model
        self.n_splits = n_splits
        self.random_state = random_state
        self.q_clip = q_clip
        self.refit_full_nuisance = refit_full_nuisance

        # Instantiate the underlying execution engine
        self.estimator_ = BasePLearner(
            bridge_params=self.bridge_params,
            final_model=self.final_model,
            n_splits=self.n_splits,
            random_state=self.random_state,
            q_clip=self.q_clip,
            refit_full_nuisance=self.refit_full_nuisance,
        )

    def fit(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        *,
        W: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
    ) -> "ProximalPLearner":
        """Unpack proxy arguments and route them to the native P-Learner signature."""
        if W is None or Z is None:
            raise ValueError(
                "ProximalPLearner requires both an outcome proxy (W) and a treatment proxy (Z)."
            )

        # Route arguments to fit(X, Z, W, A, Y)
        self.estimator_.fit(X=X, Z=Z, W=W, A=A, Y=Y)
        return self

    def predict(
        self,
        X: np.ndarray,
        *,
        W: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict the CATE (tau) for covariates X."""
        return self.estimator_.predict(X)


class KPV_Wrapper(CATEEstimator):
    def __init__(self, **kwargs):
        from benchmarks.baselines.kpv import KPVConfig, KPVEstimator

        cfg = KPVConfig(
            split_ratio=kwargs.get("split_ratio", 0.5),
            kernel_scale=kwargs.get("kernel_scale", 1.0),
            lam1=kwargs.get("lam1", None),
            lam1_min=kwargs.get("lam1_min", 1e-5),
            lam1_max=kwargs.get("lam1_max", 1e1),
            n_lam1_search=kwargs.get("n_lam1_search", 10),
            lam2=kwargs.get("lam2", None),
            lam2_min=kwargs.get("lam2_min", 1e-5),
            lam2_max=kwargs.get("lam2_max", 1e1),
            n_lam2_search=kwargs.get("n_lam2_search", 10),
            seed=kwargs.get("seed", 42),
        )
        self.model = KPVEstimator(cfg)

    def fit(self, X, A, Y, *, W=None, Z=None):
        if W is None or Z is None:
            raise ValueError("KPV requires W and Z proxies.")
        self.model.fit(X, A, Y, W=W, Z=Z)
        return self

    def predict(self, X, *, W=None, Z=None):
        return self.model.predict(X)


class PMMR_Wrapper(CATEEstimator):
    # Which prediction target the estimator returns. See PMMREstimator.predict:
    #   "marginal"    — global ATE, repeated for every test row.
    #   "paired"      — CATE(x, w) at the supplied test (X, W) pairs (needs W).
    #   "independent" — CATE(x) marginalising W over its marginal (assumes W ⊥ X).
    #   "conditional" — CATE(x) marginalising W over W | X (assumes W correlated with X).
    _VALID_STRATEGIES = {"marginal", "paired", "independent", "conditional"}

    def __init__(self, **kwargs):
        from benchmarks.baselines.pmmr import PMMRConfig, PMMREstimator

        cfg = PMMRConfig(
            kernel_scale=kwargs.get("kernel_scale", 1.0),
            lam1=kwargs.get("lam1", 1.0),
            lam2=kwargs.get("lam2", 1e-4),
            seed=kwargs.get("seed", 42),
        )
        self.model = PMMREstimator(cfg)

        self.strategy = kwargs.get("strategy", "marginal")
        if self.strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Unknown PMMR strategy {self.strategy!r}; "
                f"choose from {sorted(self._VALID_STRATEGIES)}."
            )

    def fit(self, X, A, Y, *, W=None, Z=None):
        if W is None or Z is None:
            raise ValueError("PMMR requires W and Z proxies.")
        self.model.fit(X, A, Y, W=W, Z=Z)
        return self

    def predict(self, X, *, W=None, Z=None):
        # Only the paired contrast consumes the query-time W; the other
        # strategies marginalise W internally and ignore it.
        if self.strategy == "paired":
            return self.model.predict(X, W=W, strategy="paired")
        return self.model.predict(X, strategy=self.strategy)


class NMMR_Wrapper(CATEEstimator):
    # Which prediction target the estimator returns. See NMMREstimator.predict:
    #   "paired"      — CATE(x, w) at the supplied test (X, W) pairs (needs W).
    #   "independent" — CATE(x) marginalising W over its marginal (assumes W ⊥ X).
    #   "conditional" — CATE(x) marginalising W over W | X (assumes W correlated with X).
    _VALID_STRATEGIES = {"paired", "independent", "conditional"}

    def __init__(self, **kwargs):
        from benchmarks.baselines.nmmr import NMMRConfig, NMMREstimator

        cfg = NMMRConfig(
            hidden_dim=kwargs.get("hidden_dim", 64),
            n_layers=kwargs.get("n_layers", 3),
            length_scale=kwargs.get("length_scale", 1.0),
            loss=kwargs.get("loss", "u_stat"),
            n_epochs=kwargs.get("n_epochs", 100),
            batch_size=kwargs.get("batch_size", 256),
            lr=kwargs.get("lr", 1e-3),
            weight_decay=kwargs.get("weight_decay", 1e-4),
            predict_batch_size=kwargs.get("predict_batch_size", 128),
            w_predict_batch_size=kwargs.get("w_predict_batch_size", 4096),
            x_cond_length_scale=kwargs.get("x_cond_length_scale", 1.0),
            x_cond_top_k=kwargs.get("x_cond_top_k", 512),
            seed=kwargs.get("seed", 42),
        )
        self.model = NMMREstimator(cfg)

        self.strategy = kwargs.get("strategy", "paired")
        if self.strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Unknown NMMR strategy {self.strategy!r}; "
                f"choose from {sorted(self._VALID_STRATEGIES)}."
            )

    def fit(self, X, A, Y, *, W=None, Z=None):
        if W is None or Z is None:
            raise ValueError("NMMR requires W and Z proxies.")
        self.model.fit(X, A, Y, W=W, Z=Z)
        return self

    def predict(self, X, *, W=None, Z=None):
        # Paired uses the query-time W; the marginalising strategies ignore it.
        if self.strategy == "paired":
            return self.model.predict(X, W=W)
        return self.model.predict(X, W=None, strategy=self.strategy)


class DFPV_Wrapper(CATEEstimator):
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
                _CAUSALPFN_CACHE["default"] = CausalPFN_Model(
                    device="cpu", verbose=False
                )

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

# 1. Add a global cache for TabICL models to prevent repeated initializations
_TABICL_CACHE = {}


def _get_cached_tabicl():
    """Returns a single, globally shared instance of TabICLRegressor."""
    if not _TABICL_AVAILABLE:
        raise ImportError("tabicl is not installed.")

    if "default" not in _TABICL_CACHE:
        with silent():
            # Adjust initialization parameters if your TabICL setup requires them
            _TABICL_CACHE["default"] = TabICLRegressor()

    return _TABICL_CACHE["default"]


def _make_base_model(name: str):
    """Return a fresh sklearn estimator by short name (or cached for FMs)."""
    if name == "tabicl":
        # 2. Divert to the cached singleton instance instead of creating a new one
        return _get_cached_tabicl()

    factories: dict[str, Callable] = {
        "linear": LinearRegression,
        "rf": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42
        ),
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
        "P2SLS": lambda: P2SLS(),
        "PLearner": lambda: ProximalPLearner(
            final_model=_make_base_model(cfg.get("base", "rf")),
            n_splits=cfg.get("n_splits", 3),
            random_state=cfg.get("random_state", 42),
            q_clip=cfg.get("q_clip", None),
            refit_full_nuisance=cfg.get("refit_full_nuisance", False),
        ),
        "KernelPV": lambda: KPV_Wrapper(**cfg),
        "PMMR": lambda: PMMR_Wrapper(**cfg),
        "NMMR": lambda: NMMR_Wrapper(**cfg),
        "DFPV": lambda: DFPV_Wrapper(**cfg),
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
    }
    if cls not in factories:
        raise ValueError(
            f"Unknown method class {cls!r}. "
            f"Valid choices: {', '.join(map(repr, factories.keys()))}."
        )

    return name, factories[cls]()
