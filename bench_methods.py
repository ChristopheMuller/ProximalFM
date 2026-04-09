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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from tabicl import TabICLRegressor
    _TABICL_AVAILABLE = True
except ImportError:
    _TABICL_AVAILABLE = False

import sys
import os
import torch

# --- CausalFM Setup ---
## assumes you did: "git clone https://github.com/yccm/CausalFM-toolkit.git CausalFM" beforehand
toolkit_path = os.path.abspath("CausalFM")
if toolkit_path not in sys.path:
    sys.path.append(toolkit_path)
try:
    import causalfm
    from causalfm.models import StandardCATEModel
    _CAUSALFM_AVAILABLE = True
except ImportError:
    _CAUSALFM_AVAILABLE = False

# --- DoPFN Setup ---
## assumes you did: "git clone https://github.com/yccm/DoPFN.git DoPFN" beforehand
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
    # 1. Stop PyTorch and FAISS from fighting over Mac C++ threads
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_OK"] = "True"

    import faiss

    # 2. Monkeypatch FAISS to protect it from internal data corruption
    _original_add = faiss.IndexFlatL2.add
    _original_search = faiss.IndexFlatL2.search

    def _safe_add(self, x):
        x_safe = np.ascontiguousarray(x, dtype=np.float32)
        return _original_add(self, x_safe)

    def _safe_search(self, x, k):
        x_safe = np.ascontiguousarray(x, dtype=np.float32)
        return _original_search(self, x_safe, k)

    faiss.IndexFlatL2.add = _safe_add
    faiss.IndexFlatL2.search = _safe_search

    # Alias the import so it doesn't collide with the CATEEstimator base class!
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
# Foundation Model Wrappers
# ─────────────────────────────────────────────────────────────────────────────

_CAUSALFM_CACHE = {}
_CAUSALPFN_CACHE = {}


class CausalFM_Estimator(CATEEstimator):
    """Wrapper for the CausalFM StandardCATEModel.

    Foundation models use in-context learning. The fit() method stores the
    training episode as context, and predict() evaluates the query points.
    """

    def __init__(
        self,
        checkpoint_path: str,
        use_W: bool = True,
        use_Z: bool = True,
    ):
        if not _CAUSALFM_AVAILABLE:
            raise ImportError(
                "causalfm is not installed or cannot be found. "
                "Ensure it is installed or your sys.path is configured correctly."
            )

        torch.set_num_threads(1)

        if checkpoint_path not in _CAUSALFM_CACHE:
            with silent():
                _CAUSALFM_CACHE[checkpoint_path] = StandardCATEModel.from_pretrained(
                    checkpoint_path
                )

        self.model = _CAUSALFM_CACHE[checkpoint_path]

        self.fb = FeatureBuilder(use_W=use_W, use_Z=use_Z, interaction_AX=False)

        # Placeholders for in-context training data
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
    def __init__(
        self,
        use_W: bool = True,
        use_Z: bool = True,
    ):
        if not _CAUSALPFN_AVAILABLE:
            raise ImportError("causalpfn is not installed or the FAISS patch failed.")

        torch.set_num_threads(1)

        if "default" not in _CAUSALPFN_CACHE:
            with silent():
                # verbose=False might help silence it further natively
                _CAUSALPFN_CACHE["default"] = CausalPFN_Model(device="cpu", verbose=False)

        self.model = _CAUSALPFN_CACHE["default"]
        
        # We don't build A into the features because CausalPFN takes it explicitly
        self.fb = FeatureBuilder(use_W=use_W, use_Z=use_Z, interaction_AX=False)

    def fit(self, X, A, Y, *, W=None, Z=None):
        feat = self.fb.build(X, W=W, Z=Z)

        # Enforce C-contiguous memory, float32, and 1D shapes for A and Y
        X_train = np.ascontiguousarray(feat, dtype=np.float32)
        A_train = np.ascontiguousarray(A.ravel(), dtype=np.float32)
        Y_train = np.ascontiguousarray(Y.ravel(), dtype=np.float32)

        with silent():
            self.model.fit(X_train, A_train, Y_train)
        return self

    def predict(self, X, *, W=None, Z=None):
        feat = self.fb.build(X, W=W, Z=Z)
        
        # Enforce C-contiguous float32 for inference
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
    if name == "linear":
        return LinearRegression()
    if name == "rf":
        return RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    if name == "tabicl":
        if not _TABICL_AVAILABLE:
            raise ImportError("tabicl is not installed.")
        return TabICLRegressor()
    raise ValueError(f"Unknown base model {name!r}. Choices: 'linear', 'rf', 'tabicl'.")


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

    if cls == "S_Learner":
        method = S_Learner(
            base_model=_make_base_model(cfg.get("base", "linear")),
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
            interactions=cfg.get("interactions", False),
        )

    elif cls == "T_Learner":
        base_name = cfg.get("base", "linear")
        method = T_Learner(
            base_model_factory=lambda bn=base_name: _make_base_model(bn),
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
        )

    elif cls == "P2SLS":
        method = P2SLS(interactions=cfg.get("interactions", False))

    elif cls == "CausalFM":
        method = CausalFM_Estimator(
            checkpoint_path=cfg["checkpoint"],
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
        )

    elif cls == "DoPFN":
        method = DoPFN_Estimator(
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
        )

    elif cls == "CausalPFN":
        method = CausalPFN_Estimator(
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
        )

    else:
        raise ValueError(
            f"Unknown method class {cls!r}. "
            "Valid choices: 'S_Learner', 'T_Learner', 'P2SLS', 'CausalFM', 'DoPFN', 'CausalPFN'."
        )

    return name, method
