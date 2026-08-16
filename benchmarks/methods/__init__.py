"""
benchmarks.methods — CATE estimator classes for the benchmark.

This module is the registry: the thin wrappers that adapt an implementation to
the benchmark's interface, plus `build_method` (config dict → estimator) and
`method_estimand` (config dict → what the output answers). The implementations
themselves are the sibling modules — `kpv.py`, `pmmr.py`, `nmmr.py`, `dfpv.py`,
`plearner.py` (+ the shared `kernels.py`), and `proximalfm.py`. The interface
they all implement is in `base.py`.

ProximalFM is one of them. It is *this* project's model, but it is not a
privileged one: it is configured under `methods:` like everything else, takes
its device from the same resolved `ComputeConfig`, and its predictions land in
the same cache. The `models:` block and `run.run_neural_model` that used to give
it a separate pipeline are gone.

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

from dataclasses import replace
from typing import Callable, Optional
import warnings

# The estimand taxonomy has its own module: both this package and the metrics
# layer need it, and neither may import the other (metrics must stay free of
# torch, sklearn and the competitor submodules that this file drags in).
from benchmarks.estimands import canonical_estimand, format_estimand
from benchmarks.methods.base import CATEEstimator, FeatureBuilder

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

    # FAISS and torch each ship their own OpenMP; loading both trips a
    # duplicate-runtime abort on macOS unless this is set. The BLAS *thread counts*
    # used to be pinned here too, unconditionally and invisibly from the config —
    # they now belong to `compute._pool_init`, which runs early enough in a spawned
    # worker to still have an effect and takes its value from `compute.blas_threads`.
    # (The original spelling `KMP_DUPLICATE_OK` is a typo for the variable OpenMP
    # actually reads; both are set so the behaviour is at least the intended one.)
    os.environ.setdefault("KMP_DUPLICATE_OK", "True")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

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

    # …/<repo>/benchmarks/methods/__init__.py → <repo>
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def _make_plearner(cfg: dict, compute) -> CATEEstimator:
    """Instantiate the P-learner from a config entry.

    The `outputs:` list is what makes this method worth its cost: every entry
    shares one cross-fit of the bridges, which is the whole expense. Each entry is
    either

        { name: …, base: rf|linear|tabicl }   a final-stage CATE model, or
        { name: …, kind: ate }                the proximal ATE, broadcast

    With no `outputs:`, the entry is a single CATE output named after the method
    and using the top-level `base:` (default `rf`).
    """
    from benchmarks.methods.plearner import PLearner

    for gone, fix in (
        ("refit_full_nuisance", "drop it; the bridge is never refit on all rows"),
        ("bridge_params.kernel_sigma", "use `bandwidth_scale` (median heuristic)"),
    ):
        if gone.split(".")[0] in cfg and (
            "." not in gone or gone.split(".")[1] in (cfg.get("bridge_params") or {})
        ):
            raise ValueError(
                f"`{gone}` belonged to the removed kernel-matrix P-learner. {fix}. "
                "See benchmarks/methods/plearner.py for the current knobs."
            )
    if isinstance(cfg.get("q_clip"), (list, tuple)):
        raise ValueError(
            "`q_clip` is now a single magnitude, not a (lo, hi) window: write "
            f"`q_clip: {max(abs(float(v)) for v in cfg['q_clip'])}`. A symmetric "
            "window let a sign flip through untouched; a magnitude cap does not."
        )

    tabicl_opts = cfg.get("tabicl") or None
    default_base = cfg.get("base", "rf")

    outputs = cfg.get("outputs") or [{"name": cfg["name"], "base": default_base}]
    final_models: dict[str, object] = {}
    ate_outputs: list[str] = []
    for o in outputs:
        if o.get("kind", "cate") == "ate":
            ate_outputs.append(o["name"])
        else:
            final_models[o["name"]] = _make_base_model(
                o.get("base", default_base), tabicl_opts
            )

    learner = PLearner(
        bridge=cfg.get("bridge", "kernel"),
        bridge_params=cfg.get("bridge_params") or None,
        final_models=final_models,
        n_splits=cfg.get("n_splits", 3),
        random_state=cfg.get("random_state", 42),
        q_clip=cfg.get("q_clip", None),
    )
    return _PLearner_Estimator(learner, ate_outputs)


class _PLearner_Estimator(CATEEstimator):
    """Adapts `plearner.PLearner` to the runner's one-fit-many-outputs contract.

    Thin on purpose: `PLearner` already implements the benchmark's `fit` /
    `predict` / `predict_variants` signatures, so all this adds is broadcasting
    the scalar ATE into the outputs that asked for it.
    """

    def __init__(self, learner, ate_outputs: list[str]):
        self.learner = learner
        self.ate_outputs = ate_outputs

    def fit(self, X, A, Y, *, W=None, Z=None) -> "_PLearner_Estimator":
        self.learner.fit(X, A, Y, W=W, Z=Z)
        return self

    def predict(self, X, *, W=None, Z=None) -> np.ndarray:
        return next(iter(self.predict_variants(X, W=W, Z=Z).values()))

    def predict_variants(self, X, *, W=None, Z=None) -> dict[str, np.ndarray]:
        out = self.learner.predict_variants(X)
        for name in self.ate_outputs:
            out[name] = self.learner.predict_ate(X)
        return out

    def diagnostics(self) -> dict[str, float]:
        return self.learner.diagnostics()


class KPV_Wrapper(CATEEstimator):
    # Which prediction target the estimator returns. See KPVEstimator.predict:
    #   "marginal"    — global ATE, repeated for every test row.
    #   "paired"      — CATE(x, w) at the supplied test (X, W) pairs (needs W).
    #   "independent" — CATE(x) marginalising W over its marginal (assumes W ⊥ X).
    #   "conditional" — CATE(x) marginalising W over W | X (assumes W correlated with X).
    _VALID_STRATEGIES = {"marginal", "paired", "independent", "conditional"}

    def __init__(self, **kwargs):
        from benchmarks.methods.kpv import KPVConfig, KPVEstimator

        cfg = KPVConfig(
            split_ratio=kwargs.get("split_ratio", 0.5),
            kernel_scale=kwargs.get("kernel_scale", 1.0),
            tune_kernel_scale=kwargs.get("tune_kernel_scale", False),
            kernel_scale_min=kwargs.get("kernel_scale_min", 0.5),
            kernel_scale_max=kwargs.get("kernel_scale_max", 2.0),
            n_kernel_scale_search=kwargs.get("n_kernel_scale_search", 5),
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

        self.strategy = kwargs.get("strategy", "marginal")
        if self.strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Unknown KPV strategy {self.strategy!r}; "
                f"choose from {sorted(self._VALID_STRATEGIES)}."
            )

        # Optional multi-output block: one fit → several named prediction sets,
        # each a different strategy. Reuses the single expensive KPV fit (which
        # now also runs the kernel-scale search) instead of refitting per output.
        self._outputs = kwargs.get("outputs", None)
        if self._outputs:
            for o in self._outputs:
                strat = o.get("strategy")
                if strat not in self._VALID_STRATEGIES:
                    raise ValueError(
                        f"Unknown KPV strategy {strat!r} in outputs entry {o!r}; "
                        f"choose from {sorted(self._VALID_STRATEGIES)}."
                    )

    def fit(self, X, A, Y, *, W=None, Z=None):
        if W is None or Z is None:
            raise ValueError("KPV requires W and Z proxies.")
        self.model.fit(X, A, Y, W=W, Z=Z)
        return self

    def _predict_strategy(self, strategy, X, *, W=None):
        # Only the paired contrast consumes the query-time W; the other
        # strategies marginalise W internally and ignore it.
        if strategy == "paired":
            return self.model.predict(X, W=W, mode="paired")
        return self.model.predict(X, mode=strategy)

    def predict(self, X, *, W=None, Z=None):
        return self._predict_strategy(self.strategy, X, W=W)

    def predict_variants(self, X, *, W=None, Z=None):
        if not self._outputs:
            return {None: self.predict(X, W=W, Z=Z)}
        # Single trained model reused across every requested strategy.
        return {
            o["name"]: self._predict_strategy(o["strategy"], X, W=W)
            for o in self._outputs
        }


class PMMR_Wrapper(CATEEstimator):
    # Which prediction target the estimator returns. See PMMREstimator.predict:
    #   "marginal"    — global ATE, repeated for every test row.
    #   "paired"      — CATE(x, w) at the supplied test (X, W) pairs (needs W).
    #   "independent" — CATE(x) marginalising W over its marginal (assumes W ⊥ X).
    #   "conditional" — CATE(x) marginalising W over W | X (assumes W correlated with X).
    _VALID_STRATEGIES = {"marginal", "paired", "independent", "conditional"}

    def __init__(self, **kwargs):
        from benchmarks.methods.pmmr import PMMRConfig, PMMREstimator

        cfg = PMMRConfig(
            kernel_scale=kwargs.get("kernel_scale", 1.0),
            # Default read off the dataclass so the reference-matching value
            # lives in exactly one place.
            kernel_scale_z=kwargs.get("kernel_scale_z", PMMRConfig.kernel_scale_z),
            lam1=kwargs.get("lam1", 1.0),
            lam2=kwargs.get("lam2", 1e-4),
            jitter=kwargs.get("jitter", 1e-10),
        )
        self.model = PMMREstimator(cfg)

        self.strategy = kwargs.get("strategy", "marginal")
        if self.strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Unknown PMMR strategy {self.strategy!r}; "
                f"choose from {sorted(self._VALID_STRATEGIES)}."
            )

        # Optional multi-output block: one fit → several named prediction sets,
        # each a different strategy. Reuses the single PMMR fit instead of
        # refitting per output.
        self._outputs = kwargs.get("outputs", None)
        if self._outputs:
            for o in self._outputs:
                strat = o.get("strategy")
                if strat not in self._VALID_STRATEGIES:
                    raise ValueError(
                        f"Unknown PMMR strategy {strat!r} in outputs entry {o!r}; "
                        f"choose from {sorted(self._VALID_STRATEGIES)}."
                    )

    def fit(self, X, A, Y, *, W=None, Z=None):
        if W is None or Z is None:
            raise ValueError("PMMR requires W and Z proxies.")
        self.model.fit(X, A, Y, W=W, Z=Z)
        return self

    def _predict_strategy(self, strategy, X, *, W=None):
        # Only the paired contrast consumes the query-time W; the other
        # strategies marginalise W internally and ignore it.
        if strategy == "paired":
            return self.model.predict(X, W=W, strategy="paired")
        return self.model.predict(X, strategy=strategy)

    def predict(self, X, *, W=None, Z=None):
        return self._predict_strategy(self.strategy, X, W=W)

    def predict_variants(self, X, *, W=None, Z=None):
        if not self._outputs:
            return {None: self.predict(X, W=W, Z=Z)}
        # Single trained model reused across every requested strategy.
        return {
            o["name"]: self._predict_strategy(o["strategy"], X, W=W)
            for o in self._outputs
        }


class NMMR_Wrapper(CATEEstimator):
    # Which prediction target the estimator returns. See NMMREstimator.predict:
    #   "paired"      — CATE(x, w) at the supplied test (X, W) pairs (needs W).
    #   "independent" — CATE(x) marginalising W over its marginal (assumes W ⊥ X).
    #   "conditional" — CATE(x) marginalising W over W | X (assumes W correlated with X).
    _VALID_STRATEGIES = {"paired", "independent", "conditional"}

    def __init__(self, **kwargs):
        from benchmarks.methods.nmmr import NMMRConfig, NMMREstimator
        from benchmarks.tuning import TuningConfig

        self._NMMREstimator = NMMREstimator

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
            # From the resolved ComputeConfig; NMMR used to auto-grab MPS in every
            # pool worker, which is exactly the collision ComputeConfig now refuses.
            device=kwargs.get("device", None),
        )
        # Base config; a hyperparameter search (if enabled) overrides fields on
        # a copy of this and refits the winner on the full context.
        self._base_cfg = cfg
        self._tune_cfg = TuningConfig.from_dict(kwargs.get("tune"))
        if "loss" in self._tune_cfg.grid:
            # Not deprecated — refused. Candidates are ranked by the held-out
            # moment loss of their *own* statistic (NMMREstimator.score), so a
            # grid over `loss` would compare U-statistic scores against
            # V-statistic ones, which differ by V's diagonal term (MSE/n) and
            # are not on a common scale. The statistic identifies the method
            # variant; fix it per method entry.
            raise ValueError(
                "NMMR: `loss` may not appear in `tune.grid` — the U- and "
                "V-statistic scores are not comparable, so a search over it "
                "would rank on scale rather than on fit. Set it as a "
                "method-level key instead:\n"
                "    - name: NMMR-U\n"
                "      cls:  NMMR\n"
                "      loss: u_stat\n"
                "      tune: { grid: { hidden_dim: [...], ... } }"
            )
        self.model = NMMREstimator(cfg)
        self._tune_diag: dict[str, float] = {}

        self.strategy = kwargs.get("strategy", "paired")
        if self.strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Unknown NMMR strategy {self.strategy!r}; "
                f"choose from {sorted(self._VALID_STRATEGIES)}."
            )

        self._outputs = kwargs.get("outputs", None)
        if self._outputs:
            for o in self._outputs:
                strat = o.get("strategy")
                if strat not in self._VALID_STRATEGIES:
                    raise ValueError(
                        f"Unknown NMMR strategy {strat!r} in outputs entry {o!r}; "
                        f"choose from {sorted(self._VALID_STRATEGIES)}."
                    )

    def fit(self, X, A, Y, *, W=None, Z=None):
        if W is None or Z is None:
            raise ValueError("NMMR requires W and Z proxies.")

        cfg = self._base_cfg
        self._tune_diag = {}
        if self._tune_cfg.enabled and self._tune_cfg.grid:
            best = self._tune(X, A, Y, W, Z)
            cfg = replace(self._base_cfg, **best)

        # Train the (possibly tuned) model once on the full context. Every
        # output variant later reuses this single fit — no retraining.
        self.model = self._NMMREstimator(cfg)
        self.model.fit(X, A, Y, W=W, Z=Z)
        return self

    def _tune(self, X, A, Y, W, Z) -> dict:
        """Random/grid search over `tune.grid`; return the best field overrides.

        Candidates are trained on a train split and ranked by held-out moment
        loss: a *fixed-kernel* score, so bandwidth choices stay comparable,
        evaluated with this method's own U/V statistic, so the ranking is on
        the objective the candidates were trained for. `loss` itself is not
        tunable — see `__init__` for why.
        """
        from benchmarks.tuning import (
            sample_candidates,
            search,
            search_diagnostics,
            train_val_split,
        )

        tc = self._tune_cfg
        candidates = sample_candidates(tc.grid, tc.n_trials, tc.seed)
        train_idx, val_idx = train_val_split(X.shape[0], tc.val_frac, tc.seed)

        Xtr, Atr, Ytr, Wtr, Ztr = (
            X[train_idx],
            A[train_idx],
            Y[train_idx],
            W[train_idx],
            Z[train_idx],
        )
        Xva, Ava, Yva, Wva, Zva = (
            X[val_idx],
            A[val_idx],
            Y[val_idx],
            W[val_idx],
            Z[val_idx],
        )

        def fit_score(hparams: dict) -> float:
            est = self._NMMREstimator(replace(self._base_cfg, **hparams))
            est.fit(Xtr, Atr, Ytr, W=Wtr, Z=Ztr)
            return est.score(Xva, Ava, Yva, W=Wva, Z=Zva)

        best, best_score, results = search(candidates, fit_score)
        self._tune_diag = search_diagnostics(
            tc.grid, candidates, best, best_score, results
        )
        return best

    def _predict_strategy(self, strategy, X, *, W=None):
        # Paired uses the query-time W; the marginalising strategies ignore it.
        if strategy == "paired":
            return self.model.predict(X, W=W)
        return self.model.predict(X, W=None, strategy=strategy)

    def predict(self, X, *, W=None, Z=None):
        return self._predict_strategy(self.strategy, X, W=W)

    def predict_variants(self, X, *, W=None, Z=None):
        if not self._outputs:
            return {None: self.predict(X, W=W, Z=Z)}
        # Single trained model reused across every requested strategy.
        return {
            o["name"]: self._predict_strategy(o["strategy"], X, W=W)
            for o in self._outputs
        }

    def diagnostics(self) -> dict[str, float]:
        """What the kernel carried, and where the search landed.

        Two groups, worth reading in this order:

        - ``kernel_*`` from :class:`NMMREstimator` — whether the moment
          condition was enforced at all at this episode's dimension (see
          ``NMMREstimator._record_kernel_diagnostics``). A
          ``kernel_offdiag_median`` near zero means the number reported for the
          episode is closer to a confounded regression on ``(A, W, X)`` than to
          NMMR, so check it before reading anything else here.
        - ``tune_*`` from :func:`benchmarks.tuning.search_diagnostics` — the
          selected value per grid parameter, plus the candidate count and score
          spread that say whether "selected" means anything. Absent when the
          method was configured without a ``tune:`` block.

        One fit backs every entry in ``outputs:``, so these describe the shared
        model rather than any one of paired / independent / conditional.
        """
        diag = dict(getattr(self.model, "diagnostics_", {}) or {})
        diag.update(self._tune_diag)
        return diag


class DFPV_Wrapper(CATEEstimator):
    """Wrapper for the Deep Feature Proxy Variable (DFPV) estimator."""

    # Which prediction target the estimator returns. See DFPVEstimator.predict:
    #   "marginal"    — global ATE, repeated for every test row.
    #   "paired"      — CATE(x, w) at the supplied test (X, W) pairs (needs W).
    #   "independent" — CATE(x) marginalising W over its marginal (assumes W ⊥ X).
    #   "conditional" — CATE(x) marginalising W over W | X (assumes W correlated with X).
    _VALID_STRATEGIES = {"marginal", "paired", "independent", "conditional"}

    def __init__(self, **kwargs):
        from benchmarks.methods.dfpv import DFPVConfig, DFPVEstimator

        # Device comes from the method's resolved ComputeConfig (build_method injects
        # it). This used to force CPU unconditionally, because concurrent
        # multi-process Metal access from the episode pool stalls — that constraint
        # now lives in ComputeConfig, which refuses a non-CPU device combined with
        # pool fan-out, so `compute: {device: mps, execution: inline}` is a legal and
        # useful configuration here rather than something to silently undo.
        device = kwargs.get("device", None)

        cfg = DFPVConfig(
            feature_dim=kwargs.get("feature_dim", kwargs.get("feat_dim", 8)),
            hidden_dim=kwargs.get("hidden_dim", 32),
            n_hidden_layers=kwargs.get("n_hidden_layers", 2),
            use_backdoor=kwargs.get("use_backdoor", True),
            add_stage1_intercept=kwargs.get("add_stage1_intercept", True),
            add_stage2_intercept=kwargs.get("add_stage2_intercept", True),
            lam1=kwargs.get("lam1", 0.1),
            lam2=kwargs.get("lam2", 0.1),
            n_epochs=kwargs.get("n_epochs", 50),
            stage1_iter=kwargs.get("stage1_iter", 20),
            stage2_iter=kwargs.get("stage2_iter", 1),
            lr=kwargs.get("lr", 1e-3),
            treatment_weight_decay=kwargs.get("treatment_weight_decay", 0.0),
            treatment_proxy_weight_decay=kwargs.get(
                "treatment_proxy_weight_decay", 0.0
            ),
            outcome_proxy_weight_decay=kwargs.get("outcome_proxy_weight_decay", 0.0),
            backdoor_weight_decay=kwargs.get("backdoor_weight_decay", 0.0),
            split_ratio=kwargs.get("split_ratio", 0.5),
            x_cond_length_scale=kwargs.get("x_cond_length_scale", 1.0),
            x_cond_top_k=kwargs.get("x_cond_top_k", 512),
            seed=kwargs.get("seed", 42),
            device=device,
        )
        self.model = DFPVEstimator(cfg)

        self.strategy = kwargs.get("strategy", "marginal")
        if self.strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Unknown DFPV strategy {self.strategy!r}; "
                f"choose from {sorted(self._VALID_STRATEGIES)}."
            )

        self._outputs = kwargs.get("outputs", None)
        if self._outputs:
            for o in self._outputs:
                strat = o.get("strategy")
                if strat not in self._VALID_STRATEGIES:
                    raise ValueError(
                        f"Unknown DFPV strategy {strat!r} in outputs entry {o!r}; "
                        f"choose from {sorted(self._VALID_STRATEGIES)}."
                    )

    def fit(self, X, A, Y, *, W=None, Z=None):
        if W is None or Z is None:
            raise ValueError("DFPV requires both W and Z proxies.")
        self.model.fit(X, A, Y, W=W, Z=Z)
        return self

    def _predict_strategy(self, strategy, X, *, W=None):
        # Only the paired contrast consumes the query-time W; the other
        # strategies marginalise W internally and ignore it.
        if strategy == "paired":
            return self.model.predict(X, W=W, strategy="paired")
        return self.model.predict(X, strategy=strategy)

    def predict(self, X, *, W=None, Z=None):
        return self._predict_strategy(self.strategy, X, W=W)

    def predict_variants(self, X, *, W=None, Z=None):
        # Without this the wrapper would silently accept an `outputs:` block
        # (it arrives through **cfg) and then return a single prediction keyed
        # by the method name, which the runner cannot match to the output names
        # it pre-allocated. One fit, every requested strategy — as for the other
        # three bridge wrappers.
        if not self._outputs:
            return {None: self.predict(X, W=W, Z=Z)}
        return {
            o["name"]: self._predict_strategy(o["strategy"], X, W=W)
            for o in self._outputs
        }


# ─────────────────────────────────────────────────────────────────────────────
# EconML Causal Forest (DROrthoForest)
# ─────────────────────────────────────────────────────────────────────────────


class CausalForest_Wrapper(CATEEstimator):
    """Doubly-Robust Orthogonal Forest from EconML.

    Uses DROrthoForest which combines random forests with doubly-robust
    score functions to estimate heterogeneous treatment effects.  W and Z
    proxies are optionally stacked with X as additional controls passed to
    EconML's W argument (the set of features used for nuisance estimation
    but not for heterogeneity).
    """

    def __init__(
        self,
        n_trees: int = 200,
        min_leaf_size: int = 10,
        max_depth: int = 10,
        subsample_ratio: float = 0.7,
        use_W: bool = False,
        use_Z: bool = False,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        try:
            from econml.orf import DROrthoForest
        except ImportError as e:
            raise ImportError("econml is not installed. Run: pip install econml") from e

        self.use_W = use_W
        self.use_Z = use_Z
        # n_jobs=1: the benchmark already fans episodes out across a `spawn`
        # ProcessPoolExecutor, so letting DROrthoForest spin up its own loky
        # workers (default n_jobs=-1) oversubscribes the CPU and stalls — the
        # same hazard the DFPV wrapper guards against with MPS.
        self.model = DROrthoForest(
            n_trees=n_trees,
            min_leaf_size=min_leaf_size,
            max_depth=max_depth,
            subsample_ratio=subsample_ratio,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def _build_controls(
        self, X: np.ndarray, W: Optional[np.ndarray], Z: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Stack optional proxies into EconML's W (nuisance controls)."""
        parts = []
        if self.use_W and W is not None:
            parts.append(W if W.ndim == 2 else W[:, None])
        if self.use_Z and Z is not None:
            parts.append(Z if Z.ndim == 2 else Z[:, None])
        if not parts:
            return None
        return np.hstack(parts)

    def fit(self, X, A, Y, *, W=None, Z=None):
        controls = self._build_controls(X, W, Z)
        with silent():
            self.model.fit(Y, A, X=X, W=controls)
        return self

    def predict(self, X, *, W=None, Z=None):
        return self.model.effect(X).squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# Foundation Model Wrappers
# ─────────────────────────────────────────────────────────────────────────────

_CAUSALFM_CACHE = {}
_CAUSALPFN_CACHE = {}


class CausalFM_Estimator(CATEEstimator):
    def __init__(
        self,
        checkpoint_path: str,
        use_W: bool = True,
        use_Z: bool = True,
        device: str = "cpu",
    ):
        # Lock in CausalFM, kick out NMMR
        _switch_src_context(target_repo_name="CausalFM", competitor_repo_name="NMMR")

        try:
            from causalfm.models import StandardCATEModel
        except ImportError as e:
            raise ImportError(f"CausalFM import failed even after switch: {e}")

        # No torch.set_num_threads here: `compute.apply_compute` owns thread counts
        # and has already run for this task. This call used to sit here and silently
        # undo the per-method `torch_threads` the worker had just set.
        self.device = torch.device(device)

        # Keyed by device as well as checkpoint: the same weights on two devices are
        # two distinct loaded models, and a shared session-wide pool can be asked for
        # both.
        key = (checkpoint_path, str(self.device))
        if key not in _CAUSALFM_CACHE:
            with silent():
                model = StandardCATEModel.from_pretrained(checkpoint_path)
                _CAUSALFM_CACHE[key] = model.to(self.device)

        self.model = _CAUSALFM_CACHE[key]
        self.fb = FeatureBuilder(use_W=use_W, use_Z=use_Z, interaction_AX=False)

        self.ctx_x = None
        self.ctx_a = None
        self.ctx_y = None

    def fit(self, X, A, Y, *, W=None, Z=None):
        # Build unified covariate matrix (stacking X, W, Z if configured)
        feat = self.fb.build(X, W=W, Z=Z)

        # Store as PyTorch tensors with correct shapes (n, 1) for A and Y
        self.ctx_x = torch.Tensor(feat).to(self.device)
        self.ctx_a = torch.Tensor(A).reshape(-1, 1).to(self.device)
        self.ctx_y = torch.Tensor(Y).reshape(-1, 1).to(self.device)
        return self

    def predict(self, X, *, W=None, Z=None):
        if self.ctx_x is None:
            raise ValueError("Must call fit() before predict().")

        # Build test covariate matrix
        feat = self.fb.build(X, W=W, Z=Z)
        x_test = torch.Tensor(feat).to(self.device)

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

        # `DoPFNRegressor.__init__` takes no arguments — the submodule exposes no
        # device knob — so this one is CPU whatever the method's compute says, and
        # `compute.device` is deliberately not accepted here rather than accepted
        # and ignored. Thread counts come from `compute.apply_compute`.
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
    def __init__(self, use_W: bool = True, use_Z: bool = True, device: str = "cpu"):
        if not _CAUSALPFN_AVAILABLE:
            raise ImportError("causalpfn is not installed.")

        # Was hardcoded to CPU, which was the only safe choice while every method
        # ran under the episode pool. Thread counts come from `apply_compute`.
        if device not in _CAUSALPFN_CACHE:
            with silent():
                _CAUSALPFN_CACHE[device] = CausalPFN_Model(device=device, verbose=False)

        self.model = _CAUSALPFN_CACHE[device]
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

# TabICL inference options, overridable per method with a `tabicl:` block in the
# config. The defaults are the ones that were measured to matter (n=2048 context,
# 128 queries, an S-learner's one fit + two predicts, per-episode wall time on a
# 12-core M-series, 6 worker processes at one torch thread each):
#
#   kv_cache=False (the library default)   30.6 min / 480 episodes
#   kv_cache="repr"                        22.7 min      bit-identical
#   kv_cache=True                          16.6 min      max|Δ| ~ 3e-7 (2 float32 ulp)
#
# `fit()` re-runs the whole 2048-row context through the model on *every* predict
# when there is no cache, and an S-learner predicts twice (A=1 and A=0) off one
# fit — that redundant context pass is where the time goes, not the checkpoint
# load (~0.15 s) or the ensemble size. Caching it once at fit() time is what buys
# the speedup. "repr" caches the column-embedding KV and the row-interaction
# outputs and re-runs the ICL transformer at predict time: less saving, ~24x less
# memory than "kv", and exactly reproducing the uncached numbers. True/"kv" also
# caches the ICL layers' KV projections; the residual 1e-6 relative difference is
# float32 accumulation order, not a change of estimator. Peak RSS per worker:
# 1.54 GiB ("repr") / 1.61 GiB (off) / 2.44 GiB (True) — multiply by n_workers.
#
# device: "mps" is ~5x faster than a single-threaded CPU process per episode, but
# the 6 worker processes already saturate the machine, so at pool level it is a
# wash (16.5 min) and the workers contend for one GPU. Left on CPU by default;
# set it if you drop to one worker.
_TABICL_DEFAULTS: dict = {"kv_cache": True}


def _make_tabicl(options: dict | None = None):
    """Return a fresh TabICLRegressor with the benchmark's inference options.

    Fresh, not shared: T-/X-learners call the base-model factory once per arm and
    expect independent estimators. TabICL construction is free — the checkpoint is
    loaded inside ``fit()``, not ``__init__`` — so there is nothing to cache here.
    """
    if not _TABICL_AVAILABLE:
        raise ImportError("tabicl is not installed.")

    with silent():
        return TabICLRegressor(**{**_TABICL_DEFAULTS, **(options or {})})


def _make_base_model(name: str, tabicl_options: dict | None = None):
    """Return a fresh sklearn estimator by short name."""
    if name == "tabicl":
        return _make_tabicl(tabicl_options)

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


# (kind, predict-time inputs) of each strategy supported by the proximal bridge
# methods. This is a property of the *strategy*, not of the method: `paired`
# evaluates the bridge at the query row's own W, the two marginalisations
# integrate W out and land back on tau(x), and `marginal` collapses to one
# scalar per episode.
_STRATEGY_ESTIMAND = {
    "paired": ("cate", ("x", "w")),
    "independent": ("cate", ("x",)),
    "conditional": ("cate", ("x",)),
    "marginal": ("ate", ("x",)),
}

# Classes whose estimand does not depend on any config flag.
_CLS_ESTIMAND = {
    # predict() broadcasts a single fitted ATE
    "P2SLS": ("ate", ("x",)),
    # regresses the proximal score on X; W/Z discarded at predict
    "PLearner": ("cate", ("x",)),
    # W/Z go into EconML's nuisance controls, not effect()
    "CausalForest": ("cate", ("x",)),
    # Reads the query rows' W/Z only in the sense that they are in the sequence
    # it is handed: the model masks them (`_hide_query_proxies`), so the
    # prediction is a function of x and the context alone.
    "ProximalFM": ("cate", ("x",)),
}

# Classes that stack W/Z into the design matrix used at *predict* time, so
# use_W/use_Z decide whether they answer tau(x) or tau(x, w, z).
_PROXY_FEATURE_CLASSES = (
    "S_Learner",
    "T_Learner",
    "X_Learner",
    "CausalFM",
    "DoPFN",
    "CausalPFN",
)


def method_estimand(cfg: dict, output_cfg: Optional[dict] = None) -> str:
    """What a method output answers, spelled out: ``"cate(x,w)"``, ``"ate(x)"``, …

    The kind (``cate``/``ate``) says whether the prediction varies per row; the
    parenthesis lists the variables it reads at **predict** time. Fitting a
    nuisance on the proxies does not count — ``PLearner`` is ``cate(x)`` — since
    what a table needs to know is which conditional the *prediction* answers.

    This is what routes an output to the benchmark's tracks, so it is derived
    from the config rather than declared: every pre-existing config file keeps
    working and keeps meaning what it did. An explicit ``estimand:`` key (on the
    method entry, or on the ``outputs[]`` entry for a multi-output method) wins
    — it is the escape hatch for a method whose wiring this function cannot see
    — and accepts the legacy ``x``/``xw``/``ate`` spellings.

    The distinction that matters: a ``cate(x,w)`` output is evaluated at the
    query row's *own* proxies, so it estimates ``E[Y1 - Y0 | X, W]`` — a finer
    conditional than the benchmark's primary target, and one that no ``cate(x)``
    method can match on ITE data. It is not a better estimate of tau(x); it is an
    estimate of something else.
    """
    for source in (output_cfg, cfg):
        if source and source.get("estimand"):
            try:
                return canonical_estimand(source["estimand"])
            except ValueError as exc:
                raise ValueError(f"estimand must be a valid tag: {exc}") from None

    # `is_oracle: true` concatenates the true U onto X in run._worker, so the
    # prediction really does condition on it — the tag says so. It is not
    # disqualifying: an oracle row is in the table as an upper bound.
    inputs = ["x"] + (["u"] if cfg.get("is_oracle", False) else [])

    # PLearner spells its ATE output as `kind: ate` on the outputs entry — the
    # same flag the factory reads to decide what to broadcast. Deriving the tag
    # from it keeps the config from having to say the same thing twice and drift.
    if (output_cfg or {}).get("kind") == "ate":
        return format_estimand("ate", inputs)

    strategy = (output_cfg or {}).get("strategy") or cfg.get("strategy")
    if strategy:
        if strategy not in _STRATEGY_ESTIMAND:
            raise ValueError(
                f"unknown strategy {strategy!r}; expected one of "
                f"{list(_STRATEGY_ESTIMAND)}"
            )
        kind, strategy_inputs = _STRATEGY_ESTIMAND[strategy]
        return format_estimand(kind, inputs + list(strategy_inputs))

    cls = cfg.get("cls")
    if cls in _CLS_ESTIMAND:
        kind, cls_inputs = _CLS_ESTIMAND[cls]
        return format_estimand(kind, inputs + list(cls_inputs))
    if cls in _PROXY_FEATURE_CLASSES:
        # Mirror the factory defaults below: use_W/use_Z default to True for
        # every class in this tuple, so an entry that says neither is tau(x, w, z).
        if cfg.get("use_W", True):
            inputs.append("w")
        if cfg.get("use_Z", True):
            inputs.append("z")
        return format_estimand("cate", inputs)

    # Unknown class: assume the conservative reading. A method that secretly
    # uses the query proxies and is scored as tau(x) merely looks bad on the
    # primary track; the reverse would hand it an unearned advantage.
    return format_estimand("cate", inputs)


def _make_proximalfm(cfg: dict, compute) -> CATEEstimator:
    """Instantiate the ProximalFM estimator (imported lazily: it needs torch)."""
    from benchmarks.methods.proximalfm import ProximalFM_Estimator

    return ProximalFM_Estimator(
        checkpoint=cfg["checkpoint"],
        device=compute.device,
        # Number of feature-transformed views averaged at inference; <= 1 is a
        # single plain predict.
        ensemble=int(cfg.get("ensemble", 1) or 1),
        # Bijective value transforms cycled across those views (e.g. [sign,
        # signed_log, arcsinh, quantile]); unset means permutation-only.
        transforms=cfg.get("transforms") or None,
        outputs=cfg.get("outputs") or None,
        all=cfg.get("all", False),
    )


def normalize_method_cfg(cfg: dict) -> dict:
    """Resolve a config entry's shorthands before the run acts on it.

    Called once per `methods:` entry, ahead of everything else, because a few
    shorthands decide how many prediction sets an entry yields — and the output
    *names* are what the runner pre-allocates and what the skip-if-cached check
    globs for. Today the only one is ProximalFM's `all: true` (score every
    checkpoint in a directory), which expands into an ordinary `outputs:` list.
    """
    if cfg.get("cls") == "ProximalFM":
        from benchmarks.methods.proximalfm import normalize_cfg

        return normalize_cfg(cfg)
    return cfg


def build_method(cfg: dict, compute=None) -> tuple[str, CATEEstimator]:
    """Instantiate a CATE estimator from a single YAML method-config dict.

    This is the bridge between the config file and the estimator objects.
    The run script calls this; it never imports individual estimator classes.

    Args:
        cfg : one entry from the 'methods' list in config_*.yaml.
        compute : the method's resolved `compute.ComputeConfig`. Device and
            intra-method job counts come from here and *only* from here — an
            estimator never picks a device for itself, which is what stopped the
            episode pool from putting six processes on the same GPU. Defaults to a
            single-threaded CPU budget so direct calls (tests, notebooks) still work.

    Returns:
        (display_name, estimator_instance)
    """
    from benchmarks.compute import ComputeConfig

    if compute is None:
        compute = ComputeConfig()

    name = cfg["name"]
    cls = cfg["cls"]

    # Inference knobs for a `base: tabicl` method; see _TABICL_DEFAULTS.
    tabicl_opts = cfg.get("tabicl") or None

    # The `**cfg` wrappers below read `device` out of the config dict; inject the
    # resolved one so a stale `device:` left in a YAML cannot outrank it.
    cfg = {**cfg, "device": compute.device}

    def base_factory(default: str = "linear") -> Callable[[], object]:
        bn = cfg.get("base", default)
        return lambda: _make_base_model(bn, tabicl_opts)

    factories: dict[str, Callable[[], CATEEstimator]] = {
        "S_Learner": lambda: S_Learner(
            base_model=base_factory()(),
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
            interactions=cfg.get("interactions", False),
        ),
        "T_Learner": lambda: T_Learner(
            base_model_factory=base_factory(),
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
        ),
        "X_Learner": lambda: X_Learner(
            base_model_factory=base_factory(),
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
        ),
        "P2SLS": lambda: P2SLS(),
        "PLearner": lambda: _make_plearner(cfg, compute),
        "KernelPV": lambda: KPV_Wrapper(**cfg),
        "PMMR": lambda: PMMR_Wrapper(**cfg),
        "NMMR": lambda: NMMR_Wrapper(**cfg),
        "DFPV": lambda: DFPV_Wrapper(**cfg),
        "CausalFM": lambda: CausalFM_Estimator(
            checkpoint_path=cfg["checkpoint"],
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
            device=compute.device,
        ),
        # No device: DoPFNRegressor exposes none. See DoPFN_Estimator.__init__.
        "DoPFN": lambda: DoPFN_Estimator(
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
        ),
        "CausalPFN": lambda: CausalPFN_Estimator(
            use_W=cfg.get("use_W", True),
            use_Z=cfg.get("use_Z", True),
            device=compute.device,
        ),
        # This project's own model. Device comes from `compute` like everywhere
        # else; the checkpoint is loaded once per process, not once per episode.
        "ProximalFM": lambda: _make_proximalfm(cfg, compute),
        "CausalForest": lambda: CausalForest_Wrapper(
            n_trees=cfg.get("n_trees", 200),
            min_leaf_size=cfg.get("min_leaf_size", 10),
            max_depth=cfg.get("max_depth", 10),
            subsample_ratio=cfg.get("subsample_ratio", 0.7),
            use_W=cfg.get("use_W", False),
            use_Z=cfg.get("use_Z", False),
            random_state=cfg.get("random_state", 42),
            n_jobs=compute.n_jobs,
        ),
    }
    if cls not in factories:
        raise ValueError(
            f"Unknown method class {cls!r}. "
            f"Valid choices: {', '.join(map(repr, factories.keys()))}."
        )

    return name, factories[cls]()
