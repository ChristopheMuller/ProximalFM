import sys
from pathlib import Path
import numpy as np
from itertools import product
import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Ensure project root is in sys.path to find your modules
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench_methods import build_method

# ─────────────────────────────────────────────────────────────────────────────
# Global Worker Module Function
# ─────────────────────────────────────────────────────────────────────────────


def _tuning_worker(args) -> tuple[dict, float]:
    """
    Top-level global worker function. This is critical for the 'spawn'
    start method on macOS, as nested or inner functions cannot be pickled.
    """
    params, base_cfg, X_t, A_t, Y_t, W_t, Z_t, X_v, Y_v, W_v, Z_v = args

    # Isolate environments for backend libraries inside spawned processes
    import os
    import torch

    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[k] = "1"

    # Handle Mac MPS configuration inside parallel tasks if a torch model is triggered
    if torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    test_cfg = copy.deepcopy(base_cfg)
    test_cfg.update(params)

    try:
        _, estimator = build_method(test_cfg)

        # Fit ONLY on the internal training subset
        estimator.fit(X_t, A_t, Y_t, W=W_t, Z=Z_t)

        # Predict CATE across internal validation set
        pred_cate_val = estimator.predict(X_v, W=W_v, Z=Z_v)

        # Proximal Validation Metric
        val_err = float(np.std(pred_cate_val - np.mean(Y_v)))

        # Fallback safeguard: Reject zero-variance collapsed functions
        if np.isnan(val_err) or np.std(pred_cate_val) < 1e-4:
            return params, float("inf")

        return params, val_err
    except Exception:
        return params, float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# Modular Tuning System
# ─────────────────────────────────────────────────────────────────────────────


class ProximalTuner:
    """
    Handles systematic, un-cheated hyperparameter tuning using strictly
    the training data partition executed in parallel.
    """

    def __init__(
        self,
        method_cls: str,
        param_grid: dict,
        val_split: float = 0.3,
        seed: int = 42,
        n_workers: int = 8,
    ):
        self.method_cls = method_cls
        self.param_grid = param_grid
        self.val_split = val_split
        self.seed = seed
        self.n_workers = n_workers

    def tune(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        W: np.ndarray,
        Z: np.ndarray,
        base_cfg: dict,
    ) -> dict:
        """
        Performs a parallelized internal validation split on training data.
        """
        print(f"\n[Tuner] Tuning hyperparams for {base_cfg['name']} in parallel...")

        # 1. Split training pool into internal train and validation subsets
        rng = np.random.default_rng(self.seed)
        n_samples = X.shape[0]
        indices = rng.permutation(n_samples)
        split_idx = int(n_samples * (1.0 - self.val_split))

        idx_t, idx_v = indices[:split_idx], indices[split_idx:]

        # Extract slices
        X_t, A_t, Y_t, W_t, Z_t = X[idx_t], A[idx_t], Y[idx_t], W[idx_t], Z[idx_t]
        X_v, A_v, Y_v, W_v, Z_v = X[idx_v], A[idx_v], Y[idx_v], W[idx_v], Z[idx_v]

        # 2. Build the parameter grid
        keys, values = zip(*self.param_grid.items())
        experiments = [dict(zip(keys, v)) for v in product(*values)]

        print(
            f"  - Dispatching {len(experiments)} configurations across {self.n_workers} processes..."
        )

        # Pack payloads for safe worker distribution
        tasks = [
            (params, base_cfg, X_t, A_t, Y_t, W_t, Z_t, X_v, Y_v, W_v, Z_v)
            for params in experiments
        ]

        best_err = float("inf")
        best_params = experiments[0]

        # Run process pool cleanly safely avoiding fork issues
        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            results = pool.map(_tuning_worker, tasks)
            for params, val_err in results:
                if val_err < best_err:
                    best_err = val_err
                    best_params = params

        print(f"  - Optimization Complete. Selected Params: {best_params}")
        optimized_cfg = copy.deepcopy(base_cfg)
        optimized_cfg.update(best_params)
        return optimized_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Data Generation & Execution Layer
# ─────────────────────────────────────────────────────────────────────────────


def generate_perfect_proximal_data(n_samples: int = 50_000, seed: int = 42):
    """
    Generates a large, clean, fully linear dataset satisfying the standard
    proximal causal identification assumptions.
    """
    rng = np.random.default_rng(seed)

    # 1. Independent Exogenous Variables
    X = rng.normal(0, 1, size=(n_samples, 1))
    U = rng.normal(0, 1, size=(n_samples, 1))

    # 2. Proxies
    Z = 1.5 * U + rng.normal(0, 0.5, size=(n_samples, 1))
    W = 2.0 * U + rng.normal(0, 0.5, size=(n_samples, 1))

    # 3. Treatment Assignment
    a_logit = 0.5 * X + 1.0 * U + 1.5 * Z + rng.normal(0, 0.2, size=(n_samples, 1))
    A = (a_logit > np.median(a_logit)).astype(float)

    # 4. Outcome generation with Confounding and an explicit Treatment Interaction
    base_y = 0.8 * X + 2.5 * U
    true_cate = 1.0 + 2.0 * X
    Y = base_y + true_cate * A + rng.normal(0, 0.5, size=(n_samples, 1))

    return {
        "X": X,
        "A": A.ravel(),
        "Y": Y.ravel(),
        "W": W,
        "Z": Z,
        "U": U,
        "true_cate": true_cate.ravel(),
    }


def run_sanity_check(n_samples=10_000):
    print("=" * 70)
    print("PROXIMAL LEARNING SANITY CHECK (LINEAR, LARGE-SAMPLE ASYMPTOTICS)")
    print("=" * 70)

    # 1. Generate clean data
    data = generate_perfect_proximal_data(n_samples=n_samples)
    X, A, Y, W, Z, true_cate = (
        data["X"],
        data["A"],
        data["Y"],
        data["W"],
        data["Z"],
        data["true_cate"],
    )

    # 2. Configure methods using your exact YAML specifications style
    configs = [
        # {
        #     "name": "P2SLS (No Interactions)",
        #     "cls": "P2SLS",
        #     "interactions": False
        # },
        # {
        #     "name": "P2SLS+AX (With Interactions - Correct Spec)",
        #     "cls": "P2SLS",
        #     "interactions": True
        # },
        # {
        #     "name": "S-OLS (Naive Base Learner)",
        #     "cls": "S_Learner",
        #     "base": "linear",
        #     "use_W": True,
        #     "use_Z": True,
        #     "interactions": True
        # },
        # {
        #     "name": "S-OLS (Oracle with Confounder U)",
        #     "cls": "S_Learner",
        #     "base": "linear",
        #     "use_W": False,
        #     "use_Z": False,
        #     "is_oracle": True,
        #     "interactions": True
        # },
        {
            "name": "KPV-CATE",
            "cls": "NMMRWrappers",
            "model_type": "KernelPV",
            "estimate_cate": False,
            "lam1": 0.01,
            "lam2": 0.01,
            "split_ratio": 0.5,
        },
        {
            "name": "KPV-ATE",
            "cls": "NMMRWrappers",
            "model_type": "KernelPV",
            "estimate_cate": False,
            "lam1": 0.01,
            "lam2": 0.01,
            "split_ratio": 0.5,
        },
    ]

    # 3. Evaluate each method
    for cfg in configs:
        if "tune_grid" in cfg:
            grid = cfg.pop("tune_grid")
            # Explicitly setting n_workers to 4 for local tuning scalability
            tuner = ProximalTuner(method_cls=cfg["cls"], param_grid=grid, n_workers=6)
            cfg = tuner.tune(X, A, Y, W, Z, base_cfg=cfg)

        name, estimator = build_method(cfg)

        X_train, X_test = X, X
        if cfg.get("is_oracle", False):
            X_train = np.concatenate([X, data["U"]], axis=-1)
            X_test = np.concatenate([X, data["U"]], axis=-1)

        try:
            # Fit and Predict
            estimator.fit(X_train, A, Y, W=W, Z=Z)
            pred_cate = estimator.predict(X_test, W=W, Z=Z)

            # Compute Metrics
            bias = np.mean(pred_cate - true_cate)
            rmse = np.sqrt(np.mean((pred_cate - true_cate) ** 2))
            rmedse = np.median(np.sqrt((pred_cate - true_cate) ** 2))

            print(f"\nMethod: {name}")
            print(f"  - Mean CATE Bias: {bias:11.6f}")
            print(f"  - CATE RMSE:      {rmse:11.6f}")
            print(f"  - CATE RMedSE:    {rmedse:11.6f}")

            # Print example predictions across heterogeneous slices of X
            test_x1 = np.array([[-1.0]])
            test_x2 = np.array([[1.0]])

            if cfg.get("is_oracle", False):
                test_x1 = np.array([[-1.0, 0.0]])
                test_x2 = np.array([[1.0, 0.0]])

            pred_minus1_raw = estimator.predict(
                test_x1, W=np.array([[0.0]]), Z=np.array([[0.0]])
            )
            pred_plus1_raw = estimator.predict(
                test_x2, W=np.array([[0.0]]), Z=np.array([[0.0]])
            )
            p_minus1 = np.atleast_1d(pred_minus1_raw)[0]
            p_plus1 = np.atleast_1d(pred_plus1_raw)[0]

            print(f"  - Pred at X=-1.0: {p_minus1:11.4f} (True CATE = -1.0000)")
            print(f"  - Pred at X= 1.0: {p_plus1:11.4f} (True CATE =  3.0000)")

        except Exception as e:
            print(f"\nMethod {name} failed: {e}")


if __name__ == "__main__":

    if sys.platform != "win32":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    run_sanity_check(n_samples=6_000)
