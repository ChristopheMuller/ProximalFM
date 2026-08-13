"""
Our implementation of NMMR — Nonparametric Marginal Moment Regression
================================================
Causal CATE estimator based on kernel-weighted U/V-statistic loss.

Reference: Kompa et al. "Deep Learning Methods for Proximal Inference via Maximum Moment Restriction" (NeurIPS 2022)

Model:  h(A, W, X) → Ŷ,  trained so that  E[(Y - h(A,W,X)) k(A,Z,X)] = 0
CATE:   E_W[h(1,W,X) - h(0,W,X)]  marginalized over the empirical dist of W_train

Tuning entry point: NMMRConfig — every knob lives there with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class NMMRConfig:
    # ── Architecture ──────────────────────────────────────────────────────────
    hidden_dim: int = 64  # width of each hidden layer
    n_layers: int = 3  # number of hidden layers (depth before output)

    # ── Kernel ────────────────────────────────────────────────────────────────
    # length_scale controls the RBF bandwidth for the moment kernel K(A, Z, X).
    # Smaller → kernel is more local (sharpens the moment condition).
    length_scale: float = 1.0

    # ── Loss ──────────────────────────────────────────────────────────────────
    # "u_stat": U-statistic (zeros diagonal of K, unbiased but noisier)
    # "v_stat": V-statistic (keeps diagonal, biased but lower variance)
    loss: Literal["u_stat", "v_stat"] = "u_stat"

    # ── Optimisation ──────────────────────────────────────────────────────────
    n_epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4  # L2 regularisation on network weights

    # ── Inference ─────────────────────────────────────────────────────────────
    # CATE prediction marginalises over W_train in chunks to avoid OOM.
    predict_batch_size: int = 128

    # ── Reproducibility / device ──────────────────────────────────────────────
    seed: int = 42
    # "cpu", "cuda", "mps", or None to auto-select (cuda > mps > cpu).
    #
    # Inside the benchmark this is always set explicitly, from the method's
    # resolved ComputeConfig. It has to be: auto-selection grabbed MPS in *every*
    # process of a spawn pool, and concurrent multi-process Metal access stalls.
    # `benchmarks.compute` owns the choice now; None is kept for direct use of
    # NMMREstimator outside the harness.
    device: Optional[str] = None

    # ── CATE prediction / W marginalisation ──────────────────────────────────
    w_strategy: Literal["independent", "conditional"] = "independent"

    # For conditional W|X prediction.
    x_cond_length_scale: float = 1.0
    x_cond_top_k: int | None = 512

    # Chunk over W/reference samples at prediction time.
    w_predict_batch_size: int = 1024


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────


class _MLP(nn.Module):
    """Fully-connected MLP with ReLU on hidden layers, linear output."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()

        dims = [input_dim] + [hidden_dim] * n_layers + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < n_layers:  # hidden layers only
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _rbf_kernel_matrix(X: torch.Tensor, length_scale: float = 1.0) -> torch.Tensor:
    """
    Compute the nxn RBF (Gaussian) kernel matrix for a dataset of n row-vectors.

        K[i,j] = exp( -‖X[i] - X[j]‖² / (2 * length_scale²) )

    Args:
        X:            Tensor of shape (n, d).
        length_scale: Bandwidth parameter σ.

    Returns:
        K: Tensor of shape (n, n).
    """
    # Vectorised: use broadcasting over the last two dimensions.
    # X.T has shape (d, n); unsqueeze gives (d, n, 1) and (d, 1, n).
    t = X.T  # (d, n)
    diff = t.unsqueeze(2) - t.unsqueeze(1)  # (d, n, n)
    sq_dist = (diff**2).sum(dim=0)  # (n, n)
    return torch.exp(-sq_dist / (2.0 * (length_scale + 1e-8) ** 2))


def _nmmr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    K: torch.Tensor,
    loss: Literal["u_stat", "v_stat"],
) -> torch.Tensor:
    """
    Kernel-weighted moment loss.

        L = (r^T K r) / normaliser,  r = target - pred

    U-statistic: normaliser = n(n-1), diagonal of K zeroed out (unbiased).
    V-statistic: normaliser = n²,     diagonal kept (biased).

    Args:
        pred:   (n, 1) model predictions.
        target: (n, 1) observed outcomes.
        K:      (n, n) kernel matrix; not mutated.
        loss:   which statistic to use.

    Returns:
        Scalar loss tensor.
    """
    r = target - pred  # (n, 1)
    n = r.shape[0]

    if loss == "u_stat":
        if n <= 1:
            return pred.sum() * 0.0
        K = K.clone()
        K.fill_diagonal_(0.0)
        normaliser = n * (n - 1)
    else:
        normaliser = n * n

    return (r.T @ K @ r)[0, 0] / normaliser


# ─────────────────────────────────────────────────────────────────────────────
# Estimator
# ─────────────────────────────────────────────────────────────────────────────


class NMMREstimator:
    """
    Sklearn-style CATE estimator implementing NMMR.

    Usage::

        cfg = NMMRConfig(hidden_dim=128, n_epochs=200, lr=5e-4)
        est = NMMREstimator(cfg)
        est.fit(X_train, A_train, Y_train, W=W_train, Z=Z_train)
        cate = est.predict(X_test)   # shape (n_test,)

    The ``cfg`` dataclass is the single place to tune all hyperparameters.
    """

    def __init__(self, cfg: NMMRConfig = NMMRConfig()) -> None:
        self.cfg = cfg
        self._model: _MLP | None = None
        self._X_train: torch.Tensor | None = None
        self._W_train: torch.Tensor | None = None
        self._dim_A: int = 1
        self._device = self._select_device(self.cfg.device)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        *,
        W: np.ndarray,
        Z: np.ndarray,
    ) -> "NMMREstimator":
        """
        Train the NMMR model.

        Args:
            X: Covariates,               shape (n, d_X).
            A: Treatment (binary),  shape (n,) or (n, d_A).
            Y: Observed outcome,          shape (n,).
            W: Outcome-inducing proxy,    shape (n, d_W).
            Z: Treatment-inducing proxy,  shape (n, d_Z).

        Returns:
            self (for method chaining).
        """
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        X_t, A_t, Y_t, W_t, Z_t = self._to_tensors(X, A, Y, W, Z)

        self._W_train = W_t
        self._X_train = X_t
        self._dim_A = A_t.shape[1]

        self._Z_scale = torch.std(Z_t, dim=0, unbiased=False, keepdim=True).clamp_min(
            1e-6
        )
        self._X_scale = torch.std(X_t, dim=0, unbiased=False, keepdim=True).clamp_min(
            1e-6
        )

        input_dim = self._dim_A + W_t.shape[1] + X_t.shape[1]
        self._model = _MLP(input_dim, cfg.hidden_dim, cfg.n_layers).to(self._device)

        optimizer = optim.Adam(
            self._model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        n = X_t.shape[0]
        self._model.train()

        for _ in range(cfg.n_epochs):
            perm = torch.randperm(n, device=self._device)
            for start in range(0, n, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]

                bA, bW, bX, bZ, bY = A_t[idx], W_t[idx], X_t[idx], Z_t[idx], Y_t[idx]

                optimizer.zero_grad()

                pred = self._model(torch.cat([bA, bW, bX], dim=1))

                k_input = torch.cat(
                    [
                        bA,
                        bZ / self._Z_scale,
                        bX / self._X_scale,
                    ],
                    dim=1,
                )

                K = _rbf_kernel_matrix(
                    k_input,
                    length_scale=cfg.length_scale,
                )

                loss = _nmmr_loss(pred, bY, K, cfg.loss)
                loss.backward()
                optimizer.step()

        return self

    def score(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        *,
        W: np.ndarray,
        Z: np.ndarray,
        length_scale: float = 1.0,
        loss: Literal["u_stat", "v_stat"] = "v_stat",
    ) -> float:
        """Held-out kernel moment loss (lower is better) for model selection.

        The NMMR objective is E[(Y - h(A,W,X)) k(A,Z,X)] = 0, so the natural
        validation metric is the same kernel-weighted moment loss evaluated on
        rows the model was not trained on.

        Crucially, scoring uses a *fixed* kernel bandwidth and the V-statistic
        regardless of the trained model's own ``length_scale``/``loss``. This
        keeps scores comparable across candidates whose kernel knobs differ
        during a hyperparameter search — otherwise a wider training bandwidth
        would trivially rescale the loss.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before score().")

        X_t, A_t, Y_t, W_t, Z_t = self._to_tensors(X, A, Y, W, Z)

        self._model.eval()
        with torch.no_grad():
            pred = self._model(torch.cat([A_t, W_t, X_t], dim=1))
            k_input = torch.cat(
                [A_t, Z_t / self._Z_scale, X_t / self._X_scale],
                dim=1,
            )
            K = _rbf_kernel_matrix(k_input, length_scale=length_scale)
            val = _nmmr_loss(pred, Y_t, K, loss)

        return float(val)

    def predict(
        self,
        X: np.ndarray,
        *,
        W: np.ndarray | None = None,
        strategy: Literal["independent", "conditional"] | None = None,
    ) -> np.ndarray:
        """
        Predict treatment effect h(1,W,X)-h(0,W,X).

        If W is provided, returns the paired proxy-conditional contrast.

        If W is not provided:
            - independent: average over marginal empirical W_train.
            - conditional: approximate E[. | X=x] using kernel weights over X_train.
        """
        if self._model is None or self._W_train is None:
            raise RuntimeError("Call fit() before predict().")

        X_t = self._as_2d_tensor(X, name="X")
        strategy = strategy or self.cfg.w_strategy

        if W is not None:
            W_t = self._as_2d_tensor(W, name="W")
            return self._predict_effect_paired_w(X_t, W_t)

        if strategy == "independent":
            return self._predict_effect_independent_w(X_t)

        if strategy == "conditional":
            return self._predict_effect_conditional_w(X_t)

        raise ValueError(f"Unknown W prediction strategy: {strategy!r}")

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _select_device(device: str | None = None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _as_2d_tensor(self, arr: np.ndarray, *, name: str) -> torch.Tensor:
        arr = np.asarray(arr)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if arr.ndim != 2:
            raise ValueError(f"{name} must be 1D or 2D, got shape {arr.shape}.")

        return torch.tensor(arr, dtype=torch.float32, device=self._device)

    def _to_tensors(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        W: np.ndarray,
        Z: np.ndarray,
    ) -> tuple[torch.Tensor, ...]:
        def t(arr: np.ndarray, shape: tuple | None = None) -> torch.Tensor:
            out = torch.tensor(arr, dtype=torch.float32, device=self._device)
            if shape is not None:
                out = out.view(*shape)
            return out

        n = X.shape[0]
        return (
            t(X).view(n, -1) if X.ndim == 1 else t(X),
            t(A).view(n, -1),
            t(Y).view(n, 1),
            t(W).view(n, -1) if W.ndim == 1 else t(W),
            t(Z).view(n, -1) if Z.ndim == 1 else t(Z),
        )

    def _predict_effect_paired_w(
        self,
        X_t: torch.Tensor,
        W_t: torch.Tensor,
    ) -> np.ndarray:
        assert self._model is not None

        n = X_t.shape[0]

        if W_t.shape[0] == 1:
            W_t = W_t.expand(n, -1)
        elif W_t.shape[0] != n:
            raise ValueError(
                "When W is provided to predict(), it must either have one row "
                "or the same number of rows as X."
            )

        cate = np.empty(n, dtype=np.float32)

        self._model.eval()
        with torch.no_grad():
            for start in range(0, n, self.cfg.predict_batch_size):
                X_b = X_t[start : start + self.cfg.predict_batch_size]
                W_b = W_t[start : start + self.cfg.predict_batch_size]
                b = X_b.shape[0]

                A1 = torch.ones(b, self._dim_A, device=self._device)
                A0 = torch.zeros(b, self._dim_A, device=self._device)

                y1 = self._model(torch.cat([A1, W_b, X_b], dim=1))
                y0 = self._model(torch.cat([A0, W_b, X_b], dim=1))

                cate[start : start + b] = (y1 - y0).squeeze(1).cpu().numpy()

        return cate

    def _predict_effect_independent_w(self, X_t: torch.Tensor) -> np.ndarray:
        assert self._model is not None
        assert self._W_train is not None

        n_test = X_t.shape[0]
        n_W = self._W_train.shape[0]
        cate = np.empty(n_test, dtype=np.float32)

        self._model.eval()
        with torch.no_grad():
            for start in range(0, n_test, self.cfg.predict_batch_size):
                X_b = X_t[start : start + self.cfg.predict_batch_size]
                b = X_b.shape[0]

                effect_sum = torch.zeros(b, device=self._device)
                count = 0

                for w_start in range(0, n_W, self.cfg.w_predict_batch_size):
                    W_c = self._W_train[
                        w_start : w_start + self.cfg.w_predict_batch_size
                    ]
                    diff = self._effect_grid(X_b, W_c)  # shape (b, m)

                    effect_sum += diff.sum(dim=1)
                    count += W_c.shape[0]

                cate[start : start + b] = (effect_sum / count).cpu().numpy()

        return cate

    def _effect_grid(
        self,
        X_b: torch.Tensor,
        W_ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return h(1,W_j,X_i)-h(0,W_j,X_i) for every pair (X_i, W_j).

        Shape:
            X_b:   (b, d_X)
            W_ref: (m, d_W)

        Returns:
            diff:  (b, m)
        """
        assert self._model is not None

        b = X_b.shape[0]
        m = W_ref.shape[0]

        X_exp = X_b.repeat_interleave(m, dim=0)
        W_exp = W_ref.repeat(b, 1)

        A1 = torch.ones(b * m, self._dim_A, device=self._device)
        A0 = torch.zeros(b * m, self._dim_A, device=self._device)

        y1 = self._model(torch.cat([A1, W_exp, X_exp], dim=1)).view(b, m)
        y0 = self._model(torch.cat([A0, W_exp, X_exp], dim=1)).view(b, m)

        return y1 - y0

    def _predict_effect_conditional_w(self, X_t: torch.Tensor) -> np.ndarray:
        assert self._model is not None
        assert self._W_train is not None
        assert self._X_train is not None
        assert self._X_scale is not None

        cfg = self.cfg
        n_test = X_t.shape[0]
        cate = np.empty(n_test, dtype=np.float32)

        X_train_scaled = self._X_train / self._X_scale

        self._model.eval()
        with torch.no_grad():
            for start in range(0, n_test, cfg.predict_batch_size):
                X_b = X_t[start : start + cfg.predict_batch_size]
                b = X_b.shape[0]

                X_b_scaled = X_b / self._X_scale

                # Squared distances between test X and train X.
                dist2 = torch.cdist(X_b_scaled, X_train_scaled).pow(2)

                log_weights = -dist2 / (2.0 * (cfg.x_cond_length_scale + 1e-8) ** 2)

                if cfg.x_cond_top_k is not None:
                    k = min(cfg.x_cond_top_k, self._X_train.shape[0])
                    top_log_weights, top_idx = torch.topk(log_weights, k=k, dim=1)

                    weights = torch.softmax(top_log_weights, dim=1)  # (b, k)
                    W_local = self._W_train[top_idx]  # (b, k, d_W)
                else:
                    weights = torch.softmax(log_weights, dim=1)  # (b, n_train)
                    W_local = self._W_train.unsqueeze(0).expand(b, -1, -1)

                cate_b = self._effect_rowwise_weighted_w(X_b, W_local, weights)

                cate[start : start + b] = cate_b.cpu().numpy()

        return cate

    def _effect_rowwise_weighted_w(
        self,
        X_b: torch.Tensor,
        W_local: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted conditional effect for each row of X_b.

        Args:
            X_b:      (b, d_X)
            W_local:  (b, k, d_W), local/reference W values for each X row.
            weights:  (b, k), sums to 1 across k.

        Returns:
            cate:     (b,)
        """
        assert self._model is not None

        b, k, d_W = W_local.shape

        X_exp = X_b.unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1)
        W_exp = W_local.reshape(b * k, d_W)

        A1 = torch.ones(b * k, self._dim_A, device=self._device)
        A0 = torch.zeros(b * k, self._dim_A, device=self._device)

        y1 = self._model(torch.cat([A1, W_exp, X_exp], dim=1)).view(b, k)
        y0 = self._model(torch.cat([A0, W_exp, X_exp], dim=1)).view(b, k)

        diff = y1 - y0

        return (weights * diff).sum(dim=1)
