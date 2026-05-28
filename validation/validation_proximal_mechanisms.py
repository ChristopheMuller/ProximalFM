"""
validation_proximal_mechanisms.py

A standalone validation-set generator for proximal tabular foundation models.

Each mechanism is saved as one batch containing K independently parameterized
synthetic datasets. Saved files use the same keys as your save_batch_sparse
format:

    U, X, W, Z, A, A_probs, Y_obs, Y0, Y1,
    d_U, d_X, d_W, d_Z, seq_lens, train_sizes, batch_size

Shapes before sparse saving are:

    U       : [K, T, max_dim_U]
    X       : [K, T, max_dim_X]
    W       : [K, T, max_dim_W]
    Z       : [K, T, max_dim_Z]
    A       : [K, T, 1]
    A_probs : [K, T, 1]
    Y_obs   : [K, T, 1]
    Y0      : [K, T, 1]
    Y1      : [K, T, 1]
    d_*     : [K]
    seq_lens, train_sizes : [K]

Example
-------
from validation_proximal_mechanisms import ValidationGeneratorConfig, ProximalValidationGenerator

cfg = ValidationGeneratorConfig(
    save_dir="validation_batches",
    K=10,
    seq_len=2048,
    train_size=0.5,
    min_dim_U=1,
    max_dim_U=3,
    min_dim_X=1,
    max_dim_X=8,
    min_dim_W=1,
    max_dim_W=8,
    min_dim_Z=1,
    max_dim_Z=8,
    seed=123,
)

gen = ProximalValidationGenerator(cfg)
paths = gen.save_all_mechanisms(separate_dirs=True)
print(paths)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import math

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Sparse helper compatible with the save_batch_sparse pattern in your generator.
# If you already have dense2sparse in your codebase, you can delete this helper
# and import your version instead.
# -----------------------------------------------------------------------------


def dense2sparse(x: Tensor, dims: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    """Convert padded dense rows to the compact 1D format expected by sparse2dense.

    This mirrors the sparse format used by proximalfm.prior._genload.sparse2dense:
    for each row i, keep only x[i, :dims[i]], concatenate all kept values, and
    return a single 1D tensor.

    Parameters
    ----------
    x:
        Dense tensor of shape [N, D].
    dims:
        Long tensor of shape [N], where dims[i] is the active dimensionality of
        row i. Entries with column index >= dims[i] are ignored.
    dtype:
        Value dtype for the compact tensor.
    """
    if x.ndim != 2:
        raise ValueError(f"dense2sparse expects [N, D], got shape {tuple(x.shape)}")

    device = x.device
    dims = dims.to(device=device, dtype=torch.long).view(-1)
    n, d = x.shape
    if dims.numel() != n:
        raise ValueError(f"dims has length {dims.numel()}, expected {n}")

    cols = torch.arange(d, device=device).view(1, d)
    mask = cols < dims.clamp(min=0, max=d).view(n, 1)
    return x[mask].to(dtype=dtype).contiguous()


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class ValidationGeneratorConfig:
    save_dir: str | Path

    # Number of independently parameterized datasets per mechanism batch.
    K: int = 10

    # Number of rows per synthetic dataset.
    seq_len: int = 2048

    # Either an integer number of training rows, or a float fraction of seq_len.
    train_size: int | float = 0.5

    min_dim_U: int = 1
    max_dim_U: int = 3
    min_dim_X: int = 1
    max_dim_X: int = 8
    min_dim_W: int = 1
    max_dim_W: int = 8
    min_dim_Z: int = 1
    max_dim_Z: int = 8

    # Strict positivity floor. Treatment probabilities are mapped to
    # [overlap_eps, 1 - overlap_eps]. Mechanism 8 uses this for stress tests too.
    overlap_eps: float = 0.03

    # Clamp standardized logits before sigmoid for numerical stability.
    logit_clip: float = 6.0

    seed: int = 123
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    # This implementation writes fixed-length dense batches, then sparsifies
    # U/X/W/Z on save. It intentionally does not use nested tensors.
    seq_len_per_gp: bool = False


# -----------------------------------------------------------------------------
# Main generator
# -----------------------------------------------------------------------------


class ProximalValidationGenerator:
    """Generate and save 10 proximal-validation DGF families.

    The class is intentionally self-contained. If you prefer to integrate with
    your existing prior class, copy `generate_dataset` plus the `_mechanism_*`
    methods, and dispatch on `params["mechanism_id"]` or `params["prior_type"]`.
    """

    def __init__(self, config: ValidationGeneratorConfig):
        self.cfg = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(int(config.seed))

        self.device = torch.device(config.device)
        self.dtype = config.dtype

        if config.seq_len_per_gp:
            raise NotImplementedError(
                "This validation generator currently writes fixed-length dense batches. "
                "Set seq_len_per_gp=False."
            )

        self._check_dims()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_batch(
        self,
        mechanism_id: int,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        train_size: Optional[int | float] = None,
    ) -> Tuple[Tensor, ...]:
        """Generate one mechanism batch with K independently randomized datasets.

        Returns exactly the 16 objects expected by save_batch_sparse:

            U, X, W, Z, A, A_probs, Y_obs, Y0, Y1,
            d_U, d_X, d_W, d_Z, seq_lens, train_sizes
        """
        if not 1 <= mechanism_id <= 10:
            raise ValueError("mechanism_id must be in {1, ..., 10}")

        B = int(batch_size or self.cfg.K)
        T = int(seq_len or self.cfg.seq_len)
        train_n = self._resolve_train_size(
            train_size if train_size is not None else self.cfg.train_size, T
        )

        results = []
        for ds_idx in range(B):
            params = {
                "mechanism_id": mechanism_id,
                "seq_len": T,
                "train_size": train_n,
                "dataset_seed_index": ds_idx,
            }
            results.append(self.generate_dataset(params))

        (
            U_list,
            X_list,
            W_list,
            Z_list,
            A_list,
            A_probs_list,
            Y_obs_list,
            Y0_list,
            Y1_list,
            d_U_list,
            d_X_list,
            d_W_list,
            d_Z_list,
        ) = zip(*results)

        U = torch.stack(U_list).to(self.device)
        X = torch.stack(X_list).to(self.device)
        W = torch.stack(W_list).to(self.device)
        Z = torch.stack(Z_list).to(self.device)
        A = torch.stack(A_list).to(self.device)
        A_probs = torch.stack(A_probs_list).to(self.device)
        Y_obs = torch.stack(Y_obs_list).to(self.device)
        Y0 = torch.stack(Y0_list).to(self.device)
        Y1 = torch.stack(Y1_list).to(self.device)

        d_U = torch.stack(d_U_list).to(self.device)
        d_X = torch.stack(d_X_list).to(self.device)
        d_W = torch.stack(d_W_list).to(self.device)
        d_Z = torch.stack(d_Z_list).to(self.device)

        seq_lens = torch.full((B,), T, device=self.device, dtype=torch.long)
        train_sizes = torch.full((B,), train_n, device=self.device, dtype=torch.long)

        return (
            U,
            X,
            W,
            Z,
            A,
            A_probs,
            Y_obs,
            Y0,
            Y1,
            d_U,
            d_X,
            d_W,
            d_Z,
            seq_lens,
            train_sizes,
        )

    def save_mechanism_batch(
        self,
        mechanism_id: int,
        batch_idx: int = 0,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        train_size: Optional[int | float] = None,
        save_dir: Optional[str | Path] = None,
    ) -> Path:
        """Generate and save one mechanism as one batch file."""
        old_save_dir = self.save_dir
        if save_dir is not None:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)

        batch = self.get_batch(
            mechanism_id=mechanism_id,
            batch_size=batch_size,
            seq_len=seq_len,
            train_size=train_size,
        )
        self.save_batch_sparse(batch_idx, *batch)
        path = self.save_dir / f"batch_{batch_idx:06d}.pt"

        self.save_dir = old_save_dir
        return path

    def save_all_mechanisms(
        self,
        mechanisms: Iterable[int] = range(1, 7),
        separate_dirs: bool = True,
    ) -> List[Path]:
        """Save each mechanism's K datasets as a separate batch.

        If separate_dirs=True, files are saved as:

            save_dir/mechanism_01/batch_000000.pt
            ...
            save_dir/mechanism_05/batch_000000.pt

        If separate_dirs=False, files are saved as:

            save_dir/batch_000000.pt  # mechanism 1
            ...
            save_dir/batch_000004.pt  # mechanism 5
        """
        paths: List[Path] = []
        root = self.save_dir

        for idx, mechanism_id in enumerate(mechanisms):
            if separate_dirs:
                mech_dir = root / f"mechanism_{mechanism_id:02d}"
                path = self.save_mechanism_batch(
                    mechanism_id=mechanism_id,
                    batch_idx=0,
                    save_dir=mech_dir,
                )
            else:
                path = self.save_mechanism_batch(
                    mechanism_id=mechanism_id,
                    batch_idx=idx,
                    save_dir=root,
                )
            paths.append(path)

        return paths

    def save_batch_sparse(
        self,
        batch_idx: int,
        U: Tensor,
        X: Tensor,
        W: Tensor,
        Z: Tensor,
        A: Tensor,
        A_probs: Tensor,
        Y_obs: Tensor,
        Y0: Tensor,
        Y1: Tensor,
        d_U: Tensor,
        d_X: Tensor,
        d_W: Tensor,
        d_Z: Tensor,
        seq_lens: Tensor,
        train_sizes: Tensor,
    ) -> None:
        """Save batch data in the sparse format used by your training generator."""
        B, T, _ = X.shape

        U_sparse = dense2sparse(
            U.reshape(-1, U.shape[-1]),
            d_U.repeat_interleave(T),
            dtype=torch.float32,
        )
        X_sparse = dense2sparse(
            X.reshape(-1, X.shape[-1]),
            d_X.repeat_interleave(T),
            dtype=torch.float32,
        )
        W_sparse = dense2sparse(
            W.reshape(-1, W.shape[-1]),
            d_W.repeat_interleave(T),
            dtype=torch.float32,
        )
        Z_sparse = dense2sparse(
            Z.reshape(-1, Z.shape[-1]),
            d_Z.repeat_interleave(T),
            dtype=torch.float32,
        )

        batch_file = self.save_dir / f"batch_{batch_idx:06d}.pt"
        temp_file = self.save_dir / f"batch_{batch_idx:06d}.pt.tmp"

        torch.save(
            {
                "U": U_sparse.cpu(),
                "X": X_sparse.cpu(),
                "W": W_sparse.cpu(),
                "Z": Z_sparse.cpu(),
                "A": A.cpu(),
                "A_probs": A_probs.cpu(),
                "Y_obs": Y_obs.cpu(),
                "Y0": Y0.cpu(),
                "Y1": Y1.cpu(),
                "d_U": d_U.cpu(),
                "d_X": d_X.cpu(),
                "d_W": d_W.cpu(),
                "d_Z": d_Z.cpu(),
                "seq_lens": seq_lens.cpu(),
                "train_sizes": train_sizes.cpu(),
                "batch_size": B,
            },
            temp_file,
        )
        temp_file.replace(batch_file)

    # ------------------------------------------------------------------
    # Dataset dispatch
    # ------------------------------------------------------------------

    def generate_dataset(self, params: Dict) -> Tuple[Tensor, ...]:
        mechanism_id = int(params["mechanism_id"])
        T = int(params["seq_len"])

        if mechanism_id == 1:
            return self._mechanism_1_linear_heterogeneous(T)
        if mechanism_id == 2:
            return self._mechanism_2_smooth_nonlinear(T)
        if mechanism_id == 3:
            return self._mechanism_3_lognormal_skew(T)
        if mechanism_id == 4:
            return self._mechanism_4_dense_rotation(T)
        if mechanism_id == 5:
            return self._mechanism_5_multiplicative(T)
        if mechanism_id == 6:
            return self._mechanism_6_trig_interference(T)

        raise ValueError(f"Unknown mechanism_id={mechanism_id}")

    # ------------------------------------------------------------------
    # Mechanism 1: Linear heterogeneous treatment effect
    # ------------------------------------------------------------------

    def _mechanism_1_linear_heterogeneous(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X, U, Z, W = self._linear_proxy_skeleton(T, d_U, d_X, d_W, d_Z)

        raw_logit = (
            self._linear(X, 1)
            + self._linear(U, 1)
            + self._linear(Z, 1)
            + self._rsu_scalar(-0.5, 0.5)
        )
        p = self._propensity(raw_logit)

        tau = self._rsu_scalar(0.2, 1.5) + self._linear(X, 1, low=0.1, high=0.8)
        if d_X >= 2:
            tau = tau + self._rsu_scalar(0.1, 0.8) * (X[:, :1] * X[:, 1:2])

        f0 = (
            self._linear(X, 1)
            + self._linear(U, 1)
            + self._linear(W, 1)
            + self._rsu_scalar(-0.5, 0.5)
        )
        eps_y = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 1.0))
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ------------------------------------------------------------------
    # Mechanism 2: Smooth nonlinear continuous proxies
    # ------------------------------------------------------------------

    def _mechanism_2_smooth_nonlinear(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X = self._standardize(self._randn(T, d_X))
        U = torch.tanh(self._linear(X, d_U)) + self._noise(T, d_U, sigma=0.6)

        Z = torch.sin(self._linear(U, d_Z) + self._linear(X, d_Z)) + self._noise(
            T, d_Z, sigma=0.3
        )
        W = torch.tanh(self._linear(U, d_W) + self._linear(X, d_W)) + self._noise(
            T, d_W, sigma=0.3
        )

        interaction = Z[:, :1] * X[:, :1]
        raw_logit = (
            self._linear(X, 1)
            + self._linear(torch.tanh(U), 1)
            + self._linear(Z, 1)
            + self._rsu_scalar(0.2, 1.0) * interaction
        )
        p = self._propensity(raw_logit)

        f0 = (
            torch.sin(self._linear(X, 1))
            + torch.tanh(self._linear(U, 1))
            + self._linear(W, 1)
            + 0.3 * W[:, :1].pow(2)
        )
        tau = (
            self._rsu_scalar(0.3, 1.5)
            + 0.5 * torch.sin(self._linear(X, 1))
            + 0.3 * X[:, :1].pow(2) / (1.0 + X[:, :1].pow(2))
        )

        eps_y = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 0.8))
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    def _mechanism_3_lognormal_skew(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X = self._standardize(self._randn(T, d_X))
        U = self._linear(X, d_U) + self._noise(T, d_U, sigma=0.8)

        Z_raw = (
            0.5 * self._linear(X, d_Z)
            + 0.5 * self._linear(U, d_Z)
            + self._noise(T, d_Z, sigma=0.3)
        )
        W_raw = (
            0.5 * self._linear(X, d_W)
            + 0.5 * self._linear(U, d_W)
            + self._noise(T, d_W, sigma=0.3)
        )

        Z = torch.exp(Z_raw)
        W = torch.exp(W_raw)

        raw_logit = self._linear(X, 1) + self._linear(U, 1) + self._linear(Z_raw, 1)
        p = self._propensity(raw_logit)

        f0 = torch.log1p(X.abs().sum(dim=1, keepdim=True)) * U[:, :1] + 0.1 * W_raw.pow(
            2
        ).sum(dim=1, keepdim=True)
        tau = 0.5 * X[:, :1].pow(2) + torch.tanh(self._linear(X, 1))

        eps_y = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 1.0))
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    def _mechanism_4_dense_rotation(self, T: int) -> Tuple[Tensor, ...]:

        d_U, d_X, d_W, d_Z = self._sample_dims()

        X = self._standardize(self._randn(T, d_X))
        U = self._linear(X, d_U) + self._noise(T, d_U, sigma=0.5)

        Z_signal = self._linear(U, d_U)
        Z_noise = self._noise(T, d_Z - d_U, sigma=1.0)
        Z_base = torch.cat([Z_signal, Z_noise], dim=1)

        W_signal = self._linear(U, d_U)
        W_noise = self._noise(T, d_W - d_U, sigma=1.0)
        W_base = torch.cat([W_signal, W_noise], dim=1)

        Q_Z, _ = torch.linalg.qr(self._randn(d_Z, d_Z))
        Q_W, _ = torch.linalg.qr(self._randn(d_W, d_W))

        Z = Z_base @ Q_Z
        W = W_base @ Q_W

        raw_logit = (
            self._linear(X, 1) + self._linear(U, 1) + self._rsu_scalar() * Z[:, :1]
        )
        p = self._propensity(raw_logit)

        f0 = self._linear(X, 1) + self._linear(U, 1) + self._rsu_scalar() * W[:, :1]
        tau = self._rsu_scalar(0.5, 2.0) + self._linear(X, 1, low=0.1, high=0.5)

        eps_y = self._noise(T, 1, sigma=0.5)
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    def _mechanism_5_multiplicative(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X = self._standardize(self._randn(T, d_X))
        U = self._linear(X, d_U) + self._noise(T, d_U, sigma=0.5)

        X_proj_Z = self._linear(X, d_U)
        X_proj_W = self._linear(X, d_U)

        Z_core = U * torch.sin(X_proj_Z)
        W_core = U * torch.exp(-torch.abs(X_proj_W))

        Z = self._linear(Z_core, d_Z) + self._noise(T, d_Z, sigma=0.3)
        W = self._linear(W_core, d_W) + self._noise(T, d_W, sigma=0.3)

        raw_logit = (
            self._linear(X, 1) + self._linear(U, 1) - 0.5 * Z.mean(dim=1, keepdim=True)
        )
        p = self._propensity(raw_logit)

        f0 = self._linear(X, 1) + self._linear(U, 1) + 0.5 * W.mean(dim=1, keepdim=True)
        tau = torch.ones(T, 1, dtype=self.dtype) * self._rsu_scalar(0.5, 1.5)

        eps_y = self._noise(T, 1, sigma=0.5)
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    def _mechanism_6_trig_interference(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X = self._standardize(self._randn(T, d_X))
        U = self._linear(X, d_U) + self._noise(T, d_U, sigma=0.8)

        Z = self._linear(X, d_Z) + U[:, [0 % d_U]] * self._noise(T, d_Z, sigma=0.5)
        W = self._linear(X, d_W) + U[:, [0 % d_U]] * self._noise(T, d_W, sigma=0.5)

        raw_logit = self._linear(X, 1) + self._linear(U, 1)
        p = self._propensity(raw_logit)

        # f0 involves high-frequency waves and multiplicative interactions
        X_proj = self._linear(X, 1)
        U_proj = self._linear(U, 1)
        W_proj = self._linear(W, 1)

        f0 = 2.0 * torch.sin(3.0 * X_proj * U_proj) + torch.cos(2.0 * W_proj)

        # Exponential CATE to create heavy tails in the treatment effect
        tau = torch.exp(0.5 * torch.sin(self._linear(X, 1)))

        eps_y = self._noise(T, 1, sigma=self._log_unif_scalar(0.1, 0.5))
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ------------------------------------------------------------------
    # Shared DGF helpers
    # ------------------------------------------------------------------

    def _linear_proxy_skeleton(
        self, T: int, d_U: int, d_X: int, d_W: int, d_Z: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        X = self._standardize(self._randn(T, d_X))
        U = self._linear(X, d_U) + self._noise(
            T, d_U, sigma=self._log_unif_scalar(0.4, 1.0)
        )
        Z = (
            self._linear(X, d_Z)
            + self._linear(U, d_Z)
            + self._noise(T, d_Z, sigma=self._log_unif_scalar(0.2, 1.0))
        )
        W = (
            self._linear(X, d_W)
            + self._linear(U, d_W)
            + self._noise(T, d_W, sigma=self._log_unif_scalar(0.2, 1.0))
        )
        return X, U, Z, W

    def _finalize(
        self,
        U: Tensor,
        X: Tensor,
        W: Tensor,
        Z: Tensor,
        A_probs: Tensor,
        Y0: Tensor,
        Y1: Tensor,
        d_U: int,
        d_X: int,
        d_W: int,
        d_Z: int,
    ) -> Tuple[Tensor, ...]:
        T = X.shape[0]
        A_probs = (
            A_probs.view(T, 1)
            .to(self.dtype)
            .clamp(self.cfg.overlap_eps, 1.0 - self.cfg.overlap_eps)
        )
        A = (self._rand(T, 1) < A_probs).to(self.dtype)
        Y_obs = A * Y1 + (1.0 - A) * Y0

        return (
            self._pad(U, self.cfg.max_dim_U),
            self._pad(X, self.cfg.max_dim_X),
            self._pad(W, self.cfg.max_dim_W),
            self._pad(Z, self.cfg.max_dim_Z),
            A.squeeze(-1),
            A_probs.squeeze(-1),
            Y_obs.to(self.dtype).squeeze(-1),
            Y0.to(self.dtype).squeeze(-1),
            Y1.to(self.dtype).squeeze(-1),
            torch.tensor(d_U, dtype=torch.long),
            torch.tensor(d_X, dtype=torch.long),
            torch.tensor(d_W, dtype=torch.long),
            torch.tensor(d_Z, dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # Random/math utilities
    # ------------------------------------------------------------------

    def _check_dims(self) -> None:
        c = self.cfg
        for name in ["U", "X", "W", "Z"]:
            lo = getattr(c, f"min_dim_{name}")
            hi = getattr(c, f"max_dim_{name}")
            if lo < 1 or hi < lo:
                raise ValueError(f"Invalid dimensions for {name}: min={lo}, max={hi}")
        if c.max_dim_W < max(c.min_dim_W, c.min_dim_U):
            raise ValueError(
                "max_dim_W should be >= max(min_dim_W, min_dim_U) for completeness-style validation"
            )
        if c.max_dim_Z < max(c.min_dim_Z, c.min_dim_U):
            raise ValueError(
                "max_dim_Z should be >= max(min_dim_Z, min_dim_U) for completeness-style validation"
            )

    def _sample_dims(
        self, force_d_U: Optional[int] = None
    ) -> Tuple[int, int, int, int]:
        c = self.cfg
        if force_d_U is None:
            d_U = self._randint(c.min_dim_U, c.max_dim_U)
        else:
            d_U = int(force_d_U)
            if not (1 <= d_U <= c.max_dim_U):
                raise ValueError(
                    f"force_d_U={d_U} incompatible with max_dim_U={c.max_dim_U}"
                )

        d_X = self._randint(c.min_dim_X, c.max_dim_X)

        min_W = max(c.min_dim_W, d_U)
        min_Z = max(c.min_dim_Z, d_U)
        if min_W > c.max_dim_W or min_Z > c.max_dim_Z:
            raise ValueError(
                f"Cannot enforce dim(W), dim(Z) >= dim(U). Got d_U={d_U}, "
                f"max_dim_W={c.max_dim_W}, max_dim_Z={c.max_dim_Z}."
            )
        d_W = self._randint(min_W, c.max_dim_W)
        d_Z = self._randint(min_Z, c.max_dim_Z)
        return d_U, d_X, d_W, d_Z

    def _resolve_train_size(self, train_size: int | float, T: int) -> int:
        if isinstance(train_size, float):
            if not 0.0 < train_size <= 1.0:
                raise ValueError("float train_size must be in (0, 1]")
            return max(1, min(T, int(round(train_size * T))))
        return max(1, min(T, int(train_size)))

    def _rand(self, *shape: int) -> Tensor:
        return torch.rand(*shape, generator=self.rng, dtype=self.dtype)

    def _randn(self, *shape: int) -> Tensor:
        return torch.randn(*shape, generator=self.rng, dtype=self.dtype)

    def _randint(self, low: int, high: int) -> int:
        if high < low:
            high = low
        return int(torch.randint(low, high + 1, (1,), generator=self.rng).item())

    def _uniform_scalar(self, low: float, high: float) -> float:
        return float(low + (high - low) * self._rand(1).item())

    def _log_unif_scalar(self, low: float, high: float) -> float:
        return float(
            math.exp(
                math.log(low) + (math.log(high) - math.log(low)) * self._rand(1).item()
            )
        )

    def _rsu(self, *shape: int, low: float = 0.3, high: float = 1.5) -> Tensor:
        """Random signed uniform tensor."""
        mag = low + (high - low) * self._rand(*shape)
        signs = torch.where(self._rand(*shape) < 0.5, -1.0, 1.0).to(self.dtype)
        return signs * mag

    def _rsu_scalar(self, low: float = 0.3, high: float = 1.5) -> float:
        mag = self._uniform_scalar(low, high)
        sign = -1.0 if self._rand(1).item() < 0.5 else 1.0
        return float(sign * mag)

    def _linear(
        self, x: Tensor, out_dim: int, low: float = 0.3, high: float = 1.5
    ) -> Tensor:
        if out_dim == 0:
            return self._empty(x.shape[0])
        if x.ndim != 2:
            raise ValueError(f"_linear expects [N, D], got {tuple(x.shape)}")
        in_dim = x.shape[1]
        if in_dim == 0:
            return torch.zeros(x.shape[0], out_dim, dtype=self.dtype)
        w = self._rsu(in_dim, out_dim, low=low, high=high) / math.sqrt(float(in_dim))
        return x @ w

    def _noise(self, n: int, d: int, sigma: float = 1.0) -> Tensor:
        if d == 0:
            return self._empty(n)
        return float(sigma) * self._randn(n, d)

    def _standardize(self, x: Tensor, eps: float = 1e-6) -> Tensor:
        if x.numel() == 0:
            return x
        return (x - x.mean(dim=0, keepdim=True)) / (
            x.std(dim=0, keepdim=True).clamp_min(eps)
        )

    def _propensity(self, raw_logit: Tensor) -> Tensor:
        z = self._standardize(raw_logit).clamp(
            -self.cfg.logit_clip, self.cfg.logit_clip
        )
        eps = self.cfg.overlap_eps
        return eps + (1.0 - 2.0 * eps) * torch.sigmoid(z)

    def _pad(self, x: Tensor, max_dim: int) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"_pad expects [T, d], got {tuple(x.shape)}")
        T, d = x.shape
        if d > max_dim:
            raise ValueError(f"Active dim {d} exceeds max_dim={max_dim}")
        out = torch.zeros(T, max_dim, dtype=self.dtype)
        out[:, :d] = x.to(self.dtype)
        return out

    def _empty(self, n: int) -> Tensor:
        return torch.empty(n, 0, dtype=self.dtype)
