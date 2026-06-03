"""
validation_proximal_mechanisms.py

Standalone validation-set generator for proximal tabular foundation models.

Goal
----
Generate synthetic validation datasets that are:
    1. different from the training prior,
    2. not the final held-out test mechanisms,
    3. still compatible with the proximal causal prior,
    4. more realistic than smooth Gaussian toy DGFs.

Core proximal skeleton
----------------------
Each mechanism is built to approximately obey:

    X, U generated first
    Z = treatment-inducing proxy from X, U + noise
    W = outcome-inducing proxy from X, U + noise
    A ~ Bernoulli(sigmoid(g_A(X, U, Z)))
    Y(a) = g_Y(a, X, U, W) + noise

So:
    - Z helps predict treatment.
    - W helps predict outcome.
    - U is unobserved confounding.
    - W is pre-treatment.
    - Z has no direct structural effect on Y except through A/U/X.
    - W has no direct structural effect on A.

Saved files use the same sparse layout as the prior generator save_batch_sparse
format:

    U, X, W, Z, A, A_probs, Y_obs, Y0, Y1,
    d_U, d_X, d_W, d_Z, seq_lens, train_sizes, batch_size

Dense shapes before sparse saving:

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

Mechanisms
----------
1.  Linear heterogeneous treatment effects
2.  Smooth nonlinear continuous proxies
3.  Lognormal skewed proxies
4.  Dense rotated proxy subspaces
5.  Multiplicative latent-proxy interactions
6.  Trigonometric high-frequency response surfaces
7.  Mixed tabular variables: continuous, binary, ordinal-like
8.  Latent sites / centers / cluster effects
9.  Weak and ill-conditioned proxies
10. Limited overlap and threshold policies
11. Censored, positive, heavy-tailed outcomes
12. Train-query covariate shift under the same causal mechanism

Example
-------
from validation_proximal_mechanisms import (
    ValidationGeneratorConfig,
    ProximalValidationGenerator,
)

cfg = ValidationGeneratorConfig(
    save_dir="validation_batches",
    K=32,
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

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
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
    """Convert padded dense rows to compact 1D format expected by sparse2dense.

    For each row i, keep only x[i, :dims[i]], concatenate all kept values, and
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

    if torch.any(dims < 0):
        raise ValueError("dims contains negative entries")

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
    # [overlap_eps, 1 - overlap_eps].
    overlap_eps: float = 0.03

    # Clamp standardized logits before sigmoid for numerical stability.
    logit_clip: float = 6.0

    # Extra measurement noise on observed labels only.
    # Y0 and Y1 remain clean structural potential outcomes with shared shock.
    extra_observation_noise: float = 0.03
    missing_prob: float = 0.0

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
    """Generate and save proximal-validation DGF families.

    The class is intentionally self-contained. If you prefer to integrate with
    your existing prior class, copy generate_dataset plus the _mechanism_* methods
    and dispatch on params["mechanism_id"] or params["prior_type"].
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

        Returns exactly the 15 objects expected by save_batch_sparse:

            U, X, W, Z, A, A_probs, Y_obs, Y0, Y1,
            d_U, d_X, d_W, d_Z, seq_lens, train_sizes
        """
        if not 1 <= mechanism_id <= 12:
            raise ValueError("mechanism_id must be in {1, ..., 12}")

        B = int(batch_size or self.cfg.K)
        T = int(seq_len or self.cfg.seq_len)
        train_n = self._resolve_train_size(
            train_size if train_size is not None else self.cfg.train_size, T
        )

        results = []
        for ds_idx in range(B):
            seed = int(self.cfg.seed + 100_000 * mechanism_id + ds_idx)
            params = {
                "mechanism_id": mechanism_id,
                "seq_len": T,
                "train_size": train_n,
                "dataset_seed_index": ds_idx,
                "seed": seed,
            }
            with self._with_seed(seed):
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

        self._validate_batch_shapes(
            U=U,
            X=X,
            W=W,
            Z=Z,
            A=A,
            A_probs=A_probs,
            Y_obs=Y_obs,
            Y0=Y0,
            Y1=Y1,
            d_U=d_U,
            d_X=d_X,
            d_W=d_W,
            d_Z=d_Z,
            seq_lens=seq_lens,
            train_sizes=train_sizes,
        )

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
        save_metadata: bool = True,
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

        if save_metadata:
            self._write_metadata(mechanism_id=mechanism_id, batch_idx=batch_idx)

        self.save_dir = old_save_dir
        return path

    def save_all_mechanisms(
        self,
        mechanisms: Iterable[int] = range(1, 13),
        separate_dirs: bool = True,
    ) -> List[Path]:
        """Save each mechanism's K datasets as a separate batch.

        If separate_dirs=True, files are saved as:

            save_dir/mechanism_01/batch_000000.pt
            ...
            save_dir/mechanism_12/batch_000000.pt

        If separate_dirs=False, files are saved as:

            save_dir/batch_000000.pt  # mechanism 1
            ...
            save_dir/batch_000011.pt  # mechanism 12
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
        train_n = int(params.get("train_size", max(1, T // 2)))

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
        if mechanism_id == 7:
            return self._mechanism_7_mixed_tabular(T)
        if mechanism_id == 8:
            return self._mechanism_8_latent_sites(T)
        if mechanism_id == 9:
            return self._mechanism_9_weak_proxies(T)
        if mechanism_id == 10:
            return self._mechanism_10_limited_overlap_thresholds(T)
        if mechanism_id == 11:
            return self._mechanism_11_censored_heavy_tail(T)
        if mechanism_id == 12:
            return self._mechanism_12_train_query_shift(T, train_n)

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

    # ------------------------------------------------------------------
    # Mechanism 3: Skewed lognormal proxies
    # ------------------------------------------------------------------

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

        Z = torch.exp(Z_raw.clamp(-3.0, 3.0))
        W = torch.exp(W_raw.clamp(-3.0, 3.0))

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

    # ------------------------------------------------------------------
    # Mechanism 4: Dense rotated proxy subspaces
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Mechanism 5: Multiplicative latent-proxy interactions
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Mechanism 6: Trigonometric high-frequency response surfaces
    # ------------------------------------------------------------------

    def _mechanism_6_trig_interference(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X = self._standardize(self._randn(T, d_X))
        U = self._linear(X, d_U) + self._noise(T, d_U, sigma=0.8)

        Z = self._linear(X, d_Z) + U[:, [0]] * self._noise(T, d_Z, sigma=0.5)
        W = self._linear(X, d_W) + U[:, [0]] * self._noise(T, d_W, sigma=0.5)

        raw_logit = self._linear(X, 1) + self._linear(U, 1)
        p = self._propensity(raw_logit)

        X_proj = self._linear(X, 1)
        U_proj = self._linear(U, 1)
        W_proj = self._linear(W, 1)

        f0 = 2.0 * torch.sin(3.0 * X_proj * U_proj) + torch.cos(2.0 * W_proj)

        # Heavy-tailed but bounded CATE.
        tau = torch.exp(0.5 * torch.sin(self._linear(X, 1)))

        eps_y = self._noise(T, 1, sigma=self._log_unif_scalar(0.1, 0.5))
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ------------------------------------------------------------------
    # Mechanism 7: Mixed tabular variables
    # ------------------------------------------------------------------

    def _mechanism_7_mixed_tabular(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()

        X = self._correlated_X(T, d_X)

        # Convert some covariates into binary / ordinal-looking variables.
        if d_X >= 2:
            X[:, 0:1] = self._binary_columns_from_score(
                X[:, 0:1] + 0.5 * self._randn(T, 1)
            )
        if d_X >= 4:
            q = torch.quantile(X[:, 1], torch.tensor([0.33, 0.66], dtype=self.dtype))
            X[:, 1] = torch.bucketize(X[:, 1].contiguous(), q).to(self.dtype)

        U = 0.7 * self._mlp_feature(X, d_U) + self._outlier_noise(
            T, d_U, sigma=0.5, outlier_prob=0.02, outlier_scale=5.0
        )

        Z_score = self._mlp_feature(torch.cat([X, U], dim=1), d_Z)
        W_score = self._mlp_feature(torch.cat([X, U], dim=1), d_W)

        Z = Z_score + self._noise(T, d_Z, sigma=self._log_unif_scalar(0.2, 0.8))
        W = W_score + self._noise(T, d_W, sigma=self._log_unif_scalar(0.2, 0.8))

        # Make a few proxy columns binary-like.
        if d_Z >= 2:
            Z[:, 0:1] = self._binary_columns_from_score(Z[:, 0:1])
        if d_W >= 2:
            W[:, 0:1] = self._binary_columns_from_score(W[:, 0:1])

        raw_logit = (
            0.8 * self._mlp_feature(X, 1)
            + 0.9 * self._linear(U, 1)
            + 0.7 * Z[:, :1]
            + 0.4 * X[:, :1] * Z[:, :1]
        )
        p = self._propensity(raw_logit)

        f0 = (
            self._mlp_feature(X, 1)
            + 0.8 * torch.tanh(self._linear(U, 1))
            + 0.7 * torch.tanh(self._linear(W, 1))
        )

        tau = (
            0.5
            + 0.5 * torch.sin(self._linear(X, 1))
            + 0.2 * torch.tanh(self._linear(U, 1))
        )

        eps_y = self._outlier_noise(T, 1, sigma=self._log_unif_scalar(0.15, 0.5))
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ------------------------------------------------------------------
    # Mechanism 8: Latent sites / centers / clusters
    # ------------------------------------------------------------------

    def _mechanism_8_latent_sites(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()

        n_sites = self._randint(3, 8)
        site = torch.randint(0, n_sites, (T,), generator=self.rng)

        X = self._correlated_X(T, d_X)

        site_U = self._rsu(n_sites, d_U, low=0.3, high=1.2)
        site_A = self._rsu(n_sites, 1, low=0.2, high=1.0)
        site_Y = self._rsu(n_sites, 1, low=0.2, high=1.0)

        U = (
            self._linear(X, d_U)
            + site_U[site]
            + self._noise(T, d_U, sigma=self._log_unif_scalar(0.3, 0.9))
        )

        Z = (
            self._mlp_feature(torch.cat([X, U], dim=1), d_Z)
            + 0.5 * site_A[site]
            + self._noise(T, d_Z, sigma=0.4)
        )
        W = (
            self._mlp_feature(torch.cat([X, U], dim=1), d_W)
            + 0.5 * site_Y[site]
            + self._noise(T, d_W, sigma=0.4)
        )

        raw_logit = (
            self._linear(X, 1) + self._linear(U, 1) + 0.8 * Z[:, :1] + site_A[site]
        )
        p = self._propensity(raw_logit)

        subgroup = torch.where(X[:, :1] > 0.0, 1.0, -1.0)
        tau = 0.6 + 0.4 * subgroup + 0.3 * torch.tanh(self._linear(X, 1))

        f0 = (
            self._mlp_feature(X, 1)
            + torch.tanh(self._linear(U, 1))
            + 0.8 * torch.tanh(self._linear(W, 1))
            + site_Y[site]
        )

        eps_y = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 0.7))
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ------------------------------------------------------------------
    # Mechanism 9: Weak and ill-conditioned proxies
    # ------------------------------------------------------------------

    def _mechanism_9_weak_proxies(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()

        X = self._correlated_X(T, d_X)

        U = 0.5 * self._linear(X, d_U) + self._noise(
            T, d_U, sigma=self._log_unif_scalar(0.8, 1.5)
        )

        alpha_z = self._log_unif_scalar(0.05, 0.35)
        alpha_w = self._log_unif_scalar(0.05, 0.35)

        Z_signal = self._linear(torch.cat([X, U], dim=1), d_Z)
        W_signal = self._linear(torch.cat([X, U], dim=1), d_W)

        Z = alpha_z * Z_signal + self._noise(T, d_Z, sigma=1.0)
        W = alpha_w * W_signal + self._noise(T, d_W, sigma=1.0)

        raw_logit = 0.6 * self._linear(X, 1) + 1.4 * self._linear(U, 1) + 0.4 * Z[:, :1]
        p = self._propensity(raw_logit)

        f0 = (
            0.5 * self._linear(X, 1)
            + 1.5 * torch.tanh(self._linear(U, 1))
            + 0.4 * torch.tanh(self._linear(W, 1))
        )

        tau = (
            0.4
            + 0.5 * torch.sin(self._linear(X, 1))
            + 0.2 * X[:, :1].pow(2) / (1.0 + X[:, :1].pow(2))
        )

        eps_y = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 0.8))
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ------------------------------------------------------------------
    # Mechanism 10: Limited overlap and threshold policies
    # ------------------------------------------------------------------

    def _mechanism_10_limited_overlap_thresholds(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()

        X = self._correlated_X(T, d_X)
        U = self._mlp_feature(X, d_U) + self._noise(T, d_U, sigma=0.7)

        Z = self._mlp_feature(torch.cat([X, U], dim=1), d_Z) + self._noise(
            T, d_Z, sigma=0.35
        )
        W = self._mlp_feature(torch.cat([X, U], dim=1), d_W) + self._noise(
            T, d_W, sigma=0.35
        )

        x0 = X[:, :1]
        x1 = X[:, 1:2] if d_X >= 2 else X[:, :1]

        threshold_policy = 2.5 * (x0 > 0.4).to(self.dtype) - 2.0 * (x1 < -0.4).to(
            self.dtype
        )

        raw_logit = (
            threshold_policy
            + 1.2 * torch.tanh(self._linear(U, 1))
            + 0.8 * torch.tanh(Z[:, :1])
        )

        # Still uses overlap_eps, so positivity is stressed but not violated.
        p = self._propensity(raw_logit)

        f0 = (
            self._mlp_feature(X, 1)
            + torch.tanh(self._linear(U, 1))
            + 0.7 * torch.tanh(self._linear(W, 1))
        )

        tau = (
            0.3
            + 0.8 * (x0 > 0.0).to(self.dtype)
            - 0.4 * (x1 > 0.8).to(self.dtype)
            + 0.2 * torch.sin(self._linear(X, 1))
        )

        hetero_sigma = 0.15 + 0.6 * torch.sigmoid(torch.abs(self._linear(U, 1)))
        eps_y = hetero_sigma * self._randn(T, 1)

        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ------------------------------------------------------------------
    # Mechanism 11: Censored positive heavy-tailed outcomes
    # ------------------------------------------------------------------

    def _mechanism_11_censored_heavy_tail(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()

        X = self._correlated_X(T, d_X)
        U = self._linear(X, d_U) + self._outlier_noise(
            T, d_U, sigma=0.6, outlier_prob=0.04
        )

        Z = self._mlp_feature(torch.cat([X, U], dim=1), d_Z) + self._outlier_noise(
            T, d_Z, sigma=0.3
        )
        W = self._mlp_feature(torch.cat([X, U], dim=1), d_W) + self._outlier_noise(
            T, d_W, sigma=0.3
        )

        raw_logit = (
            0.7 * self._linear(X, 1)
            + 1.0 * torch.tanh(self._linear(U, 1))
            + 0.7 * torch.tanh(Z[:, :1])
        )
        p = self._propensity(raw_logit)

        latent_base = (
            self._mlp_feature(X, 1)
            + 0.8 * torch.tanh(self._linear(U, 1))
            + 0.8 * torch.tanh(self._linear(W, 1))
        )

        tau = 0.2 + 0.6 * torch.sigmoid(self._linear(X, 1)) + 0.4 * torch.sin(X[:, :1])

        eps_y = self._outlier_noise(
            T,
            1,
            sigma=self._log_unif_scalar(0.1, 0.4),
            outlier_prob=0.05,
            outlier_scale=7.0,
        )

        # Positive skew plus censoring/flooring.
        latent0 = latent_base + eps_y
        latent1 = latent_base + tau + eps_y

        Y0 = torch.log1p(torch.exp(latent0.clamp(-20.0, 20.0)))
        Y1 = torch.log1p(torch.exp(latent1.clamp(-20.0, 20.0)))

        floor = torch.quantile(Y0, 0.10)
        cap = torch.quantile(Y1, 0.98)

        Y0 = torch.clamp(Y0, min=floor, max=cap)
        Y1 = torch.clamp(Y1, min=floor, max=cap)

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ------------------------------------------------------------------
    # Mechanism 12: Train-query covariate shift
    # ------------------------------------------------------------------

    def _mechanism_12_train_query_shift(
        self, T: int, train_n: int
    ) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()

        train_n = max(1, min(int(train_n), T - 1))
        n_q = T - train_n

        X_train = self._correlated_X(train_n, d_X)

        shift = self._rsu(1, d_X, low=0.2, high=1.0)
        scale = self._uniform_scalar(1.1, 1.8)
        X_query = scale * self._correlated_X(n_q, d_X) + shift

        X = torch.cat([X_train, X_query], dim=0)
        X = self._standardize_train_ref(X, train_n)

        U = 0.6 * self._mlp_feature(X, d_U) + self._noise(
            T, d_U, sigma=self._log_unif_scalar(0.4, 1.0)
        )

        Z = self._mlp_feature(torch.cat([X, U], dim=1), d_Z) + self._noise(
            T, d_Z, sigma=0.4
        )
        W = self._mlp_feature(torch.cat([X, U], dim=1), d_W) + self._noise(
            T, d_W, sigma=0.4
        )

        raw_logit = (
            self._linear(X, 1)
            + self._linear(U, 1)
            + 0.8 * Z[:, :1]
            + 0.3 * X[:, :1] * Z[:, :1]
        )
        p = self._propensity(raw_logit)

        f0 = (
            self._mlp_feature(X, 1)
            + torch.tanh(self._linear(U, 1))
            + 0.7 * torch.tanh(self._linear(W, 1))
        )

        tau = 0.5 + 0.5 * torch.sin(self._linear(X, 1)) + 0.3 * torch.sigmoid(X[:, :1])

        eps_y = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 0.7))
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

        U = self._inject_missing(U)
        X = self._inject_missing(X)
        W = self._inject_missing(W)
        Z = self._inject_missing(Z)

        A_probs = (
            A_probs.view(T, 1)
            .to(self.dtype)
            .clamp(self.cfg.overlap_eps, 1.0 - self.cfg.overlap_eps)
        )
        A = (self._rand(T, 1) < A_probs).to(self.dtype)

        obs_noise = float(self.cfg.extra_observation_noise) * self._randn(T, 1)
        Y_obs = A * Y1 + (1.0 - A) * Y0 + obs_noise

        self._validate_dataset(
            U=U,
            X=X,
            W=W,
            Z=Z,
            A=A,
            A_probs=A_probs,
            Y_obs=Y_obs,
            Y0=Y0,
            Y1=Y1,
            d_U=d_U,
            d_X=d_X,
            d_W=d_W,
            d_Z=d_Z,
        )

        return (
            self._pad(U, self.cfg.max_dim_U),
            self._pad(X, self.cfg.max_dim_X),
            self._pad(W, self.cfg.max_dim_W),
            self._pad(Z, self.cfg.max_dim_Z),
            A.to(self.dtype),
            A_probs.to(self.dtype),
            Y_obs.to(self.dtype),
            Y0.to(self.dtype),
            Y1.to(self.dtype),
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

        if not (0.0 <= c.missing_prob < 1.0):
            raise ValueError("missing_prob must be in [0, 1)")

        if not (0.0 <= c.extra_observation_noise < 10.0):
            raise ValueError("extra_observation_noise looks invalid")

        if not (0.0 < c.overlap_eps < 0.5):
            raise ValueError("overlap_eps must be in (0, 0.5)")

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
            if not 0.0 < train_size < 1.0:
                if train_size == 1.0:
                    return T
                raise ValueError("float train_size must be in (0, 1]")
            return max(1, min(T - 1, int(round(train_size * T))))
        return max(1, min(T, int(train_size)))

    def _with_seed(self, seed: int):
        outer = self

        class SeedContext:
            def __init__(self, seed_value: int):
                self.seed_value = seed_value
                self.old_rng: Optional[torch.Generator] = None

            def __enter__(self):
                self.old_rng = outer.rng
                outer.rng = torch.Generator(device="cpu")
                outer.rng.manual_seed(int(self.seed_value))

            def __exit__(self, exc_type, exc, tb):
                outer.rng = self.old_rng

        return SeedContext(seed)

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
        if low <= 0.0 or high <= 0.0:
            raise ValueError("log-uniform bounds must be positive")
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
        # Supports negative ranges too, e.g. low=-0.5, high=0.5.
        if low < 0.0:
            return self._uniform_scalar(low, high)
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

    def _mlp_feature(
        self, x: Tensor, out_dim: int, hidden: Optional[int] = None
    ) -> Tensor:
        """Small random two-layer feature map."""
        if out_dim == 0:
            return self._empty(x.shape[0])
        if hidden is None:
            hidden = max(4, min(16, 2 * x.shape[1]))
        bias = self._rsu(1, hidden, low=0.0, high=0.5)
        h = torch.tanh(self._linear(x, hidden, low=0.2, high=1.2) + bias)
        return self._linear(h, out_dim, low=0.2, high=1.2)

    def _correlated_X(self, T: int, d_X: int) -> Tensor:
        """Low-rank correlated continuous covariates."""
        r = min(max(2, d_X // 2), 4)
        factors = self._randn(T, r)
        loadings = self._rsu(r, d_X, low=0.2, high=1.2)
        eps = 0.4 * self._randn(T, d_X)
        return self._standardize(factors @ loadings / math.sqrt(float(r)) + eps)

    def _noise(self, n: int, d: int, sigma: float = 1.0) -> Tensor:
        if d == 0:
            return self._empty(n)
        return float(sigma) * self._randn(n, d)

    def _outlier_noise(
        self,
        n: int,
        d: int,
        sigma: float = 1.0,
        outlier_prob: float = 0.03,
        outlier_scale: float = 8.0,
    ) -> Tensor:
        if d == 0:
            return self._empty(n)
        base = sigma * self._randn(n, d)
        mask = (self._rand(n, d) < outlier_prob).to(self.dtype)
        outliers = sigma * outlier_scale * self._randn(n, d)
        return base + mask * outliers

    def _standardize(self, x: Tensor, eps: float = 1e-6) -> Tensor:
        if x.numel() == 0:
            return x
        return (x - x.mean(dim=0, keepdim=True)) / (
            x.std(dim=0, keepdim=True).clamp_min(eps)
        )

    def _standardize_train_ref(
        self, x: Tensor, train_n: int, eps: float = 1e-6
    ) -> Tensor:
        if x.numel() == 0:
            return x
        ref = x[:train_n]
        mu = ref.mean(dim=0, keepdim=True)
        std = ref.std(dim=0, keepdim=True).clamp_min(eps)
        return (x - mu) / std

    def _propensity(self, raw_logit: Tensor) -> Tensor:
        z = self._standardize(raw_logit).clamp(
            -self.cfg.logit_clip, self.cfg.logit_clip
        )
        eps = self.cfg.overlap_eps
        return eps + (1.0 - 2.0 * eps) * torch.sigmoid(z)

    def _binary_columns_from_score(self, score: Tensor) -> Tensor:
        p = torch.sigmoid(self._standardize(score))
        return (self._rand(*p.shape) < p).to(self.dtype)

    def _inject_missing(self, x: Tensor) -> Tensor:
        """Simple mean-impute-to-zero missingness after standardization.

        This deliberately does not add a missingness mask, because the purpose is
        to stress whether the model can tolerate realistic zero/imputed values in
        tabular features. Keep missing_prob low for main validation.
        """
        p = float(self.cfg.missing_prob)
        if p <= 0.0 or x.numel() == 0:
            return x
        mask = self._rand(*x.shape) < p
        out = x.clone()
        out[mask] = 0.0
        return out

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

    # ------------------------------------------------------------------
    # Validation / metadata
    # ------------------------------------------------------------------

    def _validate_dataset(
        self,
        *,
        U: Tensor,
        X: Tensor,
        W: Tensor,
        Z: Tensor,
        A: Tensor,
        A_probs: Tensor,
        Y_obs: Tensor,
        Y0: Tensor,
        Y1: Tensor,
        d_U: int,
        d_X: int,
        d_W: int,
        d_Z: int,
    ) -> None:
        T = X.shape[0]
        expected = {
            "U": (T, d_U),
            "X": (T, d_X),
            "W": (T, d_W),
            "Z": (T, d_Z),
            "A": (T, 1),
            "A_probs": (T, 1),
            "Y_obs": (T, 1),
            "Y0": (T, 1),
            "Y1": (T, 1),
        }
        tensors = {
            "U": U,
            "X": X,
            "W": W,
            "Z": Z,
            "A": A,
            "A_probs": A_probs,
            "Y_obs": Y_obs,
            "Y0": Y0,
            "Y1": Y1,
        }
        for name, shape in expected.items():
            if tuple(tensors[name].shape) != shape:
                raise ValueError(
                    f"{name} has shape {tuple(tensors[name].shape)}, expected {shape}"
                )
            if not torch.isfinite(tensors[name]).all():
                raise ValueError(f"{name} contains NaN or Inf")

        if not ((A_probs >= 0.0).all() and (A_probs <= 1.0).all()):
            raise ValueError("A_probs outside [0, 1]")
        if not ((A == 0.0) | (A == 1.0)).all():
            raise ValueError("A is not binary")

    def _validate_batch_shapes(self, **tensors: Tensor) -> None:
        U = tensors["U"]
        X = tensors["X"]
        W = tensors["W"]
        Z = tensors["Z"]
        A = tensors["A"]
        A_probs = tensors["A_probs"]
        Y_obs = tensors["Y_obs"]
        Y0 = tensors["Y0"]
        Y1 = tensors["Y1"]

        B, T, _ = X.shape

        expected_shapes = {
            "U": (B, T, self.cfg.max_dim_U),
            "X": (B, T, self.cfg.max_dim_X),
            "W": (B, T, self.cfg.max_dim_W),
            "Z": (B, T, self.cfg.max_dim_Z),
            "A": (B, T, 1),
            "A_probs": (B, T, 1),
            "Y_obs": (B, T, 1),
            "Y0": (B, T, 1),
            "Y1": (B, T, 1),
        }
        actuals = {
            "U": U,
            "X": X,
            "W": W,
            "Z": Z,
            "A": A,
            "A_probs": A_probs,
            "Y_obs": Y_obs,
            "Y0": Y0,
            "Y1": Y1,
        }
        for name, expected in expected_shapes.items():
            actual = tuple(actuals[name].shape)
            if actual != expected:
                raise ValueError(f"{name} shape {actual}, expected {expected}")

    def _write_metadata(self, mechanism_id: int, batch_idx: int) -> None:
        metadata = {
            "mechanism_id": int(mechanism_id),
            "batch_idx": int(batch_idx),
            "config": {
                k: (str(v) if isinstance(v, Path) or isinstance(v, torch.dtype) else v)
                for k, v in asdict(self.cfg).items()
            },
            "mechanism_name": self.mechanism_name(mechanism_id),
        }
        path = self.save_dir / f"batch_{batch_idx:06d}_metadata.json"
        tmp = self.save_dir / f"batch_{batch_idx:06d}_metadata.json.tmp"
        with open(tmp, "w") as f:
            json.dump(metadata, f, indent=2)
        Path(tmp).replace(path)

    @staticmethod
    def mechanism_name(mechanism_id: int) -> str:
        names = {
            1: "linear_heterogeneous",
            2: "smooth_nonlinear",
            3: "lognormal_skew",
            4: "dense_rotation",
            5: "multiplicative",
            6: "trig_interference",
            7: "mixed_tabular",
            8: "latent_sites",
            9: "weak_proxies",
            10: "limited_overlap_thresholds",
            11: "censored_heavy_tail",
            12: "train_query_shift",
        }
        return names.get(int(mechanism_id), f"unknown_{mechanism_id}")
