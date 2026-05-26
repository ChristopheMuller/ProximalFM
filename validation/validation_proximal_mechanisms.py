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
        mechanisms: Iterable[int] = range(1, 11),
        separate_dirs: bool = True,
    ) -> List[Path]:
        """Save each mechanism's K datasets as a separate batch.

        If separate_dirs=True, files are saved as:

            save_dir/mechanism_01/batch_000000.pt
            ...
            save_dir/mechanism_10/batch_000000.pt

        If separate_dirs=False, files are saved as:

            save_dir/batch_000000.pt  # mechanism 1
            ...
            save_dir/batch_000009.pt  # mechanism 10
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
            return self._mechanism_1_linear_gaussian(T)
        if mechanism_id == 2:
            return self._mechanism_2_linear_heterogeneous(T)
        if mechanism_id == 3:
            return self._mechanism_3_smooth_nonlinear(T)
        if mechanism_id == 4:
            return self._mechanism_4_categorical(T)
        if mechanism_id == 5:
            return self._mechanism_5_mixed_tabular(T)
        if mechanism_id == 6:
            return self._mechanism_6_latent_effect_modification(T)
        if mechanism_id == 7:
            return self._mechanism_7_weak_redundant_omitted_edges(T)
        if mechanism_id == 8:
            return self._mechanism_8_positivity_stress(T)
        if mechanism_id == 9:
            return self._mechanism_9_multimodal_mixture(T)
        if mechanism_id == 10:
            return self._mechanism_10_two_timepoint_summary(T)

        raise ValueError(f"Unknown mechanism_id={mechanism_id}")

    # ------------------------------------------------------------------
    # Mechanism 1: Linear-Gaussian proximal baseline
    # ------------------------------------------------------------------

    def _mechanism_1_linear_gaussian(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X, U, Z, W = self._linear_proxy_skeleton(T, d_U, d_X, d_W, d_Z)

        raw_logit = (
            self._linear(X, 1)
            + self._linear(U, 1)
            + self._linear(Z, 1)
            + self._rsu_scalar(-0.5, 0.5)
        )
        p = self._propensity(raw_logit)

        tau = self._rsu_scalar(0.3, 2.0) * torch.ones(T, 1, dtype=self.dtype)
        f0 = (
            self._rsu_scalar(-0.5, 0.5)
            + self._linear(X, 1)
            + self._linear(U, 1)
            + self._linear(W, 1)
        )
        eps_y = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 1.0))
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ------------------------------------------------------------------
    # Mechanism 2: Linear heterogeneous treatment effect
    # ------------------------------------------------------------------

    def _mechanism_2_linear_heterogeneous(self, T: int) -> Tuple[Tensor, ...]:
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
    # Mechanism 3: Smooth nonlinear continuous proxies
    # ------------------------------------------------------------------

    def _mechanism_3_smooth_nonlinear(self, T: int) -> Tuple[Tensor, ...]:
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
    # Mechanism 4: Categorical latent confounder with categorical proxies
    # ------------------------------------------------------------------

    def _mechanism_4_categorical(self, T: int) -> Tuple[Tensor, ...]:
        # U is stored as one scalar code column, but internally has M categories.
        d_U, d_X, d_W, d_Z = self._sample_dims(force_d_U=1)
        max_categories = max(1, min(5, d_Z, d_W))
        M = max_categories

        X = self._standardize(self._randn(T, d_X))

        if M == 1:
            # Degenerate fallback if max_dim_W=max_dim_Z=1. Prefer max dims >=2.
            u_idx = torch.zeros(T, dtype=torch.long)
            U_oh = torch.ones(T, 1, dtype=self.dtype)
        else:
            u_logits = self._linear(X, M) + self._randn(1, M) * 0.2
            u_idx = torch.multinomial(
                torch.softmax(u_logits, dim=1), 1, generator=self.rng
            ).squeeze(1)
            U_oh = F.one_hot(u_idx, num_classes=M).to(self.dtype)

        denom = float(max(M - 1, 1))
        U = 2.0 * (u_idx.to(self.dtype).view(T, 1) / denom) - 1.0

        kappa_z = self._uniform_scalar(1.0, 4.0)
        kappa_w = self._uniform_scalar(1.0, 4.0)

        z_logits = kappa_z * U_oh + self._linear(X, M, low=0.05, high=0.4)
        w_logits = kappa_w * U_oh + self._linear(X, M, low=0.05, high=0.4)

        z_idx = torch.multinomial(
            torch.softmax(z_logits, dim=1), 1, generator=self.rng
        ).squeeze(1)
        w_idx = torch.multinomial(
            torch.softmax(w_logits, dim=1), 1, generator=self.rng
        ).squeeze(1)

        Z = self._noise(T, d_Z, sigma=0.05)
        W = self._noise(T, d_W, sigma=0.05)
        Z[:, :M] = F.one_hot(z_idx, num_classes=M).to(self.dtype)
        W[:, :M] = F.one_hot(w_idx, num_classes=M).to(self.dtype)

        alpha_u = self._rsu(M, 1)
        beta_u = self._rsu(M, 1)
        raw_logit = self._linear(X, 1) + U_oh @ alpha_u + self._linear(Z, 1)
        p = self._propensity(raw_logit)

        tau = self._rsu_scalar(0.2, 1.5) + self._linear(X, 1, low=0.1, high=0.7)
        f0 = self._linear(X, 1) + U_oh @ beta_u + self._linear(W, 1)
        eps_y = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 0.8))
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ------------------------------------------------------------------
    # Mechanism 5: Mixed tabular data: continuous, binary, categorical, count
    # ------------------------------------------------------------------

    def _mechanism_5_mixed_tabular(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()

        d_cont = max(1, d_X // 2)
        d_bin = max(0, (d_X - d_cont) // 2)
        d_ord = d_X - d_cont - d_bin

        X_cont = self._standardize(self._randn(T, d_cont))
        parts = [X_cont]

        if d_bin > 0:
            p_bin = torch.sigmoid(self._linear(X_cont, d_bin))
            X_bin = (self._rand(T, d_bin) < p_bin).to(self.dtype)
            parts.append(X_bin)

        if d_ord > 0:
            latent_ord = self._linear(X_cont, d_ord) + self._noise(T, d_ord, sigma=0.8)
            X_ord = torch.bucketize(
                latent_ord, torch.tensor([-0.7, 0.7], dtype=self.dtype)
            ).to(self.dtype)
            X_ord = X_ord - 1.0  # values in {-1, 0, 1}
            parts.append(X_ord)

        X = torch.cat(parts, dim=1)
        X = X[:, :d_X]

        U = self._linear(X, d_U) + self._noise(T, d_U, sigma=0.7)

        # Z: first block continuous, second block binary.
        d_Z_cont = max(1, d_Z // 2)
        d_Z_bin = d_Z - d_Z_cont
        Z_cont = (
            self._linear(X, d_Z_cont)
            + self._linear(U, d_Z_cont)
            + self._noise(T, d_Z_cont, sigma=0.4)
        )
        z_parts = [Z_cont]
        if d_Z_bin > 0:
            p_zb = torch.sigmoid(self._linear(X, d_Z_bin) + self._linear(U, d_Z_bin))
            Z_bin = (self._rand(T, d_Z_bin) < p_zb).to(self.dtype)
            z_parts.append(Z_bin)
        Z = torch.cat(z_parts, dim=1)

        # W: count-like standardized block plus ordinal block.
        d_W_count = max(1, d_W // 2)
        d_W_ord = d_W - d_W_count
        rate_log = (self._linear(X, d_W_count) + self._linear(U, d_W_count)).clamp(
            -2.0, 2.0
        )
        rate = torch.exp(rate_log)
        W_count_raw = torch.poisson(rate, generator=self.rng)
        W_count = (W_count_raw - rate) / torch.sqrt(rate + 1e-4)
        w_parts = [W_count]
        if d_W_ord > 0:
            latent_w = (
                self._linear(X, d_W_ord)
                + self._linear(U, d_W_ord)
                + self._noise(T, d_W_ord, sigma=0.8)
            )
            W_ord = torch.bucketize(
                latent_w, torch.tensor([-1.0, 0.0, 1.0], dtype=self.dtype)
            ).to(self.dtype)
            W_ord = (W_ord - 1.5) / 1.5
            w_parts.append(W_ord)
        W = torch.cat(w_parts, dim=1)

        raw_logit = self._linear(X, 1) + self._linear(U, 1) + self._linear(Z, 1)
        p = self._propensity(raw_logit)

        tau = self._rsu_scalar(0.2, 1.5) + self._linear(X, 1, low=0.05, high=0.5)
        f0 = torch.tanh(self._linear(X, 1)) + self._linear(U, 1) + self._linear(W, 1)
        eps_y = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 1.0))
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ------------------------------------------------------------------
    # Mechanism 6: Latent effect modification
    # ------------------------------------------------------------------

    def _mechanism_6_latent_effect_modification(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X, U, Z, W = self._linear_proxy_skeleton(T, d_U, d_X, d_W, d_Z)

        raw_logit = self._linear(X, 1) + self._linear(U, 1) + self._linear(Z, 1)
        p = self._propensity(raw_logit)

        tau = (
            self._rsu_scalar(0.2, 1.2)
            + self._linear(X, 1, low=0.05, high=0.6)
            + self._linear(U, 1, low=0.05, high=0.6)
        )
        f0 = self._linear(X, 1) + self._linear(U, 1) + self._linear(W, 1)
        eps_y = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 1.0))
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ------------------------------------------------------------------
    # Mechanism 7: Weak/redundant proxies and omitted Z->A/W->Y edges
    # ------------------------------------------------------------------

    def _mechanism_7_weak_redundant_omitted_edges(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X = self._standardize(self._randn(T, d_X))
        U = self._linear(X, d_U) + self._noise(T, d_U, sigma=0.8)

        d_Z_rel = max(1, min(d_Z, math.ceil(d_Z / 2)))
        d_W_rel = max(1, min(d_W, math.ceil(d_W / 2)))

        weak = self._rand(1).item() < 0.30
        proxy_scale = (
            self._uniform_scalar(0.1, 0.4) if weak else self._uniform_scalar(0.6, 1.5)
        )

        Z_rel = (
            self._linear(X, d_Z_rel)
            + proxy_scale * self._linear(U, d_Z_rel)
            + self._noise(T, d_Z_rel, sigma=0.7)
        )
        W_rel = (
            self._linear(X, d_W_rel)
            + proxy_scale * self._linear(U, d_W_rel)
            + self._noise(T, d_W_rel, sigma=0.7)
        )

        Z_noise = self._randn(T, d_Z - d_Z_rel) if d_Z > d_Z_rel else self._empty(T)
        W_noise = self._randn(T, d_W - d_W_rel) if d_W > d_W_rel else self._empty(T)
        Z = torch.cat([Z_rel, Z_noise], dim=1)
        W = torch.cat([W_rel, W_noise], dim=1)

        # Omitted Z -> A edge: treatment depends on X and U, but not directly on Z.
        raw_logit = self._linear(X, 1) + self._linear(U, 1)
        p = self._propensity(raw_logit)

        # Omitted W -> Y edge: outcome depends on X and U, but not directly on W.
        tau = self._rsu_scalar(0.2, 1.5) + self._linear(X, 1, low=0.05, high=0.6)
        f0 = self._linear(X, 1) + self._linear(U, 1)
        eps_y = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 1.0))
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ------------------------------------------------------------------
    # Mechanism 8: Positivity/overlap stress test
    # ------------------------------------------------------------------

    def _mechanism_8_positivity_stress(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X = self._standardize(self._randn(T, d_X))
        U = torch.tanh(self._linear(X, d_U)) + self._noise(T, d_U, sigma=0.5)
        Z = torch.sin(self._linear(U, d_Z) + self._linear(X, d_Z)) + self._noise(
            T, d_Z, sigma=0.3
        )
        W = torch.tanh(self._linear(U, d_W) + self._linear(X, d_W)) + self._noise(
            T, d_W, sigma=0.3
        )

        raw = (
            self._linear(X, 1)
            + self._linear(U, 1)
            + self._linear(Z, 1)
            + Z[:, :1] * X[:, :1]
        )
        raw = self._standardize(raw)
        kappa = self._uniform_scalar(1.5, 4.0)
        p = self.cfg.overlap_eps + (1.0 - 2.0 * self.cfg.overlap_eps) * torch.sigmoid(
            kappa * raw
        )
        p = p.clamp(self.cfg.overlap_eps, 1.0 - self.cfg.overlap_eps)

        tau = self._rsu_scalar(0.2, 1.5) + 0.4 * torch.sin(self._linear(X, 1))
        f0 = torch.tanh(self._linear(X, 1)) + self._linear(U, 1) + self._linear(W, 1)
        eps_y = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 0.8))
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ------------------------------------------------------------------
    # Mechanism 9: Multimodal latent mixture
    # ------------------------------------------------------------------

    def _mechanism_9_multimodal_mixture(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        R = self._randint(2, 4)

        pi = torch.softmax(self._randn(R), dim=0)
        S = torch.multinomial(pi.expand(T, R), 1, generator=self.rng).squeeze(1)

        X = torch.zeros(T, d_X, dtype=self.dtype)
        U = torch.zeros(T, d_U, dtype=self.dtype)
        Z = torch.zeros(T, d_Z, dtype=self.dtype)
        W = torch.zeros(T, d_W, dtype=self.dtype)
        raw_logit = torch.zeros(T, 1, dtype=self.dtype)
        tau = torch.zeros(T, 1, dtype=self.dtype)
        f0 = torch.zeros(T, 1, dtype=self.dtype)

        for r in range(R):
            mask = S == r
            nr = int(mask.sum().item())
            if nr == 0:
                continue

            mu_x = self._randn(1, d_X) * self._uniform_scalar(0.5, 1.5)
            sig_x = self._uniform_scalar(0.5, 1.5)
            X_r = mu_x + sig_x * self._randn(nr, d_X)
            X_r = self._standardize(X_r)

            U_r = (
                self._rsu_scalar(0.3, 1.2)
                + self._linear(X_r, d_U)
                + self._noise(nr, d_U, sigma=0.6)
            )
            Z_r = (
                self._linear(X_r, d_Z)
                + self._linear(U_r, d_Z)
                + torch.sin(self._linear(U_r, d_Z, low=0.1, high=0.8))
                + self._noise(nr, d_Z, sigma=0.4)
            )
            W_r = (
                self._linear(X_r, d_W)
                + self._linear(U_r, d_W)
                + torch.tanh(self._linear(U_r, d_W, low=0.1, high=0.8))
                + self._noise(nr, d_W, sigma=0.4)
            )

            raw_r = (
                self._linear(X_r, 1)
                + self._linear(U_r, 1)
                + self._linear(Z_r, 1)
                + self._rsu_scalar(-0.5, 0.5)
            )
            tau_r = (
                self._rsu_scalar(0.1, 1.4)
                + self._linear(X_r, 1, low=0.05, high=0.6)
                + self._linear(U_r, 1, low=0.05, high=0.6)
            )
            f0_r = (
                torch.sin(self._linear(X_r, 1))
                + torch.tanh(self._linear(U_r, 1))
                + self._linear(W_r, 1)
            )

            X[mask] = X_r
            U[mask] = U_r
            Z[mask] = Z_r
            W[mask] = W_r
            raw_logit[mask] = raw_r
            tau[mask] = tau_r
            f0[mask] = f0_r

        p = self._propensity(raw_logit)
        eps_y = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 1.0))
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ------------------------------------------------------------------
    # Mechanism 10: Two-timepoint proximal-summary mechanism
    # ------------------------------------------------------------------

    def _mechanism_10_two_timepoint_summary(self, T: int) -> Tuple[Tensor, ...]:
        """Two-timepoint DGF squeezed into point-treatment batch format.

        A0 is a historical treatment included as a pre-A1 covariate when there is
        room. The saved A is the final treatment A1. Y0/Y1 are potential outcomes
        under interventions A1=0/1, holding the generated pre-A1 history fixed.
        """
        d_U, d_X, d_W, d_Z = self._sample_dims()

        d_X0 = max(1, d_X // 2)
        # Reserve one X column for A0 if possible.
        has_a0_col = d_X >= 3
        d_X1 = d_X - d_X0 - (1 if has_a0_col else 0)
        d_X1 = max(0, d_X1)

        d_Z0 = max(1, d_Z // 2)
        d_Z1 = d_Z - d_Z0
        d_W0 = max(1, d_W // 2)
        d_W1 = d_W - d_W0

        X0 = self._standardize(self._randn(T, d_X0))
        U0 = self._linear(X0, d_U) + self._noise(T, d_U, sigma=0.6)
        Z0 = (
            self._linear(X0, d_Z0)
            + self._linear(U0, d_Z0)
            + self._noise(T, d_Z0, sigma=0.4)
        )
        W0 = (
            self._linear(X0, d_W0)
            + self._linear(U0, d_W0)
            + self._noise(T, d_W0, sigma=0.4)
        )

        p0 = self._propensity(
            self._linear(X0, 1) + self._linear(U0, 1) + self._linear(Z0, 1)
        )
        A0 = (self._rand(T, 1) < p0).to(self.dtype)

        if d_X1 > 0:
            X1 = (
                self._linear(X0, d_X1)
                + self._linear(U0, d_X1)
                + self._rsu_scalar(0.2, 1.0) * A0
                + self._noise(T, d_X1, sigma=0.5)
            )
        else:
            X1 = self._empty(T)

        U1 = 0.6 * U0 + self._linear(X1, d_U) + self._noise(T, d_U, sigma=0.5)

        if d_Z1 > 0:
            Z1 = (
                self._linear(X1, d_Z1)
                + self._linear(U1, d_Z1)
                + self._noise(T, d_Z1, sigma=0.4)
            )
            Z = torch.cat([Z0, Z1], dim=1)
        else:
            Z1 = self._empty(T)
            Z = Z0

        if d_W1 > 0:
            W1 = (
                self._linear(X1, d_W1)
                + self._linear(U1, d_W1)
                + self._noise(T, d_W1, sigma=0.4)
            )
            W = torch.cat([W0, W1], dim=1)
        else:
            W1 = self._empty(T)
            W = W0

        x_parts = [X0]
        if d_X1 > 0:
            x_parts.append(X1)
        if has_a0_col:
            x_parts.append(A0)
        X = torch.cat(x_parts, dim=1)[:, :d_X]

        raw_logit = (
            self._linear(X, 1)
            + self._linear(U1, 1)
            + self._linear(Z, 1)
            + self._rsu_scalar(0.2, 1.0) * A0
        )
        p = self._propensity(raw_logit)

        tau = (
            self._rsu_scalar(0.2, 1.5)
            + self._linear(X, 1, low=0.05, high=0.6)
            + self._rsu_scalar(0.1, 0.7) * A0
        )
        f0 = (
            self._linear(X, 1)
            + self._linear(U0, 1)
            + self._linear(U1, 1)
            + self._linear(W, 1)
            + self._rsu_scalar(0.1, 0.7) * A0
        )
        eps_y = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 1.0))
        Y0 = f0 + eps_y
        Y1 = f0 + tau + eps_y

        return self._finalize(U1, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

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
