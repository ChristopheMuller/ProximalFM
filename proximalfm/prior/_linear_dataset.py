"""Modern linear-Gaussian diagnostic prior.

This module deliberately uses the same orchestration and batch contract as the
main MLP prior.  The only thing that differs is the SCM mechanism itself:
``X`` is exogenous, ``U`` is a linear Gaussian cause of the proxies and
outcomes, and the treatment is assigned through a probit link.

"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from ._dataset import DisablePrinting, SCMPrior, _mc_reduce, _scalar_params
from ._hp_sampling import HpSamplerList
from ._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP

# These are the scalar defaults used by inspect_eval_linear.  For vector
# dimensions, each output loading column is normalised to the corresponding
# scalar magnitude, so increasing a dimension changes representation capacity
# without silently changing the intended mechanism scale.
LINEAR_DEFAULTS = {
    "beta_ux": 0.5,
    "sigma_u": 1.0,
    "proxy_loading": 1.0,
    "proxy_noise_std": 0.2,
    "alpha_0": 0.0,
    "alpha_x": 0.25,
    "alpha_u": 1.0,
    "sigma_a": 1.0,
    "beta_0": 0.0,
    "beta_x": 0.5,
    "beta_u": 1.0,
    "sigma_y": 1.0,
    "tau_0": 1.0,
    "tau_x": 0.5,
}

_LINEAR_SAMPLED_HP = {"propensity_shift", "propensity_scale"}


class LinearSCMPrior(SCMPrior):
    """Generate the controlled vector linear-Gaussian SCM.

    The constructor intentionally mirrors :class:`SCMPrior`.  Shared prior
    options such as the fixed query split, MC target mode, memory cap, device,
    and parallelism therefore have identical semantics for both prior types.
    """

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_dim_U: int = 3,
        max_dim_U: int = 5,
        min_dim_W: int = 3,
        max_dim_W: int = 5,
        min_dim_Z: int = 3,
        max_dim_Z: int = 5,
        min_dim_X: int = 5,
        max_dim_X: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        n_query: int = 128,
        replay_small: bool = False,
        prior_type: str = "linear",
        mc_cate: int = 200,
        mc_targets_on_context: bool = False,
        fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
    ):
        if prior_type != "linear":
            raise ValueError(
                "LinearSCMPrior only supports prior_type='linear', got "
                f"{prior_type!r}"
            )
        super().__init__(
            batch_size=batch_size,
            batch_size_per_gp=batch_size_per_gp,
            batch_size_per_subgp=batch_size_per_subgp,
            min_dim_U=min_dim_U,
            max_dim_U=max_dim_U,
            min_dim_W=min_dim_W,
            max_dim_W=max_dim_W,
            min_dim_Z=min_dim_Z,
            max_dim_Z=max_dim_Z,
            min_dim_X=min_dim_X,
            max_dim_X=max_dim_X,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            seq_len_per_gp=seq_len_per_gp,
            n_query=n_query,
            replay_small=replay_small,
            prior_type="linear",
            mc_cate=mc_cate,
            mc_targets_on_context=mc_targets_on_context,
            fixed_hp=fixed_hp,
            sampled_hp=sampled_hp,
            n_jobs=n_jobs,
            num_threads_per_generate=num_threads_per_generate,
            device=device,
        )

    def hp_sampling(self) -> Dict[str, Any]:
        """Sample only shared propensity controls for the linear diagnostic.

        The default sampled config also contains MLP, tree, categorical, and
        graph-subsampling controls.  They are intentionally ignored here so
        the diagnostic remains continuous and canonical while retaining the
        shared propensity behaviour.
        """

        sampled = {
            key: value
            for key, value in self.sampled_hp.items()
            if key in _LINEAR_SAMPLED_HP
        }
        if not sampled:
            return {}
        return HpSamplerList(sampled, device=self.device).sample()

    @staticmethod
    def _loading(in_dim: int, out_dim: int, magnitude: float, device: str) -> Tensor:
        """Draw a matrix whose non-zero output columns have fixed norm."""

        if in_dim == 0 or out_dim == 0 or magnitude == 0:
            return torch.zeros((in_dim, out_dim), device=device)
        raw = torch.randn((in_dim, out_dim), device=device)
        norms = raw.norm(dim=0, keepdim=True).clamp_min(torch.finfo(raw.dtype).eps)
        return raw / norms * float(magnitude)

    @staticmethod
    def _normal_cdf(x: Tensor) -> Tensor:
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _param(params: Dict[str, Any], name: str) -> float:
        return float(params.get(f"linear_{name}", LINEAR_DEFAULTS[name]))

    @staticmethod
    def _observed_indices(T_ctx: int, T_q: int, K: int, device: str) -> Tensor:
        """Select the observed world, without consuming RNG when ``K == 1``."""

        if K == 1:
            query_idx = T_ctx + torch.arange(T_q, device=device)
        else:
            query_world = torch.randint(K, size=(T_q,), device=device)
            query_idx = T_ctx + torch.arange(T_q, device=device) * K + query_world
        if T_ctx == 0:
            return query_idx
        context_idx = torch.arange(T_ctx, device=device)
        return torch.cat((context_idx, query_idx))

    @staticmethod
    def _per_row_estimate(params: Dict[str, Any]) -> int:
        """Conservative live-element estimate for linear MC generation."""

        d_u = int(params["dim_U"])
        d_x = int(params["dim_X"])
        d_w = int(params["dim_W"])
        d_z = int(params["dim_Z"])
        # X, U, W, Y0/Y1/CATE and a few transient coefficient/noise blocks.
        return max(32, 4 * d_u + 4 * d_x + 3 * d_w + 2 * d_z + 16)

    @staticmethod
    def _diagnostics(
        params: Dict[str, Any],
        *,
        attempts: int,
        K: int,
        A_probs: Tensor,
        loadings: Dict[str, Tensor],
    ) -> Dict[str, Any]:
        return {
            "params": _scalar_params(params),
            "attempts": int(attempts),
            "K_eff": int(K),
            "propensity": {
                "mean": float(A_probs.mean()),
                "min": float(A_probs.min()),
                "max": float(A_probs.max()),
            },
            "noise_scales": {
                name: LinearSCMPrior._param(params, name)
                for name in ("sigma_u", "proxy_noise_std", "sigma_a", "sigma_y")
            },
            "loadings": {
                name: {
                    "shape": tuple(int(v) for v in matrix.shape),
                    "frobenius_norm": float(matrix.norm()),
                }
                for name, matrix in loadings.items()
            },
            "structure": {
                "X": [],
                "U": ["X"],
                "W": ["U"],
                "Z": ["U"],
                "A": ["X", "U", "epsilon_A"],
                "Y0": ["X", "U", "epsilon_Y"],
                "Y1": ["Y0", "X"],
            },
        }

    @torch.no_grad()
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, ...]:
        """Generate one linear episode in the shared 18-field format."""

        T = int(params["seq_len"])
        K_requested = int(params.get("mc_cate", self.mc_cate))
        if K_requested < 1:
            raise ValueError("mc_cate must be >= 1")

        T_ctx = (
            0
            if params.get("mc_targets_on_context", False)
            else int(max(0, min(int(params.get("train_size", 0)), T - 1)))
        )
        T_q = T - T_ctx
        K = self._cap_mc_samples(K_requested, T_ctx, T_q, params)
        N = T_ctx + T_q * K
        obs_idx = self._observed_indices(T_ctx, T_q, K, self.device)

        d_u = int(params["dim_U"])
        d_x = int(params["dim_X"])
        d_w = int(params["dim_W"])
        d_z = int(params["dim_Z"])

        sigma_u = self._param(params, "sigma_u")
        proxy_noise_std = self._param(params, "proxy_noise_std")
        sigma_a = self._param(params, "sigma_a")
        sigma_y = self._param(params, "sigma_y")
        if sigma_u <= 0 or sigma_a <= 0 or sigma_y < 0 or proxy_noise_std < 0:
            raise ValueError("linear Gaussian noise scales have invalid values")

        max_attempts = max(1, int(params.get("completeness_max_attempts", 50)))
        for attempt in range(1, max_attempts + 1):
            # The covariates are drawn once per episode and expanded over query
            # worlds. This is the fixed-X condition required by the MC-CATE.
            X = torch.randn((T, d_x), device=self.device)
            X_rep = torch.empty((N, d_x), device=self.device)
            if T_ctx:
                X_rep[:T_ctx] = X[:T_ctx]
            X_rep[T_ctx:] = X[T_ctx:, None, :].expand(T_q, K, d_x).reshape(T_q * K, d_x)

            B_U = self._loading(d_x, d_u, self._param(params, "beta_ux"), self.device)
            B_W = self._loading(
                d_u, d_w, self._param(params, "proxy_loading"), self.device
            )
            B_Z = self._loading(
                d_u, d_z, self._param(params, "proxy_loading"), self.device
            )
            B_AX = self._loading(d_x, 1, self._param(params, "alpha_x"), self.device)
            B_AU = self._loading(d_u, 1, self._param(params, "alpha_u"), self.device)
            B_YX = self._loading(d_x, 1, self._param(params, "beta_x"), self.device)
            B_YU = self._loading(d_u, 1, self._param(params, "beta_u"), self.device)
            B_TAU = self._loading(d_x, 1, self._param(params, "tau_x"), self.device)

            U_mc = X_rep @ B_U + sigma_u * torch.randn((N, d_u), device=self.device)
            U_obs = U_mc[obs_idx]
            W_obs = U_obs @ B_W + proxy_noise_std * torch.randn(
                (T, d_w), device=self.device
            )
            Z_obs = U_obs @ B_Z + proxy_noise_std * torch.randn(
                (T, d_z), device=self.device
            )

            score = (self._param(params, "alpha_0") + X @ B_AX + U_obs @ B_AU).squeeze(
                -1
            )
            score = score * float(params.get("propensity_scale", 1.0)) + float(
                params.get("propensity_shift", 0.0)
            )
            A_probs = self._normal_cdf(score / sigma_a).clamp(
                min=float(params.get("propensity_epsilon", 1e-3)),
                max=1.0 - float(params.get("propensity_epsilon", 1e-3)),
            )
            A = torch.bernoulli(A_probs).long()

            y0_mc = (
                self._param(params, "beta_0")
                + X_rep @ B_YX
                + U_mc @ B_YU
                + sigma_y * torch.randn((N, 1), device=self.device)
            ).squeeze(-1)
            tau_mc = self._param(params, "tau_0") + X_rep @ B_TAU
            tau_mc = tau_mc.squeeze(-1)
            y1_mc = y0_mc + tau_mc
            # The canonical SCM shares Y(0)'s noise between both arms, so its
            # CATE is exactly tau(X). Use that structural expression for S2
            # rather than subtracting two float32 tensors and manufacturing a
            # tiny round-off variance.
            cate_mc = tau_mc

            pool_rows = d_x == 0
            if T_ctx == 0:
                MU0, MU1, S2 = _mc_reduce(y0_mc, y1_mc, cate_mc, T, K, pool_rows)
            else:
                nan_pad = torch.full(
                    (T_ctx,), float("nan"), device=self.device, dtype=y0_mc.dtype
                )
                mu0_q, mu1_q, s2_q = _mc_reduce(
                    y0_mc[T_ctx:],
                    y1_mc[T_ctx:],
                    cate_mc[T_ctx:],
                    T_q,
                    K,
                    pool_rows,
                )
                MU0 = torch.cat((nan_pad, mu0_q))
                MU1 = torch.cat((nan_pad, mu1_q))
                S2 = torch.cat((nan_pad, s2_q))

            Y0 = y0_mc[obs_idx]
            Y1 = y1_mc[obs_idx]
            Y_obs = torch.where(A == 1, Y1, Y0)

            U = U_obs.unsqueeze(0)
            X_b = X.unsqueeze(0)
            W = W_obs.unsqueeze(0)
            Z = Z_obs.unsqueeze(0)
            d_U = torch.tensor([d_u], device=self.device, dtype=torch.long)
            d_X = torch.tensor([d_x], device=self.device, dtype=torch.long)
            d_W = torch.tensor([d_w], device=self.device, dtype=torch.long)
            d_Z = torch.tensor([d_z], device=self.device, dtype=torch.long)

            U, d_U = self.filter_constant_features_and_pad(U, d_U, params["max_dim_U"])
            X_b, d_X = self.filter_constant_features_and_pad(
                X_b, d_X, params["max_dim_X"]
            )
            W, d_W = self.filter_constant_features_and_pad(W, d_W, params["max_dim_W"])
            Z, d_Z = self.filter_constant_features_and_pad(Z, d_Z, params["max_dim_Z"])

            x_ok = bool((d_X > 0).all()) or d_x == 0
            basic_ok = x_ok and bool((d_W > 0).all()) and bool((d_Z > 0).all())
            if params.get("enforce_completeness_heuristic", True):
                proximal_ok = bool(
                    (d_U > 0).all() and (torch.minimum(d_W, d_Z) >= d_U).all()
                )
            else:
                proximal_ok = True

            if basic_ok and (proximal_ok or attempt == max_attempts):
                loadings = {
                    "B_U": B_U,
                    "B_W": B_W,
                    "B_Z": B_Z,
                    "B_AX": B_AX,
                    "B_AU": B_AU,
                    "B_YX": B_YX,
                    "B_YU": B_YU,
                    "B_TAU": B_TAU,
                }
                diagnostics = None
                if params.get("collect_mechanism_diagnostics", False):
                    diagnostics = self._diagnostics(
                        params,
                        attempts=attempt,
                        K=K,
                        A_probs=A_probs,
                        loadings=loadings,
                    )
                return (
                    U.squeeze(0),
                    X_b.squeeze(0),
                    W.squeeze(0),
                    Z.squeeze(0),
                    A,
                    A_probs,
                    Y_obs,
                    Y0,
                    Y1,
                    MU0,
                    MU1,
                    d_U.squeeze(0),
                    d_X.squeeze(0),
                    d_W.squeeze(0),
                    d_Z.squeeze(0),
                    S2,
                    torch.tensor(K, device=self.device, dtype=torch.long),
                    diagnostics,
                )

        raise RuntimeError("Unable to generate a valid linear diagnostic episode")


# Private imports in downstream code used this name.  Keep it as an alias,
# but do not preserve the old ratio-split implementation.
LinearPrior = LinearSCMPrior


class LinearPriorDataset(IterableDataset):
    """Thin compatibility wrapper around :class:`LinearSCMPrior`."""

    def __init__(self, **kwargs):
        super().__init__()
        self.prior = LinearSCMPrior(**kwargs)
        self.batch_size = self.prior.batch_size
        self.device = self.prior.device

    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, ...]:
        return self.prior.get_batch(batch_size)

    def __iter__(self) -> "LinearPriorDataset":
        return self

    def __next__(self) -> Tuple[Tensor, ...]:
        with DisablePrinting():
            return self.get_batch()

    def __repr__(self) -> str:
        return (
            "LinearPriorDataset(\n"
            f"  batch_size: {self.batch_size}\n"
            f"  features U: {self.prior.min_dim_U} - {self.prior.max_dim_U}\n"
            f"  features W: {self.prior.min_dim_W} - {self.prior.max_dim_W}\n"
            f"  features Z: {self.prior.min_dim_Z} - {self.prior.max_dim_Z}\n"
            f"  features X: {self.prior.min_dim_X} - {self.prior.max_dim_X}\n"
            f"  n_query: {self.prior.n_query}\n"
            f"  mc_cate: {self.prior.mc_cate}\n"
            f"  device: {self.device}\n"
            ")"
        )
