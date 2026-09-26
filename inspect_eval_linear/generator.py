"""Scalar linear-Gaussian proximal SCM with an analytic adjustment ladder.

The only mechanism-level quantity varied by the standard experiment is the
shared measurement-noise standard deviation of ``W`` and ``Z``.  The proxies
have no direct structural effect on treatment or outcome, so changing their
noise changes their information about ``U`` and nothing else in the DGP.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class LinearGeneratorConfig:
    beta_ux: float = 0.5
    sigma_u: float = 1.0
    proxy_loading: float = 1.0
    alpha_0: float = 0.0
    alpha_x: float = 0.25
    alpha_u: float = 1.0
    sigma_a: float = 1.0
    beta_0: float = 0.0
    beta_x: float = 0.5
    beta_u: float = 1.0
    sigma_y: float = 1.0
    tau_0: float = 1.0
    tau_x: float = 0.5
    mc_cate: int = 256
    mc_mem_budget: int = 2_000_000_000
    dtype: torch.dtype = torch.float64

    def validate(self) -> None:
        if self.sigma_u <= 0:
            raise ValueError("sigma_u must be > 0")
        if self.proxy_loading == 0:
            raise ValueError("proxy_loading must be non-zero")
        if self.sigma_a <= 0:
            raise ValueError("sigma_a must be > 0")
        if self.sigma_y < 0:
            raise ValueError("sigma_y must be >= 0")
        if self.mc_cate <= 1:
            raise ValueError("mc_cate must be > 1 for adjustment inspection")
        if self.mc_mem_budget <= 0:
            raise ValueError("mc_mem_budget must be > 0")


def proxy_reliability(
    proxy_noise_std: float, *, sigma_u: float = 1.0, proxy_loading: float = 1.0
) -> float:
    """Return the partial reliability ``R2(U, proxy | X)``."""
    if proxy_noise_std < 0:
        raise ValueError("proxy_noise_std must be >= 0")
    signal = (proxy_loading * sigma_u) ** 2
    return signal / (signal + proxy_noise_std**2)


class LinearProximalGenerator:
    """Generate canonical 24-field inspection episodes."""

    POOL_FIELDS = (
        "U X W Z A A_probs Y_obs Y0 Y1 d_U d_X d_W d_Z seq_lens "
        "train_sizes MU0 MU1 S2 K_eff Q_naive_X Q_adj_W Q_adj_Z "
        "Q_adj_WZ Q_true"
    ).split()

    # Conservative accounting for simultaneously-live float64 vectors.
    _COLS_PER_REPLICATE_ROW = 24

    def __init__(self, config: LinearGeneratorConfig):
        config.validate()
        self.cfg = config

    @staticmethod
    def _normal_cdf(x: Tensor) -> Tensor:
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def observed_propensity(
        self, x: Tensor, proxies: Tensor | None, proxy_noise_std: float
    ) -> Tensor:
        """Analytic ``P(A=1 | X, proxies)`` under the Gaussian SCM.

        ``proxies`` has zero, one, or two columns.  Given X, every column is an
        independent measurement ``lambda*U + noise`` of the same scalar U.
        """
        c = self.cfg
        x = x.reshape(-1, 1).to(c.dtype)
        if proxies is None or proxies.shape[1] == 0:
            post_mean = c.beta_ux * x
            post_var = torch.full_like(x, c.sigma_u**2)
        else:
            proxies = proxies.to(c.dtype)
            if proxy_noise_std < 0:
                raise ValueError("proxy_noise_std must be >= 0")
            if proxy_noise_std == 0:
                # Either perfect proxy identifies U exactly; averaging the two
                # identical noiseless measurements only adds roundoff.
                post_mean = proxies[:, :1] / c.proxy_loading
                post_var = torch.zeros_like(post_mean)
            else:
                prior_var = c.sigma_u**2
                noise_var = proxy_noise_std**2
                n_proxy = proxies.shape[1]
                precision = 1.0 / prior_var + n_proxy * c.proxy_loading**2 / noise_var
                variance = 1.0 / precision
                post_mean = variance * (
                    (c.beta_ux * x) / prior_var
                    + c.proxy_loading * proxies.sum(dim=1, keepdim=True) / noise_var
                )
                post_var = torch.full_like(post_mean, variance)

        score_mean = c.alpha_0 + c.alpha_x * x + c.alpha_u * post_mean
        score_var = c.sigma_a**2 + c.alpha_u**2 * post_var
        return self._normal_cdf(score_mean / score_var.sqrt()).clamp(1e-9, 1 - 1e-9)

    def posterior_u_variance(self, proxy_noise_std: float, n_proxies: int) -> float:
        """Conditional variance of U given X and ``n_proxies`` measurements."""
        if n_proxies < 0:
            raise ValueError("n_proxies must be >= 0")
        if proxy_noise_std < 0:
            raise ValueError("proxy_noise_std must be >= 0")
        if n_proxies and proxy_noise_std == 0:
            return 0.0
        precision = 1.0 / self.cfg.sigma_u**2
        if n_proxies:
            precision += n_proxies * self.cfg.proxy_loading**2 / proxy_noise_std**2
        return 1.0 / precision

    @staticmethod
    def _hajek(pi: Tensor, e: Tensor, y0: Tensor, y1: Tensor) -> Tensor:
        w1 = pi / e
        w0 = (1.0 - pi) / (1.0 - e)
        mu1 = (w1 * y1).sum(1) / w1.sum(1).clamp_min(1e-15)
        mu0 = (w0 * y0).sum(1) / w0.sum(1).clamp_min(1e-15)
        return mu1 - mu0

    @staticmethod
    def _pad_context(values: Tensor, n_ctx: int, dtype: torch.dtype) -> Tensor:
        return torch.cat(
            [torch.full((n_ctx,), float("nan"), dtype=dtype), values]
        ).unsqueeze(1)

    def generate_dataset(
        self,
        *,
        seq_len: int,
        train_size: int,
        proxy_noise_std: float,
        seed: int,
    ) -> Tuple[Tensor, ...]:
        """Generate one episode, including query-replicate population targets."""
        c = self.cfg
        if proxy_noise_std < 0:
            raise ValueError("proxy_noise_std must be >= 0")
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        n_ctx = max(1, min(int(train_size), seq_len - 1))
        n_q = seq_len - n_ctx
        K = c.mc_cate
        n_rep = n_ctx + n_q * K
        if n_rep * self._COLS_PER_REPLICATE_ROW > c.mc_mem_budget:
            raise RuntimeError(
                f"mc_cate={K} needs {n_rep:,} replicate rows, exceeding "
                f"mc_mem_budget={c.mc_mem_budget:,}; lower mc_cate or n_q"
            )

        rng = torch.Generator(device="cpu").manual_seed(int(seed))
        randn = lambda *shape: torch.randn(*shape, generator=rng, dtype=c.dtype)
        logical_x = randn(seq_len, 1)
        X = torch.empty((n_rep, 1), dtype=c.dtype)
        X[:n_ctx] = logical_x[:n_ctx]
        X[n_ctx:].view(n_q, K, 1)[:] = logical_x[n_ctx:, None, :]

        U = c.beta_ux * X + c.sigma_u * randn(n_rep, 1)
        W = c.proxy_loading * U + proxy_noise_std * randn(n_rep, 1)
        Z = c.proxy_loading * U + proxy_noise_std * randn(n_rep, 1)

        fine_score = c.alpha_0 + c.alpha_x * X + c.alpha_u * U
        pi = self._normal_cdf(fine_score / c.sigma_a).clamp(1e-9, 1 - 1e-9)
        eps_y = c.sigma_y * randn(n_rep, 1)
        base = c.beta_0 + c.beta_x * X + c.beta_u * U + eps_y
        tau = c.tau_0 + c.tau_x * X
        Y0_mc = base
        Y1_mc = base + tau

        sl = slice(n_ctx, None)
        xq, wq, zq = X[sl], W[sl], Z[sl]
        piq = pi[sl].view(n_q, K)
        y0q = Y0_mc[sl].view(n_q, K)
        y1q = Y1_mc[sl].view(n_q, K)
        q_true = c.tau_0 + c.tau_x * logical_x[n_ctx:, 0]
        q_naive = (piq * y1q).sum(1) / piq.sum(1) - (
            ((1.0 - piq) * y0q).sum(1) / (1.0 - piq).sum(1)
        )

        e_w = self.observed_propensity(xq, wq, proxy_noise_std).view(n_q, K)
        e_z = self.observed_propensity(xq, zq, proxy_noise_std).view(n_q, K)
        e_wz = self.observed_propensity(
            xq, torch.cat([wq, zq], dim=1), proxy_noise_std
        ).view(n_q, K)
        q_w = self._hajek(piq, e_w, y0q, y1q)
        q_z = self._hajek(piq, e_z, y0q, y1q)
        q_wz = self._hajek(piq, e_wz, y0q, y1q)

        # Select one replicate as the observed world for each query row.
        obs_k = torch.randint(K, (n_q,), generator=rng)
        obs_idx = torch.cat(
            [torch.arange(n_ctx), n_ctx + torch.arange(n_q) * K + obs_k]
        )
        U_obs, X_obs, W_obs, Z_obs = (t[obs_idx] for t in (U, X, W, Z))
        pi_obs, Y0, Y1 = (t[obs_idx] for t in (pi, Y0_mc, Y1_mc))
        A = (torch.rand(seq_len, 1, generator=rng, dtype=c.dtype) < pi_obs).to(c.dtype)
        Y_obs = A * Y1 + (1.0 - A) * Y0

        nan_ctx = torch.full((n_ctx,), float("nan"), dtype=c.dtype)
        mu0 = self._pad_context(y0q.mean(1), n_ctx, c.dtype)
        mu1 = self._pad_context(y1q.mean(1), n_ctx, c.dtype)
        # The paired contrast is deterministic given X in this SCM.
        s2 = torch.cat([nan_ctx, torch.zeros(n_q, dtype=c.dtype)]).unsqueeze(1)
        ladder = tuple(
            self._pad_context(q, n_ctx, c.dtype)
            for q in (q_naive, q_w, q_z, q_wz, q_true)
        )

        f32 = lambda t: t.to(torch.float32)
        one = torch.tensor(1, dtype=torch.long)
        return (
            f32(U_obs),
            f32(X_obs),
            f32(W_obs),
            f32(Z_obs),
            f32(A),
            f32(pi_obs),
            f32(Y_obs),
            f32(Y0),
            f32(Y1),
            one,
            one.clone(),
            one.clone(),
            one.clone(),
            torch.tensor(seq_len, dtype=torch.long),
            torch.tensor(n_ctx, dtype=torch.long),
            f32(mu0),
            f32(mu1),
            f32(s2),
            torch.tensor(K, dtype=torch.long),
            *(f32(q) for q in ladder),
        )

    def get_batch(
        self,
        *,
        proxy_noise_std: float,
        batch_size: int,
        seq_len: int,
        train_size: int,
        seeds: list[int],
    ) -> Tuple[Tensor, ...]:
        if len(seeds) != batch_size:
            raise ValueError("seeds must contain one seed per dataset")
        episodes = [
            self.generate_dataset(
                seq_len=seq_len,
                train_size=train_size,
                proxy_noise_std=proxy_noise_std,
                seed=seed,
            )
            for seed in seeds
        ]
        return tuple(torch.stack(field) for field in zip(*episodes))
