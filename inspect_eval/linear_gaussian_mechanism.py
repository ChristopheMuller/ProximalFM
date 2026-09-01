"""The smallest useful proximal SCM.

Every variable is scalar and every continuous structural equation is linear
with independent Normal noise.  Coefficients and noise scales are sampled once
per dataset.  Treatment is binary, with a linear logit:

    X = eps_X
    U = b_UX X + eps_U
    Z = b_Z0 + b_ZX X + b_ZU U + eps_Z
    W = b_W0 + b_WX X + b_WU U + eps_W
    A ~ Bernoulli(sigmoid(b_A0 + b_AX X + b_AU U))
    Y(a) = b_Y0 + b_YX X + b_YU U + a (tau_0 + tau_X X) + eps_Y

Thus Z has no arrow into A, W has no arrow into Y, and neither proxy enters the
other proxy or the outcome/treatment mechanism.  Their association with A and
Y is entirely through the shared latent confounder U (and X).

The output contract is identical to
``synt_eval.realistic_proximal_mechanisms.RealisticProximalGenerator``.  In
replicate mode, only query rows are replicated and the target is
``E[Y(a) | X]``.  Because both arms share U and eps_Y, the exact CATE is
``tau_0 + tau_X X`` and its paired Monte Carlo variance is zero up to floating
point roundoff.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from synt_eval.realistic_proximal_mechanisms import (
    RealisticGeneratorConfig,
    RealisticProximalGenerator,
)

Tensor = torch.Tensor
N_MECHANISMS = 1


class LinearGaussianGenerator(RealisticProximalGenerator):
    """Generate independent draws from the one-dimensional linear SCM."""

    def __init__(self, config: RealisticGeneratorConfig):
        dims = (
            config.min_dim_U,
            config.max_dim_U,
            config.min_dim_X,
            config.max_dim_X,
            config.min_dim_W,
            config.max_dim_W,
            config.min_dim_Z,
            config.max_dim_Z,
        )
        if dims != (1,) * 8:
            raise ValueError(
                "inspect_eval requires U, X, W, and Z to be one-dimensional"
            )
        super().__init__(config)
        self.sampled_coefficients: list[Dict[str, float]] = []

    def get_batch(self, *args, **kwargs) -> Tuple[Tensor, ...]:
        self.sampled_coefficients = []
        return super().get_batch(*args, **kwargs)

    def generate_dataset(self, params: Dict) -> Tuple[Tensor, ...]:
        if int(params["mechanism_id"]) != 1:
            raise ValueError("inspect_eval has only mechanism_id=1")
        T = int(params["seq_len"])
        train_n = int(params.get("train_size", max(1, T // 2)))
        self._rep = self._make_rep(T, train_n)
        return self._linear_gaussian(T)

    @staticmethod
    def mechanism_name(mechanism_id: int) -> str:
        return (
            "linear_gaussian" if int(mechanism_id) == 1 else f"unknown_{mechanism_id}"
        )

    @staticmethod
    def mechanism_story(mechanism_id: int) -> str:
        if int(mechanism_id) != 1:
            return ""
        return (
            "Scalar linear-Gaussian proximal SCM with pure proxies: Z does not "
            "enter treatment and W does not enter the outcome."
        )

    def _coef(self, low: float = 0.3, high: float = 1.3) -> float:
        """A non-negligible random signed coefficient."""
        return self._rsu_scalar(low=low, high=high)

    def _linear_gaussian(self, T: int) -> Tuple[Tensor, ...]:
        N = self._rep.N

        c = {
            "b_ux": self._coef(),
            "b_z0": self._uniform_scalar(-0.5, 0.5),
            "b_zx": self._coef(),
            "b_zu": self._coef(0.7, 1.5),
            "b_w0": self._uniform_scalar(-0.5, 0.5),
            "b_wx": self._coef(),
            "b_wu": self._coef(0.7, 1.5),
            "b_a0": self._uniform_scalar(-1.0, 1.0),
            "b_ax": self._coef(),
            "b_au": self._coef(0.7, 1.5),
            "b_y0": self._uniform_scalar(-1.0, 1.0),
            "b_yx": self._coef(),
            "b_yu": self._coef(0.7, 1.5),
            "tau_0": self._coef(0.4, 1.4),
            "tau_x": self._coef(0.1, 0.8),
            "sd_x": self._uniform_scalar(0.7, 1.3),
            "sd_u": self._uniform_scalar(0.5, 1.2),
            "sd_z": self._uniform_scalar(0.5, 1.2),
            "sd_w": self._uniform_scalar(0.5, 1.2),
            "sd_y": self._uniform_scalar(0.5, 1.2),
        }
        self.sampled_coefficients.append(dict(c))

        # X is sampled on logical rows and held fixed across query replicates.
        X = self._replicate(c["sd_x"] * self._randn(T, 1))
        U = c["b_ux"] * X + c["sd_u"] * self._randn(N, 1)

        # Z is a leaf for both the treatment and CATE target, so it only needs
        # the observed world. W is replicated even though it is a pure proxy,
        # matching the project's standard target-marginalisation contract.
        X_obs, U_obs = self._obs(X), self._obs(U)
        Z = (
            c["b_z0"]
            + c["b_zx"] * X_obs
            + c["b_zu"] * U_obs
            + c["sd_z"] * self._randn(T, 1)
        )
        W = c["b_w0"] + c["b_wx"] * X + c["b_wu"] * U + c["sd_w"] * self._randn(N, 1)

        logits = c["b_a0"] + c["b_ax"] * X_obs + c["b_au"] * U_obs
        eps = float(self.cfg.overlap_eps)
        A_probs = eps + (1.0 - 2.0 * eps) * torch.sigmoid(
            logits.clamp(-self.cfg.logit_clip, self.cfg.logit_clip)
        )

        outcome_noise = c["sd_y"] * self._randn(N, 1)
        Y0 = c["b_y0"] + c["b_yx"] * X + c["b_yu"] * U + outcome_noise
        tau = c["tau_0"] + c["tau_x"] * X
        Y1 = Y0 + tau

        return self._finalize(U, X, W, Z, A_probs, Y0, Y1, 1, 1, 1, 1)


GeneratorConfig = RealisticGeneratorConfig
