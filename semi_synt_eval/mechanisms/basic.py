"""Small, extensible linear and nonlinear proximal SCM families."""

from __future__ import annotations

import math
from typing import Callable

import torch

from semi_synt_eval.mechanisms.base import Mechanism, MechanismConfig

Tensor = torch.Tensor


def _randn(generator: torch.Generator, *shape: int) -> Tensor:
    return torch.randn(*shape, generator=generator)


def _signed(generator: torch.Generator, *shape: int) -> Tensor:
    magnitude = 0.5 + torch.rand(*shape, generator=generator)
    sign = torch.where(torch.rand(*shape, generator=generator) < 0.5, -1.0, 1.0)
    return magnitude * sign


class BasicMechanism(Mechanism):
    """Random-coefficient SCM with either linear or smooth nonlinear links."""

    def __init__(
        self,
        d_x: int,
        d_u: int,
        config: MechanismConfig,
        generator: torch.Generator,
        *,
        nonlinear: bool,
    ) -> None:
        self.config = config
        self.generator = generator
        self.nonlinear = nonlinear
        d_proxy = d_u
        self.d_w = d_proxy
        self.d_z = d_proxy

        # Diagonal U loadings keep every latent direction represented, while
        # the random perturbation and X paths make the proxies nontrivial.
        eye = torch.eye(d_u)
        self.bw_u = eye + 0.2 * _randn(generator, d_u, d_proxy)
        self.bz_u = eye + 0.2 * _randn(generator, d_u, d_proxy)
        self.bw_x = _signed(generator, d_x, d_proxy) / math.sqrt(d_x)
        self.bz_x = _signed(generator, d_x, d_proxy) / math.sqrt(d_x)
        self.ba_x = _signed(generator, d_x, 1) / math.sqrt(d_x)
        self.ba_u = _signed(generator, d_u, 1) / math.sqrt(d_u)
        self.ba_z = _signed(generator, d_proxy, 1) / math.sqrt(d_proxy)
        self.by_x = _signed(generator, d_x, 1) / math.sqrt(d_x)
        self.by_u = _signed(generator, d_u, 1) / math.sqrt(d_u)
        self.by_w = _signed(generator, d_proxy, 1) / math.sqrt(d_proxy)
        self.btau = _signed(generator, d_x, 1) / math.sqrt(d_x)
        self.treatment_shift = float(
            torch.empty(1).uniform_(-0.5, 0.5, generator=generator)
        )

    def _link(self, value: Tensor) -> Tensor:
        return torch.tanh(value) if self.nonlinear else value

    def proxies(self, x: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        scale = float(self.config.proxy_noise)
        w_signal = u @ self.bw_u + x @ self.bw_x
        z_signal = u @ self.bz_u + x @ self.bz_x
        w = self._link(w_signal) + scale * _randn(self.generator, *w_signal.shape)
        z = self._link(z_signal) + scale * _randn(self.generator, *z_signal.shape)
        return w, z

    def treatment_probability(self, x: Tensor, u: Tensor, z: Tensor) -> Tensor:
        logit = x @ self.ba_x + u @ self.ba_u + z @ self.ba_z
        if self.nonlinear:
            logit = torch.sin(x @ self.ba_x) + torch.tanh(u @ self.ba_u) + z @ self.ba_z
        logit = float(self.config.treatment_scale) * logit + self.treatment_shift
        return torch.sigmoid(logit.clamp(-8.0, 8.0))

    def cate(self, x: Tensor) -> Tensor:
        score = x @ self.btau
        if self.nonlinear:
            score = torch.tanh(score) + 0.25 * torch.sin(x[:, :1])
        return 0.5 + float(self.config.effect_scale) * score

    def potential_outcomes(
        self, x: Tensor, u: Tensor, w: Tensor
    ) -> tuple[Tensor, Tensor]:
        baseline = x @ self.by_x + u @ self.by_u + w @ self.by_w
        if self.nonlinear:
            baseline = (
                torch.sin(x @ self.by_x)
                + torch.tanh(u @ self.by_u)
                + 0.5 * (w @ self.by_w).square()
            )
        noise = float(self.config.outcome_noise) * _randn(self.generator, x.shape[0], 1)
        y0 = baseline + noise
        tau = self.cate(x)
        assert tau is not None
        y1 = y0 + tau
        return y0, y1


_BUILDERS: dict[str, Callable[..., BasicMechanism]] = {
    "linear": lambda *args, **kwargs: BasicMechanism(*args, **kwargs, nonlinear=False),
    "nonlinear": lambda *args, **kwargs: BasicMechanism(
        *args, **kwargs, nonlinear=True
    ),
}


def available_mechanisms() -> list[str]:
    return sorted(_BUILDERS)


def build_mechanism(
    name: str,
    d_x: int,
    d_u: int,
    config: MechanismConfig,
    generator: torch.Generator,
) -> BasicMechanism:
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise KeyError(
            f"unknown mechanism {name!r}; available: {', '.join(available_mechanisms())}"
        ) from None
    return builder(d_x, d_u, config, generator)
