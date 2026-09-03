"""Registry for the closed-form empirical-root SCM families."""

from __future__ import annotations

from typing import Type

import torch

from semi_synt_eval._config import MechanismConfig
from semi_synt_eval.mechanisms.base import Mechanism
from semi_synt_eval.mechanisms.linear import LinearMechanism
from semi_synt_eval.mechanisms.linear_quadratic import LinearQuadraticMechanism

_BUILDERS: dict[str, Type[Mechanism]] = {
    "linear": LinearMechanism,
    "linear_quadratic": LinearQuadraticMechanism,
}


def available_mechanisms() -> list[str]:
    return sorted(_BUILDERS)


def build_mechanism(
    name: str,
    d_x: int,
    d_u: int,
    generator: torch.Generator,
    *,
    config: MechanismConfig,
) -> Mechanism:
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise KeyError(
            f"unknown mechanism {name!r}; available: {', '.join(available_mechanisms())}"
        ) from None
    return builder(d_x, d_u, config, generator)


__all__ = [
    "Mechanism",
    "MechanismConfig",
    "available_mechanisms",
    "build_mechanism",
]
