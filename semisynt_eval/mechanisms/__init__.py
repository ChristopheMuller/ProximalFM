"""Registry for empirical-root SCM families."""

from __future__ import annotations

from typing import Type

import torch

from semisynt_eval._config import CaseConfig
from semisynt_eval.mechanisms.base import Mechanism
from semisynt_eval.mechanisms.linear import LinearMechanism
from semisynt_eval.mechanisms.nonlinear import NonlinearMechanism

_BUILDERS: dict[str, Type[Mechanism]] = {
    "linear": LinearMechanism,
    "nonlinear": NonlinearMechanism,
}


def available_mechanisms() -> list[str]:
    return sorted(_BUILDERS)


def build_mechanism(
    name: str,
    d_x: int,
    d_u: int,
    generator: torch.Generator,
    *,
    config: CaseConfig,
) -> Mechanism:
    if name != config.family:
        raise ValueError(
            f"case {config.name!r} selects family {config.family!r}, not {name!r}"
        )
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise KeyError(
            f"unknown mechanism {name!r}; available: "
            f"{', '.join(available_mechanisms())}"
        ) from None
    return builder(d_x, d_u, config, generator)


__all__ = ["CaseConfig", "Mechanism", "available_mechanisms", "build_mechanism"]
