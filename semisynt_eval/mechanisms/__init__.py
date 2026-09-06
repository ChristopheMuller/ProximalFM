"""Registry for empirical-root SCM families."""

from __future__ import annotations

from typing import Type

import torch

from semisynt_eval._config import CaseConfig
from semisynt_eval.mechanisms.base import Mechanism, MechanismOutput
from semisynt_eval.mechanisms.linear import LinearMechanism
from semisynt_eval.mechanisms.nonlinear import NonlinearMechanism

_BUILDERS: dict[str, Type[Mechanism]] = {
    "linear": LinearMechanism,
    "nonlinear": NonlinearMechanism,
}


def available_mechanisms() -> list[str]:
    return sorted(_BUILDERS)


def _resolve_builder(name: str, config: CaseConfig) -> Type[Mechanism]:
    if name != config.family:
        raise ValueError(
            f"case {config.name!r} selects family {config.family!r}, not {name!r}"
        )
    try:
        return _BUILDERS[name]
    except KeyError:
        raise KeyError(
            f"unknown mechanism {name!r}; available: "
            f"{', '.join(available_mechanisms())}"
        ) from None


def mechanism_target_capability(name: str, *, config: CaseConfig) -> str:
    """Return the declared target contract without fitting the mechanism."""
    return _resolve_builder(name, config).target_capability


def fit_mechanism(
    name: str,
    x_cal: torch.Tensor,
    u_cal: torch.Tensor,
    generator: torch.Generator,
    *,
    config: CaseConfig,
) -> Mechanism:
    builder = _resolve_builder(name, config)
    return builder.fit(x_cal, u_cal, config, generator)


__all__ = [
    "CaseConfig",
    "Mechanism",
    "MechanismOutput",
    "available_mechanisms",
    "fit_mechanism",
    "mechanism_target_capability",
]
