"""Registry for empirical-root SCM mechanisms and their function families."""

from __future__ import annotations

from typing import Type

import torch

from semisynt_eval._config import CaseConfig
from semisynt_eval.mechanisms.base import Mechanism, MechanismOutput
from semisynt_eval.mechanisms.random_basis import LinearMechanism, NonlinearMechanism

_BUILDERS: dict[str, dict[str, Type[Mechanism]]] = {
    "random_basis": {
        "linear": LinearMechanism,
        "nonlinear": NonlinearMechanism,
    },
}


def available_mechanisms() -> list[str]:
    return sorted(_BUILDERS)


def _resolve_builder(name: str, config: CaseConfig) -> Type[Mechanism]:
    if name != config.mechanism:
        raise ValueError(
            f"case {config.name!r} selects mechanism {config.mechanism!r}, "
            f"not {name!r}"
        )
    try:
        families = _BUILDERS[name]
    except KeyError:
        raise KeyError(
            f"unknown mechanism {name!r}; available: "
            f"{', '.join(available_mechanisms())}"
        ) from None
    try:
        return families[config.family]
    except KeyError:
        raise KeyError(
            f"mechanism {name!r} does not implement family {config.family!r}; "
            f"available: {', '.join(sorted(families))}"
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
