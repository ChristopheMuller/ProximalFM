"""Registry for empirical-root SCM mechanisms and their function families."""

from __future__ import annotations

from typing import Type

import torch

from eval_datasets.semisynt_eval._config import CaseConfig
from eval_datasets.semisynt_eval.mechanisms.base import Mechanism, MechanismOutput
from eval_datasets.semisynt_eval.mechanisms.calibrated_latent_proxy import (
    CalibratedLatentProxyMechanism,
)

_BUILDERS: dict[str, dict[str, Type[Mechanism]]] = {
    "calibrated_latent_proxy": {
        "linear": CalibratedLatentProxyMechanism,
        "nonlinear": CalibratedLatentProxyMechanism,
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


def fit_mechanism_with_identity(
    name: str,
    x_cal: torch.Tensor,
    u_cal: torch.Tensor,
    *,
    config: CaseConfig,
    master_seed: int,
    source_id: str,
    replicate: int,
) -> Mechanism:
    """Fit a mechanism from explicit paired-build identity."""
    builder = _resolve_builder(name, config)
    fit_with_identity = getattr(builder, "fit_with_identity", None)
    if fit_with_identity is None:
        raise TypeError(f"mechanism {name!r} does not support paired fitting")
    return fit_with_identity(
        x_cal,
        u_cal,
        config,
        master_seed=master_seed,
        source_id=source_id,
        replicate=replicate,
    )


__all__ = [
    "CaseConfig",
    "Mechanism",
    "MechanismOutput",
    "available_mechanisms",
    "fit_mechanism",
    "fit_mechanism_with_identity",
    "mechanism_target_capability",
]
