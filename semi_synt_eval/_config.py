"""Typed validation for the YAML-owned semi-synthetic mechanism recipes.

This module intentionally contains no generative defaults. All values that can
change generated data must be present in the input YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

CoefficientDistribution = Literal["normal", "uniform"]


@dataclass(frozen=True)
class CoefficientConfig:
    distribution: CoefficientDistribution
    scale: float
    normalize_by_input_dim: bool


@dataclass(frozen=True)
class ProxyConfig:
    use_x: bool
    use_u: bool
    noise_std: float
    normalize_output: bool
    coefficients: CoefficientConfig


@dataclass(frozen=True)
class TreatmentConfig:
    use_x: bool
    use_u: bool
    use_z: bool
    normalize_score: bool
    logit_scale: float
    logit_clip: float
    intercept_low: float
    intercept_high: float
    overlap_epsilon: float
    coefficients: CoefficientConfig


@dataclass(frozen=True)
class EffectConfig:
    use_x: bool
    intercept: float
    scale: float
    coefficients: CoefficientConfig


@dataclass(frozen=True)
class OutcomeConfig:
    use_x: bool
    use_u: bool
    use_w: bool
    noise_std: float
    normalize_baseline: bool
    coefficients: CoefficientConfig
    effect: EffectConfig


@dataclass(frozen=True)
class PowerAllocationConfig:
    proxy_quadratic_probability: float
    treatment_quadratic_probability: float
    outcome_quadratic_probability: float
    effect_quadratic_probability: float


@dataclass(frozen=True)
class MechanismConfig:
    proxies: ProxyConfig
    treatment: TreatmentConfig
    outcome: OutcomeConfig
    powers: PowerAllocationConfig


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, location: str) -> None:
    actual = set(value)
    missing, unknown = expected - actual, actual - expected
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise ValueError(f"invalid {location}: {', '.join(details)}")


def _coefficients(value: Mapping[str, Any], location: str) -> CoefficientConfig:
    _exact_keys(
        value,
        {"distribution", "scale", "normalize_by_input_dim"},
        location=location,
    )
    return CoefficientConfig(**value)


def mechanism_config_from_dict(
    value: Mapping[str, Any], *, location: str
) -> MechanismConfig:
    """Strictly parse one mechanism; omitted values never fall back silently."""
    _exact_keys(value, {"proxies", "treatment", "outcome", "powers"}, location=location)

    proxies = dict(value["proxies"])
    _exact_keys(
        proxies,
        {"use_x", "use_u", "noise_std", "normalize_output", "coefficients"},
        location=f"{location}.proxies",
    )
    proxy_coefficients = _coefficients(
        proxies.pop("coefficients"), f"{location}.proxies.coefficients"
    )

    treatment = dict(value["treatment"])
    _exact_keys(
        treatment,
        {
            "use_x",
            "use_u",
            "use_z",
            "normalize_score",
            "logit_scale",
            "logit_clip",
            "intercept_low",
            "intercept_high",
            "overlap_epsilon",
            "coefficients",
        },
        location=f"{location}.treatment",
    )
    treatment_coefficients = _coefficients(
        treatment.pop("coefficients"), f"{location}.treatment.coefficients"
    )

    outcome = dict(value["outcome"])
    _exact_keys(
        outcome,
        {
            "use_x",
            "use_u",
            "use_w",
            "noise_std",
            "normalize_baseline",
            "coefficients",
            "effect",
        },
        location=f"{location}.outcome",
    )
    outcome_coefficients = _coefficients(
        outcome.pop("coefficients"), f"{location}.outcome.coefficients"
    )
    effect = dict(outcome.pop("effect"))
    _exact_keys(
        effect,
        {"use_x", "intercept", "scale", "coefficients"},
        location=f"{location}.outcome.effect",
    )
    effect_coefficients = _coefficients(
        effect.pop("coefficients"), f"{location}.outcome.effect.coefficients"
    )

    powers = dict(value["powers"])
    _exact_keys(
        powers,
        {
            "proxy_quadratic_probability",
            "treatment_quadratic_probability",
            "outcome_quadratic_probability",
            "effect_quadratic_probability",
        },
        location=f"{location}.powers",
    )
    return MechanismConfig(
        proxies=ProxyConfig(coefficients=proxy_coefficients, **proxies),
        treatment=TreatmentConfig(coefficients=treatment_coefficients, **treatment),
        outcome=OutcomeConfig(
            coefficients=outcome_coefficients,
            effect=EffectConfig(coefficients=effect_coefficients, **effect),
            **outcome,
        ),
        powers=PowerAllocationConfig(**powers),
    )


def load_mechanism_configs(
    value: Mapping[str, Any], *, available: set[str]
) -> dict[str, MechanismConfig]:
    """Validate that the YAML has exactly one recipe per registered family."""
    _exact_keys(value, available, location="mechanisms")
    return {
        name: mechanism_config_from_dict(config, location=f"mechanisms.{name}")
        for name, config in value.items()
    }
