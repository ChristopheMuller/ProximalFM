"""Strict typed validation for format-v5 semi-synthetic recipes.

The recipe is the complete scientific specification of the calibrated latent
proxy mechanism. Python validates and resolves it; it supplies no hidden
data-affecting defaults.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Literal, Mapping

Family = Literal["linear", "nonlinear"]
FAMILIES = ("linear", "nonlinear")
LEVEL_NAMES = ("low", "high")
NONLINEAR_TRANSFORMS = ("identity", "asinh", "sine", "quadratic")


@dataclass(frozen=True)
class LatentConfig:
    innovation_sd: float
    whitening_ridge: float
    direction_sd: float


@dataclass(frozen=True)
class BasisConfig:
    nonlinear_transforms: tuple[str, ...]
    max_interactions: int
    coefficient_distribution: str
    coefficient_scale: float
    normalize_by_basis_dim: bool
    min_scale: float


@dataclass(frozen=True)
class ProxyConfig:
    x_scale: float


@dataclass(frozen=True)
class TreatmentConfig:
    x_scale: float
    intercept_low: float
    intercept_high: float
    propensity_lower: float
    propensity_upper: float


@dataclass(frozen=True)
class OutcomeConfig:
    noise_sd: float


@dataclass(frozen=True)
class CateConfig:
    intercept: float
    scale: float


@dataclass(frozen=True)
class ConfoundingLevel:
    alpha: float
    gamma: float


@dataclass(frozen=True)
class ProxyLevel:
    reliability: float


@dataclass(frozen=True)
class CommonConfig:
    mechanism: str
    mechanism_version: str
    latent: LatentConfig
    basis: BasisConfig
    proxies: ProxyConfig
    treatment: TreatmentConfig
    outcome: OutcomeConfig
    cate: CateConfig
    confounding_levels: dict[str, ConfoundingLevel]
    proxy_levels: dict[str, ProxyLevel]


@dataclass(frozen=True)
class CaseConfig:
    """One resolved cell of the family x confounding x proxy matrix."""

    name: str
    mechanism: str
    mechanism_version: str
    family: Family
    confounding: str
    proxy: str
    alpha: float
    gamma: float
    reliability: float
    latent: LatentConfig
    basis: BasisConfig
    proxies: ProxyConfig
    treatment: TreatmentConfig
    outcome: OutcomeConfig
    cate: CateConfig


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


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def _finite(value: Any, location: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{location} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{location} must be finite")
    return value


def _positive(value: Any, location: str) -> float:
    value = _finite(value, location)
    if value <= 0:
        raise ValueError(f"{location} must be positive")
    return value


def _nonnegative(value: Any, location: str) -> float:
    value = _finite(value, location)
    if value < 0:
        raise ValueError(f"{location} must be non-negative")
    return value


def _parse_common(value: Mapping[str, Any]) -> CommonConfig:
    _exact_keys(
        value,
        {
            "mechanism",
            "mechanism_version",
            "latent",
            "basis",
            "proxies",
            "treatment",
            "outcome",
            "cate",
            "levels",
        },
        location="common",
    )
    mechanism = value["mechanism"]
    if mechanism != "calibrated_latent_proxy":
        raise ValueError("common.mechanism must be calibrated_latent_proxy")
    mechanism_version = value["mechanism_version"]
    if not isinstance(mechanism_version, str) or not mechanism_version.strip():
        raise ValueError("common.mechanism_version must be a non-empty string")

    latent = dict(_mapping(value["latent"], "common.latent"))
    _exact_keys(
        latent,
        {"innovation_sd", "whitening_ridge", "direction_sd"},
        location="common.latent",
    )
    innovation_sd = _nonnegative(latent["innovation_sd"], "common.latent.innovation_sd")
    if innovation_sd >= 1:
        raise ValueError("common.latent.innovation_sd must lie in [0, 1)")
    latent_config = LatentConfig(
        innovation_sd=innovation_sd,
        whitening_ridge=_positive(
            latent["whitening_ridge"], "common.latent.whitening_ridge"
        ),
        direction_sd=_positive(latent["direction_sd"], "common.latent.direction_sd"),
    )

    basis = dict(_mapping(value["basis"], "common.basis"))
    _exact_keys(
        basis,
        {
            "nonlinear_transforms",
            "max_interactions",
            "coefficient_distribution",
            "coefficient_scale",
            "normalize_by_basis_dim",
            "min_scale",
        },
        location="common.basis",
    )
    transforms = basis["nonlinear_transforms"]
    if not isinstance(transforms, list) or tuple(transforms) != NONLINEAR_TRANSFORMS:
        raise ValueError(
            "common.basis.nonlinear_transforms must be exactly "
            f"{list(NONLINEAR_TRANSFORMS)}"
        )
    max_interactions = basis["max_interactions"]
    if isinstance(max_interactions, bool) or not isinstance(max_interactions, int):
        raise ValueError("common.basis.max_interactions must be an integer")
    if max_interactions < 0:
        raise ValueError("common.basis.max_interactions must be non-negative")
    if basis["coefficient_distribution"] != "normal":
        raise ValueError("common.basis.coefficient_distribution must be normal")
    if not isinstance(basis["normalize_by_basis_dim"], bool):
        raise ValueError("common.basis.normalize_by_basis_dim must be boolean")
    basis_config = BasisConfig(
        nonlinear_transforms=tuple(transforms),
        max_interactions=max_interactions,
        coefficient_distribution="normal",
        coefficient_scale=_positive(
            basis["coefficient_scale"], "common.basis.coefficient_scale"
        ),
        normalize_by_basis_dim=basis["normalize_by_basis_dim"],
        min_scale=_positive(basis["min_scale"], "common.basis.min_scale"),
    )

    proxies = dict(_mapping(value["proxies"], "common.proxies"))
    _exact_keys(proxies, {"x_scale"}, location="common.proxies")
    proxy_config = ProxyConfig(
        x_scale=_nonnegative(proxies["x_scale"], "common.proxies.x_scale")
    )

    treatment = dict(_mapping(value["treatment"], "common.treatment"))
    _exact_keys(
        treatment,
        {
            "x_scale",
            "intercept_low",
            "intercept_high",
            "propensity_lower",
            "propensity_upper",
        },
        location="common.treatment",
    )
    intercept_low = _finite(
        treatment["intercept_low"], "common.treatment.intercept_low"
    )
    intercept_high = _finite(
        treatment["intercept_high"], "common.treatment.intercept_high"
    )
    if intercept_low > intercept_high:
        raise ValueError("common.treatment intercept bounds are reversed")
    propensity_lower = _finite(
        treatment["propensity_lower"], "common.treatment.propensity_lower"
    )
    propensity_upper = _finite(
        treatment["propensity_upper"], "common.treatment.propensity_upper"
    )
    if not 0 <= propensity_lower < propensity_upper <= 1:
        raise ValueError(
            "common.treatment propensity bounds must satisfy " "0 <= lower < upper <= 1"
        )
    treatment_config = TreatmentConfig(
        x_scale=_nonnegative(treatment["x_scale"], "common.treatment.x_scale"),
        intercept_low=intercept_low,
        intercept_high=intercept_high,
        propensity_lower=propensity_lower,
        propensity_upper=propensity_upper,
    )

    outcome = dict(_mapping(value["outcome"], "common.outcome"))
    _exact_keys(outcome, {"noise_sd"}, location="common.outcome")
    outcome_config = OutcomeConfig(
        noise_sd=_nonnegative(outcome["noise_sd"], "common.outcome.noise_sd")
    )

    cate = dict(_mapping(value["cate"], "common.cate"))
    _exact_keys(cate, {"intercept", "scale"}, location="common.cate")
    cate_config = CateConfig(
        intercept=_finite(cate["intercept"], "common.cate.intercept"),
        scale=_positive(cate["scale"], "common.cate.scale"),
    )

    levels = dict(_mapping(value["levels"], "common.levels"))
    _exact_keys(levels, {"confounding", "proxy"}, location="common.levels")
    raw_confounding = _mapping(levels["confounding"], "common.levels.confounding")
    raw_proxy = _mapping(levels["proxy"], "common.levels.proxy")
    expected_levels = set(LEVEL_NAMES)
    if set(raw_confounding) != expected_levels:
        raise ValueError("common.levels.confounding must contain exactly low and high")
    if set(raw_proxy) != expected_levels:
        raise ValueError("common.levels.proxy must contain exactly low and high")

    confounding_levels = {}
    for name, raw in raw_confounding.items():
        location = f"common.levels.confounding.{name}"
        item = dict(_mapping(raw, location))
        _exact_keys(item, {"alpha", "gamma"}, location=location)
        confounding_levels[str(name)] = ConfoundingLevel(
            alpha=_nonnegative(item["alpha"], f"{location}.alpha"),
            gamma=_nonnegative(item["gamma"], f"{location}.gamma"),
        )

    proxy_levels = {}
    for name, raw in raw_proxy.items():
        location = f"common.levels.proxy.{name}"
        item = dict(_mapping(raw, location))
        _exact_keys(item, {"reliability"}, location=location)
        reliability = _finite(item["reliability"], f"{location}.reliability")
        if not 0 < reliability < 1:
            raise ValueError(f"{location}.reliability must lie in (0, 1)")
        proxy_levels[str(name)] = ProxyLevel(reliability=reliability)

    return CommonConfig(
        mechanism=mechanism,
        mechanism_version=mechanism_version,
        latent=latent_config,
        basis=basis_config,
        proxies=proxy_config,
        treatment=treatment_config,
        outcome=outcome_config,
        cate=cate_config,
        confounding_levels=confounding_levels,
        proxy_levels=proxy_levels,
    )


def load_case_configs(
    common_value: Mapping[str, Any], cases_value: Mapping[str, Any]
) -> dict[str, CaseConfig]:
    """Resolve and validate every named matrix case."""
    common = _parse_common(common_value)
    if not cases_value:
        raise ValueError("cases must contain at least one named case")
    resolved = {}
    for name, raw in cases_value.items():
        if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
            raise ValueError(f"invalid case name {name!r}")
        location = f"cases.{name}"
        case = dict(_mapping(raw, location))
        _exact_keys(case, {"family", "confounding", "proxy"}, location=location)
        family = case["family"]
        if family not in FAMILIES:
            raise ValueError(f"{location}.family must be linear or nonlinear")
        confounding = case["confounding"]
        if confounding not in common.confounding_levels:
            raise ValueError(
                f"{location}.confounding must name one of "
                f"{sorted(common.confounding_levels)}"
            )
        proxy = case["proxy"]
        if proxy not in common.proxy_levels:
            raise ValueError(
                f"{location}.proxy must name one of {sorted(common.proxy_levels)}"
            )
        confounding_level = common.confounding_levels[confounding]
        proxy_level = common.proxy_levels[proxy]
        resolved[name] = CaseConfig(
            name=name,
            mechanism=common.mechanism,
            mechanism_version=common.mechanism_version,
            family=family,
            confounding=confounding,
            proxy=proxy,
            alpha=confounding_level.alpha,
            gamma=confounding_level.gamma,
            reliability=proxy_level.reliability,
            latent=common.latent,
            basis=common.basis,
            proxies=common.proxies,
            treatment=common.treatment,
            outcome=common.outcome,
            cate=common.cate,
        )

    counts = Counter(
        (case.family, case.confounding, case.proxy) for case in resolved.values()
    )
    expected = set(product(FAMILIES, LEVEL_NAMES, LEVEL_NAMES))
    missing = sorted(expected - set(counts))
    duplicates = sorted(cell for cell, count in counts.items() if count > 1)
    if missing or duplicates:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if duplicates:
            details.append(f"duplicates={duplicates}")
        raise ValueError(
            "cases must contain each family x confounding x proxy cell exactly "
            f"once: {', '.join(details)}"
        )
    return resolved
