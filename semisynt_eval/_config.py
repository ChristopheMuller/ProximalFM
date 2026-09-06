"""Strict typed validation for format-v5 semi-synthetic recipes.

All data-affecting defaults live in YAML. This module only validates and
resolves the explicit ``common`` block with each named entry in ``cases``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Literal, Mapping

CoefficientDistribution = Literal["normal", "uniform"]
Family = Literal["linear", "nonlinear"]
TRANSFORMS = (
    "identity",
    "quadratic",
    "signed_log1p",
    "clipped_exp",
    "sine",
)


@dataclass(frozen=True)
class CoefficientConfig:
    distribution: CoefficientDistribution
    scale: float
    normalize_by_input_dim: bool


@dataclass(frozen=True)
class NonlinearConfig:
    probabilities: tuple[float, ...]
    exp_clip: float


@dataclass(frozen=True)
class ProxyEquationConfig:
    use_x: bool
    use_u: bool
    interactions: bool
    x_strength: float
    normalize_output: bool


@dataclass(frozen=True)
class TreatmentEquationConfig:
    use_x: bool
    use_u: bool
    use_z: bool
    interactions: bool
    normalize_score: bool
    logit_scale: float
    logit_clip: float
    intercept_low: float
    intercept_high: float
    overlap_epsilon: float


@dataclass(frozen=True)
class OutcomeEquationConfig:
    use_x: bool
    use_u: bool
    use_w: bool
    interactions: bool
    noise_std: float
    normalize_baseline: bool


@dataclass(frozen=True)
class EffectEquationConfig:
    use_x: bool
    interactions: bool
    intercept: float
    scale: float


@dataclass(frozen=True)
class CommonConfig:
    interactions_default: bool
    coefficients: CoefficientConfig
    nonlinear: NonlinearConfig
    w: ProxyEquationConfig
    z: ProxyEquationConfig
    a: TreatmentEquationConfig
    y: OutcomeEquationConfig
    tau: EffectEquationConfig


@dataclass(frozen=True)
class CaseConfig:
    name: str
    family: Family
    a_u_variance_share: float
    y_u_variance_share: float
    w_u_reliability: float
    z_u_reliability: float
    coefficients: CoefficientConfig
    nonlinear: NonlinearConfig
    w: ProxyEquationConfig
    z: ProxyEquationConfig
    a: TreatmentEquationConfig
    y: OutcomeEquationConfig
    tau: EffectEquationConfig


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


def _finite_nonnegative(value: float, location: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{location} must be finite and non-negative")
    return value


def _interaction(value: Any, default: bool, location: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{location} must be true, false, or null")
    return value


def _parse_common(value: Mapping[str, Any]) -> CommonConfig:
    _exact_keys(
        value,
        {"interactions_default", "coefficients", "nonlinear", "equations"},
        location="common",
    )
    default = value["interactions_default"]
    if not isinstance(default, bool):
        raise ValueError("common.interactions_default must be boolean")

    coefficients = dict(value["coefficients"])
    _exact_keys(
        coefficients,
        {"distribution", "scale", "normalize_by_input_dim"},
        location="common.coefficients",
    )
    if coefficients["distribution"] not in ("normal", "uniform"):
        raise ValueError("common.coefficients.distribution must be normal or uniform")
    coefficients["scale"] = _finite_nonnegative(
        coefficients["scale"], "common.coefficients.scale"
    )
    coefficient_config = CoefficientConfig(**coefficients)

    nonlinear = dict(value["nonlinear"])
    _exact_keys(nonlinear, {"probabilities", "exp_clip"}, location="common.nonlinear")
    probabilities = dict(nonlinear["probabilities"])
    _exact_keys(
        probabilities, set(TRANSFORMS), location="common.nonlinear.probabilities"
    )
    probs = tuple(float(probabilities[name]) for name in TRANSFORMS)
    if any(not math.isfinite(p) or p < 0 for p in probs) or not math.isclose(
        sum(probs), 1.0, abs_tol=1e-8
    ):
        raise ValueError(
            "common.nonlinear.probabilities must be non-negative and sum to 1"
        )
    exp_clip = float(nonlinear["exp_clip"])
    if not math.isfinite(exp_clip) or exp_clip <= 0:
        raise ValueError("common.nonlinear.exp_clip must be finite and positive")
    nonlinear_config = NonlinearConfig(probabilities=probs, exp_clip=exp_clip)

    equations = dict(value["equations"])
    _exact_keys(equations, {"w", "z", "a", "y", "tau"}, location="common.equations")

    def proxy(name: str) -> ProxyEquationConfig:
        item = dict(equations[name])
        _exact_keys(
            item,
            {
                "use_x",
                "use_u",
                "interactions",
                "x_strength",
                "normalize_output",
            },
            location=f"common.equations.{name}",
        )
        item["interactions"] = _interaction(
            item["interactions"],
            default,
            f"common.equations.{name}.interactions",
        )
        item["x_strength"] = _finite_nonnegative(
            item["x_strength"], f"common.equations.{name}.x_strength"
        )
        if not (item["use_x"] or item["use_u"]):
            raise ValueError(f"common.equations.{name} needs X and/or U")
        return ProxyEquationConfig(**item)

    a = dict(equations["a"])
    _exact_keys(
        a,
        {
            "use_x",
            "use_u",
            "use_z",
            "interactions",
            "normalize_score",
            "logit_scale",
            "logit_clip",
            "intercept_low",
            "intercept_high",
            "overlap_epsilon",
        },
        location="common.equations.a",
    )
    a["interactions"] = _interaction(
        a["interactions"], default, "common.equations.a.interactions"
    )
    if not (a["use_x"] or a["use_u"] or a["use_z"]):
        raise ValueError("common.equations.a needs X, U, and/or Z")
    a["logit_scale"] = _finite_nonnegative(
        a["logit_scale"], "common.equations.a.logit_scale"
    )
    if float(a["logit_clip"]) <= 0:
        raise ValueError("common.equations.a.logit_clip must be positive")
    if float(a["intercept_low"]) > float(a["intercept_high"]):
        raise ValueError("common.equations.a intercept bounds are reversed")
    if not 0 <= float(a["overlap_epsilon"]) < 0.5:
        raise ValueError("common.equations.a.overlap_epsilon must lie in [0, 0.5)")
    treatment = TreatmentEquationConfig(**a)

    y = dict(equations["y"])
    _exact_keys(
        y,
        {
            "use_x",
            "use_u",
            "use_w",
            "interactions",
            "noise_std",
            "normalize_baseline",
        },
        location="common.equations.y",
    )
    y["interactions"] = _interaction(
        y["interactions"], default, "common.equations.y.interactions"
    )
    if not (y["use_x"] or y["use_u"] or y["use_w"]):
        raise ValueError("common.equations.y needs X, U, and/or W")
    y["noise_std"] = _finite_nonnegative(y["noise_std"], "common.equations.y.noise_std")
    outcome = OutcomeEquationConfig(**y)

    tau = dict(equations["tau"])
    _exact_keys(
        tau,
        {"use_x", "interactions", "intercept", "scale"},
        location="common.equations.tau",
    )
    tau["interactions"] = _interaction(
        tau["interactions"], default, "common.equations.tau.interactions"
    )
    tau["scale"] = _finite_nonnegative(tau["scale"], "common.equations.tau.scale")
    effect = EffectEquationConfig(**tau)

    return CommonConfig(
        interactions_default=default,
        coefficients=coefficient_config,
        nonlinear=nonlinear_config,
        w=proxy("w"),
        z=proxy("z"),
        a=treatment,
        y=outcome,
        tau=effect,
    )


def _with_equation_overrides(
    common: CommonConfig, overrides: Mapping[str, Any], *, location: str
) -> CommonConfig:
    unknown_equations = set(overrides) - {"w", "z", "a", "y", "tau"}
    if unknown_equations:
        raise ValueError(f"invalid {location}: unknown {sorted(unknown_equations)}")
    result = common
    allowed = {
        "w": set(ProxyEquationConfig.__dataclass_fields__),
        "z": set(ProxyEquationConfig.__dataclass_fields__),
        "a": set(TreatmentEquationConfig.__dataclass_fields__),
        "y": set(OutcomeEquationConfig.__dataclass_fields__),
        "tau": set(EffectEquationConfig.__dataclass_fields__),
    }
    for name, raw in overrides.items():
        patch = dict(raw)
        unknown = set(patch) - allowed[name]
        if unknown:
            raise ValueError(f"invalid {location}.{name}: unknown {sorted(unknown)}")
        if "interactions" in patch:
            patch["interactions"] = _interaction(
                patch["interactions"],
                common.interactions_default,
                f"{location}.{name}.interactions",
            )
        result = replace(result, **{name: replace(getattr(result, name), **patch)})
    if not (result.w.use_x or result.w.use_u) or not (result.z.use_x or result.z.use_u):
        raise ValueError(f"{location}: proxy equations need X and/or U")
    if not (result.a.use_x or result.a.use_u or result.a.use_z):
        raise ValueError(f"{location}.a needs X, U, and/or Z")
    if not (result.y.use_x or result.y.use_u or result.y.use_w):
        raise ValueError(f"{location}.y needs X, U, and/or W")
    for name in ("w", "z"):
        if getattr(result, name).x_strength < 0:
            raise ValueError(f"{location}.{name}.x_strength must be non-negative")
    if result.a.logit_scale < 0 or result.a.logit_clip <= 0:
        raise ValueError(f"{location}.a logit scale/clip are invalid")
    if result.a.intercept_low > result.a.intercept_high:
        raise ValueError(f"{location}.a intercept bounds are reversed")
    if not 0 <= result.a.overlap_epsilon < 0.5:
        raise ValueError(f"{location}.a.overlap_epsilon must lie in [0, 0.5)")
    if result.y.noise_std < 0:
        raise ValueError(f"{location}.y.noise_std must be non-negative")
    if result.tau.scale < 0:
        raise ValueError(f"{location}.tau.scale must be non-negative")
    return result


def load_case_configs(
    common_value: Mapping[str, Any], cases_value: Mapping[str, Any]
) -> dict[str, CaseConfig]:
    """Resolve and validate every named matrix case."""
    common = _parse_common(common_value)
    if not cases_value:
        raise ValueError("cases must contain at least one named case")
    resolved: dict[str, CaseConfig] = {}
    for name, raw in cases_value.items():
        if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
            raise ValueError(f"invalid case name {name!r}")
        case = dict(raw)
        _exact_keys(
            case,
            {
                "family",
                "a_u_variance_share",
                "y_u_variance_share",
                "w_u_reliability",
                "z_u_reliability",
                "equations",
            },
            location=f"cases.{name}",
        )
        family = case["family"]
        if family not in ("linear", "nonlinear"):
            raise ValueError(f"cases.{name}.family must be linear or nonlinear")
        merged = _with_equation_overrides(
            common, case["equations"], location=f"cases.{name}.equations"
        )
        strengths = {
            key: float(case[key])
            for key in (
                "a_u_variance_share",
                "y_u_variance_share",
                "w_u_reliability",
                "z_u_reliability",
            )
        }
        for key, scalar in strengths.items():
            if not math.isfinite(scalar) or not 0 <= scalar <= 1:
                raise ValueError(f"cases.{name}.{key} must lie in [0, 1]")
        _validate_case_strengths(name, merged, strengths)
        resolved[name] = CaseConfig(
            name=name,
            family=family,
            coefficients=merged.coefficients,
            nonlinear=merged.nonlinear,
            w=merged.w,
            z=merged.z,
            a=merged.a,
            y=merged.y,
            tau=merged.tau,
            **strengths,
        )
    return resolved


def _validate_case_strengths(
    name: str, common: CommonConfig, strengths: Mapping[str, float]
) -> None:
    for equation, use_u, has_observed in (
        ("a", common.a.use_u, common.a.use_x or common.a.use_z),
        ("y", common.y.use_u, common.y.use_x or common.y.use_w),
    ):
        share = strengths[f"{equation}_u_variance_share"]
        if not use_u and share != 0:
            raise ValueError(
                f"cases.{name}: {equation} U share must be 0 when U is disabled"
            )
        if use_u and not has_observed and share != 1:
            raise ValueError(
                f"cases.{name}: {equation} U share must be 1 when U is the only parent"
            )
        if use_u and has_observed and not 0 < share < 1:
            raise ValueError(
                f"cases.{name}: {equation} U share must be strictly between 0 and 1"
            )
    for equation in ("w", "z"):
        use_u = getattr(common, equation).use_u
        reliability = strengths[f"{equation}_u_reliability"]
        if use_u and not 0 < reliability <= 1:
            raise ValueError(
                f"cases.{name}: {equation} reliability must lie in (0, 1] when U is enabled"
            )
        if not use_u and reliability != 0:
            raise ValueError(
                f"cases.{name}: {equation} reliability must be 0 when U is disabled"
            )
