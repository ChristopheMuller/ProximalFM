"""Native calibrated latent-proxy mechanism based on the reference mathematics.

Production code never imports the read-only ``semisynt_eval.reference_mechanism``
package.
For source latent rows and independent innovations, the fitted mechanism is

``U = sqrt(1 - innovation_sd**2) U_source + innovation_sd eta_U``
``V = (g(U) - calibration_mean) T``
``L = V c``.

``T`` is a calibration-fitted ridge whitener and ``c`` is normalized so that
``L`` has the configured calibration standard deviation. The proxy channels
are rotated, noisy measurements of ``V`` plus fitted X components. Treatment
and outcomes share ``L`` but do not use a proxy as a direct parent. Therefore
``Y1 - Y0`` is the analytic CATE, and the rotated W signal yields an analytic
proximal outcome bridge.

All calculations stay in float64. Conversion to canonical float32 storage
belongs at the existing ``Episode`` boundary. This class remains unregistered
until the explicit paired lifecycle in ``ToDoCalibratedLatentProxy.md`` step 6
is implemented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any

import torch

from semisynt_eval._config import (
    BasisConfig,
    CaseConfig,
    CateConfig,
    Family,
    LatentConfig,
    OutcomeConfig,
    ProxyConfig,
    TreatmentConfig,
)
from semisynt_eval.mechanisms.base import Mechanism, MechanismOutput

Tensor = torch.Tensor


def stable_seed(*parts: object) -> int:
    """Derive the reference-compatible unsigned 64-bit named-stream seed."""
    token = "\0".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


@dataclass(frozen=True)
class NamedRandomStreams:
    """Deterministic random streams for one fitted mechanism identity.

    Fit state is family-specific. Calibration and evaluation innovations omit
    the family deliberately, so the linear and nonlinear mechanisms for one
    source/replicate use the same underlying innovations. Every accessor
    returns a fresh generator, making results independent of call order.
    """

    master_seed: int
    source_id: str
    replicate: int
    family: Family

    def __post_init__(self) -> None:
        if not isinstance(self.master_seed, int) or isinstance(self.master_seed, bool):
            raise TypeError("master_seed must be an integer")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(self.replicate, int) or isinstance(self.replicate, bool):
            raise TypeError("replicate must be an integer")
        if self.replicate < 1:
            raise ValueError("replicate must be one-based and positive")
        if self.family not in ("linear", "nonlinear"):
            raise ValueError(f"unknown family {self.family!r}")

    @staticmethod
    def _generator(*parts: object) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(stable_seed(*parts))

    def fit_state(self) -> torch.Generator:
        return self._generator(
            self.master_seed,
            self.source_id,
            self.replicate,
            self.family,
            "fit",
        )

    def calibration_innovation(self) -> torch.Generator:
        return self._generator(
            self.master_seed,
            self.source_id,
            self.replicate,
            "cal_innovation",
        )

    def evaluation(self, name: str) -> torch.Generator:
        if name not in {"u", "w", "z", "y", "a"}:
            raise ValueError(f"unknown evaluation innovation {name!r}")
        return self._generator(
            self.master_seed,
            self.source_id,
            self.replicate,
            "evaluation",
            name,
        )


def _matrix(value: Tensor, name: str) -> Tensor:
    value = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix, got shape {tuple(value.shape)}")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} must have positive row and column counts")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _column_scale(value: Tensor, minimum: float) -> Tensor:
    return value.std(dim=0, unbiased=False).clamp_min(float(minimum))


@dataclass(frozen=True)
class FittedBasis:
    """Calibration-fitted X basis shared by all mechanism components."""

    family: Family
    transforms: Tensor
    interaction_pairs: tuple[tuple[int, int], ...]
    center: Tensor
    scale: Tensor


@dataclass(frozen=True)
class FittedComponent:
    """One normalized random projection of the fixed X basis."""

    coefficients: Tensor
    center: Tensor
    scale: Tensor


@dataclass(frozen=True)
class EvaluationInnovations:
    """Named evaluation streams reused across paired factorial cases."""

    u: Tensor
    w: Tensor
    z: Tensor
    y: Tensor
    a: Tensor


@dataclass(frozen=True)
class FittedFamilyConfig:
    """Case-invariant settings shared by one fitted function family."""

    mechanism: str
    family: Family
    latent: LatentConfig
    basis: BasisConfig
    proxies: ProxyConfig
    treatment: TreatmentConfig
    outcome: OutcomeConfig
    cate: CateConfig

    @classmethod
    def from_case(cls, case: CaseConfig) -> "FittedFamilyConfig":
        return cls(
            mechanism=case.mechanism,
            family=case.family,
            latent=case.latent,
            basis=case.basis,
            proxies=case.proxies,
            treatment=case.treatment,
            outcome=case.outcome,
            cate=case.cate,
        )

    def validate_case(self, case: CaseConfig) -> None:
        if FittedFamilyConfig.from_case(case) != self:
            raise ValueError(
                f"case {case.name!r} does not match the fitted family settings"
            )
        if not 0 < case.reliability < 1:
            raise ValueError("proxy reliability must lie in (0, 1)")


@dataclass(frozen=True)
class PreparedEvaluation:
    """Case-invariant evaluation state for one source/replicate/family."""

    X: Tensor
    U_source: Tensor
    U: Tensor
    V: Tensor
    latent_index: Tensor
    components: dict[str, Tensor]
    CATE: Tensor
    innovations: EvaluationInnovations
    source_row_ids: Tensor


class CalibratedLatentProxyMechanism(Mechanism):
    """Calibration-fitted latent-proxy mechanism for one factorial case.

    The current registry contract provides a generator whose initial seed is
    the episode identity. Independent named streams are derived from it, so
    call order cannot change fitted state or generated variables. Step 6 will
    pass source/replicate identity explicitly and reuse a fitted object.
    """

    target_capability = "exact_cate"
    _COMPONENT_NAMES = ("a", "y", "tau", "w", "z")

    config: FittedFamilyConfig
    default_case: CaseConfig
    family: Family
    d_x: int
    d_u: int
    seed: int
    source_id: str
    replicate: int
    streams: NamedRandomStreams
    basis: FittedBasis
    components: dict[str, FittedComponent]
    latent_center: Tensor
    latent_covariance_eigenvalues: Tensor
    whitener: Tensor
    latent_direction: Tensor
    rotation_w: Tensor
    rotation_z: Tensor
    proxy_signal_sd_w: Tensor
    proxy_signal_sd_z: Tensor
    treatment_intercept: float

    @classmethod
    def fit(
        cls,
        x_cal: Tensor,
        u_cal: Tensor,
        config: Any,
        generator: torch.Generator,
    ) -> "CalibratedLatentProxyMechanism":
        """Compatibility adapter for the neutral mechanism registry contract.

        The current interface carries only a generator, so its initial seed is
        used as the master seed with a placeholder source identity. Step 6 will
        call :meth:`fit_with_identity` with the real source and replicate.
        """
        if not isinstance(config, CaseConfig):
            raise TypeError("config must be a resolved CaseConfig")
        return cls.fit_with_identity(
            x_cal,
            u_cal,
            config,
            master_seed=int(generator.initial_seed()),
            source_id="generator-adapter",
            replicate=1,
        )

    @classmethod
    def fit_with_identity(
        cls,
        x_cal: Tensor,
        u_cal: Tensor,
        config: CaseConfig,
        *,
        master_seed: int,
        source_id: str,
        replicate: int,
    ) -> "CalibratedLatentProxyMechanism":
        """Fit using explicit identity and independent named streams."""
        if not isinstance(config, CaseConfig):
            raise TypeError("config must be a resolved CaseConfig")
        if config.mechanism != "calibrated_latent_proxy":
            raise ValueError("config does not select calibrated_latent_proxy")
        if config.family not in ("linear", "nonlinear"):
            raise ValueError(f"unknown family {config.family!r}")

        x_cal = _matrix(x_cal, "x_cal")
        u_cal = _matrix(u_cal, "u_cal")
        if x_cal.shape[0] != u_cal.shape[0]:
            raise ValueError("x_cal and u_cal must have matching rows")
        family_config = FittedFamilyConfig.from_case(config)
        family_config.validate_case(config)

        instance = cls()
        instance.config = family_config
        instance.default_case = config
        instance.family = config.family
        instance.d_x = x_cal.shape[1]
        instance.d_u = u_cal.shape[1]
        instance.streams = NamedRandomStreams(
            master_seed=master_seed,
            source_id=source_id,
            replicate=replicate,
            family=config.family,
        )
        instance.seed = instance.streams.master_seed
        instance.source_id = instance.streams.source_id
        instance.replicate = instance.streams.replicate

        fit_rng = instance.streams.fit_state()
        cal_rng = instance.streams.calibration_innovation()
        latent_cal = instance._augment_latent(u_cal, cal_rng)
        raw_latent = instance.latent_transform(latent_cal)
        instance.latent_center = raw_latent.mean(dim=0)
        centered = raw_latent - instance.latent_center
        covariance = centered.T @ centered / centered.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        instance.latent_covariance_eigenvalues = eigenvalues
        inverse_root = (eigenvalues + config.latent.whitening_ridge).rsqrt()
        instance.whitener = (eigenvectors * inverse_root) @ eigenvectors.T
        v_cal = instance.latent_coordinates(latent_cal)

        direction = torch.randn(instance.d_u, generator=fit_rng, dtype=torch.float64)
        direction_scale = _column_scale(v_cal @ direction, config.basis.min_scale)
        instance.latent_direction = (
            config.latent.direction_sd * direction / direction_scale
        )
        instance.rotation_w = instance._orthogonal(instance.d_u, fit_rng)
        instance.rotation_z = instance._orthogonal(instance.d_u, fit_rng)
        instance.proxy_signal_sd_w = _column_scale(
            v_cal @ instance.rotation_w.T, config.basis.min_scale
        )
        instance.proxy_signal_sd_z = _column_scale(
            v_cal @ instance.rotation_z.T, config.basis.min_scale
        )

        instance.basis = instance._fit_basis(x_cal, instance.family, config, fit_rng)
        design_cal = instance.basis_design(x_cal)
        widths = {
            "a": 1,
            "y": 1,
            "tau": 1,
            "w": instance.d_u,
            "z": instance.d_u,
        }
        instance.components = {
            name: instance._fit_component(design_cal, widths[name], config, fit_rng)
            for name in instance._COMPONENT_NAMES
        }
        instance.treatment_intercept = float(
            torch.empty((), dtype=torch.float64).uniform_(
                config.treatment.intercept_low,
                config.treatment.intercept_high,
                generator=fit_rng,
            )
        )
        return instance

    @staticmethod
    def _fit_basis(
        x_cal: Tensor,
        family: Family,
        config: CaseConfig,
        generator: torch.Generator,
    ) -> FittedBasis:
        width = x_cal.shape[1]
        transforms = (
            torch.zeros(width, dtype=torch.long)
            if family == "linear"
            else torch.randint(
                len(config.basis.nonlinear_transforms),
                (width,),
                generator=generator,
            )
        )
        candidates = tuple(
            (left, right) for left in range(width) for right in range(left + 1, width)
        )
        pairs: tuple[tuple[int, int], ...] = ()
        if family == "nonlinear" and candidates:
            count = min(config.basis.max_interactions, len(candidates))
            order = torch.randperm(len(candidates), generator=generator)[:count]
            pairs = tuple(candidates[int(index)] for index in order)
        raw = CalibratedLatentProxyMechanism._raw_basis(
            x_cal, transforms, pairs, config
        )
        return FittedBasis(
            family=family,
            transforms=transforms,
            interaction_pairs=pairs,
            center=raw.mean(dim=0),
            scale=_column_scale(raw, config.basis.min_scale),
        )

    @staticmethod
    def _raw_basis(
        x: Tensor,
        transforms: Tensor,
        pairs: tuple[tuple[int, int], ...],
        config: CaseConfig,
    ) -> Tensor:
        columns = []
        names = config.basis.nonlinear_transforms
        for column, transform_index in enumerate(transforms.tolist()):
            value = x[:, column]
            transform = names[transform_index]
            if transform == "identity":
                result = value
            elif transform == "asinh":
                result = torch.asinh(value)
            elif transform == "sine":
                result = torch.sin(value)
            elif transform == "quadratic":
                result = value.square()
            else:
                raise ValueError(f"unknown basis transform {transform!r}")
            columns.append(result)
        columns.extend(x[:, left] * x[:, right] for left, right in pairs)
        result = torch.stack(columns, dim=1)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("X basis produced non-finite values")
        return result

    @staticmethod
    def _fit_component(
        design_cal: Tensor,
        width: int,
        config: CaseConfig,
        generator: torch.Generator,
    ) -> FittedComponent:
        if width < 1:
            raise ValueError("component output width must be positive")
        coefficient_scale = config.basis.coefficient_scale
        if config.basis.normalize_by_basis_dim:
            coefficient_scale /= math.sqrt(design_cal.shape[1])
        coefficients = coefficient_scale * torch.randn(
            design_cal.shape[1], width, generator=generator, dtype=torch.float64
        )
        raw = design_cal @ coefficients
        return FittedComponent(
            coefficients=coefficients,
            center=raw.mean(dim=0),
            scale=_column_scale(raw, config.basis.min_scale),
        )

    @staticmethod
    def _orthogonal(width: int, generator: torch.Generator) -> Tensor:
        draw = torch.randn(width, width, generator=generator, dtype=torch.float64)
        q, r = torch.linalg.qr(draw)
        signs = torch.where(torch.diagonal(r) >= 0, 1.0, -1.0)
        return q * signs

    def _augment_latent(self, u_source: Tensor, generator: torch.Generator) -> Tensor:
        innovation_sd = self.config.latent.innovation_sd
        innovation = torch.randn(
            u_source.shape, generator=generator, dtype=torch.float64
        )
        return (
            math.sqrt(1.0 - innovation_sd**2) * u_source + innovation_sd * innovation
        )

    def basis_design(self, x: Tensor) -> Tensor:
        x = _matrix(x, "x")
        if x.shape[1] != self.d_x:
            raise ValueError(f"x has width {x.shape[1]}, expected {self.d_x}")
        raw = self._raw_basis(
            x,
            self.basis.transforms,
            self.basis.interaction_pairs,
            self.config,
        )
        return (raw - self.basis.center) / self.basis.scale

    def component(self, x: Tensor, name: str) -> Tensor:
        try:
            fitted = self.components[name]
        except KeyError:
            raise KeyError(
                f"unknown component {name!r}; expected {self._COMPONENT_NAMES}"
            ) from None
        raw = self.basis_design(x) @ fitted.coefficients
        return (raw - fitted.center) / fitted.scale

    def latent_transform(self, u: Tensor) -> Tensor:
        return u if self.family == "linear" else torch.asinh(u)

    def latent_coordinates(self, u: Tensor) -> Tensor:
        u = _matrix(u, "u")
        if u.shape[1] != self.d_u:
            raise ValueError(f"u has width {u.shape[1]}, expected {self.d_u}")
        return (self.latent_transform(u) - self.latent_center) @ self.whitener

    def wrap_proxy(self, value: Tensor) -> Tensor:
        return value if self.family == "linear" else torch.asinh(value)

    def unwrap_proxy(self, value: Tensor) -> Tensor:
        return value if self.family == "linear" else torch.sinh(value)

    def innovations(self, n_rows: int) -> EvaluationInnovations:
        """Materialize named streams without consuming fitted state."""
        if n_rows < 1:
            raise ValueError("n_rows must be positive")

        def normal(name: str, width: int | None = None) -> Tensor:
            shape = (n_rows,) if width is None else (n_rows, width)
            rng = self.streams.evaluation(name)
            return torch.randn(shape, generator=rng, dtype=torch.float64)

        a_rng = self.streams.evaluation("a")
        return EvaluationInnovations(
            u=normal("u", self.d_u),
            w=normal("w", self.d_u),
            z=normal("z", self.d_u),
            y=normal("y"),
            a=torch.rand(n_rows, generator=a_rng, dtype=torch.float64),
        )

    def cate(self, x: Tensor) -> Tensor:
        return self.config.cate.intercept + self.config.cate.scale * self.component(
            x, "tau"
        )

    def bridge(self, x: Tensor, w: Tensor, a: Tensor, *, gamma: float) -> Tensor:
        x = _matrix(x, "x")
        w = _matrix(w, "w")
        if x.shape[0] != w.shape[0] or w.shape[1] != self.d_u:
            raise ValueError("x and w must have matching rows and fitted widths")
        treatment = torch.as_tensor(a, dtype=torch.float64, device="cpu")
        if treatment.ndim == 0:
            treatment = treatment.expand(x.shape[0], 1)
        elif treatment.ndim == 1:
            treatment = treatment[:, None]
        if treatment.shape != (x.shape[0], 1):
            raise ValueError("a must be scalar or have shape (n,) or (n, 1)")
        residual = self.unwrap_proxy(w) - self.config.proxies.x_scale * self.component(
            x, "w"
        )
        unbiased_v = residual @ self.rotation_w
        latent_index = unbiased_v @ self.latent_direction
        return (
            self.component(x, "y")
            + float(gamma) * latent_index[:, None]
            + treatment * self.cate(x)
        )

    def prepare_evaluation(
        self,
        x: Tensor,
        u_source: Tensor,
        source_row_ids: Tensor,
    ) -> PreparedEvaluation:
        """Compute all state shared across cases of this fitted family."""
        x = _matrix(x, "x")
        u_source = _matrix(u_source, "u_source")
        if x.shape[0] != u_source.shape[0]:
            raise ValueError("x and u_source must have matching rows")
        if x.shape[1] != self.d_x or u_source.shape[1] != self.d_u:
            raise ValueError("evaluation widths do not match fitted widths")
        source_row_ids = torch.as_tensor(source_row_ids, dtype=torch.long, device="cpu")
        if source_row_ids.ndim != 1 or source_row_ids.shape[0] != x.shape[0]:
            raise ValueError("source_row_ids must have shape (n_rows,)")

        noise = self.innovations(x.shape[0])
        innovation_sd = self.config.latent.innovation_sd
        latent = (
            math.sqrt(1.0 - innovation_sd**2) * u_source + innovation_sd * noise.u
        )
        v = self.latent_coordinates(latent)
        latent_index = v @ self.latent_direction
        components = {name: self.component(x, name) for name in self._COMPONENT_NAMES}
        cate = self.config.cate.intercept + self.config.cate.scale * components["tau"]
        return PreparedEvaluation(
            X=x,
            U_source=u_source,
            U=latent,
            V=v,
            latent_index=latent_index,
            components=components,
            CATE=cate,
            innovations=noise,
            source_row_ids=source_row_ids,
        )

    def generate_case(
        self, prepared: PreparedEvaluation, case: CaseConfig
    ) -> MechanismOutput:
        """Apply case strengths to frozen fitted and prepared evaluation state."""
        self.config.validate_case(case)
        x = prepared.X
        noise = prepared.innovations
        multiplier = math.sqrt((1.0 - case.reliability) / case.reliability)
        w_raw = (
            prepared.V @ self.rotation_w.T
            + self.config.proxies.x_scale * prepared.components["w"]
            + multiplier * self.proxy_signal_sd_w * noise.w
        )
        z_raw = (
            prepared.V @ self.rotation_z.T
            + self.config.proxies.x_scale * prepared.components["z"]
            + multiplier * self.proxy_signal_sd_z * noise.z
        )
        w, z = self.wrap_proxy(w_raw), self.wrap_proxy(z_raw)

        logits = (
            self.treatment_intercept
            + self.config.treatment.x_scale * prepared.components["a"].squeeze(1)
            + case.alpha * prepared.latent_index
        )
        propensity = torch.sigmoid(logits).clamp(
            self.config.treatment.propensity_lower,
            self.config.treatment.propensity_upper,
        )
        a = (noise.a < propensity).to(torch.float64)[:, None]
        y0 = (
            prepared.components["y"]
            + case.gamma * prepared.latent_index[:, None]
            + self.config.outcome.noise_sd * noise.y[:, None]
        )
        y1 = y0 + prepared.CATE
        y_obs = torch.where(a.bool(), y1, y0)
        bridge0 = self.bridge(x, w, torch.tensor(0), gamma=case.gamma)
        bridge1 = self.bridge(x, w, torch.tensor(1), gamma=case.gamma)
        result = MechanismOutput(
            W=w,
            Z=z,
            A=a,
            A_probs=propensity[:, None],
            Y_obs=y_obs,
            Y0=y0,
            Y1=y1,
            CATE=prepared.CATE,
            U=prepared.U,
            U_source=prepared.U_source,
            V=prepared.V,
            bridge0=bridge0,
            bridge1=bridge1,
        )
        expected_shapes = {
            "W": (x.shape[0], self.d_u),
            "Z": (x.shape[0], self.d_u),
            "U": (x.shape[0], self.d_u),
            "U_source": (x.shape[0], self.d_u),
            "V": (x.shape[0], self.d_u),
            "A": (x.shape[0], 1),
            "A_probs": (x.shape[0], 1),
            "Y_obs": (x.shape[0], 1),
            "Y0": (x.shape[0], 1),
            "Y1": (x.shape[0], 1),
            "CATE": (x.shape[0], 1),
            "bridge0": (x.shape[0], 1),
            "bridge1": (x.shape[0], 1),
        }
        for name, expected_shape in expected_shapes.items():
            value = getattr(result, name)
            if value is None or value.shape != expected_shape:
                actual_shape = None if value is None else tuple(value.shape)
                raise ValueError(
                    f"invalid generated {name} shape {actual_shape}; "
                    f"expected {expected_shape}"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"generated {name} contains non-finite values")
        return result

    def generate(self, x: Tensor, u: Tensor) -> MechanismOutput:
        """Compatibility adapter using the case supplied when fitting."""
        row_ids = torch.arange(torch.as_tensor(x).shape[0], dtype=torch.long)
        prepared = self.prepare_evaluation(x, u, row_ids)
        return self.generate_case(prepared, self.default_case)

    def metadata(self) -> dict[str, object]:
        """Return JSON-serializable fitted state sufficient for an audit."""
        components = {
            name: {
                "coefficients": fitted.coefficients.tolist(),
                "center": fitted.center.tolist(),
                "scale": fitted.scale.tolist(),
            }
            for name, fitted in self.components.items()
        }
        return {
            "seed": self.seed,
            "source_id": self.source_id,
            "replicate": self.replicate,
            "family": self.family,
            "d_X": self.d_x,
            "d_U": self.d_u,
            "d_W": self.d_u,
            "d_Z": self.d_u,
            "resolved_config": asdict(self.config),
            "latent_source_variance_weight": 1.0
            - self.config.latent.innovation_sd**2,
            "latent_center": self.latent_center.tolist(),
            "latent_covariance_eigenvalues": self.latent_covariance_eigenvalues.tolist(),
            "whitener": self.whitener.tolist(),
            "latent_direction": self.latent_direction.tolist(),
            "B_W": self.rotation_w.tolist(),
            "B_Z": self.rotation_z.tolist(),
            "proxy_signal_sd_w": self.proxy_signal_sd_w.tolist(),
            "proxy_signal_sd_z": self.proxy_signal_sd_z.tolist(),
            "basis": {
                "transforms": self.basis.transforms.tolist(),
                "transform_names": list(self.config.basis.nonlinear_transforms),
                "interaction_pairs": [
                    list(pair) for pair in self.basis.interaction_pairs
                ],
                "center": self.basis.center.tolist(),
                "scale": self.basis.scale.tolist(),
            },
            "components": components,
            "treatment_intercept": self.treatment_intercept,
        }
