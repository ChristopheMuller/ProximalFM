"""Native calibrated latent-proxy mechanism.

This module is the production home for the mechanism described in
``ToDoAyu.md``.  The implementation must remain independent of the read-only
``semisynt_eval.ayu`` reference package: equivalence belongs in tests, not in
production imports.

The fitted law is

``U = sqrt(1 - innovation_sd**2) * U_source + innovation_sd * eta_U``
``V = (g(U) - latent_center) @ whitener``
``L = V @ latent_direction``

with rotated noisy measurements of ``V`` for ``W`` and ``Z``.  Treatment and
both potential outcomes share ``L``; treatment has no direct ``Z`` parent and
the potential outcomes have no direct ``W`` parent.  Consequently
``Y1 - Y0 = CATE(X)`` exactly.

The class is intentionally not registered yet.  Its outline makes the fitted
state and pure evaluation stages explicit while preserving the builder's
fail-closed behavior until the native implementation is complete.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import torch

from semisynt_eval._config import CaseConfig, Family
from semisynt_eval.mechanisms.base import Mechanism, MechanismOutput

Tensor = torch.Tensor


def stable_seed(*parts: object) -> int:
    """Derive a deterministic unsigned 64-bit seed from named identity parts."""
    token = "\0".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


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


class CalibratedLatentProxyMechanism(Mechanism):
    """Ayu-equivalent mechanism fitted only on calibration rows.

    One instance will ultimately be fitted per ``(source, replicate, family)``
    and reused for all four confounding/proxy cells in that family.  The
    orchestration changes needed to provide that identity and reuse are kept
    separate from this native mechanism implementation.
    """

    target_capability = "exact_cate"

    config: CaseConfig
    family: Family
    d_x: int
    d_u: int
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
        """Fit every data-dependent quantity using calibration rows only.

        Implementation order:

        1. validate float64 calibration inputs and the resolved case;
        2. fit the X basis and normalized component projections;
        3. draw calibration latent innovation and fit the ridge whitener;
        4. normalize the shared latent direction;
        5. construct oriented orthogonal proxy rotations and signal scales.

        ``generator`` is retained by the current neutral interface.  The
        pairing work will replace implicit generator state with stable source,
        replicate, family, and stream identities before this class is
        registered.
        """
        raise NotImplementedError("native calibrated latent-proxy fit is pending")

    @staticmethod
    def _fit_basis(
        x_cal: Tensor,
        family: Family,
        config: CaseConfig,
        generator: torch.Generator,
    ) -> FittedBasis:
        """Fit linear or bounded nonlinear basis terms and calibration scales."""
        raise NotImplementedError

    @staticmethod
    def _fit_component(
        design_cal: Tensor,
        width: int,
        config: CaseConfig,
        generator: torch.Generator,
    ) -> FittedComponent:
        """Draw and calibrate one of f_A, f_Y, f_tau, f_W, or f_Z."""
        raise NotImplementedError

    @staticmethod
    def _orthogonal(width: int, generator: torch.Generator) -> Tensor:
        """Construct a full-rank rotation using Ayu's deterministic QR signs."""
        raise NotImplementedError

    def basis_design(self, x: Tensor) -> Tensor:
        """Evaluate the frozen, centered, and scaled X basis rowwise."""
        raise NotImplementedError

    def component(self, x: Tensor, name: str) -> Tensor:
        """Evaluate a fitted normalized X component without recalibration."""
        raise NotImplementedError

    def latent_transform(self, u: Tensor) -> Tensor:
        """Apply identity for linear or asinh for nonlinear mechanisms."""
        raise NotImplementedError

    def latent_coordinates(self, u: Tensor) -> Tensor:
        """Map innovation-augmented U through the frozen ridge whitener."""
        raise NotImplementedError

    def wrap_proxy(self, value: Tensor) -> Tensor:
        """Apply the family-specific observed-proxy transform."""
        raise NotImplementedError

    def unwrap_proxy(self, value: Tensor) -> Tensor:
        """Invert the family-specific observed-proxy transform."""
        raise NotImplementedError

    def innovations(
        self, n_rows: int, *, identity: tuple[object, ...]
    ) -> EvaluationInnovations:
        """Materialize named u/w/z/y/a streams for paired case generation."""
        raise NotImplementedError

    def cate(self, x: Tensor) -> Tensor:
        """Return the analytic conditional effect from the fitted tau component."""
        raise NotImplementedError

    def bridge(self, x: Tensor, w: Tensor, a: Tensor, *, gamma: float) -> Tensor:
        """Evaluate the analytic outcome bridge using the W rotation inverse."""
        raise NotImplementedError

    def generate(self, x: Tensor, u: Tensor) -> MechanismOutput:
        """Generate one case from fixed state and paired evaluation innovations."""
        raise NotImplementedError(
            "native calibrated latent-proxy generation is pending"
        )

    def metadata(self) -> dict[str, object]:
        """Serialize all fitted state required for an independent audit."""
        raise NotImplementedError
