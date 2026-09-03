"""Interfaces that enforce the proximal DAG by construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import torch

Tensor = torch.Tensor
TargetCapability = Literal["exact_cate", "ite_only"]


@dataclass(frozen=True)
class MechanismConfig:
    proxy_noise: float = 0.75
    treatment_scale: float = 1.0
    outcome_noise: float = 0.5
    effect_scale: float = 1.0


class Mechanism(ABC):
    """One episode's sampled SCM.

    Separate method signatures are the exclusion restrictions: treatment never
    receives W and potential outcomes never receive Z.
    """

    target_capability: TargetCapability = "exact_cate"

    @abstractmethod
    def proxies(self, x: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        """Generate the outcome and treatment proxy blocks."""

    @abstractmethod
    def treatment_probability(self, x: Tensor, u: Tensor, z: Tensor) -> Tensor:
        """Return P(A=1|X,U,Z), before the dataset overlap clamp."""

    @abstractmethod
    def potential_outcomes(
        self, x: Tensor, u: Tensor, w: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return paired potential outcomes, using shared structural noise."""

    def cate(self, x: Tensor) -> Optional[Tensor]:
        """Return exact CATE, or ``None`` for an ITE-only mechanism."""
        del x
        return None
