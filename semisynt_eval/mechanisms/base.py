"""Mechanism-neutral fit/generate contracts for semi-synthetic SCMs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class MechanismOutput:
    """Synthetic variables produced rowwise by one fitted mechanism."""

    W: Tensor
    Z: Tensor
    A: Tensor
    A_probs: Tensor
    Y_obs: Tensor
    Y0: Tensor
    Y1: Tensor
    CATE: Optional[Tensor]


class Mechanism(ABC):
    """A mechanism fitted on calibration rows and applied to evaluation rows."""

    target_capability = "exact_cate"

    @classmethod
    @abstractmethod
    def fit(
        cls,
        x_cal: Tensor,
        u_cal: Tensor,
        config: Any,
        generator: torch.Generator,
    ) -> "Mechanism":
        """Fit all data-dependent state using calibration rows only."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, x: Tensor, u: Tensor) -> MechanismOutput:
        """Apply fixed mechanism state to evaluation rows without refitting."""
        raise NotImplementedError
