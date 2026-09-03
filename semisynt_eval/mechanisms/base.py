"""Interfaces and closed-form building blocks for semi-synthetic mechanisms."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Optional

import torch

from semisynt_eval._config import CoefficientConfig

Tensor = torch.Tensor


class Mechanism(ABC):
    """Causal interface shared by all empirical-root SCMs.

    The method signatures encode the proximal DAG: W/Z cannot depend on A,
    treatment cannot depend on W, and outcomes cannot depend on Z.
    """

    target_capability = "exact_cate"

    @abstractmethod
    def proxies(self, x: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def treatment_probability(self, x: Tensor, u: Tensor, z: Tensor) -> Tensor:
        pass

    @abstractmethod
    def potential_outcomes(
        self, x: Tensor, u: Tensor, w: Tensor
    ) -> tuple[Tensor, Tensor]:
        pass

    def cate(self, x: Tensor) -> Optional[Tensor]:
        """Return the analytic CATE, or ``None`` for an ITE-only mechanism."""
        return None


def normalize_columns(value: Tensor, eps: float = 1e-6) -> Tensor:
    """Centre and rescale each generated column using its episode moments."""
    mean = value.mean(dim=0, keepdim=True)
    std = value.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (value - mean) / std


def sample_coefficients(
    generator: torch.Generator,
    n_inputs: int,
    n_outputs: int,
    config: CoefficientConfig,
) -> Tensor:
    """Draw one coefficient matrix from a fully recorded distribution."""
    if n_inputs < 1 or n_outputs < 1:
        raise ValueError("regressions require at least one input and output")
    if config.scale < 0:
        raise ValueError("coefficient scale must be non-negative")
    shape = (n_inputs, n_outputs)
    if config.distribution == "normal":
        coefficients = torch.randn(*shape, generator=generator)
    elif config.distribution == "uniform":
        coefficients = 2.0 * torch.rand(*shape, generator=generator) - 1.0
    else:
        raise ValueError(
            f"unknown coefficient distribution {config.distribution!r}; "
            "available: normal, uniform"
        )
    scale = float(config.scale)
    if config.normalize_by_input_dim:
        scale /= math.sqrt(n_inputs)
    return coefficients * scale


class RandomPolynomialRegression:
    """A closed-form sum of independently powered input terms.

    For output ``j`` the equation is
    ``f_j(v) = sum_i coefficient[i,j] * v_i ** power[i,j]``, where every
    power is fixed to one or two when the mechanism is instantiated.
    There are deliberately no hidden activations or interaction terms.
    """

    def __init__(self, coefficients: Tensor, powers: Tensor) -> None:
        if coefficients.shape != powers.shape:
            raise ValueError("coefficients and powers must have identical shapes")
        self.coefficients = coefficients
        self.powers = powers

    @classmethod
    def sample(
        cls,
        generator: torch.Generator,
        n_inputs: int,
        n_outputs: int,
        coefficients: CoefficientConfig,
        *,
        quadratic_probability: float,
    ) -> "RandomPolynomialRegression":
        if not 0.0 <= quadratic_probability <= 1.0:
            raise ValueError("quadratic probability must lie in [0, 1]")
        weights = sample_coefficients(generator, n_inputs, n_outputs, coefficients)
        quadratic = torch.rand(*weights.shape, generator=generator) < float(
            quadratic_probability
        )
        powers = torch.where(quadratic, 2, 1)
        return cls(weights, powers)

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.coefficients.shape[0]:
            raise ValueError(
                f"expected (*, {self.coefficients.shape[0]}) inputs, "
                f"got {tuple(inputs.shape)}"
            )
        terms = inputs.unsqueeze(-1).pow(self.powers.unsqueeze(0))
        return (terms * self.coefficients.unsqueeze(0)).sum(dim=1)
