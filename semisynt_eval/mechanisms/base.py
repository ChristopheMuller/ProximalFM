"""Causal interfaces and normalized random-basis regressions."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Optional, Sequence

import torch

from semisynt_eval._config import CoefficientConfig, NonlinearConfig, TRANSFORMS

Tensor = torch.Tensor


class Mechanism(ABC):
    """Causal interface whose signatures enforce the proximal exclusions."""

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
        return None


def normalize_columns(value: Tensor, eps: float = 1e-6) -> Tensor:
    # Pre-scale before taking moments so finite but very large transformed
    # values cannot overflow the variance calculation. The scale cancels in
    # the standardized result.
    scale = value.abs().amax(dim=0, keepdim=True).clamp_min(1.0)
    scaled = value / scale
    mean = scaled.mean(dim=0, keepdim=True)
    std = scaled.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (scaled - mean) / std


def sample_coefficients(
    generator: torch.Generator,
    n_inputs: int,
    n_outputs: int,
    config: CoefficientConfig,
) -> Tensor:
    if n_inputs < 1 or n_outputs < 1:
        raise ValueError("regressions require at least one input and output")
    shape = (n_inputs, n_outputs)
    if config.distribution == "normal":
        result = torch.randn(*shape, generator=generator)
    elif config.distribution == "uniform":
        result = 2 * torch.rand(*shape, generator=generator) - 1
    else:  # pragma: no cover - schema validation owns this branch
        raise ValueError(f"unknown coefficient distribution {config.distribution!r}")
    scale = float(config.scale)
    if config.normalize_by_input_dim:
        scale /= math.sqrt(n_inputs)
    return result * scale


def apply_transform(value: Tensor, transform: str, *, exp_clip: float) -> Tensor:
    """Apply one domain-safe unary basis function without normalization."""
    if transform == "identity":
        return value
    if transform == "quadratic":
        # Squaring after a common positive rescaling is equivalent once the
        # basis column is standardized, and avoids overflow on extreme input.
        scale = value.abs().max().clamp_min(1.0)
        return (value / scale).square()
    if transform == "signed_log1p":
        return value.sign() * torch.log1p(value.abs())
    if transform == "clipped_exp":
        return torch.exp(value.clamp(-exp_clip, exp_clip))
    if transform == "sine":
        return torch.sin(value)
    raise ValueError(f"unknown transform {transform!r}")


class RandomBasisRegression:
    """Random normalized unary terms and optional pairwise interactions.

    Inputs are named blocks. Contributions from terms involving ``U`` are
    returned separately from terms using observed parents only, which lets the
    caller calibrate direct latent strength after sampling a function.
    """

    def __init__(
        self,
        *,
        block_names: Sequence[str],
        block_widths: Sequence[int],
        n_outputs: int,
        family: str,
        interactions: bool,
        coefficients: CoefficientConfig,
        nonlinear: NonlinearConfig,
        generator: torch.Generator,
    ) -> None:
        if len(block_names) != len(block_widths) or not block_names:
            raise ValueError(
                "regression blocks must have matching non-empty names and widths"
            )
        self.block_names = tuple(block_names)
        self.block_widths = tuple(int(width) for width in block_widths)
        self.n_inputs = sum(self.block_widths)
        self.n_outputs = int(n_outputs)
        self.family = family
        self.exp_clip = nonlinear.exp_clip
        input_blocks = [
            block
            for block, width in zip(self.block_names, self.block_widths)
            for _ in range(width)
        ]
        pairs = (
            [(i, j) for i in range(self.n_inputs) for j in range(i + 1, self.n_inputs)]
            if interactions
            else []
        )
        self.pairs = tuple(pairs)
        self.term_is_latent = torch.tensor(
            [block == "u" for block in input_blocks]
            + [input_blocks[i] == "u" or input_blocks[j] == "u" for i, j in pairs],
            dtype=torch.bool,
        )
        n_terms = self.n_inputs + len(self.pairs)
        self.coefficients = sample_coefficients(
            generator, n_terms, self.n_outputs, coefficients
        )
        self.transforms = self._sample_transforms(
            generator, (n_terms, self.n_outputs), nonlinear
        )
        self.interaction_transforms = self._sample_transforms(
            generator, (len(self.pairs), self.n_outputs), nonlinear
        )

    def _sample_transforms(
        self,
        generator: torch.Generator,
        shape: tuple[int, int],
        nonlinear: NonlinearConfig,
    ) -> Tensor:
        if shape[0] == 0:
            return torch.empty(shape, dtype=torch.long)
        if self.family == "linear":
            return torch.zeros(shape, dtype=torch.long)
        probabilities = torch.tensor(nonlinear.probabilities, dtype=torch.float)
        return torch.multinomial(
            probabilities,
            shape[0] * shape[1],
            replacement=True,
            generator=generator,
        ).reshape(shape)

    def _terms(self, value: Tensor, indices: Tensor) -> Tensor:
        outputs = []
        for output in range(self.n_outputs):
            transformed = torch.stack(
                [
                    apply_transform(
                        value[:, term],
                        TRANSFORMS[int(indices[term, output])],
                        exp_clip=self.exp_clip,
                    )
                    for term in range(value.shape[1])
                ],
                dim=1,
            )
            outputs.append(normalize_columns(transformed))
        return torch.stack(outputs, dim=-1)

    def contributions(
        self, **blocks: Tensor
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        if tuple(blocks) != self.block_names:
            raise ValueError(f"expected blocks {self.block_names}, got {tuple(blocks)}")
        inputs = torch.cat([blocks[name] for name in self.block_names], dim=1)
        if inputs.ndim != 2 or inputs.shape[1] != self.n_inputs:
            raise ValueError(
                f"expected (*, {self.n_inputs}) inputs, got {tuple(inputs.shape)}"
            )
        main = self._terms(inputs, self.transforms[: self.n_inputs])
        terms = [main]
        if self.pairs:
            left = inputs[:, [i for i, _ in self.pairs]]
            right = inputs[:, [j for _, j in self.pairs]]
            left_terms = self._terms(left, self.transforms[self.n_inputs :])
            right_terms = self._terms(right, self.interaction_transforms)
            interaction = left_terms * right_terms
            interaction = normalize_columns(interaction.flatten(1)).reshape_as(
                interaction
            )
            terms.append(interaction)
        design = torch.cat(terms, dim=1)
        weighted = design * self.coefficients.unsqueeze(0)

        def reduce(mask: Tensor) -> Optional[Tensor]:
            if not bool(mask.any()):
                return None
            return normalize_columns(weighted[:, mask].sum(dim=1))

        return reduce(self.term_is_latent), reduce(~self.term_is_latent)


def combine_variance_share(
    latent: Optional[Tensor], observed: Optional[Tensor], share: float
) -> Tensor:
    """Combine unit-variance components with requested component variances."""
    if latent is None:
        if observed is None or share != 0:
            raise ValueError("latent-free regression requires U share 0")
        return observed
    if observed is None:
        if share != 1:
            raise ValueError("latent-only regression requires U share 1")
        return latent
    return math.sqrt(share) * latent + math.sqrt(1 - share) * observed
