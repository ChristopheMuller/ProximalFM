"""Retired random-basis SCM, quarantined pending deletion.

This module preserves the retired implementation for comparison while
the calibrated latent proxy mechanism is ported. It is not selectable by the
current recipe. Nothing here belongs to the shared mechanism contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Optional, Sequence

import torch

from semisynt_eval.mechanisms.base import Mechanism, MechanismOutput

Tensor = torch.Tensor
CoefficientDistribution = Literal["normal", "uniform"]
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


def normalize_columns(value: Tensor, eps: float = 1e-6) -> Tensor:
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
    else:
        raise ValueError(f"unknown coefficient distribution {config.distribution!r}")
    scale = float(config.scale)
    if config.normalize_by_input_dim:
        scale /= math.sqrt(n_inputs)
    return result * scale


def apply_transform(value: Tensor, transform: str, *, exp_clip: float) -> Tensor:
    if transform == "identity":
        return value
    if transform == "quadratic":
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
    """Random normalized unary terms and optional pairwise interactions."""

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
    if latent is None:
        if observed is None or share != 0:
            raise ValueError("latent-free regression requires U share 0")
        return observed
    if observed is None:
        if share != 1:
            raise ValueError("latent-only regression requires U share 1")
        return latent
    return math.sqrt(share) * latent + math.sqrt(1 - share) * observed


class BasisMechanism(Mechanism):
    """Legacy family-selected additive/bilinear SCM."""

    family = "linear"

    @classmethod
    def fit(
        cls,
        x_cal: Tensor,
        u_cal: Tensor,
        config: Any,
        generator: torch.Generator,
    ) -> "BasisMechanism":
        if x_cal.ndim != 2 or u_cal.ndim != 2:
            raise ValueError("calibration X and U must both be matrices")
        if x_cal.shape[0] != u_cal.shape[0] or x_cal.shape[0] < 1:
            raise ValueError("calibration X and U must have matching non-empty rows")
        if not torch.isfinite(x_cal).all() or not torch.isfinite(u_cal).all():
            raise ValueError("calibration X and U must be finite")
        return cls(x_cal.shape[1], u_cal.shape[1], config, generator)

    def __init__(
        self,
        d_x: int,
        d_u: int,
        config: Any,
        generator: torch.Generator,
    ) -> None:
        if d_x < 1 or d_u < 1:
            raise ValueError("d_x and d_u must both be positive")
        self.config = config
        self.generator = generator
        self.d_x, self.d_u = d_x, d_u
        self.d_w = self.d_z = d_u
        self.w_regression = self._regression(config.w, {"x": d_x, "u": d_u}, d_u)
        self.z_regression = self._regression(config.z, {"x": d_x, "u": d_u}, d_u)
        self.treatment_regression = self._regression(
            config.a, {"x": d_x, "u": d_u, "z": self.d_z}, 1
        )
        self.outcome_regression = self._regression(
            config.y, {"x": d_x, "u": d_u, "w": self.d_w}, 1
        )
        self.effect_regression = (
            self._regression(config.tau, {"x": d_x}, 1) if config.tau.use_x else None
        )
        self.treatment_intercept = float(
            torch.empty(1).uniform_(
                config.a.intercept_low,
                config.a.intercept_high,
                generator=generator,
            )
        )

    def _regression(self, equation, possible: dict[str, int], n_outputs: int):
        active = [
            (block, width)
            for block, width in possible.items()
            if getattr(equation, "use_" + block)
        ]
        return RandomBasisRegression(
            block_names=[name for name, _ in active],
            block_widths=[width for _, width in active],
            n_outputs=n_outputs,
            family=self.family,
            interactions=equation.interactions,
            coefficients=self.config.coefficients,
            nonlinear=self.config.nonlinear,
            generator=self.generator,
        )

    @staticmethod
    def _active(equation, **blocks: Tensor) -> dict[str, Tensor]:
        return {
            name: value
            for name, value in blocks.items()
            if getattr(equation, "use_" + name)
        }

    def _proxy(
        self,
        regression: RandomBasisRegression,
        equation,
        reliability: float,
        x: Tensor,
        u: Tensor,
    ) -> Tensor:
        latent, observed = regression.contributions(**self._active(equation, x=x, u=u))
        if latent is None:
            signal = observed
            noise_scale = 1.0
        else:
            signal = latent
            if observed is not None:
                signal = signal + float(equation.x_strength) * observed
            noise_scale = math.sqrt((1.0 - reliability) / reliability)
        value = signal + noise_scale * torch.randn(
            x.shape[0], regression.n_outputs, generator=self.generator
        )
        return normalize_columns(value) if equation.normalize_output else value

    def proxies(self, x: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self._proxy(
                self.w_regression,
                self.config.w,
                self.config.w_u_reliability,
                x,
                u,
            ),
            self._proxy(
                self.z_regression,
                self.config.z,
                self.config.z_u_reliability,
                x,
                u,
            ),
        )

    def treatment_probability(self, x: Tensor, u: Tensor, z: Tensor) -> Tensor:
        config = self.config.a
        latent, observed = self.treatment_regression.contributions(
            **self._active(config, x=x, u=u, z=z)
        )
        score = combine_variance_share(latent, observed, self.config.a_u_variance_share)
        if config.normalize_score:
            score = normalize_columns(score)
        logits = float(config.logit_scale) * score + self.treatment_intercept
        return torch.sigmoid(
            logits.clamp(-float(config.logit_clip), float(config.logit_clip))
        )

    def cate(self, x: Tensor) -> Tensor:
        config = self.config.tau
        if self.effect_regression is None:
            signal = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        else:
            _, signal = self.effect_regression.contributions(x=x)
        return float(config.intercept) + float(config.scale) * signal

    def potential_outcomes(
        self, x: Tensor, u: Tensor, w: Tensor
    ) -> tuple[Tensor, Tensor]:
        config = self.config.y
        latent, observed = self.outcome_regression.contributions(
            **self._active(config, x=x, u=u, w=w)
        )
        baseline = combine_variance_share(
            latent, observed, self.config.y_u_variance_share
        )
        baseline = baseline + float(config.noise_std) * torch.randn(
            x.shape[0], 1, generator=self.generator
        )
        if config.normalize_baseline:
            baseline = normalize_columns(baseline)
        return baseline, baseline + self.cate(x)

    def generate(self, x: Tensor, u: Tensor) -> MechanismOutput:
        w, z = self.proxies(x, u)
        propensity = self.treatment_probability(x, u, z).clamp(
            float(self.config.a.overlap_epsilon),
            1.0 - float(self.config.a.overlap_epsilon),
        )
        a = (torch.rand(x.shape[0], 1, generator=self.generator) < propensity).float()
        y0, y1 = self.potential_outcomes(x, u, w)
        cate = self.cate(x)
        return MechanismOutput(
            W=w,
            Z=z,
            A=a,
            A_probs=propensity,
            Y_obs=a * y1 + (1.0 - a) * y0,
            Y0=y0,
            Y1=y1,
            CATE=cate,
        )


class LinearMechanism(BasisMechanism):
    family = "linear"


class NonlinearMechanism(BasisMechanism):
    family = "nonlinear"
