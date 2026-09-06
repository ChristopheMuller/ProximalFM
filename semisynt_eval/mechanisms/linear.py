"""Normalized random-basis proximal SCMs."""

from __future__ import annotations

import math

import torch

from semisynt_eval._config import CaseConfig
from semisynt_eval.mechanisms.base import (
    Mechanism,
    RandomBasisRegression,
    combine_variance_share,
    normalize_columns,
)

Tensor = torch.Tensor


class BasisMechanism(Mechanism):
    """Family-selected additive/bilinear SCM with calibrated components."""

    family = "linear"

    @classmethod
    def fit(
        cls,
        x_cal: Tensor,
        u_cal: Tensor,
        config: CaseConfig,
        generator: torch.Generator,
    ) -> "BasisMechanism":
        """Construct the legacy mechanism behind the fitted lifecycle.

        Calibration-derived normalizers will be added here in a later step.
        For now, accepting the disjoint calibration tensors rewires callers
        without changing the random-basis equations.
        """
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
        config: CaseConfig,
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


class LinearMechanism(BasisMechanism):
    family = "linear"
