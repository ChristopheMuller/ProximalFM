"""Closed-form random linear proximal SCM."""

from __future__ import annotations

import torch

from semi_synt_eval._config import MechanismConfig
from semi_synt_eval.mechanisms.base import (
    Mechanism,
    RandomPolynomialRegression,
    normalize_columns,
)

Tensor = torch.Tensor


def _selected_blocks(
    *,
    x: Tensor,
    u: Tensor | None = None,
    proxy: Tensor | None = None,
    use_x: bool,
    use_u: bool,
    use_proxy: bool,
) -> Tensor:
    blocks = []
    if use_x:
        blocks.append(x)
    if use_u:
        if u is None:
            raise ValueError("configured latent input was not supplied")
        blocks.append(u)
    if use_proxy:
        if proxy is None:
            raise ValueError("configured proxy input was not supplied")
        blocks.append(proxy)
    if not blocks:
        raise ValueError("each mechanism equation needs at least one input block")
    return torch.cat(blocks, dim=1)


class LinearMechanism(Mechanism):
    """Linear equations with random coefficients and episode normalization."""

    def __init__(
        self,
        d_x: int,
        d_u: int,
        config: MechanismConfig,
        generator: torch.Generator,
    ) -> None:
        if d_x < 1 or d_u < 1:
            raise ValueError("d_x and d_u must both be positive")
        self.config = config
        self.generator = generator
        self.d_w = d_u
        self.d_z = d_u
        self._validate_config()

        proxy_p, treatment_p, outcome_p, effect_p = self._power_probabilities()
        proxy_inputs = int(config.proxies.use_x) * d_x + int(config.proxies.use_u) * d_u
        treatment_inputs = (
            int(config.treatment.use_x) * d_x
            + int(config.treatment.use_u) * d_u
            + int(config.treatment.use_z) * self.d_z
        )
        outcome_inputs = (
            int(config.outcome.use_x) * d_x
            + int(config.outcome.use_u) * d_u
            + int(config.outcome.use_w) * self.d_w
        )
        self.w_regression = RandomPolynomialRegression.sample(
            generator,
            proxy_inputs,
            self.d_w,
            config.proxies.coefficients,
            quadratic_probability=proxy_p,
        )
        self.z_regression = RandomPolynomialRegression.sample(
            generator,
            proxy_inputs,
            self.d_z,
            config.proxies.coefficients,
            quadratic_probability=proxy_p,
        )
        self.treatment_regression = RandomPolynomialRegression.sample(
            generator,
            treatment_inputs,
            1,
            config.treatment.coefficients,
            quadratic_probability=treatment_p,
        )
        self.outcome_regression = RandomPolynomialRegression.sample(
            generator,
            outcome_inputs,
            1,
            config.outcome.coefficients,
            quadratic_probability=outcome_p,
        )
        self.effect_regression = (
            RandomPolynomialRegression.sample(
                generator,
                d_x,
                1,
                config.outcome.effect.coefficients,
                quadratic_probability=effect_p,
            )
            if config.outcome.effect.use_x
            else None
        )
        treatment = config.treatment
        self.treatment_intercept = float(
            torch.empty(1).uniform_(
                treatment.intercept_low,
                treatment.intercept_high,
                generator=generator,
            )
        )

    def _power_probabilities(self) -> tuple[float, float, float, float]:
        """The linear family assigns power one to every input term."""
        return 0.0, 0.0, 0.0, 0.0

    def _validate_config(self) -> None:
        config = self.config
        if not (config.proxies.use_x or config.proxies.use_u):
            raise ValueError("proxy equations need X and/or U")
        if not (
            config.treatment.use_x or config.treatment.use_u or config.treatment.use_z
        ):
            raise ValueError("treatment equation needs X, U, and/or Z")
        if not (config.outcome.use_x or config.outcome.use_u or config.outcome.use_w):
            raise ValueError("outcome equation needs X, U, and/or W")
        if config.proxies.noise_std < 0 or config.outcome.noise_std < 0:
            raise ValueError("noise standard deviations must be non-negative")
        if config.treatment.logit_scale < 0:
            raise ValueError("treatment logit scale must be non-negative")
        if config.treatment.logit_clip <= 0:
            raise ValueError("treatment logit clip must be positive")
        if config.treatment.intercept_low > config.treatment.intercept_high:
            raise ValueError("treatment intercept bounds are reversed")
        if not 0.0 <= config.treatment.overlap_epsilon < 0.5:
            raise ValueError("overlap_epsilon must lie in [0, 0.5)")

    def _proxy_inputs(self, x: Tensor, u: Tensor) -> Tensor:
        return _selected_blocks(
            x=x,
            u=u,
            use_x=self.config.proxies.use_x,
            use_u=self.config.proxies.use_u,
            use_proxy=False,
        )

    def proxies(self, x: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        inputs = self._proxy_inputs(x, u)
        noise_std = float(self.config.proxies.noise_std)
        w = self.w_regression(inputs) + noise_std * torch.randn(
            x.shape[0], self.d_w, generator=self.generator
        )
        z = self.z_regression(inputs) + noise_std * torch.randn(
            x.shape[0], self.d_z, generator=self.generator
        )
        if self.config.proxies.normalize_output:
            w, z = normalize_columns(w), normalize_columns(z)
        return w, z

    def treatment_probability(self, x: Tensor, u: Tensor, z: Tensor) -> Tensor:
        config = self.config.treatment
        inputs = _selected_blocks(
            x=x,
            u=u,
            proxy=z,
            use_x=config.use_x,
            use_u=config.use_u,
            use_proxy=config.use_z,
        )
        score = self.treatment_regression(inputs)
        if config.normalize_score:
            score = normalize_columns(score)
        logit = float(config.logit_scale) * score + self.treatment_intercept
        clip = float(config.logit_clip)
        return torch.sigmoid(logit.clamp(-clip, clip))

    def cate(self, x: Tensor) -> Tensor:
        effect = self.config.outcome.effect
        if self.effect_regression is None:
            return torch.full(
                (x.shape[0], 1),
                float(effect.intercept),
                dtype=x.dtype,
                device=x.device,
            )
        return float(effect.intercept) + float(effect.scale) * self.effect_regression(x)

    def potential_outcomes(
        self, x: Tensor, u: Tensor, w: Tensor
    ) -> tuple[Tensor, Tensor]:
        config = self.config.outcome
        inputs = _selected_blocks(
            x=x,
            u=u,
            proxy=w,
            use_x=config.use_x,
            use_u=config.use_u,
            use_proxy=config.use_w,
        )
        baseline = self.outcome_regression(inputs) + float(
            config.noise_std
        ) * torch.randn(x.shape[0], 1, generator=self.generator)
        if config.normalize_baseline:
            baseline = normalize_columns(baseline)
        return baseline, baseline + self.cate(x)
