"""Implementation of the proximal prior described in the spec."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Optional

import pandas as pd
import torch
from torch import Generator, Tensor

from ..Structure import NetworkSpec, ProximalEpisode, ProximalPriorConfig


class FungibleSmall:
    """
    A generative prior that samples datasets consistent with the proximal causal DAG.

    This prior simulates complex datasets featuring unobserved confounding, measured
    proxies for the confounder (Z and W), observed covariates (X), and treatment (A).
    It ensures that Assumption 4 (completeness) and other conditions of Proximal Causal
    Inference can hold for downstream tasks.

    The continuous dependencies between variables are modeled using simple linear
    relationships controlled by ProximalPriorConfig, replacing the previous BNN approach.
    """

    def __init__(
        self,
        config: Optional[ProximalPriorConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initializes the proximal prior with a specific configuration and base seed.

        Args:
            config: Configuration specifying dimensionalities and scales.
                Defaults to ProximalPriorConfig() if not provided.
            seed: A global base seed used to deterministically generate episode seeds.
        """
        self.config = config or ProximalPriorConfig()
        self.seed = seed
        self._device = torch.device(self.config.device)
        self._dtype = self.config.dtype
        if min(self.config.d_z, self.config.d_w) < self.config.d_u:
            raise ValueError(
                "Proxies Z and W must be at least as large as latent dimension d_u"
            )

    def _make_generator(self, episode_seed: Optional[int]) -> Generator:
        """
        Creates a random number generator seeded for the specific episode.
        """
        gen = torch.Generator(device=self._device)
        if self.seed is not None or episode_seed is not None:
            seed_data = f"{self.seed}-{episode_seed}".encode("utf-8")
            combined_hash = hashlib.sha256(seed_data).hexdigest()
            combined = int(combined_hash, 16) % (2**32)
            gen.manual_seed(combined)
        else:
            gen.seed()
        return gen

    def _noise(self, shape: List[int], scale: float, generator: Generator) -> Tensor:
        """
        Samples Gaussian noise with the specified shape and scale.
        """
        return (
            torch.randn(
                *shape, device=self._device, generator=generator, dtype=self._dtype
            )
            * scale
        )

    def _random_linear(
        self,
        inputs: Tensor,
        output_dim: int,
        spec: NetworkSpec,
        generator: Generator,
    ) -> Tensor:
        """
        Applies a random linear transformation (affine) to the given inputs.

        The transformation is instantiated with random weights and biases sampled from
        normal distributions with scales provided in the NetworkSpec.

        Args:
            inputs: The input tensor containing covariates or latent variables.
            output_dim: The dimension of the output tensor.
            spec: NetworkSpec configuration containing prior scales.
            generator: PyTorch random number generator for reproducibility.

        Returns:
            A tensor containing the linear transformation output.
        """
        input_dim = inputs.shape[-1]

        weight = torch.normal(
            mean=0.0,
            std=spec.weight_scale,
            size=(input_dim, output_dim),
            device=self._device,
            generator=generator,
            dtype=self._dtype,
        )
        bias = torch.normal(
            mean=0.0,
            std=spec.bias_scale,
            size=(output_dim,),
            device=self._device,
            generator=generator,
            dtype=self._dtype,
        )

        return inputs @ weight + bias

    @torch.no_grad()
    def generate_episode(
        self,
        batch_size: int,
        episode_seed: Optional[int] = None,
    ) -> ProximalEpisode:
        generator = self._make_generator(episode_seed)
        c = self.config

        X, U = self._generate_x_u(batch_size, generator)

        # --- CHANGE 1: generate Z and W from U *only* (not X), with low noise ---
        Z = self._random_linear(U, c.d_z, c.z_spec, generator)
        Z = Z + self._noise([batch_size, c.d_z], c.noise_scale_z, generator)

        W = self._random_linear(U, c.d_w, c.w_spec, generator)
        W = W + self._noise([batch_size, c.d_w], c.noise_scale_w, generator)

        # --- CHANGE 2: amplify U's role in treatment selection ---
        U_confounded = U
        treatment_input = torch.cat([U_confounded], dim=-1)
        logits = self._random_linear(
            treatment_input, 1, c.treatment_spec, generator
        ).squeeze(-1)
        logits = logits / max(c.temperature, 1e-6)
        propensity = torch.sigmoid(logits)
        propensity = torch.clamp(propensity, c.min_propensity, c.max_propensity)
        A = torch.bernoulli(propensity, generator=generator).to(self._dtype)

        # --- CHANGE 3: amplify U's role in outcome, keep W in outcome ---
        context_for_outcome = torch.cat([U_confounded], dim=-1)
        base_input = context_for_outcome

        input0 = torch.cat(
            [
                torch.zeros(batch_size, 1, device=self._device, dtype=self._dtype),
                base_input,
            ],
            dim=-1,
        )
        input1 = torch.cat(
            [
                torch.ones(batch_size, 1, device=self._device, dtype=self._dtype),
                base_input,
            ],
            dim=-1,
        )

        combined_input = torch.cat([input0, input1], dim=0)
        combined_output = self._random_linear(
            combined_input,
            1,
            c.outcome_spec,
            generator,
        ).squeeze(-1)

        Y0_raw, Y1_raw = combined_output.chunk(2, dim=0)

        shared_noise = self._noise([batch_size], c.noise_scale_y, generator)
        Y0 = Y0_raw + shared_noise
        Y1 = Y1_raw + shared_noise
        Y = torch.where(A == 0.0, Y0, Y1)

        return ProximalEpisode(
            X=X, Z=Z, W=W, A=A, Y=Y, Y0=Y0, Y1=Y1, U=U, propensity=propensity
        )

    def generate_dataset(
        self,
        num_episodes: int,
        batch_size: int,
        num_workers: int = 1,
        base_seed: int = 0,
    ) -> List[ProximalEpisode]:
        """
        Generates a collection of multiple proximal episodes.
        """
        seeds = [base_seed + i for i in range(num_episodes)]
        if num_workers <= 1:
            return [
                self.generate_episode(batch_size, episode_seed=seed) for seed in seeds
            ]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            mapper = partial(self.generate_episode, batch_size)
            return list(executor.map(mapper, seeds))

    def episode_to_dataframe(
        self, episode: ProximalEpisode, include_latents: bool = False
    ) -> pd.DataFrame:
        """
        Converts a generated ProximalEpisode into a pandas DataFrame.
        """
        columns = {}
        X_np = episode.X.cpu().numpy()
        columns.update({f"x_{i}": X_np[:, i] for i in range(X_np.shape[-1])})
        Z_np = episode.Z.cpu().numpy()
        columns.update({f"z_{i}": Z_np[:, i] for i in range(Z_np.shape[-1])})
        W_np = episode.W.cpu().numpy()
        columns.update({f"w_{i}": W_np[:, i] for i in range(W_np.shape[-1])})
        if include_latents:
            U_np = episode.U.cpu().numpy()
            columns.update({f"u_{i}": U_np[:, i] for i in range(U_np.shape[-1])})
        columns["treatment"] = episode.A.cpu().numpy()
        columns["outcome"] = episode.Y.cpu().numpy()
        columns["y0"] = episode.Y0.cpu().numpy()
        columns["y1"] = episode.Y1.cpu().numpy()
        columns["propensity"] = episode.propensity.cpu().numpy()
        return pd.DataFrame(columns)

    def _sample_regime(self, generator: Generator) -> str:
        """
        Samples a causal dependency regime for the episode.
        """
        regimes = ["u_to_x", "x_to_u", "shared"]
        idx = torch.randint(
            0, len(regimes), (1,), generator=generator, device=self._device
        ).item()
        return regimes[idx]

    def _generate_x_u(
        self, batch_size: int, generator: Generator
    ) -> tuple[Tensor, Tensor]:
        """
        Samples the observed covariates (X) and unobserved confounders (U).
        """
        c = self.config
        regime = self._sample_regime(generator)

        if regime == "u_to_x":
            U = torch.randn(
                batch_size,
                c.d_u,
                device=self._device,
                generator=generator,
                dtype=self._dtype,
            )
            X = self._random_linear(U, c.d_x, c.x_spec, generator)
            X = X + self._noise([batch_size, c.d_x], c.noise_scale_x, generator)
            return X, U

        if regime == "x_to_u":
            X_base = torch.randn(
                batch_size,
                c.d_x,
                device=self._device,
                generator=generator,
                dtype=self._dtype,
            )
            X = self._random_linear(X_base, c.d_x, c.x_spec, generator)
            X = X + self._noise([batch_size, c.d_x], c.noise_scale_x, generator)

            U = self._random_linear(X, c.d_u, c.u_spec, generator)
            U = U + self._noise([batch_size, c.d_u], c.noise_scale_x, generator)
            return X, U

        d_shared = max(c.d_x, c.d_u)
        S = torch.randn(
            batch_size,
            d_shared,
            device=self._device,
            generator=generator,
            dtype=self._dtype,
        )

        X = self._random_linear(S, c.d_x, c.x_spec, generator)
        X = X + self._noise([batch_size, c.d_x], c.noise_scale_x, generator)

        U = self._random_linear(S, c.d_u, c.u_spec, generator)
        U = U + self._noise([batch_size, c.d_u], c.noise_scale_x, generator)
        return X, U


__all__ = [
    "FungibleSmall",
    "ProximalEpisode",
    "ProximalPriorConfig",
    "NetworkSpec",
]
