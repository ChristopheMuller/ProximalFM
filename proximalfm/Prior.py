"""Implementation of the proximal prior described in the spec."""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Optional

import pandas as pd
import torch
from torch import Generator, Tensor

from .Structure import NetworkSpec, ProximalEpisode, ProximalPriorConfig


class ProximalPrior:
    """Prior that samples datasets consistent with the proximal DAG."""

    def __init__(
        self,
        config: Optional[ProximalPriorConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or ProximalPriorConfig()
        self.seed = seed
        self._device = torch.device(self.config.device)
        self._dtype = self.config.dtype
        if min(self.config.d_z, self.config.d_w) < self.config.d_u:
            raise ValueError("Proxies Z and W must be at least as large as latent dimension d_u")

    def _make_generator(self, episode_seed: Optional[int]) -> Generator:
        gen = torch.Generator(device=self._device)
        if self.seed is not None or episode_seed is not None:
            # use a deterministic hash for cross-process reproducibility
            seed_data = f"{self.seed}-{episode_seed}".encode("utf-8")
            combined_hash = hashlib.sha256(seed_data).hexdigest()
            combined = int(combined_hash, 16) % (2**32)
            gen.manual_seed(combined)
        else:
            gen.seed()
        return gen

    def _noise(self, shape: List[int], scale: float, generator: Generator) -> Tensor:
        return torch.randn(*shape, device=self._device, generator=generator, dtype=self._dtype) * scale

    def _random_bnn(
        self,
        inputs: Tensor,
        output_dim: int,
        spec: NetworkSpec,
        generator: Generator,
    ) -> Tensor:
        current = inputs
        current_dim = inputs.shape[-1]
        for _ in range(spec.num_layers - 1):
            weight = torch.normal(
                mean=0.0,
                std=spec.weight_scale,
                size=(current_dim, spec.hidden_size),
                device=self._device,
                generator=generator,
                dtype=self._dtype,
            )
            bias = torch.normal(
                mean=0.0,
                std=spec.bias_scale,
                size=(spec.hidden_size,),
                device=self._device,
                generator=generator,
                dtype=self._dtype,
            )
            current = current @ weight + bias
            current = spec.activation(current)
            current_dim = spec.hidden_size

        # output layer
        weight = torch.normal(
            mean=0.0,
            std=spec.weight_scale,
            size=(current_dim, output_dim),
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
        return current @ weight + bias

    def generate_episode(self, batch_size: int, episode_seed: Optional[int] = None, normalize_episode: bool = True) -> ProximalEpisode:
        """Generate a single proximal episode respecting all assumptions."""

        generator = self._make_generator(episode_seed)
        c = self.config

        U = torch.randn(batch_size, c.d_u, device=self._device, generator=generator, dtype=self._dtype)

        X = self._random_bnn(U, c.d_x, c.x_spec, generator)
        X = X + self._noise([batch_size, c.d_x], c.noise_scale_x, generator)

        shared_context = torch.cat([U, X], dim=-1)
        Z = self._random_bnn(shared_context, c.d_z, c.z_spec, generator)
        Z = Z + self._noise([batch_size, c.d_z], c.noise_scale_z, generator)

        W = self._random_bnn(shared_context, c.d_w, c.w_spec, generator)
        W = W + self._noise([batch_size, c.d_w], c.noise_scale_w, generator)

        treatment_input = torch.cat([U, X, Z], dim=-1)
        logits = self._random_bnn(treatment_input, 1, c.treatment_spec, generator).squeeze(-1)
        logits = logits / max(c.temperature, 1e-6)
        propensity = torch.sigmoid(logits)
        propensity = torch.clamp(propensity, c.min_propensity, c.max_propensity)
        A = torch.bernoulli(propensity, generator=generator).to(self._dtype)

        context_for_outcome = torch.cat([W, U, X], dim=-1)
        base_input = context_for_outcome

        input0 = torch.cat([torch.zeros(batch_size, 1, device=self._device, dtype=self._dtype), base_input], dim=-1)
        input1 = torch.cat([torch.ones(batch_size, 1, device=self._device, dtype=self._dtype), base_input], dim=-1)
        
        combined_input = torch.cat([input0, input1], dim=0)
        combined_output = self._random_bnn(
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

        if normalize_episode:
            X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-8)
            Z = (Z - Z.mean(dim=0)) / (Z.std(dim=0) + 1e-8)
            W = (W - W.mean(dim=0)) / (W.std(dim=0) + 1e-8)

            Y_all = torch.stack([Y0, Y1])
            y_mean = Y_all.mean()
            y_std  = Y_all.std() + 1e-8
            Y = (Y - y_mean) / y_std
            Y0 = (Y0 - y_mean) / y_std
            Y1 = (Y1 - y_mean) / y_std

        return ProximalEpisode(X=X, Z=Z, W=W, A=A, Y=Y, Y0=Y0, Y1=Y1, U=U, propensity=propensity)

    def generate_dataset(
        self,
        num_episodes: int,
        batch_size: int,
        num_workers: int = 1,
        base_seed: int = 0,
        normalize_episode: bool = True,
    ) -> List[ProximalEpisode]:
        """Generate multiple episodes, optionally parallelized with threads."""

        seeds = [base_seed + i for i in range(num_episodes)]
        if num_workers <= 1:
            return [self.generate_episode(batch_size, episode_seed=seed, normalize_episode=normalize_episode) for seed in seeds]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            mapper = partial(self.generate_episode, batch_size, normalize_episode=normalize_episode)
            return list(executor.map(mapper, seeds))

    def episode_to_dataframe(self, episode: ProximalEpisode, include_latents: bool = False) -> pd.DataFrame:
        """Convert an episode to a pandas DataFrame for logging or inspection."""

        columns = {}
        columns.update({f"x_{i}": episode.X[:, i].cpu().numpy() for i in range(episode.X.shape[-1])})
        columns.update({f"z_{i}": episode.Z[:, i].cpu().numpy() for i in range(episode.Z.shape[-1])})
        columns.update({f"w_{i}": episode.W[:, i].cpu().numpy() for i in range(episode.W.shape[-1])})
        if include_latents:
            columns.update({f"u_{i}": episode.U[:, i].cpu().numpy() for i in range(episode.U.shape[-1])})
        columns["treatment"] = episode.A.cpu().numpy()
        columns["outcome"] = episode.Y.cpu().numpy()
        columns["y0"] = episode.Y0.cpu().numpy()
        columns["y1"] = episode.Y1.cpu().numpy()
        columns["propensity"] = episode.propensity.cpu().numpy()
        return pd.DataFrame(columns)


__all__ = ["ProximalPrior", "ProximalEpisode", "ProximalPriorConfig", "NetworkSpec"]
