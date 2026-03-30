from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader, Dataset

from .Prior import ProximalPrior, ProximalPriorConfig


class ProximalDataset(Dataset):
    """
    A PyTorch Dataset that generates proximal causal episodes on-the-fly.

    Each item is one full episode (a synthetic observational dataset) with
    N = samples_per_episode rows. The variable groups X, Z, W are returned
    as separate tensors so that the transformer can apply distinct type
    embeddings to each.

    Args:
        prior_config: Configuration for the ProximalPrior. A deep copy is
            made internally so the caller's config is never mutated.
        num_episodes: Number of episodes that constitute one epoch.
        samples_per_episode: Number of rows (observations) per episode.
        seed_offset: Added to the episode index to form the episode seed,
            allowing non-overlapping train/val splits.
    """

    def __init__(
        self,
        prior_config: ProximalPriorConfig,
        num_episodes: int,
        samples_per_episode: int,
        seed_offset: int = 0,
    ) -> None:
        self.prior_config = copy.deepcopy(prior_config)
        self.prior_config.device = "cpu"

        # Validate dimensional completeness requirement up front
        if self.prior_config.d_z < self.prior_config.d_u:
            raise ValueError(
                f"d_z={self.prior_config.d_z} must be >= d_u={self.prior_config.d_u} "
                "for the completeness condition (Assumption 4) to hold."
            )
        if self.prior_config.d_w < self.prior_config.d_u:
            raise ValueError(
                f"d_w={self.prior_config.d_w} must be >= d_u={self.prior_config.d_u} "
                "for the completeness condition (Assumption 4) to hold."
            )

        self.prior = ProximalPrior(config=self.prior_config)
        self.num_episodes = num_episodes
        self.samples_per_episode = samples_per_episode
        self.seed_offset = seed_offset

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode_seed = self.seed_offset + idx

        episode = self.prior.generate_episode(
            batch_size=self.samples_per_episode,
            episode_seed=episode_seed,
        )

        ITE = episode.Y1 - episode.Y0  # (N,)

        return {
            "X": episode.X,  # (N, d_x)
            "Z": episode.Z,  # (N, d_z)
            "W": episode.W,  # (N, d_w)
            "A": episode.A.unsqueeze(-1),  # (N, 1)
            "Y": episode.Y.unsqueeze(-1),  # (N, 1)
            "Y0": episode.Y0.unsqueeze(-1),  # (N, 1)
            "Y1": episode.Y1.unsqueeze(-1),  # (N, 1)
            "ITE": ITE.unsqueeze(-1),  # (N, 1)  — training target
        }


def custom_collate(batch):
    """
    Custom collate function that stacks episodes along dimension 1 instead of 0.
    Returns tensors of shape (N, B, D) where N is samples_per_episode and B is batch_size.
    """
    elem = batch[0]
    return {
        key: torch.stack([d[key] for d in batch], dim=1)
        for key in elem.keys()
    }


def create_causal_data_loaders(
    prior_config: ProximalPriorConfig,
    batch_size: int = 16,
    samples_per_episode: int = 1000,
    train_episodes: int = 10_000,
    val_episodes: int = 1_000,
    num_workers: int = 4,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """
    Create train and validation DataLoaders for ProximalFM.

    Train and validation seeds are non-overlapping by construction:
        train episodes use seeds  [seed,              seed + train_episodes)
        val   episodes use seeds  [seed + train_episodes, seed + train_episodes + val_episodes)

    Args:
        prior_config: Config forwarded to ProximalDataset (deep-copied internally).
        batch_size: Number of episodes per batch fed to the transformer.
        samples_per_episode: Rows per episode (N in the paper).
        train_episodes: Episodes per training epoch.
        val_episodes: Episodes for validation.
        num_workers: DataLoader worker processes. Set to 0 for debugging.
        seed: Base seed controlling both episode generation and shuffle order.

    Returns:
        Dict with keys "train" and "val", each a DataLoader.
    """
    train_dataset = ProximalDataset(
        prior_config=prior_config,
        num_episodes=train_episodes,
        samples_per_episode=samples_per_episode,
        seed_offset=seed,
    )

    # Shift seed offset so validation episodes are entirely disjoint from training
    val_dataset = ProximalDataset(
        prior_config=prior_config,
        num_episodes=val_episodes,
        samples_per_episode=samples_per_episode,
        seed_offset=seed + train_episodes,
    )

    # Reproducible shuffle: pin the DataLoader's internal RNG to `seed`
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    # pin_memory accelerates CPU→GPU transfer via page-locked memory;
    # only beneficial when a CUDA device is actually available
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        generator=train_generator,
        # Each worker needs a unique seed to avoid generating identical episodes
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
        collate_fn=custom_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
        collate_fn=custom_collate,
    )

    return {"train": train_loader, "val": val_loader}


def _worker_init_fn(worker_id: int) -> None:
    """
    Give each DataLoader worker a distinct random seed.

    Without this, all workers share the same RNG state at fork time and
    would generate identical episodes when num_workers > 1.
    """
    worker_seed = torch.initial_seed() % (2**32)
    torch.manual_seed(worker_seed + worker_id)
