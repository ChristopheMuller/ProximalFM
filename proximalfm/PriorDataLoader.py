from __future__ import annotations


import torch
from torch.utils.data import DataLoader, Dataset

from .Structure import ProximalPriorConfig
from .prior import (
    BNNPrior,
    BNNPriorComplete,
    ConfoundingLinearPrior,
    LinearPrior,
    CATELinearPrior,
    FungibleSmall,
)


class ProximalDataset(Dataset):
    """
    A lightweight Dataset that only computes and returns the deterministic
    seed for each episode. The actual generation happens in the collator.
    """

    def __init__(
        self,
        num_episodes: int,
        seed_offset: int = 0,
    ) -> None:
        self.num_episodes = num_episodes
        self.base_seed_offset = seed_offset
        self._shared_reset_count = torch.tensor([0], dtype=torch.long)
        self._shared_reset_count.share_memory_()

    def set_reset_state(self, reset_count: int) -> None:
        self._shared_reset_count[0] = reset_count

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> int:
        # Just return the mathematical seed for this specific episode
        reset_count = self._shared_reset_count.item()
        return self.base_seed_offset + (reset_count * self.num_episodes) + idx


class DynamicBatchCollator:
    """
    Collator that takes a batch of episode seeds, samples a single uniform
    episode size (N) for the whole batch, and generates the tensors.
    """

    def __init__(
        self,
        prior_config: ProximalPriorConfig,
        min_episode_size: int,
        max_episode_size: int,
    ):
        self.prior_config = prior_config
        self.prior_config.device = "cpu"
        self.min_episode_size = min_episode_size
        self.max_episode_size = max_episode_size

        # Instantiate the prior once per worker
        if self.prior_config.type == "linear":
            self.prior = LinearPrior(config=self.prior_config)
        elif self.prior_config.type == "linear_confounding":
            self.prior = ConfoundingLinearPrior(config=self.prior_config)
        elif self.prior_config.type == "cate_linear":
            self.prior = CATELinearPrior(config=self.prior_config)
        elif self.prior_config.type == "fungible_small":
            self.prior = FungibleSmall(config=self.prior_config)
        elif self.prior_config.type == "bnn_complete":
            self.prior = BNNPriorComplete(config=self.prior_config)
        else:
            self.prior = BNNPrior(config=self.prior_config)

    def __call__(self, batch_seeds: list[int]) -> dict[str, torch.Tensor]:
        # 1. Deterministically sample ONE size for the entire batch
        # We use the first seed in the batch to ensure reproducibility
        g = torch.Generator(device="cpu")
        g.manual_seed(batch_seeds[0])

        batch_n = int(
            torch.randint(
                low=self.min_episode_size,
                high=self.max_episode_size + 1,
                size=(1,),
                generator=g,
            ).item()
        )

        # 2. Generate all episodes for this batch using the fixed size
        episodes_data = []
        for seed in batch_seeds:
            ep = self.prior.generate_episode(
                batch_size=batch_n,
                episode_seed=seed,
            )
            ITE = ep.Y1 - ep.Y0

            episodes_data.append(
                {
                    "X": ep.X,
                    "Z": ep.Z,
                    "W": ep.W,
                    "A": ep.A.unsqueeze(-1),
                    "Y": ep.Y.unsqueeze(-1),
                    "Y0": ep.Y0.unsqueeze(-1),
                    "Y1": ep.Y1.unsqueeze(-1),
                    "ITE": ITE.unsqueeze(-1),
                }
            )

        # 3. Stack cleanly (no padding required because sizes match!)
        return {
            key: torch.stack([d[key] for d in episodes_data], dim=0)
            for key in episodes_data[0].keys()
        }


def create_causal_data_loaders(
    prior_config: ProximalPriorConfig,
    val_prior_config: ProximalPriorConfig | None = None,
    batch_size: int = 16,
    min_episode_size: int = 500,
    max_episode_size: int = 1500,
    train_episodes: int = 10_000,
    val_episodes: int = 1_000,
    num_workers: int = 4,
    seed: int = 42,
) -> dict[str, DataLoader]:
    # Datasets are now ultra-lightweight (just math)
    train_dataset = ProximalDataset(
        num_episodes=train_episodes,
        seed_offset=seed,
    )
    val_dataset = ProximalDataset(
        num_episodes=val_episodes,
        seed_offset=seed + train_episodes,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    use_pin_memory = torch.cuda.is_available()

    # The collator handles generation and ensures N is consistent per batch
    train_collator = DynamicBatchCollator(
        prior_config=prior_config,
        min_episode_size=min_episode_size,
        max_episode_size=max_episode_size,
    )

    val_prior_config = val_prior_config if val_prior_config is not None else prior_config
    val_collator = DynamicBatchCollator(
        prior_config=val_prior_config,
        min_episode_size=min_episode_size,
        max_episode_size=max_episode_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        generator=train_generator,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
        collate_fn=train_collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
        collate_fn=val_collator,
    )
    # TODO: fix persistent worker race condition before publication

    return {"train": train_loader, "val": val_loader}


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    torch.manual_seed(worker_seed + worker_id)
