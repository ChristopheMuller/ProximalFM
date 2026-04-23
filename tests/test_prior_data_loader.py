import unittest.mock
import pytest
import torch
from torch.utils.data import DataLoader

from proximalfm.Structure import ProximalPriorConfig
from proximalfm.PriorDataLoader import (
    ProximalDataset,
    DynamicBatchCollator,
    create_causal_data_loaders,
    _worker_init_fn,
)


@pytest.fixture
def base_config() -> ProximalPriorConfig:
    # A valid configuration where d_z >= d_u and d_w >= d_u
    config = ProximalPriorConfig(
        d_x=3,
        d_u=2,
        d_z=2,
        d_w=2,
    )
    # Give it a recognizable device to test it changes
    config.device = "cuda" if torch.cuda.is_available() else "meta"
    return config


# ---------------------------------------------------------------------------
# ProximalDataset Tests (Seed Generation Only)
# ---------------------------------------------------------------------------


def test_dataset_initialization() -> None:
    num_episodes = 5

    dataset = ProximalDataset(
        num_episodes=num_episodes,
        seed_offset=123,
    )

    assert dataset.num_episodes == num_episodes
    assert dataset.base_seed_offset == 123


def test_dataset_len() -> None:
    dataset = ProximalDataset(num_episodes=42, seed_offset=0)
    assert len(dataset) == 42


def test_dataset_getitem_returns_seed() -> None:
    dataset = ProximalDataset(num_episodes=10, seed_offset=42)

    # Call __getitem__ on index 2
    # Seed math: offset (42) + reset_count (0) * num_episodes (10) + idx (2) = 44
    seed = dataset[2]

    assert isinstance(seed, int)
    assert seed == 44

    # Verify reset_count logic changes the seed
    dataset.set_reset_state(1)
    seed_reset = dataset[2]
    # offset (42) + reset_count (1) * num_episodes (10) + idx (2) = 54
    assert seed_reset == 54


# ---------------------------------------------------------------------------
# DynamicBatchCollator Tests (Data Generation)
# ---------------------------------------------------------------------------


def test_collator_validation_dz(base_config: ProximalPriorConfig) -> None:
    # Modify config to fail Assumption 4 for Z
    base_config.d_u = 3
    base_config.d_z = 2
    base_config.d_w = 3

    # Note: If your validation logic is currently inside BNNPrior, this might 
    # raise there instead of the Collator. Either way, it should raise ValueError.
    with pytest.raises(ValueError):
        DynamicBatchCollator(base_config, min_episode_size=10, max_episode_size=20)


def test_collator_validation_dw(base_config: ProximalPriorConfig) -> None:
    # Modify config to fail Assumption 4 for W
    base_config.d_u = 3
    base_config.d_z = 3
    base_config.d_w = 2

    with pytest.raises(ValueError):
        DynamicBatchCollator(base_config, min_episode_size=10, max_episode_size=20)


def test_collator_call_generates_dynamic_batch(base_config: ProximalPriorConfig) -> None:
    min_size = 15
    max_size = 25
    batch_seeds = [100, 101, 102]  # Simulating a batch size of 3

    collator = DynamicBatchCollator(
        prior_config=base_config, 
        min_episode_size=min_size, 
        max_episode_size=max_size
    )

    # Simulate the DataLoader passing a batch of seeds
    batch = collator(batch_seeds)

    expected_keys = {"X", "Z", "W", "A", "Y", "Y0", "Y1", "ITE"}
    assert set(batch.keys()) == expected_keys

    # Verify B and N
    B, N, d_x = batch["X"].shape
    assert B == len(batch_seeds)
    assert min_size <= N <= max_size
    assert d_x == base_config.d_x

    # Verify shapes of other tensors align with the sampled N
    assert batch["Z"].shape == (B, N, base_config.d_z)
    assert batch["W"].shape == (B, N, base_config.d_w)
    assert batch["A"].shape == (B, N, 1)
    assert batch["ITE"].shape == (B, N, 1)

    # Verify calculation of ITE
    assert torch.allclose(batch["ITE"], batch["Y1"] - batch["Y0"])


# ---------------------------------------------------------------------------
# DataLoader Creation Tests
# ---------------------------------------------------------------------------


def test_create_data_loaders_basic(base_config: ProximalPriorConfig) -> None:
    batch_size = 4
    min_size = 10
    max_size = 20
    train_eps = 20
    val_eps = 5
    seed = 42

    loaders = create_causal_data_loaders(
        prior_config=base_config,
        batch_size=batch_size,
        min_episode_size=min_size,
        max_episode_size=max_size,
        train_episodes=train_eps,
        val_episodes=val_eps,
        num_workers=0,
        seed=seed,
    )

    assert isinstance(loaders, dict)
    assert set(loaders.keys()) == {"train", "val"}

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    assert train_loader.batch_size == batch_size
    assert val_loader.batch_size == batch_size

    # Verify the custom collator is attached
    assert isinstance(train_loader.collate_fn, DynamicBatchCollator)
    assert isinstance(val_loader.collate_fn, DynamicBatchCollator)

    assert len(train_loader.dataset) == train_eps
    assert len(val_loader.dataset) == val_eps


def test_create_data_loaders_seed_offsets(base_config: ProximalPriorConfig) -> None:
    train_eps = 100
    seed = 42

    loaders = create_causal_data_loaders(
        prior_config=base_config,
        min_episode_size=10,
        max_episode_size=20,
        train_episodes=train_eps,
        val_episodes=50,
        num_workers=0,
        seed=seed,
    )

    train_ds = loaders["train"].dataset
    val_ds = loaders["val"].dataset

    assert isinstance(train_ds, ProximalDataset)
    assert isinstance(val_ds, ProximalDataset)

    # Verify seed offsets are disjoint
    assert train_ds.base_seed_offset == seed
    assert val_ds.base_seed_offset == seed + train_eps


@unittest.mock.patch("torch.cuda.is_available")
def test_create_data_loaders_pin_memory_cuda_available(
    mock_is_available, base_config: ProximalPriorConfig
) -> None:
    mock_is_available.return_value = True

    loaders = create_causal_data_loaders(
        prior_config=base_config,
        num_workers=0,
    )

    assert loaders["train"].pin_memory is True
    assert loaders["val"].pin_memory is True


@unittest.mock.patch("torch.cuda.is_available")
def test_create_data_loaders_pin_memory_cuda_unavailable(
    mock_is_available, base_config: ProximalPriorConfig
) -> None:
    mock_is_available.return_value = False

    loaders = create_causal_data_loaders(
        prior_config=base_config,
        num_workers=0,
    )

    assert loaders["train"].pin_memory is False
    assert loaders["val"].pin_memory is False


def test_create_data_loaders_persistent_workers(
    base_config: ProximalPriorConfig,
) -> None:
    # Test with workers > 0
    loaders_with_workers = create_causal_data_loaders(
        prior_config=base_config,
        num_workers=2,
    )
    assert loaders_with_workers["train"].persistent_workers is True
    assert loaders_with_workers["val"].persistent_workers is True

    # Test with workers == 0
    loaders_no_workers = create_causal_data_loaders(
        prior_config=base_config,
        num_workers=0,
    )
    assert loaders_no_workers["train"].persistent_workers is False
    assert loaders_no_workers["val"].persistent_workers is False


@unittest.mock.patch("torch.initial_seed")
@unittest.mock.patch("torch.manual_seed")
def test_worker_init_fn(mock_manual_seed, mock_initial_seed) -> None:
    # Use a seed larger than 2**32 to verify the modulo operation
    mock_initial_seed.return_value = 2**33 + 42

    worker_id = 5
    _worker_init_fn(worker_id)

    # initial_seed % 2**32 should be 42
    # Then it adds worker_id (5), so expected manual_seed is 47
    mock_manual_seed.assert_called_once_with(47)


def test_multiprocessing_different_data(base_config: ProximalPriorConfig) -> None:
    """
    Test that when num_workers > 0, the data generated across batches 
    iterates correctly and dynamic dimensions are respected.
    """
    batch_size = 4
    train_eps = 16
    min_size = 10
    max_size = 20

    loaders = create_causal_data_loaders(
        prior_config=base_config,
        batch_size=batch_size,
        min_episode_size=min_size,
        max_episode_size=max_size,
        train_episodes=train_eps,
        val_episodes=4,
        num_workers=2,
        seed=42,
    )

    train_loader = loaders["train"]

    total_episodes = 0
    for batch in train_loader:
        B, N, d_x = batch["X"].shape
        
        assert B == batch_size
        # The core check: the collator is dynamically picking a valid N
        assert min_size <= N <= max_size
        
        total_episodes += B

    # Total episodes processed
    assert total_episodes == train_eps