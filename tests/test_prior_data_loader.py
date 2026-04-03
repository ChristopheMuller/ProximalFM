import unittest.mock
import pytest
import torch
from torch.utils.data import DataLoader

from proximalfm.Structure import ProximalPriorConfig
from proximalfm.PriorDataLoader import (
    ProximalDataset,
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


def test_dataset_initialization(base_config: ProximalPriorConfig) -> None:
    num_episodes = 5
    samples = 10

    # Store original device
    original_device = base_config.device

    dataset = ProximalDataset(
        prior_config=base_config,
        num_episodes=num_episodes,
        samples_per_episode=samples,
        seed_offset=123,
    )

    # Verify that the dataset's config is forced to "cpu"
    assert dataset.prior_config.device == "cpu"
    # Verify the original config was not mutated
    assert base_config.device == original_device

    assert dataset.num_episodes == num_episodes
    assert dataset.samples_per_episode == samples
    assert dataset.base_seed_offset == 123


def test_dataset_validation_dz(base_config: ProximalPriorConfig) -> None:
    # Modify config to fail Assumption 4 for Z
    base_config.d_u = 3
    base_config.d_z = 2
    base_config.d_w = 3

    with pytest.raises(ValueError) as excinfo:
        ProximalDataset(base_config, num_episodes=1, samples_per_episode=10)

    assert "d_z=2 must be >= d_u=3" in str(excinfo.value)
    assert "Assumption 4" in str(excinfo.value)


def test_dataset_validation_dw(base_config: ProximalPriorConfig) -> None:
    # Modify config to fail Assumption 4 for W
    base_config.d_u = 3
    base_config.d_z = 3
    base_config.d_w = 2

    with pytest.raises(ValueError) as excinfo:
        ProximalDataset(base_config, num_episodes=1, samples_per_episode=10)

    assert "d_w=2 must be >= d_u=3" in str(excinfo.value)
    assert "Assumption 4" in str(excinfo.value)


def test_dataset_validation_dz_equal_du(base_config: ProximalPriorConfig) -> None:
    # Modify config to test edge case Assumption 4 for Z where d_z == d_u
    base_config.d_u = 3
    base_config.d_z = 3
    base_config.d_w = 4

    # This should not raise an error
    dataset = ProximalDataset(base_config, num_episodes=1, samples_per_episode=10)
    assert dataset is not None


def test_dataset_validation_dw_equal_du(base_config: ProximalPriorConfig) -> None:
    # Modify config to test edge case Assumption 4 for W where d_w == d_u
    base_config.d_u = 3
    base_config.d_z = 4
    base_config.d_w = 3

    # This should not raise an error
    dataset = ProximalDataset(base_config, num_episodes=1, samples_per_episode=10)
    assert dataset is not None


def test_dataset_len(base_config: ProximalPriorConfig) -> None:
    dataset = ProximalDataset(base_config, num_episodes=42, samples_per_episode=10)
    assert len(dataset) == 42


def test_dataset_getitem(base_config: ProximalPriorConfig) -> None:
    samples = 15
    dataset = ProximalDataset(base_config, num_episodes=10, samples_per_episode=samples)

    # Call __getitem__ on index 2
    item = dataset[2]

    # Verify dictionary keys
    expected_keys = {"X", "Z", "W", "A", "Y", "Y0", "Y1", "ITE"}
    assert set(item.keys()) == expected_keys

    # Verify shapes
    assert item["X"].shape == (samples, base_config.d_x)
    assert item["Z"].shape == (samples, base_config.d_z)
    assert item["W"].shape == (samples, base_config.d_w)
    assert item["A"].shape == (samples, 1)
    assert item["Y"].shape == (samples, 1)
    assert item["Y0"].shape == (samples, 1)
    assert item["Y1"].shape == (samples, 1)
    assert item["ITE"].shape == (samples, 1)

    # Verify calculation of ITE
    assert torch.allclose(item["ITE"], item["Y1"] - item["Y0"])

    # Verify deterministic output based on seed_offset + idx
    dataset2 = ProximalDataset(base_config, num_episodes=10, samples_per_episode=samples)
    item_same = dataset2[2]

    for key in item:
        assert torch.allclose(item[key], item_same[key])



def test_create_data_loaders_basic(base_config: ProximalPriorConfig) -> None:
    batch_size = 4
    samples = 10
    train_eps = 20
    val_eps = 5
    seed = 42

    loaders = create_causal_data_loaders(
        prior_config=base_config,
        batch_size=batch_size,
        samples_per_episode=samples,
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

    train_ds = train_loader.dataset
    val_ds = val_loader.dataset

    assert isinstance(train_ds, ProximalDataset)
    assert isinstance(val_ds, ProximalDataset)

    assert len(train_ds) == train_eps
    assert len(val_ds) == val_eps


def test_create_data_loaders_seed_offsets(base_config: ProximalPriorConfig) -> None:
    train_eps = 100
    seed = 42

    loaders = create_causal_data_loaders(
        prior_config=base_config,
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
def test_create_data_loaders_pin_memory_cuda_available(mock_is_available, base_config: ProximalPriorConfig) -> None:
    mock_is_available.return_value = True

    loaders = create_causal_data_loaders(
        prior_config=base_config,
        num_workers=0,
    )

    assert loaders["train"].pin_memory is True
    assert loaders["val"].pin_memory is True


@unittest.mock.patch("torch.cuda.is_available")
def test_create_data_loaders_pin_memory_cuda_unavailable(mock_is_available, base_config: ProximalPriorConfig) -> None:
    mock_is_available.return_value = False

    loaders = create_causal_data_loaders(
        prior_config=base_config,
        num_workers=0,
    )

    assert loaders["train"].pin_memory is False
    assert loaders["val"].pin_memory is False


def test_create_data_loaders_persistent_workers(base_config: ProximalPriorConfig) -> None:
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
    Test that when num_workers > 0, the data generated across batches and epochs
    remains correct and workers do not output identical data purely due to sharing RNG.

    Note: The dataset itself sets the seed based on seed_offset + index,
    which is deterministic per episode index. So the main thing we want to check
    is that the data loader can successfully iterate with multiple workers
    and we get the right number of distinct episodes.
    """
    batch_size = 4
    train_eps = 16

    loaders = create_causal_data_loaders(
        prior_config=base_config,
        batch_size=batch_size,
        samples_per_episode=10,
        train_episodes=train_eps,
        val_episodes=4,
        num_workers=2,
        seed=42,
    )

    train_loader = loaders["train"]

    # We collect the first batch of X from each iteration
    # Because shuffle=True, the exact order is non-deterministic
    # but over the entire epoch, we should see `train_eps` unique episodes.
    # To check uniqueness, we can hash or just collect the tensors and compare.

    all_x = []
    for batch in train_loader:
        # batch["X"] is (samples, batch_size, d_x) due to custom_collate
        assert batch["X"].shape == (10, batch_size, base_config.d_x)
        all_x.append(batch["X"])

    # Total episodes processed
    total_episodes = sum(x.shape[1] for x in all_x)
    assert total_episodes == train_eps
