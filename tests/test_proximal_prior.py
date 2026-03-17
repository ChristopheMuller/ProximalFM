import pytest
import torch

from proximalfm.Prior import ProximalPrior
from proximalfm.Structure import ProximalPriorConfig


@pytest.fixture
def prior() -> ProximalPrior:
    return ProximalPrior(seed=42)


def test_episode_shapes_and_sizes(prior: ProximalPrior) -> None:
    episode = prior.generate_episode(batch_size=16, episode_seed=0)

    assert episode.X.shape == (16, prior.config.d_x)
    assert episode.Z.shape == (16, prior.config.d_z)
    assert episode.W.shape == (16, prior.config.d_w)
    assert episode.U.shape == (16, prior.config.d_u)
    assert episode.Y.shape == (16,)
    assert episode.Y0.shape == (16,)
    assert episode.Y1.shape == (16,)
    assert episode.A.shape == (16,)
    assert episode.propensity.shape == (16,)


def test_propensity_bounds_are_respected(prior: ProximalPrior) -> None:
    episode = prior.generate_episode(batch_size=64, episode_seed=1)
    assert episode.propensity.min().item() >= prior.config.min_propensity
    assert episode.propensity.max().item() <= prior.config.max_propensity


def test_counterfactual_noise_is_shared(prior: ProximalPrior) -> None:
    episode = prior.generate_episode(batch_size=32, episode_seed=2)
    difference = episode.Y - episode.Y0
    expected = episode.A * (episode.Y1 - episode.Y0)
    assert torch.allclose(difference, expected, atol=1e-5)


def test_y_is_independent_of_z_given_axu(prior: ProximalPrior) -> None:
    episode = prior.generate_episode(batch_size=128, episode_seed=3)
    features = torch.cat([episode.A.unsqueeze(-1), episode.X, episode.U], dim=-1)
    target = episode.Y.unsqueeze(-1)
    solution = torch.linalg.lstsq(features, target).solution
    residual = target - features @ solution
    correlations = torch.abs((residual * episode.Z).mean(dim=0))
    assert torch.all(correlations < 0.25)

def test_episode_to_dataframe(prior: ProximalPrior) -> None:
    batch_size = 16
    episode = prior.generate_episode(batch_size=batch_size, episode_seed=4)

    # Test without latents
    df = prior.episode_to_dataframe(episode, include_latents=False)

    import pandas as pd
    assert isinstance(df, pd.DataFrame)
    assert len(df) == batch_size

    # Check expected columns
    expected_cols = (
        [f"x_{i}" for i in range(prior.config.d_x)] +
        [f"z_{i}" for i in range(prior.config.d_z)] +
        [f"w_{i}" for i in range(prior.config.d_w)] +
        ["treatment", "outcome", "y0", "y1", "propensity"]
    )
    for col in expected_cols:
        assert col in df.columns

    # Check that latents are NOT included
    for i in range(prior.config.d_u):
        assert f"u_{i}" not in df.columns

    # Test with latents
    df_with_latents = prior.episode_to_dataframe(episode, include_latents=True)
    for i in range(prior.config.d_u):
        assert f"u_{i}" in df_with_latents.columns
    
def test_invalid_config_dz_too_small() -> None:
    config = ProximalPriorConfig(d_u=4, d_z=3, d_w=6)
    with pytest.raises(ValueError):
        ProximalPrior(config=config)


def test_invalid_config_dw_too_small() -> None:
    config = ProximalPriorConfig(d_u=4, d_z=6, d_w=3)
    with pytest.raises(ValueError):
        ProximalPrior(config=config)

def test_generate_dataset_num_workers(prior: ProximalPrior) -> None:
    num_episodes = 4
    batch_size = 16

    # Generate dataset with num_workers=1
    episodes_1_worker = prior.generate_dataset(
        num_episodes=num_episodes,
        batch_size=batch_size,
        num_workers=1,
        base_seed=123
    )

    # Generate dataset with num_workers=2
    episodes_2_workers = prior.generate_dataset(
        num_episodes=num_episodes,
        batch_size=batch_size,
        num_workers=2,
        base_seed=123
    )

    # Verify lengths
    assert len(episodes_1_worker) == num_episodes
    assert len(episodes_2_workers) == num_episodes

    # Verify properties and determinism
    for ep1, ep2 in zip(episodes_1_worker, episodes_2_workers):
        # Verify batch size
        assert ep1.X.shape[0] == batch_size

        # Verify determinism across workers
        assert torch.allclose(ep1.X, ep2.X)
        assert torch.allclose(ep1.Z, ep2.Z)
        assert torch.allclose(ep1.W, ep2.W)
        assert torch.allclose(ep1.U, ep2.U)
        assert torch.allclose(ep1.Y, ep2.Y)
        assert torch.allclose(ep1.Y0, ep2.Y0)
        assert torch.allclose(ep1.Y1, ep2.Y1)
        assert torch.allclose(ep1.A, ep2.A)
        assert torch.allclose(ep1.propensity, ep2.propensity)
