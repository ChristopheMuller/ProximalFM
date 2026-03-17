import pytest
import torch

from proximalfm.Prior import ProximalPrior


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
