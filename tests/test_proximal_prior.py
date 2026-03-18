import unittest.mock
import pytest
import torch

from proximalfm.Prior import ProximalPrior
from proximalfm.Structure import ProximalPriorConfig, NetworkSpec


@pytest.fixture
def prior() -> ProximalPrior:
    return ProximalPrior(seed=42)


def test_episode_normalization(prior: ProximalPrior) -> None:
    # Test with normalization (default)
    prior.config.normalize_episode = True
    ep_norm = prior.generate_episode(batch_size=1000, episode_seed=6)

    # Check that variables have roughly zero mean and unit variance
    assert torch.allclose(ep_norm.X.mean(dim=0), torch.zeros_like(ep_norm.X.mean(dim=0)), atol=0.1)
    assert torch.allclose(ep_norm.X.std(dim=0), torch.ones_like(ep_norm.X.std(dim=0)), atol=0.1)

    assert torch.isclose(ep_norm.Y.mean(), torch.tensor(0.0), atol=0.1)
    # The std is based on [Y0, Y1], so Y's std will be roughly 1
    assert torch.isclose(ep_norm.Y.std(), torch.tensor(1.0), atol=0.2)

    # Test without normalization
    prior.config.normalize_episode = False
    ep_unnorm = prior.generate_episode(batch_size=1000, episode_seed=6)

    # Check that they differ, proving normalization was skipped
    assert not torch.allclose(ep_unnorm.X.mean(dim=0), torch.zeros_like(ep_unnorm.X.mean(dim=0)), atol=0.1)

def test_episode_extreme_temperature() -> None:
    # High temperature -> propensities close to 0.5
    config_high = ProximalPriorConfig(temperature=100.0, min_propensity=0.0, max_propensity=1.0)
    prior_high = ProximalPrior(config=config_high, seed=7)
    ep_high = prior_high.generate_episode(batch_size=1000, episode_seed=7)

    # Sigmoid of near 0 is 0.5
    assert torch.allclose(ep_high.propensity.mean(), torch.tensor(0.5), atol=0.05)

    # Low temperature -> extreme propensities (close to min/max bounds)
    config_low = ProximalPriorConfig(temperature=0.001, min_propensity=0.1, max_propensity=0.9)
    prior_low = ProximalPrior(config=config_low, seed=8)
    ep_low = prior_low.generate_episode(batch_size=1000, episode_seed=8)

    # We should see many clamped values at the bounds
    prop = ep_low.propensity
    num_clamped = ((torch.isclose(prop, torch.tensor(0.1))) | (torch.isclose(prop, torch.tensor(0.9)))).sum().item()
    assert num_clamped > 500  # More than half should be clamped at extremes

def test_init_defaults() -> None:
    # Ensure it can be initialized without arguments
    default_prior = ProximalPrior()
    assert isinstance(default_prior.config, ProximalPriorConfig)
    assert default_prior.seed is None

def test_episode_batch_size_one(prior: ProximalPrior) -> None:
    # A batch size of 1 is tricky for standardization, but check it doesn't crash
    # and gives NaNs gracefully due to pytorch's std() behavior on single elements
    # Since std over 1 element returns nan, and nan + 1e-8 is nan, (val - mean) / nan -> nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ep = prior.generate_episode(batch_size=1, episode_seed=9)

    assert ep.X.shape[0] == 1
    assert ep.Y.shape[0] == 1
    # Normalization with PyTorch defaults causes NaNs on single batches for X, Z, W
    assert torch.isnan(ep.X).all()
    assert torch.isnan(ep.Z).all()
    assert torch.isnan(ep.W).all()
    # Y on the other hand stacks Y0 and Y1, so its std uses 2 elements and does not return nan
    assert not torch.isnan(ep.Y).any()

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

def test_make_generator_deterministic(prior: ProximalPrior) -> None:
    # Same episode seed -> same sequence of numbers
    gen1 = prior._make_generator(episode_seed=123)
    gen2 = prior._make_generator(episode_seed=123)
    val1 = torch.rand(10, generator=gen1)
    val2 = torch.rand(10, generator=gen2)
    assert torch.allclose(val1, val2)

    # Different episode seed -> different sequence of numbers
    gen3 = prior._make_generator(episode_seed=124)
    val3 = torch.rand(10, generator=gen3)
    assert not torch.allclose(val1, val3)

def test_random_bnn_shape(prior: ProximalPrior) -> None:
    gen = prior._make_generator(episode_seed=2)
    inputs = torch.randn(10, 5, generator=gen)
    spec = NetworkSpec(num_layers=3, hidden_size=8, weight_scale=1.0, bias_scale=0.1)

    output_dim = 4
    output = prior._random_bnn(inputs=inputs, output_dim=output_dim, spec=spec, generator=gen)

    assert list(output.shape) == [10, output_dim]

def test_random_bnn_single_layer(prior: ProximalPrior) -> None:
    gen = prior._make_generator(episode_seed=3)
    inputs = torch.randn(10, 5, generator=gen)
    spec = NetworkSpec(num_layers=1, hidden_size=8, weight_scale=1.0, bias_scale=0.1)

    output_dim = 4
    output = prior._random_bnn(inputs=inputs, output_dim=output_dim, spec=spec, generator=gen)

    assert list(output.shape) == [10, output_dim]

@pytest.mark.parametrize("regime", ["u_to_x", "x_to_u", "shared"])
def test_generate_x_u_regimes(prior: ProximalPrior, regime: str) -> None:
    gen = prior._make_generator(episode_seed=5)
    batch_size = 12
    with unittest.mock.patch.object(prior, "_sample_regime", return_value=regime):
        X, U = prior._generate_x_u(batch_size=batch_size, generator=gen)

        assert list(X.shape) == [batch_size, prior.config.d_x]
        assert list(U.shape) == [batch_size, prior.config.d_u]

def test_sample_regime(prior: ProximalPrior) -> None:
    gen = prior._make_generator(episode_seed=4)
    regimes_sampled = set()
    expected_regimes = {"u_to_x", "x_to_u", "shared"}

    # Run enough times to with high probability see all three
    for _ in range(100):
        regimes_sampled.add(prior._sample_regime(gen))

    assert regimes_sampled.issubset(expected_regimes)
    assert regimes_sampled == expected_regimes

def test_noise_shape_and_scale(prior: ProximalPrior) -> None:
    gen = prior._make_generator(episode_seed=1)
    shape = [1000, 5]
    scale = 0.5
    noise = prior._noise(shape=shape, scale=scale, generator=gen)

    assert list(noise.shape) == shape
    assert torch.isclose(noise.mean(), torch.tensor(0.0), atol=0.05)
    assert torch.isclose(noise.std(), torch.tensor(scale), atol=0.05)

def test_make_generator_non_deterministic() -> None:
    # If prior has no seed and episode_seed is None, it should use completely random seed
    prior_no_seed = ProximalPrior(seed=None)
    gen1 = prior_no_seed._make_generator(episode_seed=None)
    gen2 = prior_no_seed._make_generator(episode_seed=None)
    val1 = torch.rand(10, generator=gen1)
    val2 = torch.rand(10, generator=gen2)
    assert not torch.allclose(val1, val2)

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
