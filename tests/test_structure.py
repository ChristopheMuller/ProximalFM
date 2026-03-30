import pytest
from unittest.mock import Mock

import torch
from proximalfm.Structure import NetworkSpec, ProximalPriorConfig, ProximalEpisode


def test_network_spec_defaults():
    """Test that NetworkSpec initializes with correct default values."""
    spec = NetworkSpec()
    assert spec.num_layers == 3
    assert spec.hidden_size == 32
    assert spec.activation == torch.tanh
    assert spec.weight_scale == 0.75
    assert spec.bias_scale == 0.1


def test_network_spec_custom():
    """Test that NetworkSpec correctly uses provided arguments."""
    spec = NetworkSpec(
        num_layers=5,
        hidden_size=64,
        activation=torch.relu,
        weight_scale=1.0,
        bias_scale=0.2
    )
    assert spec.num_layers == 5
    assert spec.hidden_size == 64
    assert spec.activation == torch.relu
    assert spec.weight_scale == 1.0
    assert spec.bias_scale == 0.2


def test_proximal_prior_config_defaults():
    """Test that ProximalPriorConfig initializes with correct default values and distinct factory instances."""
    config = ProximalPriorConfig()

    # Check primitive defaults
    assert config.d_u == 4
    assert config.d_x == 8
    assert config.d_z == 6
    assert config.d_w == 6
    assert config.noise_scale_x == 0.05
    assert config.noise_scale_z == 0.05
    assert config.noise_scale_w == 0.05
    assert config.noise_scale_y == 0.05
    assert config.temperature == 0.7
    assert config.min_propensity == 0.05
    assert config.max_propensity == 0.95
    assert config.device == "cpu"
    assert config.dtype == torch.float32
    assert config.normalize_episode is True

    # Check that default_factory creates distinct NetworkSpec instances
    assert isinstance(config.x_spec, NetworkSpec)
    assert isinstance(config.z_spec, NetworkSpec)
    assert isinstance(config.w_spec, NetworkSpec)
    assert isinstance(config.u_spec, NetworkSpec)

    assert config.x_spec is not config.z_spec
    assert config.w_spec is not config.u_spec

    # Check specific default_factory lambdas for treatment and outcome specs
    assert isinstance(config.treatment_spec, NetworkSpec)
    assert config.treatment_spec.hidden_size == 24
    assert config.treatment_spec.num_layers == 3  # Ensures it inherits other NetworkSpec defaults

    assert isinstance(config.outcome_spec, NetworkSpec)
    assert config.outcome_spec.hidden_size == 24
    assert config.outcome_spec.num_layers == 3


def test_proximal_episode_success():
    """Test that ProximalEpisode correctly instantiates when all arguments are provided."""
    mock_X = Mock(spec=torch.Tensor)
    mock_Z = Mock(spec=torch.Tensor)
    mock_W = Mock(spec=torch.Tensor)
    mock_A = Mock(spec=torch.Tensor)
    mock_Y = Mock(spec=torch.Tensor)
    mock_Y0 = Mock(spec=torch.Tensor)
    mock_Y1 = Mock(spec=torch.Tensor)
    mock_U = Mock(spec=torch.Tensor)
    mock_propensity = Mock(spec=torch.Tensor)

    episode = ProximalEpisode(
        X=mock_X,
        Z=mock_Z,
        W=mock_W,
        A=mock_A,
        Y=mock_Y,
        Y0=mock_Y0,
        Y1=mock_Y1,
        U=mock_U,
        propensity=mock_propensity
    )

    assert episode.X is mock_X
    assert episode.Z is mock_Z
    assert episode.W is mock_W
    assert episode.A is mock_A
    assert episode.Y is mock_Y
    assert episode.Y0 is mock_Y0
    assert episode.Y1 is mock_Y1
    assert episode.U is mock_U
    assert episode.propensity is mock_propensity


def test_proximal_episode_missing_args():
    """Test that ProximalEpisode raises a TypeError when missing required positional arguments."""
    with pytest.raises(TypeError):
        ProximalEpisode()

    # Also check when partially complete
    with pytest.raises(TypeError):
        ProximalEpisode(
            X=Mock(spec=torch.Tensor),
            Z=Mock(spec=torch.Tensor)
        )
