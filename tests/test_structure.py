from unittest.mock import MagicMock, patch
import sys

# Create mocks
mock_torch = MagicMock()
mock_torch.tanh = lambda x: x
mock_torch.float32 = "float32"

mock_pandas = MagicMock()
mock_numpy = MagicMock()
mock_matplotlib = MagicMock()
mock_einops = MagicMock()
mock_tqdm = MagicMock()

# Since we cannot avoid importing the package which depends on many things,
# we need to provide mocks during the import time if they are not present.
# We will use patch.dict for this.

with patch.dict(sys.modules, {
    "torch": mock_torch,
    "pandas": mock_pandas,
    "numpy": mock_numpy,
    "matplotlib": mock_matplotlib,
    "matplotlib.pyplot": mock_matplotlib,
    "einops": mock_einops,
    "tqdm": mock_tqdm
}):
    from proximalfm.Structure import NetworkSpec, ProximalPriorConfig, ProximalEpisode

def test_network_spec_defaults():
    spec = NetworkSpec()
    assert spec.num_layers == 3
    assert spec.hidden_size == 32
    assert spec.activation == mock_torch.tanh
    assert spec.weight_scale == 0.75
    assert spec.bias_scale == 0.1

def test_network_spec_custom():
    custom_activation = lambda x: x * 2
    spec = NetworkSpec(
        num_layers=5,
        hidden_size=64,
        activation=custom_activation,
        weight_scale=1.0,
        bias_scale=0.0
    )
    assert spec.num_layers == 5
    assert spec.hidden_size == 64
    assert spec.activation(2) == 4
    assert spec.weight_scale == 1.0
    assert spec.bias_scale == 0.0

def test_proximal_prior_config_defaults():
    config = ProximalPriorConfig()
    assert config.d_u == 4
    assert config.d_x == 8
    assert config.d_z == 6
    assert config.d_w == 6

    assert isinstance(config.x_spec, NetworkSpec)
    assert config.x_spec.hidden_size == 32

    assert config.treatment_spec.hidden_size == 24
    assert config.outcome_spec.hidden_size == 24

    assert config.noise_scale_x == 0.05
    assert config.noise_scale_z == 0.05
    assert config.noise_scale_w == 0.05
    assert config.noise_scale_y == 0.05

    assert config.temperature == 0.7
    assert config.min_propensity == 0.05
    assert config.max_propensity == 0.95
    assert config.device == "cpu"
    assert config.dtype == mock_torch.float32
    assert config.normalize_episode is True

def test_proximal_prior_config_custom():
    config = ProximalPriorConfig(d_u=10, temperature=0.5, device="cuda")
    assert config.d_u == 10
    assert config.temperature == 0.5
    assert config.device == "cuda"
    assert config.d_x == 8
    assert config.noise_scale_x == 0.05

def test_proximal_episode_instantiation():
    mock_tensor = MagicMock()

    episode = ProximalEpisode(
        X=mock_tensor,
        Z=mock_tensor,
        W=mock_tensor,
        A=mock_tensor,
        Y=mock_tensor,
        Y0=mock_tensor,
        Y1=mock_tensor,
        U=mock_tensor,
        propensity=mock_tensor
    )

    assert episode.X == mock_tensor
    assert episode.Z == mock_tensor
    assert episode.W == mock_tensor
    assert episode.A == mock_tensor
    assert episode.Y == mock_tensor
    assert episode.Y0 == mock_tensor
    assert episode.Y1 == mock_tensor
    assert episode.U == mock_tensor
    assert episode.propensity == mock_tensor
