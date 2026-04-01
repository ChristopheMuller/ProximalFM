import sys
from unittest.mock import MagicMock, patch

# Mocking dependencies before importing proximalfm modules
# We use a permanent mock in sys.modules for this test session
mock_torch = MagicMock()
mock_pandas = MagicMock()
mock_numpy = MagicMock()

# Only mock if not already present to avoid overriding in environments that have them,
# though for pure unit testing mocking is often preferred.
if "torch" not in sys.modules:
    sys.modules["torch"] = mock_torch
if "pandas" not in sys.modules:
    sys.modules["pandas"] = mock_pandas
if "numpy" not in sys.modules:
    sys.modules["numpy"] = mock_numpy

# Now we can safely import from proximalfm
import proximalfm.PriorLinear
from proximalfm.PriorLinear import ProximalPrior
from proximalfm.Structure import ProximalEpisode, ProximalPriorConfig

def test_episode_to_dataframe_linear():
    # Setup mock config and prior
    mock_config = ProximalPriorConfig(d_x=2, d_z=2, d_w=2, d_u=1)
    prior = ProximalPrior(config=mock_config)

    # Setup mock episode with Tensors that have .cpu().numpy()
    batch_size = 5

    def prepare_mock_tensor(name, shape):
        mock_t = MagicMock()
        # Mocking the chain .cpu().numpy()
        mock_cpu = MagicMock()
        mock_t.cpu.return_value = mock_cpu

        if len(shape) > 1:
            fake_np = MagicMock()
            fake_np.shape = shape
            # Handle slicing: X_np[:, i]
            # Return a simple list of floats to simulate numpy behavior for testing the dataframe dictionary construction
            fake_np.__getitem__.side_effect = lambda x: [float(i) for i in range(shape[0])]
            mock_cpu.numpy.return_value = fake_np
        else:
            # 1D case: columns["treatment"] = episode.A.cpu().numpy()
            mock_cpu.numpy.return_value = [float(i) for i in range(shape[0])]

        return mock_t

    episode = ProximalEpisode(
        X=prepare_mock_tensor("x", (batch_size, 2)),
        Z=prepare_mock_tensor("z", (batch_size, 2)),
        W=prepare_mock_tensor("w", (batch_size, 2)),
        A=prepare_mock_tensor("a", (batch_size,)),
        Y=prepare_mock_tensor("y", (batch_size,)),
        Y0=prepare_mock_tensor("y0", (batch_size,)),
        Y1=prepare_mock_tensor("y1", (batch_size,)),
        U=prepare_mock_tensor("u", (batch_size, 1)),
        propensity=prepare_mock_tensor("prop", (batch_size,))
    )

    # We patch the exact reference used in the method under test
    with patch("proximalfm.PriorLinear.pd.DataFrame") as mock_df_class:
        # Test without latents
        prior.episode_to_dataframe(episode, include_latents=False)

        # Verify pd.DataFrame was called
        mock_df_class.assert_called()
        called_dict = mock_df_class.call_args[0][0]

        expected_keys = ["x_0", "x_1", "z_0", "z_1", "w_0", "w_1", "treatment", "outcome", "y0", "y1", "propensity"]
        for key in expected_keys:
            assert key in called_dict
            assert isinstance(called_dict[key], list)
            assert all(isinstance(val, float) for val in called_dict[key])

        assert "u_0" not in called_dict

        # Test with latents
        mock_df_class.reset_mock()
        prior.episode_to_dataframe(episode, include_latents=True)
        called_dict_latents = mock_df_class.call_args[0][0]
        assert "u_0" in called_dict_latents
        assert all(isinstance(val, float) for val in called_dict_latents["u_0"])
