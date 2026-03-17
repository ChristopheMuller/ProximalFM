from unittest.mock import MagicMock
import sys

# Mock torch
mock_torch = MagicMock()
sys.modules["torch"] = mock_torch
mock_torch.device = MagicMock
mock_torch.float32 = "float32"

# Mock pandas
mock_pandas = MagicMock()
sys.modules["pandas"] = mock_pandas

# Mock numpy
mock_numpy = MagicMock()
sys.modules["numpy"] = mock_numpy

from proximalfm.Prior import ProximalPrior

def test_episode_to_dataframe_calls():
    config = MagicMock()
    config.device = "cpu"
    config.dtype = "float32"
    config.d_z = 6
    config.d_w = 6
    config.d_u = 4

    prior = ProximalPrior(config=config)

    episode = MagicMock()

    # Setup returns for .cpu().numpy()
    x_np = MagicMock()
    x_np.shape = (10, 8)
    episode.X.cpu.return_value.numpy.return_value = x_np

    z_np = MagicMock()
    z_np.shape = (10, 6)
    episode.Z.cpu.return_value.numpy.return_value = z_np

    w_np = MagicMock()
    w_np.shape = (10, 6)
    episode.W.cpu.return_value.numpy.return_value = w_np

    u_np = MagicMock()
    u_np.shape = (10, 4)
    episode.U.cpu.return_value.numpy.return_value = u_np

    # Call the method
    prior.episode_to_dataframe(episode, include_latents=True)

    # Verify that .cpu().numpy() was called exactly once for each main tensor
    assert episode.X.cpu.return_value.numpy.call_count == 1
    assert episode.Z.cpu.return_value.numpy.call_count == 1
    assert episode.W.cpu.return_value.numpy.call_count == 1
    assert episode.U.cpu.return_value.numpy.call_count == 1
