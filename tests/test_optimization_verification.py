import pytest
from unittest.mock import MagicMock, patch
import sys

# --- Fixtures ---

@pytest.fixture(autouse=True)
def mock_dependencies():
    """Automatically mock heavy dependencies for all tests in this module."""
    mock_torch = MagicMock()
    mock_pd = MagicMock()
    mock_np = MagicMock()
    
    with patch.dict(sys.modules, {
        "torch": mock_torch,
        "pandas": mock_pd,
        "numpy": mock_np
    }):
        mock_torch.device = MagicMock
        mock_torch.float32 = "float32"
        
        # We also need to patch the references in ProximalPrior if it's already imported
        # or will be imported. 
        import proximalfm.Prior
        with patch.object(proximalfm.Prior, "pd", mock_pd), \
             patch.object(proximalfm.Prior, "torch", mock_torch):
            yield

@pytest.fixture
def mock_config():
    """Provides a standard config object."""
    config = MagicMock()
    config.device = "cpu"
    config.dtype = "float32"
    config.d_z = 6
    config.d_w = 6
    config.d_u = 4
    return config

@pytest.fixture
def mock_episode():
    """Provides an episode mock with pre-configured tensor return values."""
    episode = MagicMock()
    
    # Define shapes (rows, cols)
    shapes = {
        "X": (10, 8),
        "Z": (10, 6),
        "W": (10, 6),
        "U": (10, 4),
        "A": (10,),
        "Y": (10,),
        "Y0": (10,),
        "Y1": (10,),
        "propensity": (10,)
    }
    
    for attr, shape in shapes.items():
        tensor_mock = getattr(episode, attr)
        numpy_mock = MagicMock()
        numpy_mock.shape = shape
        numpy_mock.__len__.return_value = shape[0]
        # Chain the calls: .cpu().numpy() -> returns numpy_mock
        tensor_mock.cpu.return_value.numpy.return_value = numpy_mock
        
    return episode

# --- Tests ---

def test_episode_to_dataframe_calls(mock_config, mock_episode):
    # We import inside the test or after the mock_dependencies fixture runs
    from proximalfm.Prior import ProximalPrior
    
    # Initialize
    prior = ProximalPrior(config=mock_config)

    # Execute
    prior.episode_to_dataframe(mock_episode, include_latents=True)

    # Verify: Check that .cpu().numpy() was called exactly once for each tensor
    assert mock_episode.X.cpu.return_value.numpy.call_count == 1
    assert mock_episode.Z.cpu.return_value.numpy.call_count == 1
    assert mock_episode.W.cpu.return_value.numpy.call_count == 1
    assert mock_episode.U.cpu.return_value.numpy.call_count == 1