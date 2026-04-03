import sys
from unittest.mock import MagicMock, patch, mock_open
import os

# Mock dependencies that are not available in the environment
mock_torch = MagicMock()
mock_torch.float32 = "float32"
mock_torch.tanh = "tanh"
mock_torch.device = MagicMock()
mock_torch.load = MagicMock()
mock_torch.save = MagicMock()
mock_torch.nn = MagicMock()

sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch.nn
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['torch.optim.lr_scheduler'] = MagicMock()
sys.modules['torch.utils.tensorboard'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['yaml'] = MagicMock()
sys.modules['tqdm'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['networkx'] = MagicMock()

# Prevent loading modules that trigger errors
sys.modules['proximalfm.Model'] = MagicMock()
sys.modules['proximalfm.Structure'] = MagicMock()
sys.modules['proximalfm.PriorDataLoader'] = MagicMock()

import pytest

def test_security_weights_only_true():
    # Only import CATETrainer here, after mocks
    from proximalfm.Train import CATETrainer

    config = {
        "training": {"device": "cpu", "save_dir": "saves"},
        "model": {},
        "prior": {"d_x": 1, "d_w": 1, "d_z": 1},
        "data": {}
    }

    # We need a dummy file for os.path.exists
    with open("dummy_checkpoint.pth", "w") as f:
        f.write("dummy")

    try:
        # Mocking CATETrainer's dependency classes in the Trainer's module scope
        with patch('proximalfm.Train.ProximalNanoTabICLv2'), \
             patch('proximalfm.Train.create_causal_data_loaders'), \
             patch('proximalfm.Train.SummaryWriter'), \
             patch('proximalfm.Train.ProximalPriorConfig'), \
             patch('proximalfm.Train.os.makedirs'), \
             patch('proximalfm.Train.open', mock_open(), create=True):

            trainer = CATETrainer(config)

            # Mock the model and optimizer so load_state_dict doesn't fail
            trainer.model = MagicMock()
            trainer.optimizer = MagicMock()
            trainer.scheduler = MagicMock()

            # The actual test:
            trainer._load_checkpoint("dummy_checkpoint.pth")

            # Check if torch.load was called with weights_only=True
            mock_torch.load.assert_called_once()
            _, kwargs = mock_torch.load.call_args
            assert kwargs.get("weights_only") is True, f"torch.load must be called with weights_only=True, got {kwargs.get('weights_only')}"
    finally:
        if os.path.exists("dummy_checkpoint.pth"):
            os.remove("dummy_checkpoint.pth")
