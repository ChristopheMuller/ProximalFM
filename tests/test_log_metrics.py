import sys
import pytest
from unittest.mock import MagicMock

# Define modules to mock globally for the environment
MOCK_MODULES = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.optim",
    "torch.utils.tensorboard",
    "yaml",
    "tqdm",
    "numpy",
    "einops",
    "networkx",
    "tabicl",
    "tabpfn_common_utils",
    "pydantic",
    "pydantic_settings",
    "matplotlib",
    "seaborn",
    "pandas",
    "proximalfm.Model",
    "proximalfm.Structure",
    "proximalfm.PriorDataLoader",
]

@pytest.fixture(scope="module", autouse=True)
def mock_dependencies():
    """Globally mock missing dependencies for the duration of this test module."""
    saved_modules = {}
    for module in MOCK_MODULES:
        if module in sys.modules:
            saved_modules[module] = sys.modules[module]
        sys.modules[module] = MagicMock()

    yield

    # Restore original modules
    for module in MOCK_MODULES:
        del sys.modules[module]
    sys.modules.update(saved_modules)

@pytest.fixture
def trainer():
    """Fixture to provide a CATETrainer instance with mocked attributes."""
    # Import inside fixture to ensure mocks are applied
    from proximalfm.Train import CATETrainer

    # Instantiate without calling __init__
    instance = CATETrainer.__new__(CATETrainer)
    instance.writer = MagicMock()
    instance.optimizer = MagicMock()
    instance.optimizer.param_groups = [{"lr": 0.001}]
    return instance

def test_log_metrics_basic(trainer):
    """Test log_metrics correctly handles train metrics without validation data."""
    train_metrics = {
        "loss_total": 0.5,
        "loss_mse": 0.4,
        "loss_model": 0.1,
        "alpha": 0.8,
        "grad_norm": 0.01,
        "time_s": 10.0
    }
    epoch = 5

    trainer.log_metrics(epoch, train_metrics, val_metrics=None)

    # Assert train metrics are logged
    trainer.writer.add_scalar.assert_any_call("Loss/train_total", 0.5, epoch)
    trainer.writer.add_scalar.assert_any_call("Loss/train_mse", 0.4, epoch)
    trainer.writer.add_scalar.assert_any_call("Loss/train_model", 0.1, epoch)
    trainer.writer.add_scalar.assert_any_call("Parameters/alpha", 0.8, epoch)
    trainer.writer.add_scalar.assert_any_call("Parameters/grad_norm", 0.01, epoch)
    trainer.writer.add_scalar.assert_any_call("Parameters/learning_rate", 0.001, epoch)
    trainer.writer.add_scalar.assert_any_call("Time/train_s", 10.0, epoch)

    # Assert no validation metrics are logged
    for call in trainer.writer.add_scalar.call_args_list:
        metric_name = call[0][0]
        assert not metric_name.startswith("Loss/val")
        assert not metric_name.startswith("Metrics/val")
