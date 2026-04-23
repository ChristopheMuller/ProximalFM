from proximalfm.Train import LoggerCallback
import sys
import pytest
from unittest.mock import MagicMock, patch

# Define modules to mock globally for the environment
MOCK_MODULES = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.optim",
    "wandb",
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
    
    # We no longer need instance.writer
    instance.global_step = 150 # Required for wandb.log step parameter
    instance.n_params = 10000  # Required for epoch 0 logging
    instance.optimizer = MagicMock()
    instance.optimizer.param_groups = [{"lr": 0.001}]
    return instance

@patch("proximalfm.Train.wandb")
def test_log_metrics_basic(mock_wandb, trainer):
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



    trainer.model = MagicMock()
    trainer.global_step = 10
    logger_cb = LoggerCallback({})
    logger_cb.on_init_end(trainer)
    logger_cb.on_epoch_end(trainer, epoch, train_metrics, val_metrics=None)



    # Assert wandb.log was called
    mock_wandb.log.assert_called_once()
    
    # Extract the dictionary and keyword arguments passed to wandb.log
    args, kwargs = mock_wandb.log.call_args
    logs = args[0]

    # Assert step was passed correctly to avoid the timeline collision bug
    assert kwargs.get("step") == trainer.global_step

    # Assert train metrics are in the dictionary
    assert logs["epoch"] == 5
    assert logs["Loss/train_total"] == 0.5
    assert logs["Loss/train_mse"] == 0.4
    assert logs["Loss/train_model"] == 0.1
    assert logs["Parameters/alpha"] == 0.8
    assert logs["Parameters/grad_norm"] == 0.01
    assert logs["Parameters/learning_rate"] == 0.001
    assert logs["Time/train_s"] == 10.0

    # Assert no validation metrics are logged
    for metric_name in logs.keys():
        assert not metric_name.startswith("Loss/val")
        assert not metric_name.startswith("Metrics/val")
        assert not metric_name.startswith("ATE/")
        assert not metric_name.startswith("CATE/")