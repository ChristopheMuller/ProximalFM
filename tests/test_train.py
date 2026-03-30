import pytest
import os
import torch
import numpy as np
import random
import yaml
from unittest.mock import patch, MagicMock

from proximalfm.Train import (
    set_seed,
    format_time,
    load_config,
    gaussian_nll_loss,
    CATETrainer,
    main
)


# ---------------------------------------------------------------------------
# Utility Function Tests
# ---------------------------------------------------------------------------

@patch("proximalfm.Train.torch.manual_seed")
@patch("proximalfm.Train.torch.cuda.manual_seed_all")
@patch("proximalfm.Train.torch.mps.manual_seed")
@patch("proximalfm.Train.np.random.seed")
@patch("proximalfm.Train.random.seed")
def test_set_seed(mock_random, mock_np, mock_mps, mock_cuda, mock_torch):
    with patch("proximalfm.Train.torch.cuda.is_available", return_value=True), \
         patch("proximalfm.Train.torch.backends.mps.is_available", return_value=True):
        set_seed(42)

    mock_torch.assert_called_once_with(42)
    mock_cuda.assert_called_once_with(42)
    mock_mps.assert_called_once_with(42)
    mock_np.assert_called_once_with(42)
    mock_random.assert_called_once_with(42)

def test_set_seed_no_gpu():
    with patch("proximalfm.Train.torch.manual_seed") as mock_torch, \
         patch("proximalfm.Train.torch.cuda.is_available", return_value=False), \
         patch("proximalfm.Train.torch.backends.mps.is_available", return_value=False), \
         patch("proximalfm.Train.torch.cuda.manual_seed_all") as mock_cuda, \
         patch("proximalfm.Train.torch.mps.manual_seed") as mock_mps:

        set_seed(123)
        mock_torch.assert_called_once_with(123)
        mock_cuda.assert_not_called()
        mock_mps.assert_not_called()

def test_format_time():
    assert format_time(45.678) == "45.68s"
    assert format_time(60.0) == "1m 0.00s"
    assert format_time(125.4) == "2m 5.40s"
    assert format_time(3600.0) == "1h 0m 0.00s"
    assert format_time(3665.5) == "1h 1m 5.50s"

def test_load_config_success(tmp_path):
    config_file = tmp_path / "test_config.yaml"
    test_dict = {"training": {"epochs": 10}, "data": {"batch_size": 32}}
    with open(config_file, "w") as f:
        yaml.dump(test_dict, f)

    loaded = load_config(str(config_file))
    assert loaded == test_dict

def test_load_config_not_found():
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config("non_existent_config_12345.yaml")

def test_gaussian_nll_loss():
    # Mean = 0, target = 0, log_var = 0 (var = 1)
    # L = 0.5 * (0 + 0) = 0
    mean = torch.zeros(2, 2)
    target = torch.zeros(2, 2)
    log_var = torch.zeros(2, 2)
    loss = gaussian_nll_loss(mean, log_var, target)
    assert torch.allclose(loss, torch.tensor(0.0))

    # Mean = 1, target = 2, log_var = ln(2)
    # L = 0.5 * (ln(2) + (2-1)^2 / 2) = 0.5 * (ln(2) + 0.5)
    mean = torch.ones(2)
    target = torch.ones(2) * 2
    log_var = torch.log(torch.tensor(2.0)).expand(2)
    loss = gaussian_nll_loss(mean, log_var, target)
    expected = 0.5 * (np.log(2.0) + 0.5)
    assert torch.allclose(loss, torch.tensor(expected, dtype=torch.float32), atol=1e-5)


# ---------------------------------------------------------------------------
# CATETrainer Unit Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def base_config():
    return {
        "training": {
            "device": "cpu",
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "seed": 42,
            "epochs": 2,
            "log_dir": "test_logs",
            "save_dir": "test_saves",
            "save_freq": 1,
            "early_stop": 3,
        },
        "data": {
            "batch_size": 4,
            "samples_per_episode": 10,
            "train_episodes": 20,
            "val_episodes": 5,
            "num_workers": 0,
            "context_fraction": 0.8,
        },
        "prior": {
            "d_x": 3,
            "d_z": 2,
            "d_w": 2,
            "d_u": 1,
        },
        "model": {
            "dtype": "float32",
            "ninp": 16,
            "nhead": 2,
            "nhid": 32,
            "nlayers": 2,
        }
    }

@patch("proximalfm.Train.optim.lr_scheduler.ReduceLROnPlateau")
@patch("proximalfm.Train.optim.Adam")
@patch("proximalfm.Train.SummaryWriter")
@patch("proximalfm.Train.create_causal_data_loaders")
@patch("proximalfm.Train.PerFeatureTransformerCATE")
def test_trainer_init_device_cpu(mock_model, mock_loaders, mock_writer, mock_adam, mock_scheduler, base_config):
    # Setup mock loaders
    mock_loaders.return_value = {"train": MagicMock(), "val": MagicMock()}
    mock_model.return_value.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]

    trainer = CATETrainer(base_config)

    assert trainer.device == torch.device("cpu")
    assert trainer.dtype == torch.float32
    assert trainer.single_eval_pos == 8  # 10 * 0.8
    mock_model.assert_called_once()
    mock_loaders.assert_called_once()

@patch("proximalfm.Train.optim.lr_scheduler.ReduceLROnPlateau")
@patch("proximalfm.Train.optim.Adam")
@patch("proximalfm.Train.SummaryWriter")
@patch("proximalfm.Train.create_causal_data_loaders")
@patch("proximalfm.Train.PerFeatureTransformerCATE")
def test_trainer_init_device_cuda(mock_model, mock_loaders, mock_writer, mock_adam, mock_scheduler, base_config):
    mock_loaders.return_value = {"train": MagicMock(), "val": MagicMock()}
    mock_model.return_value.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    base_config["training"]["device"] = "cuda"
    base_config["training"]["gpu"] = 0

    with patch("proximalfm.Train.torch.cuda.is_available", return_value=True):
        trainer = CATETrainer(base_config)
        assert trainer.device == torch.device("cuda:0")

@patch("proximalfm.Train.optim.lr_scheduler.ReduceLROnPlateau")
@patch("proximalfm.Train.optim.Adam")
@patch("proximalfm.Train.SummaryWriter")
@patch("proximalfm.Train.create_causal_data_loaders")
@patch("proximalfm.Train.PerFeatureTransformerCATE")
def test_trainer_init_device_mps(mock_model, mock_loaders, mock_writer, mock_adam, mock_scheduler, base_config):
    mock_loaders.return_value = {"train": MagicMock(), "val": MagicMock()}
    mock_model.return_value.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    base_config["training"]["device"] = "mps"

    with patch("proximalfm.Train.torch.backends.mps.is_available", return_value=True):
        trainer = CATETrainer(base_config)
        assert trainer.device == torch.device("mps")

@patch("proximalfm.Train.optim.lr_scheduler.ReduceLROnPlateau")
@patch("proximalfm.Train.optim.Adam")
@patch("proximalfm.Train.SummaryWriter")
@patch("proximalfm.Train.create_causal_data_loaders")
@patch("proximalfm.Train.PerFeatureTransformerCATE")
def test_trainer_move_batch(mock_model, mock_loaders, mock_writer, mock_adam, mock_scheduler, base_config):
    mock_loaders.return_value = {"train": MagicMock(), "val": MagicMock()}
    mock_model.return_value.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    trainer = CATETrainer(base_config)
    trainer.device = torch.device("cpu") # Force CPU to test move safely

    batch = {
        "A": torch.randn(2, 2),
        "B": [1, 2, 3],
        "C": "string"
    }

    with patch.object(torch.Tensor, "to", return_value=torch.zeros(2, 2)) as mock_to:
        moved_batch = trainer._move_batch(batch)
        mock_to.assert_called_once_with(trainer.device, non_blocking=True)
        assert "A" in moved_batch
        assert isinstance(moved_batch["A"], torch.Tensor)
        assert moved_batch["B"] == [1, 2, 3]
        assert moved_batch["C"] == "string"

@patch("proximalfm.Train.optim.lr_scheduler.ReduceLROnPlateau")
@patch("proximalfm.Train.optim.Adam")
@patch("proximalfm.Train.SummaryWriter")
@patch("proximalfm.Train.create_causal_data_loaders")
@patch("proximalfm.Train.PerFeatureTransformerCATE")
def test_trainer_prepare_inputs(mock_model, mock_loaders, mock_writer, mock_adam, mock_scheduler, base_config):
    mock_loaders.return_value = {"train": MagicMock(), "val": MagicMock()}
    mock_model.return_value.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    trainer = CATETrainer(base_config)

    B, N = 4, 10
    dx, dz, dw = 3, 2, 2
    batch = {
            "X": torch.randn(N, B, dx),
            "Z": torch.randn(N, B, dz),
            "W": torch.randn(N, B, dw),
            "A": torch.randn(N, B, 1),
            "Y": torch.randn(N, B, 1),
            "ITE": torch.randn(N, B, 1)
    }

    x, z, w, a, y, ite_query = trainer._prepare_inputs(batch)

    # Check sequences are N, B, D
    assert x["main"].shape == (N, B, dx)
    assert z["main"].shape == (N, B, dz)
    assert w["main"].shape == (N, B, dw)
    assert a["main"].shape == (N, B, 1)
    assert y["main"].shape == (N, B, 1)

    # Check ite_query is N_query, B, 1
    # single_eval_pos is 8, so N_query is 2
    assert ite_query.shape == (2, B, 1)


# ---------------------------------------------------------------------------
# CATETrainer Training & Validation Tests
# ---------------------------------------------------------------------------

@patch("proximalfm.Train.optim.lr_scheduler.ReduceLROnPlateau")
@patch("proximalfm.Train.optim.Adam")
@patch("proximalfm.Train.SummaryWriter")
@patch("proximalfm.Train.create_causal_data_loaders")
@patch("proximalfm.Train.PerFeatureTransformerCATE")
def test_trainer_train_epoch(mock_model, mock_loaders, mock_writer, mock_adam, mock_scheduler, base_config):
    B, N = 2, 10
    dx, dz, dw = 3, 2, 2
    mock_batch = {
            "X": torch.randn(N, B, dx),
            "Z": torch.randn(N, B, dz),
            "W": torch.randn(N, B, dw),
            "A": torch.randn(N, B, 1),
            "Y": torch.randn(N, B, 1),
            "ITE": torch.randn(N, B, 1)
    }

    # Mock data loader to return 1 batch
    mock_loaders.return_value = {"train": [mock_batch], "val": []}

    # Set config to ensure gradient clipping works
    base_config["training"]["clip_grad"] = 1.0

    trainer = CATETrainer(base_config)
    trainer.device = torch.device("cpu")

    # Mock optimizer
    mock_optimizer = MagicMock()
    trainer.optimizer = mock_optimizer

    # Mock model output with requires_grad=True
    mock_out = {
        "cate_mean": torch.randn(B, N, 1, requires_grad=True),
        "cate_log_var": torch.randn(B, N, 1, requires_grad=True)
    }
    trainer.model.return_value = mock_out

    with patch("proximalfm.Train.nn.utils.clip_grad_norm_") as mock_clip:
        loss = trainer.train_epoch(0)

        # Verify mode
        trainer.model.train.assert_called_once()

        # Verify steps
        mock_optimizer.zero_grad.assert_called_once()
        mock_optimizer.step.assert_called_once()
        mock_clip.assert_called_once()

        assert isinstance(loss, float)
        assert len(trainer.train_losses) == 1
        assert len(trainer.epoch_times) == 1

@patch("proximalfm.Train.optim.lr_scheduler.ReduceLROnPlateau")
@patch("proximalfm.Train.optim.Adam")
@patch("proximalfm.Train.SummaryWriter")
@patch("proximalfm.Train.create_causal_data_loaders")
@patch("proximalfm.Train.PerFeatureTransformerCATE")
def test_trainer_validate(mock_model, mock_loaders, mock_writer, mock_adam, mock_scheduler, base_config):
    B, N = 2, 10
    dx, dz, dw = 3, 2, 2
    mock_batch = {
            "X": torch.randn(N, B, dx),
            "Z": torch.randn(N, B, dz),
            "W": torch.randn(N, B, dw),
            "A": torch.randn(N, B, 1),
            "Y": torch.randn(N, B, 1),
            "ITE": torch.randn(N, B, 1)
    }

    mock_loaders.return_value = {"train": [], "val": [mock_batch]}

    trainer = CATETrainer(base_config)
    trainer.device = torch.device("cpu")

    mock_out = {
        "cate_mean": torch.randn(B, N, 1),
        "cate_log_var": torch.randn(B, N, 1)
    }
    trainer.model.return_value = mock_out

    val_loss, val_rmse = trainer.validate(0)

    trainer.model.eval.assert_called_once()
    trainer.scheduler.step.assert_called_once()

    assert isinstance(val_loss, float)
    assert isinstance(val_rmse, float)
    assert len(trainer.val_losses) == 1
    assert len(trainer.val_rmses) == 1
    assert len(trainer.validation_times) == 1

# ---------------------------------------------------------------------------
# CATETrainer Checkpointing & Logging Tests
# ---------------------------------------------------------------------------

@patch("proximalfm.Train.torch.save")
@patch("proximalfm.Train.os.makedirs")
@patch("proximalfm.Train.optim.lr_scheduler.ReduceLROnPlateau")
@patch("proximalfm.Train.optim.Adam")
@patch("proximalfm.Train.SummaryWriter")
@patch("proximalfm.Train.create_causal_data_loaders")
@patch("proximalfm.Train.PerFeatureTransformerCATE")
def test_trainer_save_checkpoint(mock_model, mock_loaders, mock_writer, mock_adam, mock_scheduler, mock_makedirs, mock_save, base_config):
    trainer = CATETrainer(base_config)

    # Mock states
    trainer.model.state_dict.return_value = {"model": "state"}
    trainer.optimizer.state_dict.return_value = {"opt": "state"}
    trainer.epoch_times = [10.0]
    trainer.best_val_loss = 2.0

    trainer.save_checkpoint(epoch=1, val_loss=1.5, tag="test")

    mock_makedirs.assert_called_with("test_saves", exist_ok=True)

    # Check what was saved
    call_args_list = mock_save.call_args_list
    assert len(call_args_list) == 2  # Once for epoch, once for best

    saved_dict = call_args_list[0][0][0]
    assert saved_dict["epoch"] == 1
    assert saved_dict["val_loss"] == 1.5
    assert saved_dict["epoch_times"] == [10.0]
    assert saved_dict["model_state_dict"] == {"model": "state"}
    assert saved_dict["optimizer_state_dict"] == {"opt": "state"}

    assert "model_epoch_1_test.pth" in call_args_list[0][0][1]
    assert "best_model.pth" in call_args_list[1][0][1]

    assert trainer.best_val_loss == 1.5

@patch("proximalfm.Train.optim.lr_scheduler.ReduceLROnPlateau")
@patch("proximalfm.Train.optim.Adam")
@patch("proximalfm.Train.SummaryWriter")
@patch("proximalfm.Train.create_causal_data_loaders")
@patch("proximalfm.Train.PerFeatureTransformerCATE")
def test_trainer_log_metrics(mock_model, mock_loaders, mock_writer, mock_adam, mock_scheduler, base_config):
    trainer = CATETrainer(base_config)

    # Mock parameters
    trainer.optimizer.param_groups = [{"lr": 0.001}]
    trainer.epoch_times = [15.5]

    trainer.log_metrics(epoch=5, train_loss=1.2, val_loss=1.3, rmse=0.8)

    writer = trainer.writer
    writer.add_scalar.assert_any_call("Loss/train_nll", 1.2, 5)
    writer.add_scalar.assert_any_call("Loss/val_nll", 1.3, 5)
    writer.add_scalar.assert_any_call("Metrics/val_rmse", 0.8, 5)
    writer.add_scalar.assert_any_call("Learning_rate", 0.001, 5)
    writer.add_scalar.assert_any_call("Time/epoch_s", 15.5, 5)

# ---------------------------------------------------------------------------
# Lightweight Integration Test
# ---------------------------------------------------------------------------

@patch("proximalfm.Train.SummaryWriter")
def test_trainer_train_loop_integration(mock_writer, tmp_path, base_config):
    # Set config for a fast, tiny run
    base_config["training"]["device"] = "cpu"
    base_config["training"]["epochs"] = 1
    base_config["training"]["save_dir"] = str(tmp_path / "saves")
    base_config["training"]["log_dir"] = str(tmp_path / "logs")
    base_config["data"]["train_episodes"] = 2
    base_config["data"]["val_episodes"] = 1
    base_config["data"]["samples_per_episode"] = 5
    base_config["data"]["batch_size"] = 2
    base_config["model"]["ninp"] = 8
    base_config["model"]["nhid"] = 16
    base_config["model"]["nlayers"] = 1
    base_config["model"]["nhead"] = 1

    trainer = CATETrainer(base_config)

    # Run the full train loop
    trainer.train()

    # Check that it generated losses and logged metrics
    assert len(trainer.train_losses) == 1
    assert len(trainer.val_losses) == 1
    assert len(trainer.epoch_times) == 1

    # Check that files were created
    save_dir = tmp_path / "saves"
    assert save_dir.exists()
    assert (save_dir / "model_epoch_1.pth").exists()
    assert (save_dir / "best_model.pth").exists()
    assert (save_dir / "model_epoch_1_final.pth").exists()

    # Writer closed
    trainer.writer.close.assert_called_once()

@patch("sys.argv", ["train.py"])
@patch("proximalfm.Train.CATETrainer")
@patch("proximalfm.Train.load_config")
@patch("proximalfm.Train.set_seed")
def test_main_success(mock_set_seed, mock_load, mock_trainer, base_config):
    mock_load.return_value = base_config
    mock_instance = mock_trainer.return_value
    mock_instance.total_training_time = 5.0
    mock_instance.epoch_times = [2.0, 3.0]
    mock_instance.validation_times = [0.5, 0.5]

    main()

    mock_load.assert_called_once()
    mock_set_seed.assert_called_once_with(42)
    mock_trainer.assert_called_once_with(base_config)
    mock_instance.train.assert_called_once()

@patch("sys.argv", ["train.py"])
@patch("proximalfm.Train.CATETrainer")
@patch("proximalfm.Train.load_config")
@patch("proximalfm.Train.set_seed")
def test_main_keyboard_interrupt(mock_set_seed, mock_load, mock_trainer, base_config):
    mock_load.return_value = base_config
    mock_instance = mock_trainer.return_value
    mock_instance.total_training_time = 5.0
    mock_instance.epoch_times = [2.0]
    mock_instance.validation_times = []

    # Simulate a user hitting Ctrl+C
    mock_instance.train.side_effect = KeyboardInterrupt()

    main()

    # Should not raise an exception, just print stats and exit
    mock_instance.train.assert_called_once()

@patch("sys.argv", ["train.py"])
@patch("proximalfm.Train.CATETrainer")
@patch("proximalfm.Train.load_config")
@patch("proximalfm.Train.set_seed")
def test_main_exception(mock_set_seed, mock_load, mock_trainer, base_config):
    mock_load.return_value = base_config
    mock_instance = mock_trainer.return_value
    mock_instance.total_training_time = 0.0
    mock_instance.epoch_times = []
    mock_instance.validation_times = []

    mock_instance.train.side_effect = ValueError("Some unexpected error")

    with pytest.raises(ValueError, match="Some unexpected error"):
        main()
