import os
import yaml
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from proximalfm.Train import (
    set_seed,
    format_time,
    load_config,
    CATETrainer,
)

def test_set_seed():
    # Calling set_seed should not raise any exceptions and should be reproducible
    set_seed(42)
    val1 = torch.rand(1).item()
    set_seed(42)
    val2 = torch.rand(1).item()
    assert val1 == val2

def test_format_time():
    assert format_time(30.5) == "30.50s"
    assert format_time(150.25) == "2m 30.25s"
    assert format_time(3665.12) == "1h 1m 5.12s"

def test_load_config_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")

def test_load_config(tmp_path):
    config_data = {"test": 123}
    file_path = tmp_path / "test_config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config_data, f)

    loaded = load_config(str(file_path))
    assert loaded == config_data

@pytest.fixture
def mock_config(tmp_path):
    return {
        "training": {
            "device": "cpu",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 2,
            "log_dir": str(tmp_path / "logs"),
            "save_dir": str(tmp_path / "saves"),
            "seed": 42,
            "warmup_epochs": 1,
            "clip_grad": 1.0,
            "val_freq": 1,
            "save_freq": 1,
            "early_stop": 3,
            "warmup_steps": 2,
            "exp_name": "experiment_1",
        },
        "data": {
            "batch_size": 2,
            "samples_per_episode": 10,
            "train_episodes": 4,
            "reset_freq_train": 1,
            "val_episodes": 2,
            "num_workers": 0,
            "context_fraction": 0.8,
        },
        "prior": {
            "d_x": 2,
            "d_z": 2,
            "d_w": 2,
            "d_u": 2,
        },
        "model": {
            "dtype": "float32",
            "embed_dim": 16,
            "col_num_blocks": 1,
            "row_num_blocks": 1,
            "icl_num_blocks": 1,
            "col_nhead": 2,
            "row_nhead": 2,
            "icl_nhead": 2,
        }
    }

@patch("proximalfm.Train.SummaryWriter")
@patch("proximalfm.Train.create_causal_data_loaders")
@patch("proximalfm.Train.build_model")
def test_cate_trainer_init(mock_build_model, mock_create_loaders, mock_writer, mock_config):
    mock_model_inst = MagicMock()
    mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.tensor([1.0]))]
    
    mock_build_model.return_value = mock_model_inst
    mock_create_loaders.return_value = {"train": ["train_batch"], "val": ["val_batch"]}

    trainer = CATETrainer(mock_config)

    assert trainer.device == torch.device("cpu")
    
    mock_build_model.assert_called_once()
    
    passed_args = mock_build_model.call_args[0]
    assert len(passed_args) > 0, "build_model was called without positional arguments."
    assert passed_args[0].__class__.__name__ == "ModelConfig"

    mock_create_loaders.assert_called_once()
    assert trainer.train_loader == ["train_batch"]
    assert trainer.val_loader == ["val_batch"]
    assert trainer.single_eval_pos == 8

@patch("proximalfm.Train.SummaryWriter")
@patch("proximalfm.Train.create_causal_data_loaders")
def test_prepare_inputs(mock_create_loaders, mock_writer, mock_config):
    mock_create_loaders.return_value = {"train": [], "val": []}
    trainer = CATETrainer(mock_config)

    N, B, D = 10, 2, 2
    # Create dummy DataLoader output (N, B, D)
    batch = {
        "X": torch.randn(N, B, D),
        "Z": torch.randn(N, B, D),
        "W": torch.randn(N, B, D),
        "A": torch.randn(N, B, 1),
        "Y": torch.randn(N, B, 1),
        "ITE": torch.randn(N, B, 1),
    }

    x, a, w, z, y_norm, target_norm, y_std = trainer._prepare_inputs(batch)

    # Expected shapes: NanoTab expects (B, N, D)
    assert x.shape == (B, N, D)
    assert z.shape == (B, N, D)
    assert w.shape == (B, N, D)

    # A and Y should be sliced to n_train (single_eval_pos = 8) and squeezed to (B, N)
    n_train = 8
    assert a.shape == (B, n_train)
    assert y_norm.shape == (B, n_train)

    # Target should be remaining rows: (B, N - n_train, 1)
    assert target_norm.shape == (B, N - n_train, 1)

    assert y_std.shape == (B, 1) or y_std.shape == (B,)

    assert y_std.gt(0).all()

@patch("proximalfm.Train.SummaryWriter")
@patch("proximalfm.Train.create_causal_data_loaders")
def test_train_epoch(mock_create_loaders, mock_writer, mock_config):
    N, B, D = 10, 2, 2
    batch = {
        "X": torch.randn(N, B, D),
        "Z": torch.randn(N, B, D),
        "W": torch.randn(N, B, D),
        "A": torch.randn(N, B, 1),
        "Y": torch.randn(N, B, 1),
        "ITE": torch.randn(N, B, 1),
    }

    mock_create_loaders.return_value = {"train": [batch], "val": []}
    trainer = CATETrainer(mock_config)

    # 1. Create a dummy parameter to anchor the computational graph
    dummy_param = torch.nn.Parameter(torch.randn(1))
    
    # 2. Mock model outputs
    # N_query = N - single_eval_pos = 10 - 8 = 2
    mock_out = torch.randn(B, 2, 3)  # Matches shape for 3 quantiles (or GMM outputs)
    
    # We add `dummy_param * 0.0` to force these tensors into the autograd graph
    # so `loss.backward()` successfully computes gradients for dummy_param
    mock_mean = torch.randn(B, 2, 1) + dummy_param * 0.0
    mock_log_var = torch.randn(B, 2, 1) + dummy_param * 0.0
    mock_loss = dummy_param * 0.1  # Scalar loss linked to the parameter

    # 3. Setup the MagicMock for the model
    trainer.model = MagicMock()
    trainer.model.return_value = mock_out
    trainer.model.loss.return_value = mock_loss
    trainer.model.estimate_CATE.return_value = (mock_mean, mock_log_var)
    
    # 4. Mock parameter iterations for optimizer and grad clipping
    trainer.model.parameters.return_value = [dummy_param]
    trainer.model.named_parameters.return_value = [("dummy_layer", dummy_param)]

    # Run the epoch
    train_metrics = trainer.train_epoch(epoch=0)
    
    # Assertions
    assert isinstance(train_metrics, dict)
    assert "loss_total" in train_metrics
    assert "rmse" in train_metrics
    assert "variance" in train_metrics
    assert "grad_norm" in train_metrics
    assert "alpha" in train_metrics
    assert "time_s" in train_metrics
    assert len(trainer.train_losses) == 1
    
    # Verify our new agnostic methods were actually called
    trainer.model.loss.assert_called_once()
    trainer.model.estimate_CATE.assert_called_once()

@patch("proximalfm.Train.SummaryWriter")
@patch("proximalfm.Train.create_causal_data_loaders")
def test_validate(mock_create_loaders, mock_writer, mock_config):
    N, B, D = 10, 2, 2
    batch = {
        "X": torch.randn(N, B, D),
        "Z": torch.randn(N, B, D),
        "W": torch.randn(N, B, D),
        "A": torch.randn(N, B, 1),
        "Y": torch.randn(N, B, 1),
        "ITE": torch.randn(N, B, 1),
    }

    mock_create_loaders.return_value = {"train": [], "val": [batch]}
    trainer = CATETrainer(mock_config)

    mock_out_mean = torch.randn(B, 2, 1)
    mock_out_var = torch.randn(B, 2, 1)
    trainer.model = MagicMock()
    trainer.model.estimate_CATE.return_value = (mock_out_mean, mock_out_var)

    val_metrics = trainer.validate(epoch=0)

    assert isinstance(val_metrics, dict)
    assert "rmse" in val_metrics
    assert "loss_gnll" not in val_metrics
    assert "variance" in val_metrics



    assert len(trainer.val_rmses) == 1

@patch("proximalfm.Train.SummaryWriter")
@patch("proximalfm.Train.create_causal_data_loaders")
def test_save_checkpoint(mock_create_loaders, mock_writer, mock_config, tmp_path):
    mock_create_loaders.return_value = {"train": [], "val": []}
    trainer = CATETrainer(mock_config)

    trainer.save_checkpoint(epoch=1, val_rmse=0.5)

    save_dir = os.path.join(mock_config["training"]["save_dir"], mock_config["training"]["exp_name"])
    assert os.path.exists(os.path.join(save_dir, "model_epoch_1.pth"))
    assert os.path.exists(os.path.join(save_dir, "best_model.pth"))

    # Test tracking of best val_loss
    assert trainer.best_val_loss == 0.5
    trainer.save_checkpoint(epoch=2, val_rmse=0.6)
    # best model shouldn't be overridden
    assert trainer.best_val_loss == 0.5
