import pytest
from proximalfm.train._train_config import build_parser
from proximalfm.train._run import Trainer
import torch
from unittest.mock import patch, MagicMock


@pytest.fixture
def dummy_args(tmp_path):
    parser = build_parser()
    # minimal dummy config
    # Note: embed_dim / row_nhead must be > 2 for RotaryEmbedding (16 / 4 = 4)
    args = parser.parse_args(
        [
            "--max_steps",
            "1",
            "--batch_size",
            "2",
            "--micro_batch_size",
            "2",
            "--min_seq_len",
            "10",
            "--max_seq_len",
            "20",
            "--min_train_size",
            "5",
            "--max_train_size",
            "15",
            "--embed_dim",
            "16",
            "--row_nhead",
            "4",
            "--col_num_blocks",
            "1",
            "--row_num_blocks",
            "1",
            "--icl_num_blocks",
            "1",
            "--amp",
            "False",
            "--device",
            "cpu",
            "--wandb_mode",
            "disabled",
            "--checkpoint_dir",
            str(tmp_path),
        ]
    )
    return args


@patch("wandb.init")
def test_trainer_initialization(mock_wandb_init, dummy_args):
    trainer = Trainer(dummy_args)
    assert trainer.config.max_steps == 1
    assert trainer.model is not None


def test_trainer_run_one_epoch(dummy_args):
    trainer = Trainer(dummy_args)

    # We want to patch the dataloader to return a fast dummy batch
    with patch.object(trainer, "dataloader") as mock_dataloader:
        # Create a dummy batch
        # 17 items: (U, X, W, Z, A, A_probs, Y_obs, Y0, Y1, MU0, MU1,
        #            d_U, d_X, d_W, d_Z, seq_len, train_size)

        U = torch.randn(2, 20, dummy_args.max_dim_U)
        X = torch.randn(2, 20, dummy_args.max_dim_X)
        W = torch.randn(2, 20, dummy_args.max_dim_W)
        Z = torch.randn(2, 20, dummy_args.max_dim_Z)
        A = torch.randint(0, 2, (2, 20, 1)).float()
        A_probs = torch.rand(2, 20, 1)
        Y_obs = torch.randn(2, 20)
        Y0 = torch.randn(2, 20)
        Y1 = torch.randn(2, 20)
        MU0 = torch.randn(2, 20)
        MU1 = torch.randn(2, 20)
        d_U = torch.tensor([3, 4])
        d_X = torch.tensor([5, 6])
        d_W = torch.tensor([4, 5])
        d_Z = torch.tensor([3, 4])
        seq_len = torch.tensor([15, 15])
        train_size = torch.tensor([10, 10])

        dummy_batch = (
            U,
            X,
            W,
            Z,
            A,
            A_probs,
            Y_obs,
            Y0,
            Y1,
            MU0,
            MU1,
            d_U,
            d_X,
            d_W,
            d_Z,
            seq_len,
            train_size,
        )

        # Make the mock dataloader yield this batch once
        mock_dataloader.__iter__.return_value = iter([dummy_batch])

        # Override the max_steps and save_temp_every to run just one step
        trainer.config.max_steps = 1
        trainer.config.save_temp_every = 1000
        trainer.config.save_perm_every = 1000

        # Drive the loop in "near-infinite-stream" mode: one generated batch per
        # block, reused once, so exactly one batch is consumed for the one step
        # (matches the single mocked batch above).
        trainer.config.gen_block_size = trainer.config.batch_size
        trainer.config.gen_reuse_passes = 1

        trainer.train()
        assert trainer.curr_step == 1


def test_trainer_sub_functions(dummy_args):
    trainer = Trainer(dummy_args)

    # Test validate_micro_batch
    seq_lens = torch.tensor([10, 10])
    train_sizes = torch.tensor([5, 5])
    seq_len, train_size = trainer.validate_micro_batch(seq_lens, train_sizes)
    assert seq_len == 10
    assert train_size == 5

    # Test align_micro_batch
    X = torch.randn(2, 20, 10)  # B=2, T=20, max_dim=10
    W = torch.randn(2, 20, 10)
    Z = torch.randn(2, 20, 10)
    A = torch.randn(2, 20, 1)
    Y_obs = torch.randn(2, 20)
    Y0 = torch.randn(2, 20)
    Y1 = torch.randn(2, 20)
    MU0 = torch.randn(2, 20)
    MU1 = torch.randn(2, 20)
    d_X = torch.tensor([5, 6])
    d_W = torch.tensor([4, 5])
    d_Z = torch.tensor([3, 4])

    (
        a_X,
        a_W,
        a_Z,
        a_A,
        a_Y_obs,
        a_Y0,
        a_Y1,
        a_MU0,
        a_MU1,
    ) = trainer.align_micro_batch(
        X, W, Z, A, Y_obs, Y0, Y1, MU0, MU1, d_X, d_W, d_Z, seq_len=15
    )

    # Check truncated sequence length
    assert a_X.shape[1] == 15
    assert a_A.shape[1] == 15
    assert a_Y_obs.shape[1] == 15
    assert a_MU0.shape[1] == 15
    assert a_MU1.shape[1] == 15

    # Check truncated feature dimensions (to max across the batch)
    assert a_X.shape[2] == 6
    assert a_W.shape[2] == 5
    assert a_Z.shape[2] == 4
