import math

import pytest
from proximalfm.train._train_config import build_parser
from proximalfm.train._run import Trainer
from proximalfm.prior._dataset import PriorDataset
import torch
from unittest.mock import patch, MagicMock


def _dummy_args_list(tmp_path):
    # minimal dummy config
    # Note: embed_dim / row_nhead must be > 2 for RotaryEmbedding (16 / 4 = 4)
    return [
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


@pytest.fixture
def dummy_args(tmp_path):
    parser = build_parser()
    return parser.parse_args(_dummy_args_list(tmp_path))


def _parse_args(tmp_path, extra):
    parser = build_parser()
    return parser.parse_args(_dummy_args_list(tmp_path) + extra)


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
        # 19 items: (U, X, W, Z, A, A_probs, Y_obs, Y0, Y1, MU0, MU1,
        #            d_U, d_X, d_W, d_Z, seq_len, train_size, S2, K_eff)

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
        S2 = torch.rand(2, 20)
        K_eff = torch.tensor([8, 8])

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
            S2,
            K_eff,
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
    S2 = torch.rand(2, 20)
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
        a_S2,
    ) = trainer.align_micro_batch(
        X, W, Z, A, Y_obs, Y0, Y1, MU0, MU1, S2, d_X, d_W, d_Z, seq_len=15
    )

    # Check truncated sequence length
    assert a_X.shape[1] == 15
    assert a_A.shape[1] == 15
    assert a_Y_obs.shape[1] == 15
    assert a_MU0.shape[1] == 15
    assert a_MU1.shape[1] == 15
    assert a_S2.shape[1] == 15

    # Check truncated feature dimensions (to max across the batch)
    assert a_X.shape[2] == 6
    assert a_W.shape[2] == 5
    assert a_Z.shape[2] == 4


@pytest.mark.parametrize("mc_targets_on_context", [True, False])
def test_trainer_micro_batch_with_query_only_mc(dummy_args, mc_targets_on_context):
    """The trainer must be unaffected by NaN MU values on the context prefix.

    Runs a real prior batch (not a mock) through the loss path. With
    mc_targets_on_context=False the MU targets are NaN for rows [:train_size];
    the trainer only ever slices [train_size:], so every reported loss and
    target statistic must still be finite.
    """
    dummy_args.mc_targets_on_context = mc_targets_on_context
    trainer = Trainer(dummy_args)

    prior = PriorDataset(
        batch_size=2,
        batch_size_per_gp=2,
        min_dim_U=2,
        max_dim_U=dummy_args.max_dim_U,
        min_dim_W=2,
        max_dim_W=dummy_args.max_dim_W,
        min_dim_Z=2,
        max_dim_Z=dummy_args.max_dim_Z,
        min_dim_X=2,
        max_dim_X=dummy_args.max_dim_X,
        min_seq_len=None,
        max_seq_len=64,
        # A wide context so both treatment arms are present; run_micro_batch
        # legitimately returns None when one arm is missing.
        min_train_size=24,
        max_train_size=25,
        prior_type="mlp_scm",
        mc_cate=3,
        mc_targets_on_context=mc_targets_on_context,
        n_jobs=1,
        device="cpu",
    )
    batch = prior.get_batch()

    MU0, train_sizes = batch[9], batch[16]
    train_size = int(train_sizes[0].item())
    # Guard that the parametrisation actually exercises both regimes.
    assert bool(MU0[:, :train_size].isnan().all()) is not mc_targets_on_context

    results = trainer.run_micro_batch(batch, 0, 1, is_train=False)

    assert results, "expected the micro batch to produce metrics"
    for key, value in results.items():
        assert math.isfinite(value), f"{key} is not finite: {value}"

    # bins_po has a real per-arm head, so its po/* metrics are genuine and kept.
    val_metrics = trainer.val_evaluator.compute(prefix="val/point/")
    assert any(key.startswith("val/point/po/") for key in val_metrics)


def test_trainer_micro_batch_cate_single(dummy_args):
    """The cate_single loss branch runs and reports its own finite metrics.

    Guards the loss branch and the metrics arm together: a missing metrics arm
    would raise NameError rather than return a dict.
    """
    dummy_args.loss_type = "cate_single"
    dummy_args.lambda_cate = 0.1
    trainer = Trainer(dummy_args)

    assert trainer.raw_model.icl_predictor.decoder[-1].out_features == (
        trainer.raw_model.nbins
    )

    prior = PriorDataset(
        batch_size=2,
        batch_size_per_gp=2,
        min_dim_U=2,
        max_dim_U=dummy_args.max_dim_U,
        min_dim_W=2,
        max_dim_W=dummy_args.max_dim_W,
        min_dim_Z=2,
        max_dim_Z=dummy_args.max_dim_Z,
        min_dim_X=2,
        max_dim_X=dummy_args.max_dim_X,
        min_seq_len=None,
        max_seq_len=64,
        min_train_size=24,
        max_train_size=25,
        prior_type="mlp_scm",
        mc_cate=3,
        n_jobs=1,
        device="cpu",
    )
    batch = prior.get_batch()

    results = trainer.run_micro_batch(batch, 0, 1, is_train=False)

    assert results, "expected the micro batch to produce metrics"
    assert "loss/cate_dist" in results
    assert "loss/cate" in results
    for key, value in results.items():
        assert math.isfinite(value), f"{key} is not finite: {value}"

    # cate_single has no per-arm head, so the val evaluator was updated without
    # pred_y0/pred_y1/true_y0/true_y1 and reports no po/* metrics.
    val_metrics = trainer.val_evaluator.compute(prefix="val/point/")
    assert not any(key.startswith("val/point/po/") for key in val_metrics)
    assert any(key.startswith("val/point/cate/") for key in val_metrics)


@pytest.mark.parametrize("mc_targets_on_context", [True, False])
def test_trainer_micro_batch_cate_single_noise_aware(dummy_args, mc_targets_on_context):
    """The noise-aware branch runs end to end and reports its noise diagnostics.

    Parametrised over mc_targets_on_context because that flag decides whether S2
    carries a NaN context prefix -- the path most likely to leak a NaN into the
    loss.
    """
    dummy_args.loss_type = "cate_single_noise_aware"
    dummy_args.lambda_cate = 0.1
    dummy_args.mc_targets_on_context = mc_targets_on_context
    trainer = Trainer(dummy_args)

    # Same head as cate_single: out_dim == nbins, tau grid registered.
    assert trainer.raw_model.icl_predictor.decoder[-1].out_features == (
        trainer.raw_model.nbins
    )
    assert hasattr(trainer.raw_model, "cate_bin_edges")

    prior = PriorDataset(
        batch_size=2,
        batch_size_per_gp=2,
        min_dim_U=2,
        max_dim_U=dummy_args.max_dim_U,
        min_dim_W=2,
        max_dim_W=dummy_args.max_dim_W,
        min_dim_Z=2,
        max_dim_Z=dummy_args.max_dim_Z,
        min_dim_X=2,
        max_dim_X=dummy_args.max_dim_X,
        min_seq_len=None,
        max_seq_len=64,
        min_train_size=24,
        max_train_size=25,
        prior_type="mlp_scm",
        mc_cate=8,
        mc_targets_on_context=mc_targets_on_context,
        n_jobs=1,
        device="cpu",
    )
    batch = prior.get_batch()

    S2, train_sizes = batch[17], batch[16]
    train_size = int(train_sizes[0].item())
    assert bool(S2[:, :train_size].isnan().all()) is not mc_targets_on_context

    results = trainer.run_micro_batch(batch, 0, 1, is_train=False)

    assert results, "expected the micro batch to produce metrics"
    for key in (
        "loss/cate_dist",
        "noise/mean_s",
        "noise/mean_sd_pred",
        "noise/s_over_sd_pred",
        "noise/frac_fallback",
        "noise/mean_K",
    ):
        assert key in results, f"missing {key}"
    for key, value in results.items():
        assert math.isfinite(value), f"{key} is not finite: {value}"

    # Query rows all have K=8 replicates, so nothing should take the
    # no-variance fallback regardless of how the context was generated.
    assert results["noise/frac_fallback"] == 0.0
    assert results["noise/mean_K"] == 8.0
    assert results["noise/mean_s"] > 0.0


def test_noise_aware_falls_back_when_the_variance_is_missing(dummy_args):
    """K=1 leaves no variance estimate; the loss must still produce metrics."""
    dummy_args.loss_type = "cate_single_noise_aware"
    trainer = Trainer(dummy_args)

    prior = PriorDataset(
        batch_size=2,
        batch_size_per_gp=2,
        min_dim_U=2,
        max_dim_U=dummy_args.max_dim_U,
        min_dim_W=2,
        max_dim_W=dummy_args.max_dim_W,
        min_dim_Z=2,
        max_dim_Z=dummy_args.max_dim_Z,
        min_dim_X=2,
        max_dim_X=dummy_args.max_dim_X,
        min_seq_len=None,
        max_seq_len=64,
        min_train_size=24,
        max_train_size=25,
        prior_type="mlp_scm",
        mc_cate=1,
        n_jobs=1,
        device="cpu",
    )
    results = trainer.run_micro_batch(prior.get_batch(), 0, 1, is_train=False)

    assert results["noise/frac_fallback"] == 1.0
    assert results["noise/mean_s"] == 0.0
    for key, value in results.items():
        assert math.isfinite(value), f"{key} is not finite: {value}"


def test_cate_single_checkpoint_loads_into_the_noise_aware_head(dummy_args):
    """The two loss types share a head, so weights transfer with no surgery.

    This is what makes the noise-aware objective a fine-tune of an existing
    cate_single checkpoint rather than a from-scratch run.
    """
    dummy_args.loss_type = "cate_single"
    point = Trainer(dummy_args).raw_model

    dummy_args.loss_type = "cate_single_noise_aware"
    noise_aware = Trainer(dummy_args).raw_model

    result = noise_aware.load_state_dict(point.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []

    assert torch.equal(noise_aware.cate_bin_edges, point.cate_bin_edges)
    assert torch.equal(noise_aware.cate_bin_centers, point.cate_bin_centers)


@patch("wandb.init")
def test_dist_eval_interval_minus_one_resolves_to_point_val_interval(
    mock_wandb_init, tmp_path
):
    args = _parse_args(
        tmp_path,
        ["--point_val_interval", "37", "--dist_eval_interval", "-1"],
    )
    trainer = Trainer(args)
    assert trainer.dist_eval_interval == 37


@patch("wandb.init")
def test_dist_eval_interval_zero_and_unset_dir_disable_it(mock_wandb_init, tmp_path):
    args = _parse_args(tmp_path, ["--dist_eval_interval", "0"])
    trainer = Trainer(args)
    assert trainer.dist_eval_interval == 0
    assert trainer.config.dist_eval_dir is None


@patch("wandb.init")
def test_prior_dist_best_metric_without_dist_eval_dir_warns(
    mock_wandb_init, tmp_path, capsys
):
    args = _parse_args(tmp_path, ["--best_metric", "prior/dist/crps"])
    Trainer(args)
    captured = capsys.readouterr()
    assert "prior/dist/crps" in captured.out
    assert "dist_eval_dir is None" in captured.out


@pytest.mark.parametrize(
    "best_metric",
    ["prior/point/cate/npehe", "val/point/cate/npehe", "val/dist/crps"],
)
@patch("wandb.init")
def test_non_prior_dist_best_metric_without_dist_eval_dir_does_not_warn(
    mock_wandb_init, tmp_path, capsys, best_metric
):
    args = _parse_args(tmp_path, ["--best_metric", best_metric])
    Trainer(args)
    captured = capsys.readouterr()
    assert "dist_eval_dir is None" not in captured.out


@patch("wandb.init")
def test_prior_best_metric_with_dist_eval_dir_set_does_not_warn(
    mock_wandb_init, tmp_path, capsys
):
    args = _parse_args(
        tmp_path,
        [
            "--best_metric",
            "prior/dist/crps",
            "--dist_eval_dir",
            str(tmp_path),
        ],
    )
    Trainer(args)
    captured = capsys.readouterr()
    assert "dist_eval_dir is None" not in captured.out
    assert "higher-is-better" not in captured.out
    assert "lower-is-better" not in captured.out
