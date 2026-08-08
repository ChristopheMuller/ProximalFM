import math
import os
from pathlib import Path

import pytest
from proximalfm.train._train_config import build_parser
from proximalfm.train import _run
from proximalfm.train._run import Trainer
from proximalfm.train._dist_eval import UnsupportedLossTypeError
from proximalfm.prior._dataset import PriorDataset
from proximalfm.prior._prior_config import DEFAULT_SAMPLED_HP
from posterior_eval.build_prior_frozen_set import BuildConfig, build
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


@pytest.fixture(scope="module")
def tiny_frozen_set(tmp_path_factory):
    """A tiny real frozen prior set for the ``--prior_eval_dir`` tests.

    Same shape as ``tests/test_posterior_eval.py``'s ``frozen`` fixture, just
    smaller: these tests only need the trainer wiring to fire, not meaningful
    numbers.
    """
    out = tmp_path_factory.mktemp("prior_eval_frozen_set")
    cfg = BuildConfig(
        n_datasets=2,
        n_max=16,
        n_q=4,
        mc_cate=4,
        seed=20260805,
        prior_mc_mem_budget=2_000_000_000,
    )
    build(cfg, out, datasets_per_file=2, progress=False)
    return out


@pytest.fixture(scope="module")
def tiny_replicate_set(tmp_path_factory):
    """A tiny realistic replicate set for the ``--val_eval_dir`` tests.

    Built with ``--mc_cate > 1``, so it carries a Monte-Carlo CEPO target and a
    ``frozen_meta.pt`` sidecar -- the two things that make a realistic set
    scorable distributionally at all (ROADMAP B1-B3).
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "synthetic"))
    from build_proximal_datasets import build as build_realistic
    from build_proximal_datasets import parse_args as parse_realistic_args

    root = tmp_path_factory.mktemp("val_eval_set")
    build_realistic(
        parse_realistic_args(
            [
                "--name",
                "toy",
                "--out_root",
                str(root),
                "--n_datasets",
                "2",
                "--seq_len",
                "20",
                "--prop_train",
                "0.75",
                "--mc_cate",
                "4",
                "--mechanisms",
                "1",
                "--seed",
                "20260807",
            ]
        )
    )
    return root / "toy"


def _parse_args(tmp_path, extra):
    parser = build_parser()
    return parser.parse_args(_dummy_args_list(tmp_path) + extra)


# The prior samples propensity_shift ~ U(-2.5, 2.5) and a log-scaled
# propensity_scale, so an unlucky draw makes P(A=1|.) near-degenerate and every
# context row lands in one arm. run_micro_batch then *correctly* returns None
# for the whole micro batch, and any test that asserts on its metrics fails --
# not because the code under test is wrong, but because that draw came up.
# Measured over the config below, that happens with probability ~0.8% on
# average (up to ~13% for some hyperparameter draws), which is exactly the
# order of flakiness that makes these tests depend on the global RNG stream.
#
# Dropping the two entries (falsy values are skipped by HpSamplerList) leaves
# sample_A on its shift=0 / scale=1 defaults, i.e. sigmoid of standardized
# logits. Overlap is then a property of the prior rather than of the draw: with
# a 24-row context the single-arm probability falls to ~1e-7.
_BALANCED_PROPENSITY_HP = {
    **DEFAULT_SAMPLED_HP,
    "propensity_shift": None,
    "propensity_scale": None,
}


def _overlap_prior(args, *, mc_cate, mc_targets_on_context=False):
    """A prior whose context carries both treatment arms by construction."""
    return PriorDataset(
        batch_size=2,
        batch_size_per_gp=2,
        min_dim_U=2,
        max_dim_U=args.max_dim_U,
        min_dim_W=2,
        max_dim_W=args.max_dim_W,
        min_dim_Z=2,
        max_dim_Z=args.max_dim_Z,
        min_dim_X=2,
        max_dim_X=args.max_dim_X,
        min_seq_len=None,
        max_seq_len=64,
        # A wide context, so both arms are well populated under a balanced
        # propensity rather than merely likely to be.
        min_train_size=24,
        max_train_size=25,
        prior_type="mlp_scm",
        mc_cate=mc_cate,
        mc_targets_on_context=mc_targets_on_context,
        scm_sampled_hp=_BALANCED_PROPENSITY_HP,
        n_jobs=1,
        device="cpu",
    )


def _batch_with_both_arms(prior):
    """Draw a batch and state the two-arm precondition the trainer needs.

    Asserted rather than assumed: if a future prior change reintroduces
    near-degenerate propensities, this fails where the cause is legible instead
    of downstream on an empty metrics dict.
    """
    batch = prior.get_batch()
    train_size = int(batch[16][0].item())
    A_context = batch[4][:, :train_size].squeeze(-1)
    assert bool((A_context == 0).any(dim=1).all()), "a context episode is all-treated"
    assert bool((A_context == 1).any(dim=1).all()), "a context episode is all-control"
    return batch


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

    prior = _overlap_prior(
        dummy_args, mc_cate=3, mc_targets_on_context=mc_targets_on_context
    )
    batch = _batch_with_both_arms(prior)

    MU0, train_sizes = batch[9], batch[16]
    train_size = int(train_sizes[0].item())
    # Guard that the parametrisation actually exercises both regimes.
    assert bool(MU0[:, :train_size].isnan().all()) is not mc_targets_on_context

    results = trainer.run_micro_batch(batch, 0, 1)

    assert results, "expected the micro batch to produce metrics"
    for key, value in results.items():
        assert math.isfinite(value), f"{key} is not finite: {value}"


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

    batch = _batch_with_both_arms(_overlap_prior(dummy_args, mc_cate=3))

    results = trainer.run_micro_batch(batch, 0, 1)

    assert results, "expected the micro batch to produce metrics"
    assert "loss/cate_dist" in results
    assert "loss/cate_mse" in results
    assert results["loss/cate_mse_weighted"] == pytest.approx(
        dummy_args.lambda_cate * results["loss/cate_mse"]
    )
    for key, value in results.items():
        assert math.isfinite(value), f"{key} is not finite: {value}"


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

    prior = _overlap_prior(
        dummy_args, mc_cate=8, mc_targets_on_context=mc_targets_on_context
    )
    batch = _batch_with_both_arms(prior)

    S2, train_sizes = batch[17], batch[16]
    train_size = int(train_sizes[0].item())
    assert bool(S2[:, :train_size].isnan().all()) is not mc_targets_on_context

    results = trainer.run_micro_batch(batch, 0, 1)

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

    batch = _batch_with_both_arms(_overlap_prior(dummy_args, mc_cate=1))
    results = trainer.run_micro_batch(batch, 0, 1)

    assert results["noise/frac_fallback"] == 1.0
    assert results["noise/mean_s"] == 0.0
    for key, value in results.items():
        assert math.isfinite(value), f"{key} is not finite: {value}"


def test_non_finite_loss_skips_the_micro_batch(dummy_args):
    """An inf/NaN loss must never reach backward().

    One inf turns every parameter NaN, and from then on the whole logged
    surface -- prior_dist/*, prior_point/*, val_point/* -- reads nan with no
    trace of which step did it.
    """
    dummy_args.loss_type = "cate_single_noise_aware"
    trainer = Trainer(dummy_args)

    batch = _batch_with_both_arms(_overlap_prior(dummy_args, mc_cate=4))

    with patch.object(
        _run, "compute_cate_noise_aware_nll", return_value=torch.tensor(float("inf"))
    ):
        with pytest.warns(RuntimeWarning, match="Non-finite"):
            results = trainer.run_micro_batch(batch, 0, 1)

    assert results is None
    assert all(p.grad is None for p in trainer.model.parameters())
    for param in trainer.model.parameters():
        assert torch.isfinite(param).all()


def test_non_finite_gradient_skips_the_optimizer_step(dummy_args):
    """A finite loss can still produce NaN gradients, and clipping does not help.

    ``clip_grad_norm_`` multiplies by a NaN scale and leaves the gradients NaN.
    On CUDA + AMP ``GradScaler`` would skip the step; the scaler is disabled off
    CUDA, so the explicit guard is the only thing standing between one bad batch
    and a permanently dead model.
    """
    dummy_args.loss_type = "cate_single_noise_aware"
    trainer = Trainer(dummy_args)

    batch = _batch_with_both_arms(_overlap_prior(dummy_args, mc_cate=4))
    before = [p.detach().clone() for p in trainer.model.parameters()]

    with patch.object(
        _run.nn.utils, "clip_grad_norm_", return_value=torch.tensor(float("nan"))
    ):
        with pytest.warns(RuntimeWarning, match="Non-finite gradient norm"):
            trainer.run_batch(batch)

    for param, original in zip(trainer.model.parameters(), before):
        assert torch.equal(param, original), "the optimizer stepped anyway"


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
        ["--point_val_interval", "37", "--prior_eval_interval", "-1"],
    )
    trainer = Trainer(args)
    assert trainer.prior_eval_interval == 37


@patch("wandb.init")
def test_dist_eval_interval_zero_and_unset_dir_disable_it(mock_wandb_init, tmp_path):
    args = _parse_args(tmp_path, ["--prior_eval_interval", "0"])
    trainer = Trainer(args)
    assert trainer.prior_eval_interval == 0
    assert trainer.config.prior_eval_dir is None


@pytest.mark.parametrize(
    "best_metric,missing_flag",
    [
        ("prior_dist/crps", "prior_eval_dir"),
        # prior_point/* is produced by the dist pass too, so it needs the same
        # flag -- exactly as _monitored_surface_ran routes it.
        ("prior_point/cate/npehe", "prior_eval_dir"),
        ("val_dist/crps", "val_eval_dir"),
        # And val_point/* is a side effect of the SAME val_eval_dir pass, not
        # of a separate --val_dir -- that flag no longer exists.
        ("val_point/cate/npehe", "val_eval_dir"),
    ],
)
@patch("wandb.init")
def test_best_metric_on_a_pass_that_never_runs_warns(
    mock_wandb_init, tmp_path, capsys, best_metric, missing_flag
):
    """Naming a metric whose producing pass is disabled is silent otherwise."""
    args = _parse_args(tmp_path, ["--best_metric", best_metric])
    Trainer(args)
    captured = capsys.readouterr()
    assert best_metric in captured.out
    assert f"{missing_flag} is None" in captured.out


@patch("wandb.init")
def test_prior_best_metric_with_dist_eval_dir_set_does_not_warn(
    mock_wandb_init, tmp_path, capsys, tiny_frozen_set
):
    args = _parse_args(
        tmp_path,
        [
            "--best_metric",
            "prior_dist/crps",
            "--loss_type",
            "cate_single",
            "--prior_eval_dir",
            str(tiny_frozen_set),
        ],
    )
    Trainer(args)
    captured = capsys.readouterr()
    assert "prior_eval_dir is None" not in captured.out
    assert "higher-is-better" not in captured.out
    assert "lower-is-better" not in captured.out


# --------------------------------------------------------------------------
# Step 6c: prior_validate(), single process
# --------------------------------------------------------------------------


@patch("wandb.init")
def test_bins_po_with_dist_eval_dir_raises_at_construction(
    mock_wandb_init, tmp_path, tiny_frozen_set
):
    """The step-5 reason (two marginals don't identify a posterior of their
    difference) must be fatal at construction, not discovered at the first
    prior_validate() pass hundreds of steps into training."""
    args = _parse_args(tmp_path, ["--prior_eval_dir", str(tiny_frozen_set)])
    assert args.loss_type == "bins_po"
    with pytest.raises(UnsupportedLossTypeError):
        Trainer(args)


@patch("wandb.init")
def test_dist_validate_is_a_noop_when_dir_unset(mock_wandb_init, dummy_args):
    trainer = Trainer(dummy_args)
    assert trainer.prior_validate() == {}


@patch("wandb.init")
def test_post_step_unchanged_when_dist_eval_dir_unset(mock_wandb_init, dummy_args):
    """The clean no-op: with --prior_eval_dir unset, _post_step never produces
    a prior_dist/* key, exactly as before this feature existed."""
    trainer = Trainer(dummy_args)
    results = trainer._post_step({}, prior_time=0.0, train_time=0.0)
    assert not any(k.startswith(("prior_dist/", "prior_point/")) for k in results)


@patch("wandb.init")
def test_dist_eval_lands_exactly_on_the_gated_steps(
    mock_wandb_init, tmp_path, tiny_frozen_set
):
    args = _parse_args(
        tmp_path,
        [
            "--loss_type",
            "cate_single",
            "--prior_eval_dir",
            str(tiny_frozen_set),
            "--prior_eval_interval",
            "2",
        ],
    )
    args.lambda_cate = 0.1
    trainer = Trainer(args)

    fired = {}
    for _ in range(5):
        results = trainer._post_step({}, prior_time=0.0, train_time=0.0)
        fired[trainer.curr_step] = "prior_dist/crps" in results

    # curr_step == 1 always fires (mirrors validate()'s early sanity check),
    # then every multiple of --prior_eval_interval.
    assert fired == {1: True, 2: True, 3: False, 4: True, 5: False}


@patch("wandb.init")
def test_dist_eval_pass_emits_both_prior_surfaces(
    mock_wandb_init, tmp_path, tiny_frozen_set
):
    args = _parse_args(
        tmp_path,
        [
            "--loss_type",
            "cate_single",
            "--prior_eval_dir",
            str(tiny_frozen_set),
            "--prior_eval_interval",
            "1",
        ],
    )
    args.lambda_cate = 0.1
    trainer = Trainer(args)

    results = trainer._post_step({}, prior_time=0.0, train_time=0.0)

    assert "prior_point/cate/npehe" in results
    assert "prior_dist/crps" in results
    assert "Charts/prior_eval_time" in results


@patch("wandb.init")
def test_prior_best_metric_only_saves_best_on_steps_the_dist_surface_ran(
    mock_wandb_init, tmp_path, tiny_frozen_set
):
    """prior_eval_interval coarser than point_val_interval: a prior_dist/* best
    metric must not warn (or attempt a comparison) on the steps where only
    point validation ran."""
    args = _parse_args(
        tmp_path,
        [
            "--loss_type",
            "cate_single",
            "--best_metric",
            "prior_dist/crps",
            "--best_metric_mode",
            "min",
            "--prior_eval_dir",
            str(tiny_frozen_set),
            "--prior_eval_interval",
            "3",
        ],
    )
    args.lambda_cate = 0.1
    trainer = Trainer(args)

    trainer._post_step({}, prior_time=0.0, train_time=0.0)  # step 1: dist fires
    assert trainer.best_metric_value is not None
    first_best = trainer.best_metric_value

    trainer._post_step({}, prior_time=0.0, train_time=0.0)  # step 2: dist idle
    assert trainer.best_metric_value == first_best


@patch("wandb.init")
def test_val_point_best_metric_follows_val_dist_interval_not_dist_eval_interval(
    mock_wandb_init, tmp_path, capsys, tiny_replicate_set, tiny_frozen_set
):
    """val_point/* is a side effect of the val_eval_dir pass, gated by
    val_eval_interval -- not by prior_eval_interval, the PRIOR surface's
    cadence, even though both passes run in the same _post_step call."""
    args = _parse_args(
        tmp_path,
        [
            "--loss_type",
            "cate_single",
            "--best_metric",
            "val_point/cate/npehe",
            "--best_metric_mode",
            "min",
            "--val_eval_dir",
            str(tiny_replicate_set),
            "--val_eval_interval",
            "3",
            "--prior_eval_dir",
            str(tiny_frozen_set),
            "--prior_eval_interval",
            "1",
        ],
    )
    args.lambda_cate = 0.1
    trainer = Trainer(args)

    trainer._post_step({}, prior_time=0.0, train_time=0.0)  # step 1: both fire
    assert trainer.best_metric_value is not None
    first_best = trainer.best_metric_value

    trainer._post_step(
        {}, prior_time=0.0, train_time=0.0
    )  # step 2: val_dist idle, prior_dist fires
    assert trainer.best_metric_value == first_best
    captured = capsys.readouterr()
    assert "not found in validation results" not in captured.out

    # step 3: val_dist fires again (3 % val_eval_interval == 0); must not
    # error and must still be comparing the same val_point/* key.
    trainer._post_step({}, prior_time=0.0, train_time=0.0)
    assert trainer.best_metric_value is not None


# --------------------------------------------------------------------------
# Step 6d: DDP shard + sync
# --------------------------------------------------------------------------


@patch("wandb.init")
def test_dist_validate_passes_its_shard_identity_through(
    mock_wandb_init, tmp_path, tiny_frozen_set, monkeypatch
):
    """The pass must be sharded by *this rank's* identity and synced, not run
    whole on rank 0. Single-process tests cannot observe the deadlock this
    prevents, so pin the arguments instead: reintroducing a master-only pass
    would show up here."""
    args = _parse_args(
        tmp_path,
        [
            "--loss_type",
            "cate_single",
            "--prior_eval_dir",
            str(tiny_frozen_set),
            "--prior_eval_interval",
            "1",
        ],
    )
    args.lambda_cate = 0.1
    trainer = Trainer(args)

    seen = {}
    real = _run.score_frozen_set

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(_run, "score_frozen_set", spy)
    # Pretend to be rank 1 of a world of 4; the trainer must forward that.
    trainer.ddp, trainer.ddp_rank, trainer.ddp_world_size = False, 1, 4
    trainer.prior_validate()

    assert seen["rank"] == 1
    assert seen["world_size"] == 4
    assert seen["sync_ddp"] is False


@patch("wandb.init")
def test_dist_eval_gate_does_not_depend_on_rank(
    mock_wandb_init, tmp_path, tiny_frozen_set
):
    """Every rank must reach prior_validate() on the same steps: the pass enters
    collectives, so a gate that is true on one rank and false on another hangs
    the group. Assert the gate holds with master_process cleared."""
    args = _parse_args(
        tmp_path,
        [
            "--loss_type",
            "cate_single",
            "--prior_eval_dir",
            str(tiny_frozen_set),
            "--prior_eval_interval",
            "1",
        ],
    )
    args.lambda_cate = 0.1
    trainer = Trainer(args)
    trainer.master_process = False  # a non-zero rank

    results = trainer._post_step({}, prior_time=0.0, train_time=0.0)
    assert "prior_dist/crps" in results


# --------------------------------------------------------------------------
# Step 6e: diag/sigma on the width axis
# --------------------------------------------------------------------------


@patch("wandb.init")
def test_dist_validate_emits_sigma_on_the_width_axis(
    mock_wandb_init, tmp_path, tiny_frozen_set
):
    """sigma floors the learned width, so it has to be readable next to the
    width it floors: same dict, same cadence, same units. A plot against
    loss/hl_sigma is not a comparison (micro-batch cadence, head units)."""
    args = _parse_args(
        tmp_path,
        [
            "--loss_type",
            "cate_single",
            "--prior_eval_dir",
            str(tiny_frozen_set),
            "--prior_eval_interval",
            "1",
            "--hl_start_sigma",
            "0.2",
            "--hl_final_sigma",
            "0.2",
        ],
    )
    args.lambda_cate = 0.1
    trainer = Trainer(args)

    metrics = trainer.prior_validate()

    # Both keys, one dict: that is the whole point of the step.
    assert "diag/sigma" in metrics
    assert "prior_dist/mean_sd" in metrics

    # And in mean_sd's units, not the head's: sigma times the same head ->
    # frozen-space factor the scored grid carried.
    assert metrics["diag/sigma"] == pytest.approx(
        trainer.get_hl_sigma() * metrics["prior_dist/grid_scale"]
    )
    # At n_ctx == n_max the factor is 1, so this also pins the plain value.
    assert metrics["diag/sigma"] == pytest.approx(0.2, abs=1e-4)


# ---------------------------------------------------------------------------
# val_dist/* -- the realistic replicate surface (ROADMAP B4)
# ---------------------------------------------------------------------------


def _val_eval_args(tmp_path, replicate_set, extra=None):
    args = _parse_args(
        tmp_path,
        [
            "--loss_type",
            "cate_single",
            "--val_eval_dir",
            str(replicate_set),
            "--val_eval_interval",
            "1",
        ]
        + (extra or []),
    )
    args.lambda_cate = 0.1
    return args


@patch("wandb.init")
def test_val_dist_pass_emits_both_val_surfaces(
    mock_wandb_init, tmp_path, tiny_replicate_set
):
    """One pass over --val_eval_dir emits val_point/* AND val_dist/*.

    Mirrors --prior_eval_dir producing both prior_point/* and prior_dist/* from
    one per-episode pass: point metrics are a side effect of the same forward
    pass that produces the distributional ones (score_episode's
    CausalEvaluator fan-out), not a second, separately-gated pass.
    """
    trainer = Trainer(_val_eval_args(tmp_path, tiny_replicate_set))
    results = trainer._post_step({}, prior_time=0.0, train_time=0.0)

    assert "val_dist/crps" in results
    assert "val_dist/sd_over_rmse" in results
    assert "val_dist/eval_rows" in results
    assert "val_dist/n_ctx" in results
    assert "val_point/cate/npehe" in results
    assert "val_point/cate/rmse_pooled" in results
    assert "Charts/val_eval_time" in results
    # No leakage into the prior namespace.
    assert not any(k.startswith(("prior_dist/", "prior_point/")) for k in results)


@patch("wandb.init")
def test_val_dist_and_prior_dist_do_not_overwrite_each_other(
    mock_wandb_init, tmp_path, tiny_replicate_set, tiny_frozen_set
):
    """Both distributional passes in one step, on different data, both intact.

    They are merged into one dict with ``update()``, so a shared prefix would
    silently leave only whichever ran last.
    """
    args = _val_eval_args(
        tmp_path,
        tiny_replicate_set,
        ["--prior_eval_dir", str(tiny_frozen_set), "--prior_eval_interval", "1"],
    )
    trainer = Trainer(args)
    results = trainer._post_step({}, prior_time=0.0, train_time=0.0)

    assert "prior_dist/crps" in results and "val_dist/crps" in results
    assert "prior_point/cate/npehe" in results
    # Different sets, so the numbers must not be the same object/value.
    assert results["prior_dist/crps"] != results["val_dist/crps"]


@patch("wandb.init")
def test_val_dist_lands_exactly_on_the_gated_steps(
    mock_wandb_init, tmp_path, tiny_replicate_set
):
    args = _val_eval_args(tmp_path, tiny_replicate_set)
    args.val_eval_interval = 2
    trainer = Trainer(args)
    trainer.val_eval_interval = 2

    fired = {}
    for _ in range(5):
        results = trainer._post_step({}, prior_time=0.0, train_time=0.0)
        fired[trainer.curr_step] = "val_dist/crps" in results

    assert fired == {1: True, 2: True, 3: False, 4: True, 5: False}


@patch("wandb.init")
def test_val_dist_interval_minus_one_follows_point_val_interval(
    mock_wandb_init, tmp_path, tiny_replicate_set
):
    args = _val_eval_args(
        tmp_path,
        tiny_replicate_set,
        ["--val_eval_interval", "-1", "--point_val_interval", "7"],
    )
    assert Trainer(args).val_eval_interval == 7


@patch("wandb.init")
def test_a_set_without_a_sidecar_is_refused_at_construction(
    mock_wandb_init, tmp_path, tiny_replicate_set
):
    """The absence of frozen_meta.pt is the gate, and it must fire early.

    Pointing --val_eval_dir at a plain val/ directory (or at a mc_cate=1 set)
    is the mistake this prevents: those targets are sampled ITEs, and scoring a
    posterior against them is *minimised by an over-wide posterior* rather than
    erroring. Failing at __init__ beats failing hundreds of steps in.
    """
    args = _val_eval_args(tmp_path, tiny_replicate_set / "val")
    with pytest.raises(FileNotFoundError, match="frozen_meta.pt"):
        Trainer(args)


@patch("wandb.init")
def test_val_dist_best_metric_only_saves_best_on_steps_that_pass_ran(
    mock_wandb_init, tmp_path, tiny_replicate_set
):
    """Routing: a val_dist/* best_metric must key off the val_dist pass, not
    the prior pass, even when the prior pass ran and this one didn't."""
    args = _val_eval_args(
        tmp_path,
        tiny_replicate_set,
        [
            "--best_metric",
            "val_dist/crps",
            "--best_metric_mode",
            "min",
            "--val_eval_interval",
            "2",
        ],
    )
    trainer = Trainer(args)
    assert trainer._monitored_surface_ran(ran_prior_eval=False, ran_val_eval=True)
    assert not trainer._monitored_surface_ran(ran_prior_eval=True, ran_val_eval=False)


def test_atomic_checkpoint_save(dummy_args):
    trainer = Trainer(dummy_args)
    ckpt_name = "test_atomic.ckpt"
    with patch("os.replace", wraps=os.replace) as mock_replace:
        trainer.save_checkpoint(ckpt_name)
        target_path = os.path.join(dummy_args.checkpoint_dir, ckpt_name)
        tmp_path_expected = f"{target_path}.tmp"
        mock_replace.assert_called_once_with(tmp_path_expected, target_path)
    assert os.path.exists(target_path)
    loaded = torch.load(target_path, weights_only=False)
    assert "state_dict" in loaded
