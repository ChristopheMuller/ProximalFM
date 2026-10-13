"""Smoke-level contracts for trainer wiring."""

from unittest.mock import patch

import pytest
import torch

from eval_datasets.prior_eval.build_prior_frozen_set import BuildConfig, build
from proximalfm.prior._dataset import PriorDataset
from proximalfm.prior._prior_config import DEFAULT_SAMPLED_HP
from proximalfm.train._run import Trainer
from proximalfm.train._train_config import build_parser


def _args(tmp_path, extra=()):
    return build_parser().parse_args(
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
            "--n_query",
            "5",
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
            *extra,
        ]
    )


def _batch(args, mc_cate=4):
    prior = PriorDataset(
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
        min_seq_len=16,
        max_seq_len=17,
        n_query=5,
        mc_cate=mc_cate,
        scm_sampled_hp={
            **DEFAULT_SAMPLED_HP,
            "propensity_shift": None,
            "propensity_scale": None,
        },
        n_jobs=1,
        device="cpu",
    )
    return prior.get_batch()


def test_trainer_runs_a_real_micro_batch_and_writes_checkpoint(tmp_path):
    args = _args(tmp_path, ["--loss_type", "cate_single"])
    trainer = Trainer(args)
    metrics = trainer.run_micro_batch(_batch(args), 0, 1)
    assert metrics and all(
        torch.isfinite(torch.tensor(value)) for value in metrics.values()
    )
    trainer.save_checkpoint("smoke.ckpt")
    checkpoint = tmp_path / "smoke.ckpt"
    assert checkpoint.exists() and "state_dict" in torch.load(
        checkpoint, weights_only=False
    )
    assert trainer._checkpoint_name(1) == "step-000001.ckpt"


def test_noise_aware_loss_uses_mc_batch_metadata(tmp_path):
    args = _args(tmp_path, ["--loss_type", "noise_aware"])
    metrics = Trainer(args).run_micro_batch(_batch(args), 0, 1)
    assert metrics["noise/mean_K"] == 4.0
    assert metrics["noise/frac_fallback"] == 0.0


@pytest.fixture(scope="module")
def frozen_set(tmp_path_factory):
    out = tmp_path_factory.mktemp("frozen_prior")
    build(
        BuildConfig(
            n_datasets=2,
            n_max=16,
            n_q=4,
            mc_cate=4,
            seed=7,
            prior_mc_mem_budget=2_000_000_000,
        ),
        out,
        datasets_per_file=2,
        progress=False,
    )
    return out


@patch("wandb.init")
def test_prior_evaluation_emits_point_and_distribution_metrics(
    _wandb, tmp_path, frozen_set
):
    args = _args(
        tmp_path,
        [
            "--loss_type",
            "cate_single",
            "--prior_eval_dir",
            str(frozen_set),
            "--prior_eval_interval",
            "1",
        ],
    )
    metrics = Trainer(args).prior_validate()
    assert "prior_point/cate/npehe" in metrics
    assert "prior_dist/crps" in metrics
