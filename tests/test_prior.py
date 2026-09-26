"""A few integration contracts for the synthetic prior."""

import json

import pytest
import torch

from proximalfm.prior._dataset import SCMPrior
from proximalfm.prior._genload import LoadPriorDataset, SavePriorDataset


class Args:
    def __init__(self, **values):
        self.__dict__.update(values)


def _prior(*, mc_targets_on_context=False, mc_cate=4):
    return SCMPrior(
        batch_size=2,
        batch_size_per_gp=2,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=2,
        max_dim_W=3,
        min_dim_Z=2,
        max_dim_Z=3,
        min_dim_X=2,
        max_dim_X=3,
        min_seq_len=12,
        max_seq_len=13,
        n_query=5,
        mc_cate=mc_cate,
        mc_targets_on_context=mc_targets_on_context,
        n_jobs=1,
    )


def test_batch_has_fixed_query_rows_and_valid_mc_targets():
    batch = _prior().get_batch()
    X, W, Z, A, y_obs, mu0, mu1 = (batch[i] for i in (1, 2, 3, 4, 6, 9, 10))
    seq_lens, train_sizes, s2, k_eff = batch[15:]

    assert X.shape[:2] == W.shape[:2] == Z.shape[:2] == A.shape[:2]
    assert y_obs.shape == mu0.shape == mu1.shape
    assert torch.equal(seq_lens - train_sizes, torch.full_like(train_sizes, 5))
    assert (k_eff == 4).all()
    for row, context in enumerate(train_sizes.tolist()):
        assert mu0[row, :context].isnan().all()
        assert mu1[row, :context].isnan().all()
        assert s2[row, :context].isnan().all()
        assert mu0[row, context:].isfinite().all()
        assert mu1[row, context:].isfinite().all()
        assert (s2[row, context:] >= 0).all()


def test_full_replication_has_targets_and_noise_for_every_row():
    batch = _prior(mc_targets_on_context=True).get_batch()
    for value in (batch[9], batch[10], batch[17]):
        assert value.isfinite().all()
    assert (batch[17] >= 0).all()


def test_invalid_query_count_is_rejected():
    for value in (0, -1, 1.5, True, "5"):
        with pytest.raises(ValueError, match="n_query"):
            SCMPrior(n_query=value)


def test_saved_batches_round_trip_the_mc_contract(tmp_path):
    args = Args(
        save_dir=str(tmp_path),
        batch_size=2,
        batch_size_per_gp=2,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=2,
        max_dim_W=3,
        min_dim_Z=2,
        max_dim_Z=3,
        min_dim_X=2,
        max_dim_X=3,
        min_seq_len=12,
        max_seq_len=13,
        log_seq_len=False,
        seq_len_per_gp=False,
        n_query=5,
        replay_small=False,
        prior_type="mlp_scm",
        mc_cate=3,
        mc_targets_on_context=False,
        num_batches=1,
        resume_from=0,
        n_jobs=1,
        num_threads_per_generate=1,
        device="cpu",
    )
    SavePriorDataset(args).run()
    with (tmp_path / "metadata.json").open() as handle:
        assert json.load(handle)["mc_targets_on_context"] is False

    batch = next(iter(LoadPriorDataset(tmp_path, batch_size=2, max_batches=1)))
    mu0, mu1, train_sizes, s2, k_eff = (
        batch[9],
        batch[10],
        batch[16],
        batch[17],
        batch[18],
    )
    assert (k_eff == 3).all()
    for row, context in enumerate(train_sizes.tolist()):
        assert mu0[row, :context].isnan().all() and mu1[row, :context].isnan().all()
        assert s2[row, :context].isnan().all()
        assert (
            mu0[row, context:].isfinite().all() and mu1[row, context:].isfinite().all()
        )
