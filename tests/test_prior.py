"""A few integration contracts for the synthetic prior."""

import json
from unittest.mock import patch

import pytest
import torch

from proximalfm.prior._dataset import PriorDataset, SCMPrior, _mc_reduce
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


@pytest.mark.parametrize("dim_u", [1, 3])
def test_ablation_prior_preserves_observations_and_target_prefixes(dim_u):
    context, queries = 64, 16
    batches, draws = [], []
    for k in (1, 16, 250):
        prior = PriorDataset(
            prior_type="ablation_scm",
            batch_size=1,
            min_dim_U=dim_u,
            max_dim_U=dim_u,
            min_dim_X=1,
            max_dim_X=1,
            min_dim_W=3,
            max_dim_W=3,
            min_dim_Z=4,
            max_dim_Z=4,
            max_seq_len=context,
            n_query=queries,
            mc_cate=k,
            n_jobs=1,
        )
        with torch.random.fork_rng(devices=[]), patch(
            "proximalfm.prior._ablation_dataset._mc_reduce", wraps=_mc_reduce
        ) as reduce:
            torch.manual_seed(1234)
            batch = prior.get_batch()
        assert tuple(t.item() for t in batch[11:17]) == (dim_u, 1, 3, 4, 80, 64)
        assert batch[-1].item() == k
        assert torch.all((batch[5] >= 0.1) & (batch[5] <= 0.9))
        torch.testing.assert_close(
            batch[6], torch.where(batch[4] == 1, batch[8], batch[7])
        )
        for i in (9, 10, 17):
            assert batch[i][0, :context].isnan().all()
            assert batch[i][0, context:].isfinite().all()
        paired_draws = tuple(t.reshape(queries, k) for t in reduce.call_args.args[:3])
        if k == 1:
            assert not batch[17][0, context:].any()
        else:
            expected_s2 = paired_draws[2].var(dim=1, unbiased=True) / k
            assert (expected_s2 > 0).any()
            torch.testing.assert_close(batch[17][0, context:], expected_s2)
        batches.append(batch)
        draws.append(paired_draws)

    for batch, prefix in zip(batches[:-1], draws[:-1]):
        for observed, reference in zip(batch[:9], batches[-1][:9]):
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        for short, full in zip(prefix, draws[-1]):
            torch.testing.assert_close(short, full[:, : short.shape[1]], rtol=0, atol=0)


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
