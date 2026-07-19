import contextlib
import math
import random

import numpy as np
import torch

import proximalfm.prior._dataset as _dataset_mod
from proximalfm.prior._dataset import SCMPrior, PriorDataset, DisablePrinting
from proximalfm.prior._reg2cls import BlockProfile, dominates
from proximalfm.prior._genload import (
    SavePriorDataset,
    LoadPriorDataset,
    dense2sparse,
    sparse2dense,
    SliceNestedTensor,
    cat_slice_nested_tensors,
)
from proximalfm.prior._prior_config import DEFAULT_FIXED_HP


import pytest

FAST_FIXED_HP = DEFAULT_FIXED_HP
FAST_MCMC_CATE = 2


class ArgsMock:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_scm_prior_initialization():
    prior = SCMPrior(batch_size=8, min_dim_U=2, max_dim_U=5)
    assert prior.batch_size == 8
    assert prior.min_dim_U == 2
    assert prior.max_dim_U == 5


def test_scm_prior_get_batch():
    prior = SCMPrior(
        batch_size=4,
        batch_size_per_gp=2,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=2,
        max_dim_W=4,
        min_dim_Z=2,
        max_dim_Z=4,
        min_dim_X=3,
        max_dim_X=5,
        min_seq_len=20,
        max_seq_len=30,
        min_train_size=0.1,
        max_train_size=0.8,
        prior_type="mlp_scm",
        mcmc_cate=FAST_MCMC_CATE,
        fixed_hp=FAST_FIXED_HP,
    )
    (
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
        seq_lens,
        train_sizes,
    ) = prior.get_batch()

    assert X.shape[0] == 4
    assert W.shape[0] == 4
    assert Z.shape[0] == 4
    assert A.shape[0] == 4
    assert Y_obs.shape[0] == 4
    assert Y0.shape[0] == 4
    assert Y1.shape[0] == 4
    assert MU0.shape[0] == 4
    assert MU1.shape[0] == 4

    assert d_U.shape[0] == 4
    assert d_X.shape[0] == 4
    assert d_W.shape[0] == 4
    assert d_Z.shape[0] == 4

    assert seq_lens.shape[0] == 4
    assert train_sizes.shape[0] == 4


def test_scm_prior_mix_scm():
    prior = SCMPrior(
        batch_size=4,
        batch_size_per_gp=4,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=2,
        max_dim_W=4,
        min_dim_Z=2,
        max_dim_Z=4,
        min_dim_X=3,
        max_dim_X=5,
        min_seq_len=20,
        max_seq_len=30,
        min_train_size=0.1,
        max_train_size=0.8,
        prior_type="mix_scm",
        mcmc_cate=FAST_MCMC_CATE,
        fixed_hp=FAST_FIXED_HP,
    )
    (
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
        seq_lens,
        train_sizes,
    ) = prior.get_batch()
    assert X.shape[0] == 4


def test_scm_prior_tree_scm():
    prior = SCMPrior(
        batch_size=2,
        batch_size_per_gp=2,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=2,
        max_dim_W=4,
        min_dim_Z=2,
        max_dim_Z=4,
        min_dim_X=3,
        max_dim_X=5,
        min_seq_len=20,
        max_seq_len=30,
        min_train_size=0.1,
        max_train_size=0.8,
        prior_type="tree_scm",
        mcmc_cate=FAST_MCMC_CATE,
        fixed_hp=FAST_FIXED_HP,
    )
    (
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
        seq_lens,
        train_sizes,
    ) = prior.get_batch()
    assert X.shape[0] == 2


def test_scm_prior_seq_len_per_gp():
    prior = SCMPrior(
        batch_size=4,
        batch_size_per_gp=2,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=2,
        max_dim_W=4,
        min_dim_Z=2,
        max_dim_Z=4,
        min_dim_X=3,
        max_dim_X=5,
        min_seq_len=20,
        max_seq_len=30,
        min_train_size=0.1,
        max_train_size=0.8,
        seq_len_per_gp=True,
        prior_type="mlp_scm",
        mcmc_cate=FAST_MCMC_CATE,
        fixed_hp=FAST_FIXED_HP,
    )
    (
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
        seq_lens,
        train_sizes,
    ) = prior.get_batch()

    # When seq_len_per_gp=True, the tensors should be nested tensors
    assert U.is_nested
    assert X.is_nested
    assert W.is_nested
    assert Z.is_nested
    assert A.is_nested
    assert Y_obs.is_nested
    assert Y0.is_nested
    assert Y1.is_nested
    assert MU0.is_nested
    assert MU1.is_nested

    assert X.size(0) == 4
    assert len(seq_lens) == 4


def test_prior_dataset():
    dataset = PriorDataset(
        batch_size=2,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=2,
        max_dim_W=4,
        min_dim_Z=2,
        max_dim_Z=4,
        min_dim_X=3,
        max_dim_X=5,
        min_seq_len=20,
        max_seq_len=30,
        prior_type="mlp_scm",
        mcmc_cate=FAST_MCMC_CATE,
        scm_fixed_hp=FAST_FIXED_HP,
    )

    (
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
        seq_lens,
        train_sizes,
    ) = next(iter(dataset))

    assert X.shape[0] == 2


def test_save_and_load_prior_dataset_seq_len_per_gp(tmp_path):
    args = ArgsMock(
        save_dir=str(tmp_path),
        batch_size=4,
        batch_size_per_gp=2,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=4,
        max_dim_W=4,
        min_dim_Z=4,
        max_dim_Z=4,
        min_dim_X=5,
        max_dim_X=5,
        min_seq_len=20,
        max_seq_len=30,
        log_seq_len=False,
        seq_len_per_gp=True,
        min_train_size=5,
        max_train_size=15,
        replay_small=False,
        prior_type="mlp_scm",
        mcmc_cate=2,
        num_batches=2,
        resume_from=0,
        n_jobs=1,
        num_threads_per_generate=1,
        device="cpu",
    )

    saver = SavePriorDataset(args)
    saver.run()

    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "batch_000000.pt").exists()
    assert (tmp_path / "batch_000001.pt").exists()

    loader = LoadPriorDataset(data_dir=str(tmp_path), batch_size=3, max_batches=2)

    batches = []
    try:
        while True:
            batches.append(next(loader))
    except StopIteration:
        pass

    assert len(batches) == 3

    batch1 = batches[0]
    assert batch1[0].is_nested


def test_dense_sparse_conversions():
    dense = torch.tensor(
        [[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 0.0, 0.0], [6.0, 7.0, 8.0, 9.0]]
    )
    row_lengths = torch.tensor([3, 2, 4])

    sparse = dense2sparse(dense, row_lengths)
    expected_sparse = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert torch.allclose(sparse, expected_sparse)

    recovered_dense = sparse2dense(sparse, row_lengths, max_len=4)
    assert torch.allclose(recovered_dense, dense)

    # Test different data type
    dense_int = torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long)
    row_lengths_int = torch.tensor([2, 1])

    sparse_int = dense2sparse(dense_int, row_lengths_int, dtype=torch.long)
    expected_sparse_int = torch.tensor([1, 2, 3], dtype=torch.long)
    assert torch.equal(sparse_int, expected_sparse_int)

    recovered_dense_int = sparse2dense(
        sparse_int, row_lengths_int, max_len=3, dtype=torch.long
    )
    assert torch.equal(recovered_dense_int, dense_int)


def test_slice_nested_tensor():
    t1 = torch.ones(5, 3)
    t2 = torch.zeros(4, 3)
    nt = torch.nested.nested_tensor([t1, t2])

    snt = SliceNestedTensor(nt)

    # Test length
    assert len(snt) == 2

    # Test indexing
    assert torch.equal(snt[0], t1)
    assert torch.equal(snt[1], t2)

    # Test slicing
    sliced = snt[1:]
    assert isinstance(sliced, SliceNestedTensor)
    assert len(sliced) == 1
    assert torch.equal(sliced[0], t2)


def test_cat_slice_nested_tensors():
    t1 = torch.ones(5, 3)
    t2 = torch.zeros(4, 3)
    nt1 = torch.nested.nested_tensor([t1, t2])

    t3 = torch.full((6, 3), 2.0)
    nt2 = torch.nested.nested_tensor([t3])

    snt1 = SliceNestedTensor(nt1)
    snt2 = SliceNestedTensor(nt2)

    cat_snt = cat_slice_nested_tensors([snt1, snt2])

    assert len(cat_snt) == 3
    assert torch.equal(cat_snt[0], t1)
    assert torch.equal(cat_snt[1], t2)
    assert torch.equal(cat_snt[2], t3)


def test_disable_printing(capsys):
    with DisablePrinting():
        print("This should be hidden")
    print("This should be visible")

    captured = capsys.readouterr()
    assert "This should be hidden" not in captured.out
    assert "This should be visible" in captured.out


def test_sparse2dense():
    # 1. Happy path (max_len=None)
    sparse = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    row_lengths = torch.tensor([2, 3])
    dense = sparse2dense(sparse, row_lengths)
    expected = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]])
    assert torch.equal(dense, expected)

    # 2. max_len provided
    dense_padded = sparse2dense(sparse, row_lengths, max_len=4)
    expected_padded = torch.tensor([[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 5.0, 0.0]])
    assert torch.equal(dense_padded, expected_padded)

    # 3. Empty tensor
    sparse_empty = torch.tensor([], dtype=torch.float32)
    row_lengths_empty = torch.tensor([0, 0])
    dense_empty = sparse2dense(sparse_empty, row_lengths_empty)
    assert dense_empty.shape == (2, 0)

    # 4. Error conditions
    # Not 1D
    with pytest.raises(AssertionError, match="data must be 1D"):
        sparse2dense(torch.tensor([[1.0, 2.0]]), row_lengths)

    # Length mismatch
    with pytest.raises(
        AssertionError, match="data length must match sum of row_lengths"
    ):
        sparse2dense(sparse, torch.tensor([2, 2]))


# ---------------------------------------------------------------------------
# Completeness enforcement (min(d_W, d_Z) >= d_U > 0 after column filtering)
# ---------------------------------------------------------------------------


def _make_prior(fixed_hp, max_dim=8, batch_size=24):
    """A prior with moderately high dims, where post-filter completeness
    violations occur often enough to exercise the enforcement path."""
    return SCMPrior(
        batch_size=batch_size,
        batch_size_per_gp=1,
        min_dim_U=2,
        max_dim_U=max_dim,
        min_dim_W=2,
        max_dim_W=max_dim,
        min_dim_Z=2,
        max_dim_Z=max_dim,
        min_dim_X=2,
        max_dim_X=max_dim,
        min_seq_len=200,
        max_seq_len=201,
        prior_type="mlp_scm",
        mcmc_cate=2,
        fixed_hp=fixed_hp,
        n_jobs=1,
    )


def test_enforce_completeness_heuristic_holds_by_default():
    # Default DEFAULT_FIXED_HP has enforce_completeness_heuristic=True.
    prior = _make_prior(DEFAULT_FIXED_HP)
    out = prior.get_batch()
    d_U, d_W, d_Z = out[11], out[13], out[14]

    # Every delivered dataset must be confounded and satisfy the heuristic.
    assert (d_U > 0).all()
    assert (torch.minimum(d_W, d_Z) >= d_U).all()


def test_enforce_completeness_heuristic_can_be_disabled():
    # With enforcement off, unconfounded / violating datasets are allowed
    # through again. We only assert it runs and returns the batch (violations
    # are probabilistic, so we don't require them to appear).
    fixed_hp = {**DEFAULT_FIXED_HP, "enforce_completeness_heuristic": False}
    prior = _make_prior(fixed_hp)
    out = prior.get_batch()
    d_U, d_W, d_Z = out[11], out[13], out[14]
    assert d_U.shape[0] == 24
    # d_U > 0 is no longer guaranteed; the shape contract still holds.
    assert (d_W > 0).all() and (d_Z > 0).all()


def test_completeness_give_up_terminates():
    # A max_attempts of 1 means the safety valve fires immediately: the very
    # first basically-valid draw is accepted even if completeness fails. This
    # must still terminate and return a full batch (regression guard against an
    # infinite loop in the enforcement path).
    fixed_hp = {
        **DEFAULT_FIXED_HP,
        "enforce_completeness_heuristic": True,
        "completeness_max_attempts": 1,
    }
    prior = _make_prior(fixed_hp, max_dim=10)
    out = prior.get_batch()
    d_X, d_W, d_Z = out[12], out[13], out[14]
    # Basic validity is always required, even when giving up on completeness.
    assert (d_X > 0).all() and (d_W > 0).all() and (d_Z > 0).all()


# ---------------------------------------------------------------------------
# Categorical completeness enforcement (Track F)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _capture_accepted_profiles(prior):
    """Capture the *declared* U/W/Z profiles the generator validates against.

    Reconstructing a block's profile from its realised tensor values is lossy:
    whether a column is continuous or categorical is a *decision* the generator
    records in ``num2cat`` metadata, and it cannot be recovered from the values
    alone. A genuinely continuous column can, under saturating activations plus
    outlier clamping, realise only a handful of distinct values -- indistinguishable
    from a true categorical column by any distinct-count threshold. Guessing thus
    misclassifies such columns as categorical, which spuriously fails the
    dominance check even though the generator's own (metadata-based) check passed.

    The generator computes the authoritative profiles via ``observed_block_profile``
    (on the observed, post-constant-filter blocks) immediately before accepting a
    sample. We intercept that call so the test asserts against the exact profiles
    the generator used. Relies on ``n_jobs=1`` (sequential generation) and on
    ``generate_dataset`` computing ``(U, W, Z)`` profiles, in that order, on its
    accepted attempt -- both true for the priors built by ``_make_prior``.
    """
    accepted: list[tuple[BlockProfile, BlockProfile, BlockProfile]] = []
    per_dataset: list[BlockProfile] = []
    real_obp = _dataset_mod.observed_block_profile
    real_generate = prior.generate_dataset

    def spy_obp(block, meta):
        profile = real_obp(block, meta)
        per_dataset.append(profile)
        return profile

    def spy_generate(params):
        per_dataset.clear()
        result = real_generate(params)
        # The accepted attempt records exactly (U, W, Z) just before returning,
        # so the last three profiles belong to the delivered dataset.
        accepted.append(tuple(per_dataset[-3:]))
        return result

    _dataset_mod.observed_block_profile = spy_obp
    prior.generate_dataset = spy_generate
    try:
        yield accepted
    finally:
        _dataset_mod.observed_block_profile = real_obp
        del prior.generate_dataset  # restore the bound method on the class


def test_enforce_categorical_completeness_holds():
    # Every delivered proxy must be categorically complete w.r.t. U (its profile
    # dominates U's), on top of the dimension heuristic.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    max_categories = 6
    fixed_hp = {
        **DEFAULT_FIXED_HP,
        "enforce_categorical_completeness": True,
        "max_categories": max_categories,
        "cat_prob": 0.8,  # exercise the categorical path often
        "completeness_max_attempts": 200,  # keep give-ups (which may violate) rare
    }
    prior = _make_prior(fixed_hp, max_dim=8, batch_size=24)
    with _capture_accepted_profiles(prior) as profiles:
        out = prior.get_batch()
    d_U, d_W, d_Z = out[11], out[13], out[14]

    # Dimension heuristic still holds.
    assert (d_U > 0).all()
    assert (torch.minimum(d_W, d_Z) >= d_U).all()

    # One captured (U, W, Z) profile triple per delivered dataset.
    assert len(profiles) == d_U.shape[0]

    saw_categorical_U = False
    for i, (u_prof, w_prof, z_prof) in enumerate(profiles):
        assert dominates(w_prof, u_prof), (i, w_prof, u_prof)
        assert dominates(z_prof, u_prof), (i, z_prof, u_prof)
        if u_prof.categorical:
            saw_categorical_U = True

    # The categorical machinery was actually exercised (U had categorical cols).
    assert saw_categorical_U


def test_categorical_completeness_defaults_on():
    # Track F is on by default: U is categorised and proxies are enforced.
    assert DEFAULT_FIXED_HP["corrupt_U"] is True
    assert DEFAULT_FIXED_HP["enforce_categorical_completeness"] is True
    # Finite category cap keeps categorical columns discrete and distinguishable
    # from continuous ones by distinct-value count (well below the sequence length).
    assert math.isfinite(DEFAULT_FIXED_HP["max_categories"])
    prior = _make_prior(DEFAULT_FIXED_HP)
    out = prior.get_batch()
    d_U, d_W, d_Z = out[11], out[13], out[14]
    assert (torch.minimum(d_W, d_Z) >= d_U).all()


def test_categorical_completeness_ablation_arm_runs():
    # corrupt_U on, enforce off: U is categorised but proxies are unconstrained
    # (the ablation arm). Must still run and satisfy the dimension heuristic.
    fixed_hp = {
        **DEFAULT_FIXED_HP,
        "corrupt_U": True,
        "enforce_categorical_completeness": False,
    }
    prior = _make_prior(fixed_hp)
    out = prior.get_batch()
    d_U, d_W, d_Z = out[11], out[13], out[14]
    assert (torch.minimum(d_W, d_Z) >= d_U).all()


def test_enforce_requires_corrupt_U():
    # enforce=True with corrupt_U=False is contradictory (completeness is defined
    # relative to a categorised U) and must raise.
    fixed_hp = {
        **DEFAULT_FIXED_HP,
        "corrupt_U": False,
        "enforce_categorical_completeness": True,
    }
    prior = _make_prior(fixed_hp)
    with pytest.raises(ValueError, match="requires corrupt_U"):
        prior.get_batch()


# ---------------------------------------------------------------------------
# dim_U distribution (Option B: triangular vs uniform)
# ---------------------------------------------------------------------------


def test_triangular_dim_pmf_exact():
    vals, probs = SCMPrior.triangular_dim_pmf(1, 10)
    assert vals.tolist() == list(range(1, 11))
    # p(d) ∝ (11 - d): weights 10, 9, ..., 1 normalised by 55.
    np.testing.assert_allclose(probs, np.arange(10, 0, -1) / 55.0)
    assert probs.sum() == pytest.approx(1.0)
    # Strictly decreasing: small dimensions are favoured.
    assert np.all(np.diff(probs) < 0)


def test_triangular_dim_pmf_respects_lower_bound():
    vals, probs = SCMPrior.triangular_dim_pmf(3, 6)
    assert vals.tolist() == [3, 4, 5, 6]
    np.testing.assert_allclose(probs, np.array([4, 3, 2, 1]) / 10.0)


def test_sample_dim_ranges_and_means():
    np.random.seed(0)
    tri = [SCMPrior.sample_dim(1, 10, "triangular") for _ in range(5000)]
    uni = [SCMPrior.sample_dim(1, 10, "uniform") for _ in range(5000)]

    assert min(tri) >= 1 and max(tri) <= 10
    assert min(uni) >= 1 and max(uni) <= 10
    # Theoretical means: triangular 4.0, uniform 5.5.
    assert np.mean(tri) == pytest.approx(4.0, abs=0.2)
    assert np.mean(uni) == pytest.approx(5.5, abs=0.2)
    # The whole point of Option B: triangular skews d_U smaller.
    assert np.mean(tri) < np.mean(uni)


def test_sample_dim_rejects_unknown_distribution():
    with pytest.raises(ValueError, match="Unknown dim distribution"):
        SCMPrior.sample_dim(1, 10, "nope")


def test_get_batch_respects_dim_U_distribution_flag():
    # Both settings must produce a valid batch; the default is triangular.
    for dist in ("uniform", "triangular"):
        fixed_hp = {**DEFAULT_FIXED_HP, "dim_U_distribution": dist}
        prior = _make_prior(fixed_hp)
        d_U = prior.get_batch()[11]
        assert d_U.shape[0] == 24
        assert (d_U > 0).all()  # enforcement is still on
