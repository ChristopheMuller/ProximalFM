import contextlib
import json
import math
import random
import warnings

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
from proximalfm.prior._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP


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
        S2,
        K_eff,
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
        S2,
        K_eff,
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
        S2,
        K_eff,
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
        S2,
        K_eff,
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
        S2,
        K_eff,
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


def _make_prior(fixed_hp, max_dim=8, batch_size=24, sampled_hp=DEFAULT_SAMPLED_HP):
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
        sampled_hp=sampled_hp,
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
        "completeness_max_attempts": 200,  # keep give-ups (which may violate) rare
    }
    # Pin the per-column categorisation probability high to exercise the
    # categorical path often (it is a sampled HP, so override it there).
    sampled_hp = {
        **DEFAULT_SAMPLED_HP,
        "cat_col_prob": {"distribution": "uniform", "min": 0.8, "max": 0.8},
    }
    prior = _make_prior(fixed_hp, max_dim=8, batch_size=24, sampled_hp=sampled_hp)
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


# ---------------------------------------------------------------------------
# Confounding-dim control (share-targeted U input width for f_Y / f_A)
# ---------------------------------------------------------------------------


def test_confounding_input_dim_floor_and_cap():
    # Tiny share: the count would round to 0, so the floor of 2 bites.
    assert SCMPrior.confounding_input_dim(0.001, 20) == 2
    # Share clamped at 0.75, so k is capped at 3 * n_other regardless of s.
    assert SCMPrior.confounding_input_dim(0.99, 20) == 60
    assert SCMPrior.confounding_input_dim(0.75, 20) == 60
    # Monotone non-decreasing in the share.
    ks = [SCMPrior.confounding_input_dim(s, 20) for s in np.linspace(0, 1, 50)]
    assert all(a <= b for a, b in zip(ks, ks[1:]))


def test_confounding_input_dim_round_trip():
    # Away from floor and cap, the realised input share k / (k + n) must sit
    # near the targeted share (up to integer rounding).
    for s in (0.2, 0.3, 0.5, 0.7):
        for n in (10, 20, 31):
            k = SCMPrior.confounding_input_dim(s, n)
            assert k / (k + n) == pytest.approx(s, abs=0.5 / n)


def test_split_confounding_inputs_regimes():
    perm = (3, 0, 4, 1, 2)  # dim_U = 5

    # Both sides subsample: no augmentation, nested raw-U subsets.
    idx_y, idx_a, aug_y, aug_a = SCMPrior.split_confounding_inputs(2, 4, 5, perm)
    assert idx_y == (3, 0) and idx_a == (3, 0, 4, 1)
    assert aug_y == 0 and aug_a == 0

    # Both sides augment: full U each, nested aug prefixes carry the rest.
    idx_y, idx_a, aug_y, aug_a = SCMPrior.split_confounding_inputs(7, 9, 5, perm)
    assert idx_y == perm and idx_a == perm
    assert aug_y == 2 and aug_a == 4

    # Mixed: one side subsamples, the other augments; nesting still holds.
    idx_y, idx_a, aug_y, aug_a = SCMPrior.split_confounding_inputs(3, 8, 5, perm)
    assert idx_y == (3, 0, 4) and idx_a == perm
    assert aug_y == 0 and aug_a == 3
    assert set(idx_y) <= set(idx_a)


@contextlib.contextmanager
def _capture_mechanism_widths():
    """Record (in_dim, num_outputs) of every MLPSCM the generator builds.

    ``generate_dataset`` looks MLPSCM up as a module global, so patching the
    module attribute intercepts every mechanism. f_A is the only mechanism with
    num_outputs=1 and f_Y the only one with num_outputs=2 (keep block dims >= 3
    so they cannot collide).
    """
    records: list[tuple[int, int]] = []
    real = _dataset_mod.MLPSCM

    class SpyMLPSCM(real):
        def __init__(self, in_dim, num_outputs, **kwargs):
            records.append((in_dim, num_outputs))
            super().__init__(in_dim=in_dim, num_outputs=num_outputs, **kwargs)

    _dataset_mod.MLPSCM = SpyMLPSCM
    try:
        yield records
    finally:
        _dataset_mod.MLPSCM = real


def _make_pinned_dims_prior(fixed_hp, sampled_hp, dim=5):
    """A prior with every block dimension pinned to ``dim`` so the expected
    f_A / f_Y input widths are deterministic."""
    from proximalfm.prior._prior_config import DEFAULT_SAMPLED_HP

    return SCMPrior(
        batch_size=2,
        batch_size_per_gp=1,
        min_dim_U=dim,
        max_dim_U=dim,
        min_dim_W=dim,
        max_dim_W=dim,
        min_dim_Z=dim,
        max_dim_Z=dim,
        min_dim_X=dim,
        max_dim_X=dim,
        min_seq_len=200,
        max_seq_len=201,
        prior_type="mlp_scm",
        mcmc_cate=2,
        fixed_hp=fixed_hp,
        sampled_hp={**DEFAULT_SAMPLED_HP, **sampled_hp},
        n_jobs=1,
    )


def _pinned_share(value):
    return {"distribution": "uniform", "min": value, "max": value}


def _full_direct_edges():
    """Pin the proxy direct-edge keep probabilities to 1 so f_Y / f_A see the
    full W / Z blocks and the expected input widths are deterministic."""
    return {
        "direct_edge_share_of_W_in_Y": _pinned_share(1.0),
        "direct_edge_share_of_Z_in_A": _pinned_share(1.0),
    }


def _full_covariate_edges():
    """Pin every covariate keep probability to 1 so all mechanisms (including
    f_U) see the full X block and the expected input widths are deterministic.
    Covariate edge subsampling is on by default, so any width assertion must
    pin these."""
    return {
        f"covariate_share_of_X_in_{k}": _pinned_share(1.0)
        for k in ("U", "W", "Z", "A", "Y")
    }


def _pinned_covariate_edges(value):
    """Pin every covariate keep probability to ``value``."""
    return {
        f"covariate_share_of_X_in_{k}": _pinned_share(value)
        for k in ("U", "W", "Z", "A", "Y")
    }


def _pinned_noise_shares(value=0.01):
    """Pin every eps noise share to ``value``. At the default 0.01 each eps
    block collapses to its floor (1 column; dim_U columns for eps_U), keeping
    the expected mechanism input widths deterministic and minimal."""
    return {
        f"noise_share_of_input_in_{k}": _pinned_share(value)
        for k in ("U", "W", "Z", "A", "Y")
    }


def test_confounding_dim_control_subsample_regime():
    # s = 0.01 -> k = max(2, round(0.01/0.99 * 10)) = 2 < dim_U = 5: both
    # mechanisms see 2 subsampled U columns and no augmentation, so
    # in_dim = k + dim_X + dim_{W/Z} + eps(1) = 13.
    torch.manual_seed(0)
    np.random.seed(0)
    prior = _make_pinned_dims_prior(
        DEFAULT_FIXED_HP,
        {
            "confounding_share_of_U_in_Y": _pinned_share(0.01),
            "confounding_share_of_U_in_A": _pinned_share(0.01),
            **_full_direct_edges(),
            **_full_covariate_edges(),
            **_pinned_noise_shares(),
        },
    )
    with _capture_mechanism_widths() as records:
        prior.get_batch()
    a_widths = {w for w, n_out in records if n_out == 1}
    y_widths = {w for w, n_out in records if n_out == 2}
    assert a_widths == {13} and y_widths == {13}


def test_confounding_dim_control_augment_regime():
    # s = 0.7 -> k = round(0.7/0.3 * 10) = 23 > dim_U = 5: full U plus 18
    # augmented columns, so in_dim = k + dim_X + dim_{W/Z} + eps(1) = 34.
    torch.manual_seed(0)
    np.random.seed(0)
    prior = _make_pinned_dims_prior(
        DEFAULT_FIXED_HP,
        {
            "confounding_share_of_U_in_Y": _pinned_share(0.7),
            "confounding_share_of_U_in_A": _pinned_share(0.7),
            **_full_direct_edges(),
            **_full_covariate_edges(),
            **_pinned_noise_shares(),
        },
    )
    with _capture_mechanism_widths() as records:
        prior.get_batch()
    a_widths = {w for w, n_out in records if n_out == 1}
    y_widths = {w for w, n_out in records if n_out == 2}
    assert a_widths == {34} and y_widths == {34}


def test_confounding_dim_control_off_feeds_raw_U():
    # Flag off: raw full U, no subsampling, no augmentation --
    # in_dim = dim_U + dim_X + dim_{W/Z} + eps(1) = 16.
    torch.manual_seed(0)
    np.random.seed(0)
    fixed_hp = {**DEFAULT_FIXED_HP, "confounding_dim_control": False}
    prior = _make_pinned_dims_prior(
        fixed_hp,
        {
            **_full_direct_edges(),
            **_full_covariate_edges(),
            **_pinned_noise_shares(),
        },
    )
    with _capture_mechanism_widths() as records:
        prior.get_batch()
    a_widths = {w for w, n_out in records if n_out == 1}
    y_widths = {w for w, n_out in records if n_out == 2}
    assert a_widths == {16} and y_widths == {16}


# ---------------------------------------------------------------------------
# Proxy direct-edge subsampling (optional W -> Y and Z -> A edges)
# ---------------------------------------------------------------------------


def test_proxy_edge_subsampling_pure_proxy_regime():
    # Keep probability 0: no W column enters f_Y and no Z column enters f_A
    # (the classical negative-control setting). With the U share pinned at
    # 0.01, the competing width is d_X + 0 = 5, so k = 2 and
    # in_dim = k + d_X + 0 + eps(1) = 8 for both mechanisms.
    torch.manual_seed(0)
    np.random.seed(0)
    prior = _make_pinned_dims_prior(
        DEFAULT_FIXED_HP,
        {
            "confounding_share_of_U_in_Y": _pinned_share(0.01),
            "confounding_share_of_U_in_A": _pinned_share(0.01),
            "direct_edge_share_of_W_in_Y": _pinned_share(0.0),
            "direct_edge_share_of_Z_in_A": _pinned_share(0.0),
            **_full_covariate_edges(),
            **_pinned_noise_shares(),
        },
    )
    with _capture_mechanism_widths() as records:
        out = prior.get_batch()
    a_widths = {w for w, n_out in records if n_out == 1}
    y_widths = {w for w, n_out in records if n_out == 2}
    assert a_widths == {8} and y_widths == {8}
    # The full proxy blocks are still generated, delivered and non-degenerate:
    # subsampling only prunes the direct edges, not the proxies themselves.
    d_W, d_Z = out[13], out[14]
    assert (d_W > 0).all() and (d_Z > 0).all()


def test_proxy_edge_subsampling_flag_off_feeds_full_proxies():
    # Flag off: f_A / f_Y see the full Z / W blocks regardless of the share
    # HPs -- same widths as keep probability 1 (in_dim = 2 + 5 + 5 + 1 = 13).
    torch.manual_seed(0)
    np.random.seed(0)
    fixed_hp = {**DEFAULT_FIXED_HP, "proxy_edge_subsampling": False}
    prior = _make_pinned_dims_prior(
        fixed_hp,
        {
            "confounding_share_of_U_in_Y": _pinned_share(0.01),
            "confounding_share_of_U_in_A": _pinned_share(0.01),
            "direct_edge_share_of_W_in_Y": _pinned_share(0.0),
            "direct_edge_share_of_Z_in_A": _pinned_share(0.0),
            **_full_covariate_edges(),
            **_pinned_noise_shares(),
        },
    )
    with _capture_mechanism_widths() as records:
        prior.get_batch()
    a_widths = {w for w, n_out in records if n_out == 1}
    y_widths = {w for w, n_out in records if n_out == 2}
    assert a_widths == {13} and y_widths == {13}


# ---------------------------------------------------------------------------
# Covariate direct-edge subsampling (optional X -> W, X -> Z, X -> A, X -> Y)
# ---------------------------------------------------------------------------


def _make_distinct_dims_prior(fixed_hp, sampled_hp):
    """A prior with every block dimension distinct, so each mechanism's MLP can
    be told apart by its ``num_outputs``: f_U=3, f_W=4, f_Z=6, f_A=1, f_y=2.
    Lets tests read the X-edge width of f_W / f_Z, not just f_A / f_y."""
    from proximalfm.prior._prior_config import DEFAULT_SAMPLED_HP

    return SCMPrior(
        batch_size=2,
        batch_size_per_gp=1,
        min_dim_U=3,
        max_dim_U=3,
        min_dim_W=4,
        max_dim_W=4,
        min_dim_Z=6,
        max_dim_Z=6,
        min_dim_X=5,
        max_dim_X=5,
        min_seq_len=200,
        max_seq_len=201,
        prior_type="mlp_scm",
        mcmc_cate=2,
        fixed_hp=fixed_hp,
        sampled_hp={**DEFAULT_SAMPLED_HP, **sampled_hp},
        n_jobs=1,
    )


def test_covariate_edge_subsampling_pure_regime():
    # Every covariate keep probability 0: no X column enters any peripheral
    # mechanism. With proxies full and the U share pinned low, the competing
    # width for f_A / f_Y is |X*|(0) + proxy(5) = 5, so k = 2 and
    # in_dim = 2 + 0 + 5 + eps(1) = 8.
    torch.manual_seed(0)
    np.random.seed(0)
    prior = _make_pinned_dims_prior(
        DEFAULT_FIXED_HP,
        {
            "confounding_share_of_U_in_Y": _pinned_share(0.01),
            "confounding_share_of_U_in_A": _pinned_share(0.01),
            **_full_direct_edges(),
            **_pinned_covariate_edges(0.0),
            **_pinned_noise_shares(),
        },
    )
    with _capture_mechanism_widths() as records:
        out = prior.get_batch()
    a_widths = {w for w, n_out in records if n_out == 1}
    y_widths = {w for w, n_out in records if n_out == 2}
    assert a_widths == {8} and y_widths == {8}
    # X itself is still generated and delivered at full width: subsampling only
    # prunes the direct edges into the mechanisms, not the covariate block.
    d_X = out[12]
    assert (d_X > 0).all()


def test_covariate_edge_subsampling_flag_off_feeds_full_X():
    # Flag off: every mechanism sees the full X block regardless of the share
    # HPs -- in_dim = U(2) + X(5) + W(5) + eps(1) = 13.
    torch.manual_seed(0)
    np.random.seed(0)
    fixed_hp = {**DEFAULT_FIXED_HP, "covariate_edge_subsampling": False}
    prior = _make_pinned_dims_prior(
        fixed_hp,
        {
            "confounding_share_of_U_in_Y": _pinned_share(0.01),
            "confounding_share_of_U_in_A": _pinned_share(0.01),
            **_full_direct_edges(),
            **_pinned_covariate_edges(0.0),
            **_pinned_noise_shares(),
        },
    )
    with _capture_mechanism_widths() as records:
        prior.get_batch()
    a_widths = {w for w, n_out in records if n_out == 1}
    y_widths = {w for w, n_out in records if n_out == 2}
    assert a_widths == {13} and y_widths == {13}


def test_covariate_edge_subsampling_is_independent_per_mechanism():
    # Keep X fully for W but drop it entirely for Z: the two proxy mechanisms
    # must wire X independently. With distinct dims, f_W (num_outputs=4) and
    # f_Z (num_outputs=6) are identifiable.
    #   f_W in_dim = U(3) + |X_W|(5) + eps_W(1) = 9
    #   f_Z in_dim = U(3) + |X_Z|(0) + eps_Z(1) = 4
    torch.manual_seed(0)
    np.random.seed(0)
    prior = _make_distinct_dims_prior(
        DEFAULT_FIXED_HP,
        {
            # Low U share -> subsample regime -> no f_aug to collide on n_out.
            "confounding_share_of_U_in_Y": _pinned_share(0.01),
            "confounding_share_of_U_in_A": _pinned_share(0.01),
            "covariate_share_of_X_in_W": _pinned_share(1.0),
            "covariate_share_of_X_in_Z": _pinned_share(0.0),
            "covariate_share_of_X_in_A": _pinned_share(1.0),
            "covariate_share_of_X_in_Y": _pinned_share(1.0),
            **_pinned_noise_shares(),
        },
    )
    with _capture_mechanism_widths() as records:
        prior.get_batch()
    w_widths = {w for w, n_out in records if n_out == 4}
    z_widths = {w for w, n_out in records if n_out == 6}
    assert w_widths == {9}  # X kept
    assert z_widths == {4}  # X dropped


def test_covariate_edge_subsampling_mean_kept_fraction():
    # The realised kept fraction of X should track the configured
    # covariate_share_of_X_in_W Beta mean, read live so this can't drift out
    # of sync with _prior_config.py.
    # Measured on f_W's X contribution: with the noise shares pinned to the
    # floor, in_dim(f_W) = U(3) + |X_W| + eps_W(1), so |X_W| = in_dim - 4,
    # over X of width 5.
    beta_cfg = DEFAULT_SAMPLED_HP["covariate_share_of_X_in_W"]
    expected_mean = beta_cfg["a"] / (beta_cfg["a"] + beta_cfg["b"])
    torch.manual_seed(0)
    np.random.seed(0)
    prior = _make_distinct_dims_prior(
        DEFAULT_FIXED_HP,
        {
            "confounding_share_of_U_in_Y": _pinned_share(0.01),
            "confounding_share_of_U_in_A": _pinned_share(0.01),
            **_pinned_noise_shares(),
        },
    )
    x_widths = []
    for _ in range(40):
        with _capture_mechanism_widths() as records:
            prior.get_batch()
        x_widths += [w - 4 for w, n_out in records if n_out == 4]
    frac = np.array(x_widths) / 5.0
    # 40 batches x 2 datasets = 80 draws; wide tolerance for the small sample.
    assert 0.0 <= frac.min() and frac.max() <= 1.0
    assert frac.mean() == pytest.approx(expected_mean, abs=0.12)


def test_covariate_edge_subsampling_includes_x_to_u():
    # X -> U is subsampled like the peripheral edges. With distinct dims f_U is
    # identifiable (num_outputs=3) and the noise share pinned to the floor
    # (eps_U = dim_U = 3), so f_U in_dim = |X_U*| + 3: 8 when X is kept, 3 when
    # X is dropped (U then a pure function of its exogenous noise).
    for share, expected in [(1.0, 8), (0.0, 3)]:
        torch.manual_seed(0)
        np.random.seed(0)
        prior = _make_distinct_dims_prior(
            DEFAULT_FIXED_HP,
            {
                "confounding_share_of_U_in_Y": _pinned_share(0.01),
                "confounding_share_of_U_in_A": _pinned_share(0.01),
                "covariate_share_of_X_in_U": _pinned_share(share),
                **_pinned_noise_shares(),
            },
        )
        with _capture_mechanism_widths() as records:
            prior.get_batch()
        u_widths = {w for w, n_out in records if n_out == 3}
        assert u_widths == {expected}


def test_covariate_edge_subsampling_flag_off_feeds_full_x_to_u():
    # Flag off: f_U sees the full X regardless of the X->U share --
    # in_dim = X(5) + eps_U(3) = 8.
    torch.manual_seed(0)
    np.random.seed(0)
    fixed_hp = {**DEFAULT_FIXED_HP, "covariate_edge_subsampling": False}
    prior = _make_distinct_dims_prior(
        fixed_hp,
        {
            "confounding_share_of_U_in_Y": _pinned_share(0.01),
            "confounding_share_of_U_in_A": _pinned_share(0.01),
            "covariate_share_of_X_in_U": _pinned_share(0.0),
            **_pinned_noise_shares(),
        },
    )
    with _capture_mechanism_widths() as records:
        prior.get_batch()
    u_widths = {w for w, n_out in records if n_out == 3}
    assert u_widths == {8}


# ---------------------------------------------------------------------------
# Exogenous-noise (eps) width control
# ---------------------------------------------------------------------------


def test_noise_share_widens_mechanism_inputs():
    # With dims pinned at 5, low confounding share and full edges, f_Y's
    # signal width is U_sel(2) + X(5) + W(5) = 12. A tiny noise share hits the
    # 1-column floor (in_dim 13); s = 0.5 gives eps_Y = round(1 * 12) = 12
    # (in_dim 24).
    torch.manual_seed(0)
    np.random.seed(0)
    widths = {}
    for share in (0.01, 0.5):
        prior = _make_pinned_dims_prior(
            DEFAULT_FIXED_HP,
            {
                "confounding_share_of_U_in_Y": _pinned_share(0.01),
                "confounding_share_of_U_in_A": _pinned_share(0.01),
                **_full_direct_edges(),
                **_full_covariate_edges(),
                **_pinned_noise_shares(),
                "noise_share_of_input_in_Y": _pinned_share(share),
            },
        )
        with _capture_mechanism_widths() as records:
            prior.get_batch()
        widths[share] = {w for w, n_out in records if n_out == 2}
    assert widths[0.01] == {13}
    assert widths[0.5] == {24}


def test_eps_U_width_floors_at_dim_U():
    # f_U's in_dim is |X_U*| + eps_U. Pin X -> U full (|X_U*| = dim_X = 5) to
    # isolate the eps floor. With distinct dims (dim_U=3, dim_X=5), a tiny share
    # would give a single eps column, but eps_U is floored at dim_U so U keeps
    # dim_U genuinely exogenous dimensions: in_dim = 5+3 = 8. A large share
    # overrides the floor: s = 0.7 on n_signal = |X_U*| = 5 gives
    # k = round(0.7/0.3 * 5) = 12, so in_dim = 17.
    torch.manual_seed(0)
    np.random.seed(0)
    widths = {}
    for share in (0.01, 0.7):
        prior = _make_distinct_dims_prior(
            DEFAULT_FIXED_HP,
            {
                "confounding_share_of_U_in_Y": _pinned_share(0.01),
                "confounding_share_of_U_in_A": _pinned_share(0.01),
                "covariate_share_of_X_in_U": _pinned_share(1.0),
                **_pinned_noise_shares(),
                "noise_share_of_input_in_U": _pinned_share(share),
            },
        )
        with _capture_mechanism_widths() as records:
            prior.get_batch()
        widths[share] = {w for w, n_out in records if n_out == 3}
    assert widths[0.01] == {8}
    assert widths[0.7] == {17}


def test_mechanisms_receive_only_the_dither_layer_noise():
    # All meaningful randomness enters as eps input columns; within-layer noise
    # is pinned to the tiny fixed dither (DEFAULT_FIXED_HP["noise_std"], 1e-3),
    # whose only job is to keep continuous columns at ~T*K distinct values
    # under saturating/discretising activations so the distinct-count profile
    # classification cannot misread them as categorical. Every params-built
    # mechanism must carry exactly the dither; the U augmentation network
    # (built with explicit args, its output never profiled) stays at 0.
    torch.manual_seed(0)
    np.random.seed(0)
    dither = DEFAULT_FIXED_HP["noise_std"]
    assert 0.0 < dither <= 1e-2  # a tie-breaker, not a noise model
    stds = []
    real = _dataset_mod.MLPSCM

    class SpyMLPSCM(real):
        def __init__(self, in_dim, num_outputs, **kwargs):
            super().__init__(in_dim=in_dim, num_outputs=num_outputs, **kwargs)
            stds.append(self.noise_std)

    _dataset_mod.MLPSCM = SpyMLPSCM
    try:
        prior = _make_pinned_dims_prior(DEFAULT_FIXED_HP, {})
        prior.get_batch()
    finally:
        _dataset_mod.MLPSCM = real
    assert stds and all(s in (dither, 0.0) for s in stds)
    assert any(s == dither for s in stds)


# ---------------------------------------------------------------------------
# Query-only Monte Carlo targets (mc_targets_on_context)
# ---------------------------------------------------------------------------


def _make_mc_split_prior(mc_targets_on_context, train_frac=0.5, seq_len=96, K=3):
    """A small prior with the train/test split pinned to ``train_frac``."""
    return SCMPrior(
        batch_size=2,
        batch_size_per_gp=2,
        min_dim_U=1,
        max_dim_U=5,
        min_dim_W=1,
        max_dim_W=5,
        min_dim_Z=1,
        max_dim_Z=5,
        min_dim_X=1,
        max_dim_X=5,
        min_seq_len=None,
        max_seq_len=seq_len,
        min_train_size=train_frac,
        max_train_size=train_frac + 0.001,
        prior_type="mlp_scm",
        mcmc_cate=K,
        mc_targets_on_context=mc_targets_on_context,
        fixed_hp=FAST_FIXED_HP,
        n_jobs=1,
    )


def _capture_params(prior):
    """The exact params dict ``get_batch`` builds, without generating anything."""
    captured = {}
    real = SCMPrior.generate_dataset

    class _Stop(Exception):
        pass

    def spy(self, params):
        captured.update(params)
        raise _Stop

    SCMPrior.generate_dataset = spy
    try:
        prior.get_batch()
    except _Stop:
        pass
    finally:
        SCMPrior.generate_dataset = real
    return captured


def _seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def test_mc_targets_on_context_off_matches_full_at_zero_context():
    # With no context rows, every row is a query row, so the query-only path
    # must consume the RNG stream identically and produce bit-identical output.
    prior = _make_mc_split_prior(mc_targets_on_context=True)
    _seed(0)
    params = _capture_params(prior)

    _seed(1)
    full = prior.generate_dataset({**params, "train_size": 0})
    _seed(1)
    query_only = prior.generate_dataset(
        {**params, "train_size": 0, "mc_targets_on_context": False}
    )

    assert len(full) == len(query_only)
    for a, b in zip(full[:-1], query_only[:-1]):  # last entry is the diagnostics dict
        assert torch.equal(a, b)


def test_mc_targets_on_context_off_nans_context_prefix():
    prior = _make_mc_split_prior(mc_targets_on_context=False, train_frac=0.5)
    out = prior.get_batch()
    MU0, MU1, train_sizes = out[9], out[10], out[16]

    train_size = int(train_sizes[0].item())
    assert 0 < train_size < MU0.shape[1]

    for MU in (MU0, MU1):
        assert MU[:, :train_size].isnan().all()
        assert MU[:, train_size:].isfinite().all()

    # Only the CEPO targets carry the NaN marker; the observational tuple and
    # the realized potential outcomes stay fully defined on every row.
    for tensor in out[:9]:
        assert tensor.isnan().sum() == 0


def test_mc_targets_on_context_default_is_full():
    prior = _make_mc_split_prior(mc_targets_on_context=True, train_frac=0.5)
    assert prior.mc_targets_on_context is True
    out = prior.get_batch()
    MU0, MU1 = out[9], out[10]
    assert MU0.isfinite().all()
    assert MU1.isfinite().all()


def test_mc_targets_on_context_survives_train_size_at_or_past_seq_len():
    # train_size is only metadata elsewhere, so int-valued ranges can put it at
    # or beyond seq_len. The generator must still leave at least one query row.
    prior = _make_mc_split_prior(mc_targets_on_context=False)
    _seed(2)
    params = _capture_params(prior)
    T = params["seq_len"]

    for train_size in (T, T + 50):
        _seed(3)
        out = prior.generate_dataset({**params, "train_size": train_size})
        MU0, MU1 = out[9], out[10]
        assert MU0.shape == (T,)
        # Exactly one query row survives the clamp.
        assert MU0[:-1].isnan().all() and math.isfinite(float(MU0[-1]))
        assert MU1[:-1].isnan().all() and math.isfinite(float(MU1[-1]))


def test_cap_mc_samples_matches_legacy_formula_without_context():
    # The rewritten cap solves N(K) = T_ctx + T_q*K <= budget/per_row. At
    # T_ctx=0, T_q=T it must reproduce the historical budget // (T * per_row)
    # exactly -- floor division gives (a // b) // c == a // (b * c).
    prior = SCMPrior(batch_size=1)
    for T in (128, 1024, 4096):
        for K in (1, 8, 200):
            for budget in (0, 10_000, 1_000_000, 500_000_000):
                params = {
                    "prior_mc_mem_budget": budget,
                    "num_layers": 4,
                    "hidden_dim": 20,
                    "dim_X": 7,
                    "dim_U": 3,
                    "dim_W": 5,
                    "dim_Z": 5,
                }
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    got = prior._cap_mc_samples(K, 0, T, params)
                if budget <= 0:
                    assert got == K
                    continue
                per_row = prior._per_row_estimate(params)
                assert got == max(1, min(K, int(budget // max(1, T * per_row))))


def test_cap_mc_samples_allows_larger_k_with_context():
    # Half the rows carrying no replicates roughly doubles the K that fits in
    # the same element budget.
    prior = SCMPrior(batch_size=1)
    params = {
        "prior_mc_mem_budget": 2_000_000,
        "num_layers": 4,
        "hidden_dim": 20,
        "dim_X": 7,
        "dim_U": 3,
        "dim_W": 5,
        "dim_Z": 5,
    }
    T = 1024
    T_ctx = T // 2
    T_q = T - T_ctx
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        full = prior._cap_mc_samples(10_000, 0, T, params)
        split = prior._cap_mc_samples(10_000, T_ctx, T_q, params)

    assert split > full
    # Halving the replicated rows buys ~2x the K (exactly 2x only up to the
    # floor divisions and the T_ctx rows subtracted off the budget first).
    assert 1.5 * full <= split <= 2.5 * full

    # The contract that matters: the resulting row count fits the budget.
    per_row = prior._per_row_estimate(params)
    assert T * full * per_row <= params["prior_mc_mem_budget"]
    assert (T_ctx + T_q * split) * per_row <= params["prior_mc_mem_budget"]


# ---------------------------------------------------------------------------
# Target-leaf mechanisms (f_Z / f_A skip the Monte Carlo replicates)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _capture_mechanism_rows():
    """Record ``(num_outputs, n_rows)`` of every MLPSCM forward pass.

    ``generate_dataset`` looks MLPSCM up as a module global, so patching the
    module attribute intercepts every mechanism. The row count is what this
    guards: only the mechanisms feeding MU0/MU1 may see the replicated block.
    """
    records: list[tuple[int, int]] = []
    real = _dataset_mod.MLPSCM

    class SpyMLPSCM(real):
        def forward(self, input):
            records.append((self.num_outputs, input.shape[0]))
            return super().forward(input)

    _dataset_mod.MLPSCM = SpyMLPSCM
    try:
        yield records
    finally:
        _dataset_mod.MLPSCM = real


def test_target_leaf_mechanisms_skip_mc_replicates():
    # Z and A are leaves w.r.t. the Monte Carlo targets -- the proximal DAG
    # keeps Z out of f_W/f_Y and A out of f_Y -- so every replicate of theirs
    # beyond the observed one would be computed only to be discarded. f_Z and
    # f_A must therefore run on the T observed rows while f_U / f_W / f_Y run
    # on the full T*K block. This is the guard against silently regressing that
    # (it costs ~40% of generation time).
    _seed(0)
    T, K = 64, 8
    prior = SCMPrior(
        batch_size=1,
        batch_size_per_gp=1,
        # Distinct block dims so num_outputs identifies the mechanism, and
        # confounding_dim_control off so no augmentation net collides with them.
        min_dim_U=3,
        max_dim_U=3,
        min_dim_W=4,
        max_dim_W=4,
        min_dim_Z=5,
        max_dim_Z=5,
        min_dim_X=6,
        max_dim_X=6,
        min_seq_len=T,
        max_seq_len=T + 1,
        prior_type="mlp_scm",
        mcmc_cate=K,
        fixed_hp={**FAST_FIXED_HP, "confounding_dim_control": False},
        n_jobs=1,
    )
    with _capture_mechanism_rows() as records:
        prior.get_batch()

    rows = {}
    for num_outputs, n_rows in records:
        rows.setdefault(num_outputs, set()).add(n_rows)

    assert rows[6] == {T}  # f_X, never replicated
    assert rows[3] == {T * K}  # f_U
    assert rows[4] == {T * K}  # f_W
    assert rows[2] == {T * K}  # f_Y
    assert rows[5] == {T}  # f_Z -- observed rows only
    assert rows[1] == {T}  # f_A -- observed rows only


def test_target_leaf_mechanisms_preserve_observed_contract():
    # Moving the Z/A branch off the replicate block must not change what a
    # dataset delivers: one row per sequence position, A a valid draw from the
    # clamped propensity, and Y_obs selected by the treatment drawn in the same
    # world as Z.
    _seed(1)
    epsilon = FAST_FIXED_HP["propensity_epsilon"]
    prior = _make_mc_split_prior(mc_targets_on_context=False, train_frac=0.5)
    out = prior.get_batch()
    Z, A, A_probs = out[3], out[4], out[5]
    Y_obs, Y0, Y1 = out[6], out[7], out[8]
    MU0, d_Z = out[9], out[14]
    T = MU0.shape[1]

    assert Z.shape[1] == T
    assert A.shape == (A.shape[0], T) and A_probs.shape == A.shape
    assert set(A.unique().tolist()) <= {0, 1}
    assert bool(((A_probs >= epsilon) & (A_probs <= 1.0 - epsilon)).all())
    assert bool((d_Z > 0).all())
    assert torch.equal(Y_obs, torch.where(A == 1, Y1, Y0))


def test_save_and_load_prior_dataset_query_only_mc(tmp_path):
    # The NaN context prefix must survive the sparse on-disk round-trip, and
    # the flag must be recorded in metadata so a val/ directory is
    # self-describing about how its MU targets were produced.
    train_size = 12
    args = ArgsMock(
        save_dir=str(tmp_path),
        batch_size=2,
        batch_size_per_gp=2,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=4,
        max_dim_W=4,
        min_dim_Z=4,
        max_dim_Z=4,
        min_dim_X=5,
        max_dim_X=5,
        min_seq_len=None,
        max_seq_len=40,
        log_seq_len=False,
        seq_len_per_gp=False,
        min_train_size=train_size,
        max_train_size=train_size + 1,
        replay_small=False,
        prior_type="mlp_scm",
        mcmc_cate=3,
        mc_targets_on_context=False,
        num_batches=1,
        resume_from=0,
        n_jobs=1,
        num_threads_per_generate=1,
        device="cpu",
    )

    SavePriorDataset(args).run()

    with open(tmp_path / "metadata.json") as f:
        assert json.load(f)["mc_targets_on_context"] is False

    loader = LoadPriorDataset(data_dir=str(tmp_path), batch_size=2, max_batches=1)
    batch = next(iter(loader))
    MU0, MU1, train_sizes = batch[9], batch[10], batch[16]

    t = int(train_sizes[0].item())
    assert t == train_size
    for MU in (MU0, MU1):
        assert MU[:, :t].isnan().all()
        assert MU[:, t:].isfinite().all()


# ---------------------------------------------------------------------------
# Target-noise variance (S2) and the realised replicate count (K_eff)
# ---------------------------------------------------------------------------


def _save_args(tmp_path, num_batches=1, mcmc_cate=4, **overrides):
    """A small dense-path SavePriorDataset config."""
    return ArgsMock(
        save_dir=str(tmp_path),
        batch_size=2,
        batch_size_per_gp=2,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=4,
        max_dim_W=4,
        min_dim_Z=4,
        max_dim_Z=4,
        min_dim_X=5,
        max_dim_X=5,
        min_seq_len=None,
        max_seq_len=40,
        log_seq_len=False,
        seq_len_per_gp=False,
        min_train_size=0.5,
        max_train_size=0.51,
        replay_small=False,
        prior_type="mlp_scm",
        mcmc_cate=mcmc_cate,
        num_batches=num_batches,
        resume_from=0,
        n_jobs=1,
        num_threads_per_generate=1,
        device="cpu",
        **overrides,
    )


def test_s2_tracks_the_mu_nan_convention():
    # S2 describes the noise on MU1 - MU0, so it is defined exactly where those
    # are: NaN on the context prefix when only query rows get replicates.
    prior = _make_mc_split_prior(mc_targets_on_context=False, train_frac=0.5)
    out = prior.get_batch()
    MU0, train_sizes, S2 = out[9], out[16], out[17]

    train_size = int(train_sizes[0].item())
    assert 0 < train_size < S2.shape[1]
    assert S2.shape == MU0.shape

    assert S2[:, :train_size].isnan().all()
    assert S2[:, train_size:].isfinite().all()
    # A variance is non-negative wherever it is defined.
    assert (S2[:, train_size:] >= 0).all()


def test_s2_is_finite_everywhere_with_full_replication():
    prior = _make_mc_split_prior(mc_targets_on_context=True, train_frac=0.5)
    S2 = prior.get_batch()[17]
    assert S2.isfinite().all()
    assert (S2 >= 0).all()


def test_s2_is_zero_without_replicates():
    # K = 1 admits no variance estimate. Zeros, not NaN: the loss must degrade
    # to its plain form rather than treating the row as undefined.
    prior = _make_mc_split_prior(mc_targets_on_context=True, K=1)
    out = prior.get_batch()
    S2, K_eff = out[17], out[18]

    assert (S2 == 0).all()
    assert (K_eff == 1).all()


def test_s2_scales_as_one_over_k():
    # The defining property of a Monte Carlo mean's sampling variance. Held on
    # one fixed mechanism (same params, same seed) so only K varies -- across
    # independently drawn mechanisms the comparison would be meaningless.
    prior = _make_mc_split_prior(mc_targets_on_context=True, K=8)
    _seed(0)
    params = _capture_params(prior)

    means = {}
    for K in (8, 64):
        _seed(11)
        out = prior.generate_dataset({**params, "mcmc_cate": K})
        means[K] = float(out[15].mean())
        assert int(out[16]) == K

    ratio = means[8] / means[64]
    # 8x more replicates. Both endpoints are themselves noisy estimates, so this
    # is a wide band around 8 -- it catches a wrong normalisation (e.g. dividing
    # by K^2 or not at all), not a few percent of bias.
    assert 4.0 < ratio < 16.0


def test_s2_is_the_paired_variance_not_the_sum_of_arm_variances():
    # y0_mc and y1_mc come out of one f_y pass over the same replicates, so they
    # are strongly positively correlated. Var(y1) + Var(y0) would therefore
    # over-state the noise on their difference. Assert we are below that bound.
    prior = _make_mc_split_prior(mc_targets_on_context=True, K=16)
    _seed(4)
    params = _capture_params(prior)
    _seed(5)
    out = prior.generate_dataset(params)

    S2 = out[15]
    Y0, Y1 = out[7], out[8]

    # Var(Y1 - Y0) over rows bounds the per-row paired variance loosely, but the
    # sharp statement is simply that the estimator produced something finite and
    # far below the unpaired construction on the same draws.
    assert torch.isfinite(S2).all()
    assert (S2 >= 0).all()

    unpaired_bound = (Y1.var() + Y0.var()) / 16.0
    assert float(S2.mean()) < float(unpaired_bound)


def test_k_eff_reports_the_capped_value_not_the_request():
    # prior_mc_mem_budget silently caps mcmc_cate behind a once-per-process
    # warning, so K_eff is the only way downstream code learns how many
    # replicates the targets actually averaged.
    prior = _make_mc_split_prior(mc_targets_on_context=True, seq_len=128, K=64)
    _seed(6)
    params = _capture_params(prior)

    tight = {**params, "prior_mc_mem_budget": 200_000}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _seed(7)
        out = prior.generate_dataset(tight)

    K_eff = int(out[16])
    expected = prior._cap_mc_samples(64, 0, params["seq_len"], tight)

    assert K_eff == expected
    assert K_eff < 64


def test_s2_and_k_eff_round_trip_through_disk(tmp_path):
    args = _save_args(tmp_path, num_batches=1)
    with DisablePrinting():
        SavePriorDataset(args).run()

    loader = LoadPriorDataset(tmp_path, batch_size=args.batch_size, max_batches=1)
    out = next(iter(loader))

    S2, K_eff = out[17], out[18]
    assert S2.shape == out[9].shape
    assert S2.isfinite().all() and (S2 >= 0).all()
    assert (K_eff > 0).all()

    raw = torch.load(tmp_path / "batch_000000.pt", weights_only=True)
    assert torch.equal(raw["S2"], S2)
    assert torch.equal(raw["K_eff"], K_eff)


def test_legacy_batches_without_s2_load_as_noise_free(tmp_path):
    # Files predating target-noise tracking -- and the synthetic/ validation
    # sets, which have no replicates at all -- must stay loadable. They read as
    # "no measurable MC noise", which is what the loss falls back on.
    args = _save_args(tmp_path, num_batches=1)
    with DisablePrinting():
        SavePriorDataset(args).run()

    batch_file = tmp_path / "batch_000000.pt"
    raw = torch.load(batch_file, weights_only=True)
    del raw["S2"], raw["K_eff"]
    torch.save(raw, batch_file)

    import proximalfm.prior._genload as _genload_mod

    _genload_mod._LEGACY_TARGET_VAR_WARNED = False

    loader = LoadPriorDataset(tmp_path, batch_size=args.batch_size, max_batches=1)
    with pytest.warns(RuntimeWarning, match="no S2/K_eff"):
        out = next(iter(loader))

    S2, K_eff = out[17], out[18]
    assert S2.shape == out[9].shape
    assert (S2 == 0).all()
    assert (K_eff == 0).all()
