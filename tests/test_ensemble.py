import math
import sys

import pytest
import torch

from proximalfm._model.ensemble import (
    VALUE_TRANSFORMS,
    RecursionLimitManager,
    FeatureShuffler,
    FeatureEnsemble,
    apply_permutation,
    apply_value_transform,
    _context_standardize,
    _quantile_to_normal,
)


# ---------------------------------------------------------
# RecursionLimitManager
# ---------------------------------------------------------


def test_recursion_limit_manager_sets_and_restores():
    original = sys.getrecursionlimit()
    with RecursionLimitManager(original + 5000) as mgr:
        assert sys.getrecursionlimit() == original + 5000
        assert mgr.original_limit == original
    assert sys.getrecursionlimit() == original


def test_recursion_limit_manager_restores_on_exception():
    original = sys.getrecursionlimit()
    with pytest.raises(ValueError):
        with RecursionLimitManager(original + 5000):
            assert sys.getrecursionlimit() == original + 5000
            raise ValueError("boom")
    assert sys.getrecursionlimit() == original


# ---------------------------------------------------------
# FeatureShuffler
# ---------------------------------------------------------


def _is_permutation(perm, n):
    return sorted(perm) == list(range(n))


def test_shuffler_none_returns_identity():
    shuffler = FeatureShuffler(n_elements=5, method="none")
    patterns = shuffler.shuffle(n_estimators=8)
    assert patterns == [list(range(5))]


def test_shuffler_single_estimator_or_one_element():
    # n_estimators == 1 is a degenerate case: identity only.
    shuffler = FeatureShuffler(n_elements=6, method="random")
    patterns = shuffler.shuffle(n_estimators=1)
    assert patterns == [list(range(6))]

    # n_elements <= 1 is also degenerate.
    shuffler = FeatureShuffler(n_elements=1, method="latin")
    assert shuffler.shuffle(n_estimators=3) == [[0]]


def test_shuffler_shift_produces_all_circular_shifts():
    shuffler = FeatureShuffler(n_elements=4, method="shift")
    patterns = shuffler.shuffle(n_estimators=10)
    assert len(patterns) == 4
    for p in patterns:
        assert _is_permutation(p, 4)
    # First shift moves the last element to the front.
    assert patterns[0] == [0, 1, 2, 3]
    assert patterns[1] == [3, 0, 1, 2]


def test_shuffler_random_are_valid_permutations():
    shuffler = FeatureShuffler(n_elements=8, method="random", random_state=0)
    patterns = shuffler.shuffle(n_estimators=5)
    assert len(patterns) == 5
    for p in patterns:
        assert _is_permutation(p, 8)


def test_shuffler_random_is_reproducible():
    a = FeatureShuffler(n_elements=8, method="random", random_state=123).shuffle(5)
    b = FeatureShuffler(n_elements=8, method="random", random_state=123).shuffle(5)
    assert a == b


def test_shuffler_latin_produces_latin_square():
    n = 5
    shuffler = FeatureShuffler(n_elements=n, method="latin", random_state=7)
    patterns = shuffler.shuffle(n_estimators=n)
    assert len(patterns) == n
    for p in patterns:
        assert _is_permutation(p, n)
    # In a Latin square, each column also contains every symbol exactly once.
    for col in range(n):
        column = sorted(row[col] for row in patterns)
        assert column == list(range(n))


def test_shuffler_latin_falls_back_to_random_when_too_large():
    shuffler = FeatureShuffler(
        n_elements=10, method="latin", max_elements_for_latin=4, random_state=1
    )
    patterns = shuffler.shuffle(n_estimators=3)
    # Fallback uses the random branch (n_elements > 5): one perm per estimator.
    assert len(patterns) == 3
    for p in patterns:
        assert _is_permutation(p, 10)


def test_shuffler_unknown_method_raises():
    shuffler = FeatureShuffler(n_elements=4, method="bogus")
    with pytest.raises(ValueError):
        shuffler.shuffle(n_estimators=4)


# ---------------------------------------------------------
# apply_permutation
# ---------------------------------------------------------


def test_apply_permutation_none_is_noop():
    x = torch.randn(2, 3, 4)
    out = apply_permutation(x, None)
    assert out is x


def test_apply_permutation_identity_returns_input():
    x = torch.randn(2, 3, 4)
    out = apply_permutation(x, [0, 1, 2, 3])
    assert torch.equal(out, x)


def test_apply_permutation_reorders_columns():
    x = torch.arange(2 * 1 * 3, dtype=torch.float).reshape(2, 1, 3)
    # x[..., :] == [[0,1,2],[3,4,5]]
    out = apply_permutation(x, [2, 0, 1])
    expected = x[:, :, [2, 0, 1]]
    assert torch.equal(out, expected)


def test_apply_permutation_length_mismatch_raises():
    x = torch.randn(2, 3, 4)
    with pytest.raises(ValueError):
        apply_permutation(x, [0, 1, 2])


def test_apply_permutation_respects_padding_via_d():
    # F=4, but batch elem 0 has 2 valid cols, batch elem 1 has 4 valid cols.
    x = torch.arange(2 * 1 * 4, dtype=torch.float).reshape(2, 1, 4)
    # x = [[0,1,2,3],[4,5,6,7]]
    perm = [3, 2, 1, 0]
    d = torch.tensor([2, 4])
    out = apply_permutation(x, perm, d=d)

    # Row 0: only cols <2 are valid -> among {0,1} kept in perm order,
    # padding cols {2,3} appended as-is. perm restricted to <2 is [1, 0].
    assert out[0, 0].tolist() == [1, 0, 2, 3]
    # Row 1: all valid -> full reverse.
    assert out[1, 0].tolist() == [7, 6, 5, 4]


# ---------------------------------------------------------
# Value transforms
# ---------------------------------------------------------


def test_context_standardize_zero_mean_unit_std_on_context():
    x = torch.randn(2, 50, 3) * 5.0 + 3.0
    train_size = 40
    out = _context_standardize(x, train_size)
    ctx = out[:, :train_size, :]
    assert torch.allclose(ctx.mean(dim=1), torch.zeros(2, 3), atol=1e-5)
    assert torch.allclose(ctx.std(dim=1, unbiased=False), torch.ones(2, 3), atol=1e-4)


def test_context_standardize_leaves_constant_columns_unscaled():
    x = torch.full((1, 10, 1), 7.0)
    out = _context_standardize(x, train_size=8)
    # Constant column: std guarded to 1, so output is just (x - mean) = 0.
    assert torch.allclose(out, torch.zeros_like(out))


def test_apply_value_transform_none_is_identity():
    x = torch.randn(2, 10, 3)
    out = apply_value_transform(x, "none", train_size=8)
    assert out is x
    out_none = apply_value_transform(x, None, train_size=8)
    assert out_none is x


def test_apply_value_transform_sign_negates_standardized():
    x = torch.randn(2, 10, 3)
    z = _context_standardize(x, train_size=8)
    out = apply_value_transform(x, "sign", train_size=8)
    assert torch.allclose(out, -z)


def test_apply_value_transform_signed_log_and_arcsinh_are_odd():
    x = torch.randn(2, 20, 3)
    for name in ("signed_log", "arcsinh"):
        z = _context_standardize(x, train_size=15)
        out = apply_value_transform(x, name, train_size=15)
        # Monotone increasing and sign-preserving on the standardized values.
        assert torch.all(torch.sign(out) == torch.sign(z))


def test_apply_value_transforms_preserve_shape():
    x = torch.randn(2, 30, 4)
    for name in VALUE_TRANSFORMS:
        out = apply_value_transform(x, name, train_size=20)
        assert out.shape == x.shape, name
        assert torch.isfinite(out).all(), name


def test_apply_value_transform_unknown_raises():
    x = torch.randn(1, 5, 2)
    with pytest.raises(ValueError):
        apply_value_transform(x, "square", train_size=3)


def test_quantile_to_normal_is_roughly_gaussian_and_monotone():
    torch.manual_seed(0)
    x = torch.randn(1, 500, 1) * 3.0 + 2.0
    train_size = 500
    out = _quantile_to_normal(x, train_size)
    # Rank transform is monotone: argsort of input matches argsort of output.
    in_order = torch.argsort(x[0, :, 0])
    out_order = torch.argsort(out[0, :, 0])
    assert torch.equal(in_order, out_order)
    # Output should be approximately standard normal.
    assert abs(out.mean().item()) < 0.1
    assert abs(out.std(unbiased=False).item() - 1.0) < 0.15


# ---------------------------------------------------------
# FeatureEnsemble
# ---------------------------------------------------------


def test_normalize_transforms_defaults_to_none():
    assert FeatureEnsemble._normalize_transforms(None) == ["none"]
    assert FeatureEnsemble._normalize_transforms([]) == ["none"]


def test_normalize_transforms_puts_none_first_and_dedups():
    out = FeatureEnsemble._normalize_transforms(["arcsinh", "none", "arcsinh", "sign"])
    assert out[0] == "none"
    assert out == ["none", "arcsinh", "sign"]


def test_normalize_transforms_rejects_unknown():
    with pytest.raises(ValueError):
        FeatureEnsemble._normalize_transforms(["not_a_transform"])


def test_generate_returns_n_estimators_views():
    ens = FeatureEnsemble(n_estimators=6, random_state=0)
    views = ens.generate(d_X_width=5, d_Z_width=3, d_W_width=2)
    assert len(views) == 6
    for v in views:
        assert set(v.keys()) == {"X", "Z", "W", "transform"}


def test_generate_first_view_is_canonical():
    ens = FeatureEnsemble(
        n_estimators=4,
        value_transforms=["arcsinh", "sign"],
        shuffle_X=True,
        shuffle_Z=True,
        shuffle_W=True,
        transform_X=True,
        random_state=0,
    )
    views = ens.generate(d_X_width=4, d_Z_width=3, d_W_width=2)
    first = views[0]
    assert first["transform"] == "none"
    assert first["X"] == [0, 1, 2, 3]
    assert first["Z"] == [0, 1, 2]
    assert first["W"] == [0, 1]


def test_generate_disabled_blocks_are_none():
    ens = FeatureEnsemble(
        n_estimators=4,
        shuffle_X=True,
        shuffle_Z=False,
        shuffle_W=False,
        random_state=0,
    )
    views = ens.generate(d_X_width=5, d_Z_width=4, d_W_width=3)
    for v in views:
        assert v["Z"] is None
        assert v["W"] is None
    # X is enabled with width > 1, so at least the canonical identity is present.
    assert views[0]["X"] == [0, 1, 2, 3, 4]


def test_generate_width_one_block_not_permuted():
    ens = FeatureEnsemble(n_estimators=4, shuffle_X=True, random_state=0)
    views = ens.generate(d_X_width=1, d_Z_width=1, d_W_width=1)
    for v in views:
        assert v["X"] is None


def test_generate_cycles_value_transforms():
    ens = FeatureEnsemble(
        n_estimators=5,
        value_transforms=["sign", "arcsinh"],
        random_state=0,
    )
    views = ens.generate(d_X_width=4, d_Z_width=2, d_W_width=2)
    # Normalized order is ["none", "sign", "arcsinh"], cycled across 5 views.
    transforms = [v["transform"] for v in views]
    assert transforms == ["none", "sign", "arcsinh", "none", "sign"]


def test_generate_is_reproducible_with_seed():
    a = FeatureEnsemble(n_estimators=6, random_state=42).generate(6, 4, 3)
    b = FeatureEnsemble(n_estimators=6, random_state=42).generate(6, 4, 3)
    assert a == b


def test_generate_permutations_are_valid():
    ens = FeatureEnsemble(
        n_estimators=8,
        shuffle_X=True,
        shuffle_Z=True,
        shuffle_W=True,
        random_state=3,
    )
    views = ens.generate(d_X_width=6, d_Z_width=4, d_W_width=3)
    for v in views:
        assert _is_permutation(v["X"], 6)
        assert _is_permutation(v["Z"], 4)
        assert _is_permutation(v["W"], 3)
