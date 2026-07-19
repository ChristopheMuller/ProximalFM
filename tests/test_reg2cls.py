import torch
import pytest

import proximalfm.prior._reg2cls as reg2cls
from proximalfm.prior._reg2cls import (
    standard_scaling,
    torch_nanstd,
    outlier_removing,
    num2cat,
    process_features,
    MulticlassAssigner,
    BlockProfile,
    block_profile,
    dominates,
    observed_block_profile,
)


def test_standard_scaling_basic():
    # Shape (T, H)
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    scaled = standard_scaling(input_tensor)

    # Check mean is approximately 0
    mean_scaled = torch.nanmean(scaled, dim=0)
    torch.testing.assert_close(mean_scaled, torch.zeros(2))

    # Check std is approximately 1 (with ddof=1)
    std_scaled = torch_nanstd(scaled, dim=0, ddof=1)
    torch.testing.assert_close(std_scaled, torch.ones(2))


def test_standard_scaling_with_nans():
    input_tensor = torch.tensor(
        [[1.0, float("nan")], [3.0, 4.0], [5.0, 6.0], [float("nan"), 8.0]]
    )

    scaled = standard_scaling(input_tensor)

    # NaNs should remain NaNs in the output
    assert torch.isnan(scaled[0, 1])
    assert torch.isnan(scaled[3, 0])

    # Calculate expected values manually for non-NaN elements
    # Column 0 non-NaNs: 1.0, 3.0, 5.0. Mean = 3.0, Std = 2.0
    # Column 1 non-NaNs: 4.0, 6.0, 8.0. Mean = 6.0, Std = 2.0

    expected_col0 = (torch.tensor([1.0, 3.0, 5.0]) - 3.0) / 2.0
    torch.testing.assert_close(scaled[[0, 1, 2], 0], expected_col0)

    expected_col1 = (torch.tensor([4.0, 6.0, 8.0]) - 6.0) / 2.0
    torch.testing.assert_close(scaled[[1, 2, 3], 1], expected_col1)


def test_standard_scaling_clipping():
    # Large outlier to trigger clipping
    input_tensor = torch.tensor([[1.0], [2.0], [3.0], [1000.0]])

    # Using a small clip value
    clip_val = 1.5
    scaled = standard_scaling(input_tensor, clip_value=clip_val)

    # Check that maximum value is bounded by clip_val
    # Note: torch.nanmax doesn't exist in all PyTorch versions,
    # so we filter out NaNs manually if necessary, or just use max since we know there are no NaNs here
    assert torch.max(scaled).item() <= clip_val
    assert torch.min(scaled).item() >= -clip_val

    # Specifically, the 1000.0 should be clipped to 1.5
    assert torch.isclose(scaled[3, 0], torch.tensor(clip_val), atol=1e-5)


def test_standard_scaling_single_element():
    # T=1 case
    input_tensor = torch.tensor([[5.0, 10.0]])

    scaled = standard_scaling(input_tensor)

    # Mean is the element itself, so numerator is 0.
    # Denom is clipped to 1e-6. Result should be 0.
    torch.testing.assert_close(scaled, torch.zeros(1, 2))


def test_standard_scaling_zero_variance():
    # Input with zero variance
    input_tensor = torch.tensor([[5.0], [5.0], [5.0]])

    scaled = standard_scaling(input_tensor)

    # Mean is 5.0. Numerator is 0.
    # Denom is 0, but clipped to 1e-6. Result should be 0.
    torch.testing.assert_close(scaled, torch.zeros(3, 1))


# ---------------------------------------------------------------------------
# outlier_removing
# ---------------------------------------------------------------------------


def test_outlier_removing_clamps_extreme_value():
    # 49 tightly-packed values in [-1, 1] plus one extreme outlier.
    tight = torch.linspace(-1.0, 1.0, 49)
    input_tensor = torch.cat([tight, torch.tensor([100.0])]).unsqueeze(1)

    out = outlier_removing(input_tensor, threshold=4.0)

    # Shape is preserved (no padding / dropping).
    assert out.shape == input_tensor.shape

    # The extreme value is clamped well below its original magnitude, while the
    # in-range values (all within the robust bounds) are left untouched.
    assert out[-1, 0] < 100.0
    assert out.max().item() < 10.0
    torch.testing.assert_close(out[:49], input_tensor[:49])


def test_outlier_removing_preserves_nans_and_does_not_crash():
    input_tensor = torch.tensor([[1.0], [2.0], [float("nan")], [3.0], [500.0]])

    out = outlier_removing(input_tensor, threshold=4.0)

    assert out.shape == input_tensor.shape
    # NaN input stays NaN.
    assert torch.isnan(out[2, 0])
    # No NaN is introduced anywhere the input was finite.
    finite_in = ~torch.isnan(input_tensor)
    assert not torch.isnan(out[finite_in]).any()


# ---------------------------------------------------------------------------
# num2cat
# ---------------------------------------------------------------------------


def test_num2cat_identity_when_cat_prob_zero():
    torch.manual_seed(0)
    X = torch.randn(64, 5)
    original = X.clone()

    out, meta = num2cat(X, {"cat_prob": 0.0})

    # No column is touched, the tensor is returned as-is, and metadata is empty.
    torch.testing.assert_close(out, original)
    assert meta == {}


def test_num2cat_respects_max_categories():
    torch.manual_seed(0)
    max_categories = 4
    T, H = 200, 6

    # A converted column can have at most `max_categories` unique values; an
    # untouched continuous column keeps all T distinct values. Check the
    # invariant holds for every column across several forced-conversion runs.
    for _ in range(20):
        X = torch.randn(T, H)
        out, meta = num2cat(X, {"cat_prob": 1.0, "max_categories": max_categories})
        assert out.shape == (T, H)
        for col in range(H):
            n_unique = out[:, col].unique().numel()
            assert n_unique <= max_categories or n_unique == T
        # Reported category counts also honour the cap.
        assert all(2 <= c <= max_categories for c in meta.values())


def test_num2cat_metadata_matches_converted_columns():
    torch.manual_seed(0)
    T, H = 200, 5
    max_categories = 6

    # Metadata must list exactly the categorised columns, with the right counts.
    for _ in range(30):
        X = torch.randn(T, H)
        out, meta = num2cat(X, {"cat_prob": 1.0, "max_categories": max_categories})
        for col in range(H):
            n_unique = out[:, col].unique().numel()
            if col in meta:
                # A reported categorical column holds integer labels and its
                # realised distinct count never exceeds the reported count.
                torch.testing.assert_close(out[:, col], torch.round(out[:, col]))
                assert n_unique <= meta[col]
            else:
                # Unreported columns are left continuous (all T distinct here).
                assert n_unique == T


def test_num2cat_actually_converts_some_columns():
    torch.manual_seed(0)
    T, H = 200, 5
    max_categories = 4

    # Over many forced-eligible calls, at least one column must be discretized
    # (i.e. collapse from T distinct continuous values to <= max_categories).
    converted = False
    for _ in range(30):
        X = torch.randn(T, H)
        out, meta = num2cat(X, {"cat_prob": 1.0, "max_categories": max_categories})
        if meta:
            converted = True
            for col in meta:
                vals = out[:, col]
                torch.testing.assert_close(vals, torch.round(vals))
    assert converted


# ---------------------------------------------------------------------------
# block_profile / dominates (categorical completeness heuristic, Track F)
# ---------------------------------------------------------------------------


def test_block_profile_derives_continuous_and_sorted_cats():
    # 5 total columns, columns 1 and 3 categorised into 2 and 5 categories.
    prof = block_profile(5, {1: 2, 3: 5})
    assert prof.continuous == 3
    # Category counts are sorted descending for a canonical form.
    assert prof.categorical == (5, 2)


def test_block_profile_all_continuous():
    prof = block_profile(4, {})
    assert prof == BlockProfile(continuous=4, categorical=())


def test_dominates_reduces_to_dimension_heuristic_when_no_cats():
    # With nothing categorised, dominance is exactly c_P >= c_U.
    assert dominates(BlockProfile(5, ()), BlockProfile(5, ()))
    assert dominates(BlockProfile(6, ()), BlockProfile(5, ()))
    assert not dominates(BlockProfile(4, ()), BlockProfile(5, ()))


def test_dominates_worked_examples_from_heuristic():
    # U = 2 continuous + 1 categorical(3)  (the worked example in the plan).
    u = BlockProfile(continuous=2, categorical=(3,))

    # Each row of the plan's truth table.
    assert dominates(BlockProfile(3, ()), u)  # 3 continuous: wildcard covers [3]
    assert dominates(BlockProfile(4, ()), u)  # surplus continuous
    assert dominates(BlockProfile(3, (2,)), u)  # wildcard covers [3]; cat(2) surplus
    assert dominates(BlockProfile(2, (3,)), u)  # exact categorical match
    assert dominates(BlockProfile(2, (4,)), u)  # richer categorical
    assert dominates(BlockProfile(2, (3, 2)), u)  # cat(3) dominates; cat(2) surplus

    # 1 continuous + 3 categorical: fails the continuous slot (c_P=1 < c_U=2).
    assert not dominates(BlockProfile(1, (9, 9, 9)), u)
    # 2 continuous + 1 categorical(2): c_P ok but cats [2] cannot cover [3].
    assert not dominates(BlockProfile(2, (2,)), u)


def test_observed_block_profile_drops_constant_and_measures_cardinality():
    # Columns: 0 continuous, 1 categorical(3), 2 constant (dropped), 3 categorical(2).
    T = 60
    col0 = torch.randn(T)
    col1 = torch.tensor([0.0, 1.0, 2.0] * (T // 3))
    col2 = torch.full((T,), 5.0)  # constant -> filtered out
    col3 = torch.tensor([0.0, 1.0] * (T // 2))
    block = torch.stack([col0, col1, col2, col3], dim=1)
    meta = {1: 3, 3: 2}  # constant col2 is not categorical; col0 continuous

    prof = observed_block_profile(block, meta)
    # col2 dropped; col0 continuous survives; col1->3 cats, col3->2 cats.
    assert prof.continuous == 1
    assert prof.categorical == (3, 2)


def test_observed_block_profile_drops_collapsed_categorical():
    # A "categorical" column that realised a single value is constant -> dropped.
    T = 40
    col0 = torch.randn(T)
    col1 = torch.full((T,), 1.0)  # categorical in meta but collapsed to 1 value
    block = torch.stack([col0, col1], dim=1)
    prof = observed_block_profile(block, {1: 4})
    assert prof == BlockProfile(continuous=1, categorical=())


def test_dominates_multi_categorical_requirement():
    # U = 1 continuous + categoricals [4, 3].
    u = BlockProfile(continuous=1, categorical=(4, 3))
    # Exact element-wise match after sorting descending.
    assert dominates(BlockProfile(1, (4, 3)), u)
    assert dominates(BlockProfile(1, (5, 3)), u)
    # One wildcard (extra continuous) can cover the largest requirement.
    assert dominates(BlockProfile(2, (3,)), u)  # wildcard->4, cat(3)->3
    # [4, 2] fails: the 2 cannot cover the required 3.
    assert not dominates(BlockProfile(1, (4, 2)), u)
    # Too few categorical slots and no spare wildcard.
    assert not dominates(BlockProfile(1, (9,)), u)


# ---------------------------------------------------------------------------
# process_features
# ---------------------------------------------------------------------------


def test_process_features_preserves_shape_no_padding():
    torch.manual_seed(0)
    X = torch.randn(128, 3)

    out, meta = process_features(X, {"cat_prob": 0.0})

    # No padding, permutation, or feature-count change.
    assert out.shape == X.shape
    # cat_prob=0 -> everything stays continuous, no categorisation metadata.
    assert meta == {}
    assert block_profile(out.shape[1], meta) == BlockProfile(
        continuous=3, categorical=()
    )


def test_process_features_standardizes_continuous_input():
    torch.manual_seed(0)
    X = torch.randn(500, 4) * 7.0 + 3.0

    # cat_prob=0 -> pipeline is outlier_removing + standard_scaling only.
    out, _ = process_features(X, {"cat_prob": 0.0})

    mean = torch.nanmean(out, dim=0)
    std = torch_nanstd(out, dim=0, ddof=1)
    torch.testing.assert_close(mean, torch.zeros(4), atol=1e-5, rtol=0)
    torch.testing.assert_close(std, torch.ones(4), atol=0.05, rtol=0)


def test_process_features_finite_with_categorization():
    torch.manual_seed(0)
    X = torch.randn(256, 5)

    out, meta = process_features(X, {"cat_prob": 1.0, "max_categories": 5})

    assert out.shape == X.shape
    assert torch.isfinite(out).all()
    # Metadata columns account for a subset of the full feature count.
    prof = block_profile(out.shape[1], meta)
    assert prof.continuous + len(prof.categorical) == 5


# ---------------------------------------------------------------------------
# process_features / num2cat construct-by-design (require_profile)
# ---------------------------------------------------------------------------


def test_num2cat_enforced_profile_dominates_requirement():
    torch.manual_seed(0)
    T = 300
    max_categories = 8

    # A spread of U-profiles the proxy must dominate, all feasible given H cols.
    requirements = [
        BlockProfile(continuous=2, categorical=(3,)),
        BlockProfile(continuous=1, categorical=(4, 3)),
        BlockProfile(continuous=3, categorical=()),
        BlockProfile(continuous=0, categorical=(5, 2, 2)),
    ]

    for req in requirements:
        need = req.continuous + len(req.categorical)
        for H in (need, need + 2, need + 4):
            for _ in range(15):
                X = torch.randn(T, H)
                _, meta = num2cat(
                    X,
                    {"cat_prob": 1.0, "max_categories": max_categories},
                    require_profile=req,
                )
                prof = block_profile(H, meta)
                assert dominates(prof, req), (req, H, prof)


def test_process_features_enforced_profile_dominates_and_finite():
    torch.manual_seed(0)
    req = BlockProfile(continuous=2, categorical=(4, 3))

    for _ in range(20):
        X = torch.randn(256, 7)
        out, meta = process_features(
            X, {"cat_prob": 0.5, "max_categories": 10}, require_profile=req
        )
        assert out.shape == (256, 7)
        assert torch.isfinite(out).all()
        assert dominates(block_profile(out.shape[1], meta), req)


def test_num2cat_enforced_all_continuous_requirement_keeps_enough_continuous():
    torch.manual_seed(0)
    # U all continuous -> proxy just needs c_P >= c_U, even under forced cat.
    req = BlockProfile(continuous=4, categorical=())
    for _ in range(30):
        X = torch.randn(200, 6)
        _, meta = num2cat(
            X, {"cat_prob": 1.0, "max_categories": 5}, require_profile=req
        )
        prof = block_profile(6, meta)
        assert prof.continuous >= 4
        assert dominates(prof, req)


def test_num2cat_enforced_raises_when_infeasible():
    torch.manual_seed(0)
    # Requirement needs 4 columns (2 continuous + 2 categorical) but block has 3.
    req = BlockProfile(continuous=2, categorical=(3, 2))
    X = torch.randn(100, 3)
    with pytest.raises(ValueError):
        num2cat(X, {"cat_prob": 1.0, "max_categories": 5}, require_profile=req)


def test_num2cat_enforced_continuous_coverage_can_avoid_categorical():
    torch.manual_seed(0)
    # proxy_cat_cover_prob=0 -> every U categorical requirement is covered by a
    # continuous wildcard, so the proxy can be fully continuous yet still dominate.
    req = BlockProfile(continuous=1, categorical=(4, 3))
    saw_fully_continuous = False
    for _ in range(30):
        X = torch.randn(200, 5)
        _, meta = num2cat(
            X,
            {"cat_prob": 0.0, "max_categories": 8, "proxy_cat_cover_prob": 0.0},
            require_profile=req,
        )
        prof = block_profile(5, meta)
        assert dominates(prof, req)  # still complete, via wildcards
        if len(prof.categorical) == 0:
            saw_fully_continuous = True
    # With cat_prob=0 and cover_prob=0, no column is ever categorised.
    assert saw_fully_continuous


def test_num2cat_enforced_full_categorical_coverage_mirrors_u():
    torch.manual_seed(0)
    # proxy_cat_cover_prob=1 -> every requirement gets a categorical column, so
    # the proxy has at least as many categorical columns as U.
    req = BlockProfile(continuous=2, categorical=(5, 3))
    for _ in range(30):
        X = torch.randn(200, 6)
        _, meta = num2cat(
            X,
            {"cat_prob": 0.0, "max_categories": 10, "proxy_cat_cover_prob": 1.0},
            require_profile=req,
        )
        prof = block_profile(6, meta)
        assert dominates(prof, req)
        assert len(prof.categorical) >= len(req.categorical)


def test_num2cat_enforced_dominates_across_cover_probs():
    torch.manual_seed(0)
    # Whatever the coverage probability, the result must always dominate.
    req = BlockProfile(continuous=1, categorical=(4, 3, 2))
    for cover in (0.0, 0.25, 0.5, 0.75, 1.0):
        for _ in range(15):
            X = torch.randn(200, 7)
            _, meta = num2cat(
                X,
                {
                    "cat_prob": 0.5,
                    "max_categories": 8,
                    "proxy_cat_cover_prob": cover,
                },
                require_profile=req,
            )
            assert dominates(block_profile(7, meta), req)


# ---------------------------------------------------------------------------
# MulticlassAssigner (used internally by num2cat)
# ---------------------------------------------------------------------------


def test_multiclass_assigner_output_range_and_shape():
    torch.manual_seed(0)
    num_classes = 3
    x = torch.randn(300)

    assigner = MulticlassAssigner(num_classes, mode="rank", ordered_prob=0.2)
    out = assigner(x)

    assert out.shape == x.shape
    assert out.min().item() >= 0
    assert out.max().item() <= num_classes - 1
    # Labels are whole numbers.
    torch.testing.assert_close(out.float(), torch.round(out.float()))


def test_multiclass_assigner_requires_at_least_two_classes():
    with pytest.raises(ValueError):
        MulticlassAssigner(1)


# ---------------------------------------------------------------------------
# Guard: the target-side reg-to-cls transforms were removed
# ---------------------------------------------------------------------------


def test_removed_target_transforms_are_gone():
    assert not hasattr(reg2cls, "Reg2Cls")
    assert not hasattr(reg2cls, "BalancedBinarize")
