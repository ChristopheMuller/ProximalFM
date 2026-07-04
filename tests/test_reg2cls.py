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

    out = num2cat(X, {"cat_prob": 0.0})

    # No column is touched, and the tensor is returned as-is.
    torch.testing.assert_close(out, original)


def test_num2cat_respects_max_categories():
    torch.manual_seed(0)
    max_categories = 4
    T, H = 200, 6

    # A converted column can have at most `max_categories` unique values; an
    # untouched continuous column keeps all T distinct values. Check the
    # invariant holds for every column across several forced-conversion runs.
    for _ in range(20):
        X = torch.randn(T, H)
        out = num2cat(X, {"cat_prob": 1.0, "max_categories": max_categories})
        assert out.shape == (T, H)
        for col in range(H):
            n_unique = out[:, col].unique().numel()
            assert n_unique <= max_categories or n_unique == T


def test_num2cat_actually_converts_some_columns():
    torch.manual_seed(0)
    T, H = 200, 5
    max_categories = 4

    # Over many forced-eligible calls, at least one column must be discretized
    # (i.e. collapse from T distinct continuous values to <= max_categories).
    converted = False
    for _ in range(30):
        X = torch.randn(T, H)
        out = num2cat(X, {"cat_prob": 1.0, "max_categories": max_categories})
        for col in range(H):
            n_unique = out[:, col].unique().numel()
            if n_unique <= max_categories:
                converted = True
                # Discretized columns hold integer-valued class labels.
                vals = out[:, col]
                torch.testing.assert_close(vals, torch.round(vals))
    assert converted


# ---------------------------------------------------------------------------
# process_features
# ---------------------------------------------------------------------------


def test_process_features_preserves_shape_no_padding():
    torch.manual_seed(0)
    X = torch.randn(128, 3)

    out = process_features(X, {"cat_prob": 0.0})

    # No padding, permutation, or feature-count change.
    assert out.shape == X.shape


def test_process_features_standardizes_continuous_input():
    torch.manual_seed(0)
    X = torch.randn(500, 4) * 7.0 + 3.0

    # cat_prob=0 -> pipeline is outlier_removing + standard_scaling only.
    out = process_features(X, {"cat_prob": 0.0})

    mean = torch.nanmean(out, dim=0)
    std = torch_nanstd(out, dim=0, ddof=1)
    torch.testing.assert_close(mean, torch.zeros(4), atol=1e-5, rtol=0)
    torch.testing.assert_close(std, torch.ones(4), atol=0.05, rtol=0)


def test_process_features_finite_with_categorization():
    torch.manual_seed(0)
    X = torch.randn(256, 5)

    out = process_features(X, {"cat_prob": 1.0, "max_categories": 5})

    assert out.shape == X.shape
    assert torch.isfinite(out).all()


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
