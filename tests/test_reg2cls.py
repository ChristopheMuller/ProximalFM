import torch
import pytest

from proximalfm.prior._reg2cls import standard_scaling, torch_nanstd


def test_standard_scaling_basic():
    # Shape (T, H)
    input_tensor = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0]
    ])

    scaled = standard_scaling(input_tensor)

    # Check mean is approximately 0
    mean_scaled = torch.nanmean(scaled, dim=0)
    torch.testing.assert_close(mean_scaled, torch.zeros(2))

    # Check std is approximately 1 (with ddof=1)
    std_scaled = torch_nanstd(scaled, dim=0, ddof=1)
    torch.testing.assert_close(std_scaled, torch.ones(2))


def test_standard_scaling_with_nans():
    input_tensor = torch.tensor([
        [1.0, float('nan')],
        [3.0, 4.0],
        [5.0, 6.0],
        [float('nan'), 8.0]
    ])

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
    input_tensor = torch.tensor([
        [1.0],
        [2.0],
        [3.0],
        [1000.0]
    ])

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
    input_tensor = torch.tensor([
        [5.0],
        [5.0],
        [5.0]
    ])

    scaled = standard_scaling(input_tensor)

    # Mean is 5.0. Numerator is 0.
    # Denom is 0, but clipped to 1e-6. Result should be 0.
    torch.testing.assert_close(scaled, torch.zeros(3, 1))
