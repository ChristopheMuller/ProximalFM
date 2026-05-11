import pytest
import torch

from proximalfm.train._eval import compute_po_metrics


@pytest.mark.parametrize("batch_size, seq_len", [(1, 10), (4, 5), (8, 20)])
def test_compute_po_metrics_perfect_prediction(batch_size, seq_len):
    """Test that when predictions exactly match true values, NRMSE is 0."""
    pred_y0 = torch.rand(batch_size, seq_len)
    pred_y1 = torch.rand(batch_size, seq_len)

    true_y0 = pred_y0.clone()
    true_y1 = pred_y1.clone()

    y0_scale = torch.ones(batch_size, 1)
    y1_scale = torch.ones(batch_size, 1)

    metrics = compute_po_metrics(
        pred_y0, pred_y1, true_y0, true_y1, y0_scale, y1_scale
    )

    assert "po/nrmse_y0" in metrics
    assert "po/nrmse_y1" in metrics
    assert metrics["po/nrmse_y0"] == pytest.approx(0.0)
    assert metrics["po/nrmse_y1"] == pytest.approx(0.0)


@pytest.mark.parametrize("offset, scale", [(1.0, 1.0), (2.0, 2.0), (0.5, 0.1)])
def test_compute_po_metrics_known_values(offset, scale):
    """Test NRMSE computation against manually calculated expected values."""
    batch_size, seq_len = 2, 5

    true_y0 = torch.zeros(batch_size, seq_len)
    true_y1 = torch.zeros(batch_size, seq_len)

    # offset makes MSE = offset^2, RMSE = offset
    pred_y0 = true_y0 + offset
    pred_y1 = true_y1 + offset

    y0_scale = torch.ones(batch_size, 1) * scale
    y1_scale = torch.ones(batch_size, 1) * scale

    metrics = compute_po_metrics(
        pred_y0, pred_y1, true_y0, true_y1, y0_scale, y1_scale
    )

    expected_nrmse = offset / scale

    assert metrics["po/nrmse_y0"] == pytest.approx(expected_nrmse)
    assert metrics["po/nrmse_y1"] == pytest.approx(expected_nrmse)


def test_compute_po_metrics_variable_batch_elements():
    """Test that NRMSE is correctly averaged over the batch."""
    batch_size, seq_len = 2, 10

    true_y0 = torch.zeros(batch_size, seq_len)
    true_y1 = torch.zeros(batch_size, seq_len)

    # Batch element 0: offset 3 -> RMSE 3
    # Batch element 1: offset 4 -> RMSE 4
    pred_y0 = torch.zeros(batch_size, seq_len)
    pred_y0[0, :] = 3.0
    pred_y0[1, :] = 4.0

    pred_y1 = torch.zeros(batch_size, seq_len)
    pred_y1[0, :] = 3.0
    pred_y1[1, :] = 4.0

    y0_scale = torch.ones(batch_size, 1)
    y1_scale = torch.ones(batch_size, 1)

    metrics = compute_po_metrics(
        pred_y0, pred_y1, true_y0, true_y1, y0_scale, y1_scale
    )

    # Mean RMSE across batch: (3 + 4) / 2 = 3.5
    assert metrics["po/nrmse_y0"] == pytest.approx(3.5)
    assert metrics["po/nrmse_y1"] == pytest.approx(3.5)
