import pytest
import torch

from proximalfm.train._eval import CausalEvaluator


@pytest.fixture
def evaluator():
    """Fixture to provide a clean instance of CausalEvaluator for each test."""
    return CausalEvaluator()


# ---------------------------------------------------------
# Potential Outcomes (PO) Metrics Tests
# ---------------------------------------------------------


@pytest.mark.parametrize("batch_size, seq_len", [(1, 10), (4, 5), (8, 20)])
def test_compute_po_metrics_perfect_prediction(evaluator, batch_size, seq_len):
    """Test that when predictions exactly match true values, NRMSE is 0."""
    pred_y0 = torch.rand(batch_size, seq_len)
    pred_y1 = torch.rand(batch_size, seq_len)

    true_y0 = pred_y0.clone()
    true_y1 = pred_y1.clone()

    # Now using y_obs_std to match the updated signature
    y_obs_std = torch.ones(batch_size, 1)

    metrics = evaluator.compute_po_metrics(
        pred_y0, pred_y1, true_y0, true_y1, y_obs_std
    )

    assert "po/nrmse_y0" in metrics
    assert "po/nrmse_y1" in metrics
    assert metrics["po/nrmse_y0"] == pytest.approx(0.0)
    assert metrics["po/nrmse_y1"] == pytest.approx(0.0)


@pytest.mark.parametrize("offset, scale", [(1.0, 1.0), (2.0, 2.0), (0.5, 0.1)])
def test_compute_po_metrics_known_values(evaluator, offset, scale):
    """Test NRMSE computation against manually calculated expected values."""
    batch_size, seq_len = 2, 5

    true_y0 = torch.zeros(batch_size, seq_len)
    true_y1 = torch.zeros(batch_size, seq_len)

    # offset makes MSE = offset^2, RMSE = offset
    pred_y0 = true_y0 + offset
    pred_y1 = true_y1 + offset

    y_obs_std = torch.ones(batch_size, 1) * scale

    metrics = evaluator.compute_po_metrics(
        pred_y0, pred_y1, true_y0, true_y1, y_obs_std
    )

    expected_nrmse = offset / scale

    assert metrics["po/nrmse_y0"] == pytest.approx(expected_nrmse)
    assert metrics["po/nrmse_y1"] == pytest.approx(expected_nrmse)


def test_compute_po_metrics_variable_batch_elements(evaluator):
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

    y_obs_std = torch.ones(batch_size, 1)

    metrics = evaluator.compute_po_metrics(
        pred_y0, pred_y1, true_y0, true_y1, y_obs_std
    )

    # Mean RMSE across batch: (3 + 4) / 2 = 3.5
    assert metrics["po/nrmse_y0"] == pytest.approx(3.5)
    assert metrics["po/nrmse_y1"] == pytest.approx(3.5)


# ---------------------------------------------------------
# CATE Metrics Tests
# ---------------------------------------------------------


@pytest.mark.parametrize("batch_size, seq_len", [(1, 10), (4, 5)])
def test_compute_cate_metrics_perfect_prediction(evaluator, batch_size, seq_len):
    """Test that when CATE predictions exactly match true values, errors are 0."""
    pred_cate = torch.rand(batch_size, seq_len)
    true_cate = pred_cate.clone()
    y_obs_std = torch.ones(batch_size, 1)

    metrics = evaluator.compute_cate_metrics(pred_cate, true_cate, y_obs_std)

    # Updated keys to match "cate/" instead of "val_cate/"
    assert "cate/nrmse" in metrics
    assert metrics["cate/nrmse"] == pytest.approx(0.0)


@pytest.mark.parametrize("offset, scale", [(1.0, 1.0), (3.0, 1.5)])
def test_compute_cate_metrics_known_values(evaluator, offset, scale):
    """Test CATE RMSE and NRMSE computation against expected values."""
    batch_size, seq_len = 2, 5

    true_cate = torch.zeros(batch_size, seq_len)
    pred_cate = true_cate + offset  # MSE = offset^2, RMSE = offset

    y_obs_std = torch.ones(batch_size, 1) * scale

    metrics = evaluator.compute_cate_metrics(pred_cate, true_cate, y_obs_std)

    expected_nrmse = offset / scale

    assert metrics["cate/nrmse"] == pytest.approx(expected_nrmse)


# ---------------------------------------------------------
# Integration Test
# ---------------------------------------------------------


def test_evaluate_all(evaluator):
    """Test that evaluate_all combines PO and CATE metrics seamlessly."""
    batch_size, seq_len = 2, 5

    # Dummies
    pred_y0 = torch.zeros(batch_size, seq_len) + 1.0
    pred_y1 = torch.zeros(batch_size, seq_len) + 2.0
    true_y0 = torch.zeros(batch_size, seq_len)
    true_y1 = torch.zeros(batch_size, seq_len)

    pred_cate = torch.zeros(batch_size, seq_len) + 3.0
    true_cate = torch.zeros(batch_size, seq_len)

    y0_scale = torch.ones(batch_size, 1)
    y1_scale = torch.ones(batch_size, 1)
    y_obs_std = torch.ones(batch_size, 1)

    # Note: evaluate_all in your snippet still accepts y0_scale and y1_scale,
    # even though compute_po_metrics now ignores them.
    metrics = evaluator.evaluate_all(
        pred_y0,
        pred_y1,
        true_y0,
        true_y1,
        pred_cate,
        true_cate,
        y0_scale,
        y1_scale,
        y_obs_std,
    )

    # Check that all keys from both sub-functions are present
    expected_keys = ["po/nrmse_y0", "po/nrmse_y1", "cate/nrmse"]
    for key in expected_keys:
        assert key in metrics

    # Check basic math on the combined output
    assert metrics["po/nrmse_y0"] == pytest.approx(1.0)
    assert metrics["po/nrmse_y1"] == pytest.approx(2.0)
    assert metrics["cate/nrmse"] == pytest.approx(3.0)
