import pytest
import torch

from proximalfm.train._eval import CausalEvaluator


@pytest.fixture
def evaluator():
    """Fixture to provide a clean instance of CausalEvaluator for each test."""
    return CausalEvaluator()


# ---------------------------------------------------------
# Basic State & Lifecycle Tests
# ---------------------------------------------------------


def test_evaluator_initial_state(evaluator):
    """Test that a new evaluator is empty and computes to an empty dict."""
    assert evaluator.empty
    assert not evaluator.ready(min_episodes=1)
    assert evaluator.compute() == {}


def test_evaluator_reset(evaluator):
    """Test that reset clears accumulated state."""
    batch_size, seq_len = 2, 5

    # Push dummy data to change state
    evaluator.update(
        pred_y0=torch.zeros(batch_size, seq_len),
        pred_y1=torch.zeros(batch_size, seq_len),
        true_y0=torch.zeros(batch_size, seq_len),
        true_y1=torch.zeros(batch_size, seq_len),
        pred_cate=torch.zeros(batch_size, seq_len),
        true_cate=torch.zeros(batch_size, seq_len),
    )

    assert not evaluator.empty
    assert evaluator.num_episodes == batch_size

    # Reset and verify
    evaluator.reset()
    assert evaluator.empty
    assert evaluator.num_episodes == 0
    assert evaluator.compute() == {}


# ---------------------------------------------------------
# Core Metric Computation Tests
# ---------------------------------------------------------


def test_update_perfect_prediction(evaluator):
    """Test that when predictions exactly match true values, RMSE is 0."""
    batch_size, seq_len = 4, 5
    pred_y0 = torch.rand(batch_size, seq_len)
    pred_y1 = torch.rand(batch_size, seq_len)
    pred_cate = torch.rand(batch_size, seq_len)

    evaluator.update(
        pred_y0=pred_y0,
        pred_y1=pred_y1,
        true_y0=pred_y0.clone(),
        true_y1=pred_y1.clone(),
        pred_cate=pred_cate,
        true_cate=pred_cate.clone(),
    )

    metrics = evaluator.compute()

    # Implementation uses 'rmse', not 'nrmse'
    assert metrics["po/rmse_y0"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["po/rmse_y1"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["cate/rmse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["eval_episodes"] == batch_size


def test_update_without_per_arm_predictions_omits_po_metrics(evaluator):
    """A cate_single-shaped update (no per-arm head) reports cate/ate but no po/*."""
    batch_size, seq_len = 4, 5
    pred_cate = torch.rand(batch_size, seq_len)

    evaluator.update(pred_cate=pred_cate, true_cate=pred_cate.clone())

    metrics = evaluator.compute()

    assert metrics["cate/rmse"] == pytest.approx(0.0, abs=1e-6)
    assert "ate/rmse" in metrics
    assert not any(key.startswith("po/") for key in metrics)


def test_update_known_values(evaluator):
    """Test RMSE computation against manually calculated expected values."""
    batch_size, seq_len = 2, 5
    offset = 2.0

    true_y0 = torch.zeros(batch_size, seq_len)
    true_y1 = torch.zeros(batch_size, seq_len)
    true_cate = torch.zeros(batch_size, seq_len)

    # Adding offset makes MSE = offset^2, so RMSE = offset
    evaluator.update(
        pred_y0=true_y0 + offset,
        pred_y1=true_y1 + offset,
        true_y0=true_y0,
        true_y1=true_y1,
        pred_cate=true_cate + offset,
        true_cate=true_cate,
    )

    metrics = evaluator.compute()

    assert metrics["po/rmse_y0"] == pytest.approx(offset)
    assert metrics["po/rmse_y1"] == pytest.approx(offset)
    assert metrics["cate/rmse"] == pytest.approx(offset)


# ---------------------------------------------------------
# Streaming & Accumulation Tests
# ---------------------------------------------------------


def test_update_streaming_accumulation(evaluator):
    """Test that RMSE is correctly averaged over multiple sequential micro-batches."""
    batch_size, seq_len = 2, 10

    # Batch 1: Offset 3 -> RMSE 3.0
    evaluator.update(
        pred_y0=torch.ones(batch_size, seq_len) * 3.0,
        pred_y1=torch.ones(batch_size, seq_len) * 3.0,
        true_y0=torch.zeros(batch_size, seq_len),
        true_y1=torch.zeros(batch_size, seq_len),
        pred_cate=torch.ones(batch_size, seq_len) * 3.0,
        true_cate=torch.zeros(batch_size, seq_len),
    )

    # Batch 2: Offset 5 -> RMSE 5.0
    evaluator.update(
        pred_y0=torch.ones(batch_size, seq_len) * 5.0,
        pred_y1=torch.ones(batch_size, seq_len) * 5.0,
        true_y0=torch.zeros(batch_size, seq_len),
        true_y1=torch.zeros(batch_size, seq_len),
        pred_cate=torch.ones(batch_size, seq_len) * 5.0,
        true_cate=torch.zeros(batch_size, seq_len),
    )

    metrics = evaluator.compute()

    # Mean RMSE: ((3.0 * 2) + (5.0 * 2)) / 4 = 4.0
    assert metrics["po/rmse_y0"] == pytest.approx(4.0)
    assert metrics["cate/rmse"] == pytest.approx(4.0)
    assert metrics["eval_episodes"] == batch_size * 2


def test_extra_metrics_and_prefixes(evaluator):
    """Test that extra_metrics are correctly accumulated and prefixes are applied."""
    batch_size, seq_len = 2, 5

    evaluator.update(
        pred_y0=torch.zeros(batch_size, seq_len),
        pred_y1=torch.zeros(batch_size, seq_len),
        true_y0=torch.zeros(batch_size, seq_len),
        true_y1=torch.zeros(batch_size, seq_len),
        pred_cate=torch.zeros(batch_size, seq_len),
        true_cate=torch.zeros(batch_size, seq_len),
        extra_metrics={"loss/total": 1.5, "loss/penalty": torch.tensor(0.5)},
    )

    # Compute with a prefix
    metrics = evaluator.compute(prefix="synt_point/")

    assert "synt_point/loss/total" in metrics
    assert "synt_point/loss/penalty" in metrics
    # Note: the implementation multiplies extra_metrics by batch_size and then
    # divides by num_episodes in compute(), effectively averaging them.
    assert metrics["synt_point/loss/total"] == pytest.approx(1.5)
    assert metrics["synt_point/loss/penalty"] == pytest.approx(0.5)
    assert metrics["synt_point/eval_episodes"] == batch_size
