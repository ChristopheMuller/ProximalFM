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
        y_obs_std=torch.ones(batch_size, 1),
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


@pytest.mark.parametrize("batch_size, seq_len", [(1, 10), (4, 5), (8, 20)])
def test_update_perfect_prediction(evaluator, batch_size, seq_len):
    """Test that when predictions exactly match true values, NRMSE is 0."""
    pred_y0 = torch.rand(batch_size, seq_len)
    pred_y1 = torch.rand(batch_size, seq_len)
    pred_cate = torch.rand(batch_size, seq_len)

    y_obs_std = torch.ones(batch_size, 1)

    evaluator.update(
        pred_y0=pred_y0,
        pred_y1=pred_y1,
        true_y0=pred_y0.clone(),
        true_y1=pred_y1.clone(),
        pred_cate=pred_cate,
        true_cate=pred_cate.clone(),
        y_obs_std=y_obs_std,
    )

    metrics = evaluator.compute()

    assert metrics["po/nrmse_y0"] == pytest.approx(0.0)
    assert metrics["po/nrmse_y1"] == pytest.approx(0.0)
    assert metrics["cate/nrmse"] == pytest.approx(0.0)
    assert metrics["eval/episodes"] == batch_size


@pytest.mark.parametrize("offset, scale", [(1.0, 1.0), (2.0, 2.0), (0.5, 0.1)])
def test_update_known_values(evaluator, offset, scale):
    """Test NRMSE computation against manually calculated expected values."""
    batch_size, seq_len = 2, 5

    true_y0 = torch.zeros(batch_size, seq_len)
    true_y1 = torch.zeros(batch_size, seq_len)
    true_cate = torch.zeros(batch_size, seq_len)

    # offset makes MSE = offset^2, RMSE = offset
    pred_y0 = true_y0 + offset
    pred_y1 = true_y1 + offset
    pred_cate = true_cate + offset

    y_obs_std = torch.ones(batch_size, 1) * scale

    evaluator.update(
        pred_y0=pred_y0,
        pred_y1=pred_y1,
        true_y0=true_y0,
        true_y1=true_y1,
        pred_cate=pred_cate,
        true_cate=true_cate,
        y_obs_std=y_obs_std,
    )

    metrics = evaluator.compute()

    expected_nrmse = offset / scale

    assert metrics["po/nrmse_y0"] == pytest.approx(expected_nrmse)
    assert metrics["po/nrmse_y1"] == pytest.approx(expected_nrmse)
    assert metrics["cate/nrmse"] == pytest.approx(expected_nrmse)


# ---------------------------------------------------------
# Streaming & Accumulation Tests
# ---------------------------------------------------------


def test_update_streaming_accumulation(evaluator):
    """Test that NRMSE is correctly averaged over multiple sequential micro-batches."""
    batch_size, seq_len = 2, 10

    # Batch 1: Offset 3 -> RMSE 3.
    evaluator.update(
        pred_y0=torch.ones(batch_size, seq_len) * 3.0,
        pred_y1=torch.ones(batch_size, seq_len) * 3.0,
        true_y0=torch.zeros(batch_size, seq_len),
        true_y1=torch.zeros(batch_size, seq_len),
        pred_cate=torch.ones(batch_size, seq_len) * 3.0,
        true_cate=torch.zeros(batch_size, seq_len),
        y_obs_std=torch.ones(batch_size, 1),
    )

    # Batch 2: Offset 5 -> RMSE 5.
    evaluator.update(
        pred_y0=torch.ones(batch_size, seq_len) * 5.0,
        pred_y1=torch.ones(batch_size, seq_len) * 5.0,
        true_y0=torch.zeros(batch_size, seq_len),
        true_y1=torch.zeros(batch_size, seq_len),
        pred_cate=torch.ones(batch_size, seq_len) * 5.0,
        true_cate=torch.zeros(batch_size, seq_len),
        y_obs_std=torch.ones(batch_size, 1),
    )

    metrics = evaluator.compute()

    # Mean RMSE across all 4 episodes: (3 + 3 + 5 + 5) / 4 = 4.0
    assert metrics["po/nrmse_y0"] == pytest.approx(4.0)
    assert metrics["po/nrmse_y1"] == pytest.approx(4.0)
    assert metrics["cate/nrmse"] == pytest.approx(4.0)
    assert metrics["eval/episodes"] == batch_size * 2


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
        y_obs_std=torch.ones(batch_size, 1),
        extra_metrics={"loss/total": 1.5, "loss/penalty": torch.tensor(0.5)},
    )

    # Compute with a prefix
    metrics = evaluator.compute(prefix="val_")

    assert "val_loss/total" in metrics
    assert "val_loss/penalty" in metrics
    assert metrics["val_loss/total"] == pytest.approx(1.5)
    assert metrics["val_loss/penalty"] == pytest.approx(0.5)
    assert metrics["val_eval/episodes"] == batch_size
