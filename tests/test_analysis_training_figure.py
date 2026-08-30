"""Tests for the training-curve figure and its shared design helpers.

No network: the figure builder takes a :class:`History` and is handed a synthetic
one, so what is exercised is the binning, the derived reference values and the
rendering -- everything that could be silently wrong in a picture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis_training import design, loss_figure
from analysis_training.run_config import parse_script
from analysis_training.wandb_history import History

STEPS = 4000


@pytest.fixture
def run():
    return parse_script(loss_figure.DEFAULT_SCRIPT, {})


@pytest.fixture
def history():
    """A synthetic run: every key the figure reads, on a plausible scale."""
    step = np.arange(1, STEPS + 1, dtype=float)
    decay = np.exp(-step / 500.0)
    frame = pd.DataFrame(
        {
            "_step": step,
            "_runtime": step * 3.0,
            "_timestamp": 1.7e9 + step * 3.0,
            "loss/cate_dist": 2.3 * decay - 0.9 * (1 - decay),
            "loss/cate_mse": 1.5 * decay + 0.03,
            "loss/cate_mse_weighted": 0.5 * (1.5 * decay + 0.03),
            "loss/total": 2.3 * decay - 0.9 * (1 - decay) + 0.5 * (1.5 * decay + 0.03),
            "loss/hl_sigma": np.clip(0.5 - step * 1e-4, 0.01, 0.5),
            "target/cate_std": np.full_like(step, 1.3),
            "lr": np.minimum(step / 2500.0, 1.0) * 1e-4,
        }
    )
    return History(frame=frame, keys=tuple(frame.columns))


# --------------------------------------------------------------------------- #
# binning
# --------------------------------------------------------------------------- #


def test_binned_reduces_to_median_and_iqr():
    steps = np.repeat([10.0, 1000.0], 4)
    values = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    out = design.binned(steps, values, n_bins=2, log=True)
    assert out.median.tolist() == [2.5, 25.0]
    assert out.p25.tolist() == [1.75, 17.5]
    assert out.p75.tolist() == [3.25, 32.5]


def test_binned_median_ignores_outliers_a_mean_would_follow():
    """The objective's first steps spike far above the trend; that is why the
    reduction is a median."""
    steps = np.ones(11)
    values = np.array([1.0] * 10 + [1000.0])
    out = design.binned(steps, values, n_bins=1, log=True)
    assert out.median[0] == 1.0


def test_binned_last_step_is_not_a_bin_of_its_own():
    """`digitize` on the interior edges only, so the top edge folds into the
    last bin instead of opening an empty one."""
    steps = np.arange(1.0, 101.0)
    out = design.binned(steps, steps, n_bins=10, log=False)
    assert len(out.median) == 10
    assert out.median[-1] > out.median[-2]


def test_binned_x_is_where_the_data_is():
    """A sparse bin is drawn at its data's step, not at its edge."""
    steps = np.array([1.0, 2.0, 900.0, 1000.0])
    out = design.binned(steps, np.ones(4), n_bins=2, log=True)
    assert out.x[0] == pytest.approx(1.5)
    assert out.x[1] == pytest.approx(950.0)


def test_binned_span_filters_and_empty_span_raises():
    steps = np.arange(1.0, 101.0)
    out = design.binned(steps, steps, n_bins=5, log=False, span=(50, 100))
    assert out.median.min() >= 50
    with pytest.raises(ValueError, match="no samples"):
        design.binned(steps, steps, span=(1000, 2000))


# --------------------------------------------------------------------------- #
# derived reference values
# --------------------------------------------------------------------------- #


def test_uniform_loss_is_the_entropy_of_the_grid(run):
    """The do-nothing value of a cross-entropy against a *density* is
    log(v_max - v_min) -- the run's plus/minus 5 grid gives log 10, which is
    what the head's first logged step (2.3037) actually sits on."""
    assert loss_figure.uniform_loss(run) == pytest.approx(np.log(10.0))
    assert run.config.cate_vmax - run.config.cate_vmin == 10


def test_sigma_window_is_warmup_then_decay(run):
    """The shaded span is where sigma *moves*: constant through the warmup, then
    the linear decay."""
    lo, hi = loss_figure.sigma_window(run)
    assert (lo, hi) == (
        run.config.hl_warmup_sigma_steps,
        run.config.hl_warmup_sigma_steps + run.config.hl_linear_decay_steps,
    )
    assert (lo, hi) == (500, 3500)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def test_loss_figure_content_layout_and_shared_styling(history, run):
    fig = loss_figure.build_figure(history, run, n_bins=20)
    try:
        axes = fig.get_axes()
        assert len(axes) == 5
        titles = [ax.get_title(loc="left") for ax in axes[:4]]
        assert [t[:3] for t in titles] == ["(a)", "(b)", "(c)", "(d)"]
        overview, plateau, schedules, mean_term_detail = fig.get_axes()[:4]
        assert overview.get_subplotspec().colspan == schedules.get_subplotspec().colspan
        assert overview.get_xscale() == schedules.get_xscale() == "log"
        assert overview.get_xlim() == pytest.approx(schedules.get_xlim())
        assert _has_span(overview) and _has_span(schedules)
        assert _has_span(mean_term_detail)
        assert not _has_span(plateau)
        expected = design.binned(
            history.frame["_step"],
            history.frame["loss/cate_mse_weighted"],
            log=True,
            n_bins=20,
        )
        [curve] = mean_term_detail.get_lines()
        assert curve.get_ydata() == pytest.approx(expected.median)
        assert curve.get_color() == design.series("loss/cate_mse_weighted").color
        assert mean_term_detail.get_yscale() == "log"

        with_legend = [ax for ax in axes if ax.get_legend() is not None]
        assert len(with_legend) == 1
        labels = [t.get_text() for t in with_legend[0].get_legend().get_texts()]
        assert labels == [design.series(k).label for k in loss_figure.COMPONENTS]
        assert fig.get_figwidth() == pytest.approx(design.TEXT_WIDTH_IN)
    finally:
        design.plt.close(fig)


def _has_span(ax) -> bool:
    return any(
        patch.get_alpha() == design.SCHEDULE_SPAN["alpha"] for patch in ax.patches
    )


def test_panel_tag_clears_the_centred_title(history, run):
    """Matplotlib keeps one Text per title location, so a tag that only *sets*
    the left title leaves both rendering on top of each other."""
    fig, ax = design.plt.subplots()
    try:
        ax.set_title("objective")
        design.panel_tag(ax, "a")
        assert ax.get_title(loc="center") == ""
        assert ax.get_title(loc="left") == "(a) objective"
    finally:
        design.plt.close(fig)


def test_series_registry_refuses_an_unknown_key():
    with pytest.raises(KeyError, match="No series style"):
        design.series("loss/does_not_exist")
