"""Tests for the distributional validation figure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_training import distributional_figure
from analysis_training.wandb_history import History


def history() -> History:
    step = np.array([1.0, 50.0, 500.0, 5_000.0, 75_000.0])
    frame = pd.DataFrame(
        {
            "_step": step,
            "synt_dist/crps": [0.8, 0.6, 0.4, 0.3, 0.2],
            "synt_dist/sd_over_rmse": [0.5, 0.7, 0.9, 0.98, 1.0],
            "synt_dist/cov50_pooled": [0.4, 0.45, 0.48, 0.5, 0.5],
            "synt_dist/cov50/p25": [0.25, 0.3, 0.35, 0.38, 0.4],
            "synt_dist/cov50/p75": [0.6, 0.65, 0.68, 0.7, 0.65],
            "synt_dist/cov90_pooled": [0.6, 0.7, 0.8, 0.87, 0.9],
            "synt_dist/cov90/p25": [0.4, 0.5, 0.65, 0.75, 0.82],
            "synt_dist/cov90/p75": [0.8, 0.9, 0.95, 0.98, 0.97],
            "synt_dist/width90": [2.0, 1.8, 1.4, 1.0, 0.8],
            "synt_dist/width50": [1.0, 0.9, 0.7, 0.5, 0.4],
        }
    )
    return History(frame=frame, keys=tuple(frame.columns))


def test_builds_four_labelled_panels():
    fig = distributional_figure.build_figure(history())
    try:
        axes = fig.get_axes()
        assert len(axes) == 4
        titles = [ax.get_title(loc="left") for ax in axes]
        assert [title[:3] for title in titles] == ["(a)", "(b)", "(c)", "(d)"]
        assert all(ax.get_xscale() == "log" for ax in axes)
        assert axes[1].get_yscale() == "log"
        assert axes[3].get_yscale() == "log"
    finally:
        distributional_figure.plt.close(fig)


def test_coverage_panel_has_two_curves_and_episode_bands_outside_legend():
    fig = distributional_figure.build_figure(history())
    try:
        axes = fig.get_axes()
        assert len(axes[0].collections) == 0
        assert len(axes[1].collections) == 0
        assert len(axes[2].collections) == 2
        assert len(axes[2].lines) == 4
        labels = [text.get_text() for text in axes[2].get_legend().get_texts()]
        assert labels == ["90% coverage", "50% coverage"]
        assert len(axes[3].collections) == 0
        assert len(axes[3].lines) == 2
    finally:
        distributional_figure.plt.close(fig)


def test_trajectories_are_log_binned_for_display():
    fig = distributional_figure.build_figure(history(), n_bins=2)
    try:
        axes = fig.get_axes()
        assert len(axes[0].lines[0].get_xdata()) == 2
        assert len(axes[2].lines[0].get_xdata()) == 2
    finally:
        distributional_figure.plt.close(fig)


def test_calibration_reference_lines_are_present():
    fig = distributional_figure.build_figure(history())
    try:
        axes = fig.get_axes()
        assert any(np.allclose(line.get_ydata(), 1.0) for line in axes[1].lines)
        assert any(np.allclose(line.get_ydata(), 0.9) for line in axes[2].lines)
        assert any(np.allclose(line.get_ydata(), 0.5) for line in axes[2].lines)
    finally:
        distributional_figure.plt.close(fig)
