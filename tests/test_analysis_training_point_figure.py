"""Tests for the point-estimate validation figure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_training import point_figure
from analysis_training.wandb_history import History


def history() -> History:
    step = np.array([1.0, 50.0, 500.0, 5_000.0, 75_000.0])
    frame = pd.DataFrame(
        {
            "_step": step,
            "synt_point/ate/rmse": [1.0, 0.8, 0.6, 0.5, 0.4],
            "synt_point/ate/bias_norm": [0.4, 0.2, 0.1, 0.05, 0.01],
            "synt_point/ate/bias_norm/p25": [0.1, 0.0, -0.1, -0.08, -0.05],
            "synt_point/ate/bias_norm/p75": [0.8, 0.5, 0.3, 0.2, 0.1],
            "synt_point/cate/npehe": [1.2, 1.0, 0.8, 0.7, 0.6],
            "synt_point/cate/npehe/p25": [1.0, 0.8, 0.6, 0.5, 0.4],
            "synt_point/cate/npehe/p75": [1.4, 1.2, 1.0, 0.9, 0.8],
            "synt_point/cate/r2_shape": [-0.2, 0.0, 0.3, 0.5, 0.6],
            "synt_point/cate/r2_shape/p25": [-0.5, -0.2, 0.0, 0.2, 0.3],
            "synt_point/cate/r2_shape/p75": [0.1, 0.2, 0.5, 0.7, 0.8],
        }
    )
    return History(frame=frame, keys=tuple(frame.columns))


def test_builds_four_labelled_panels():
    fig = point_figure.build_figure(history())
    try:
        axes = fig.get_axes()
        assert len(axes) == 4
        titles = [ax.get_title(loc="left") for ax in axes]
        assert [title[:3] for title in titles] == ["(a)", "(b)", "(c)", "(d)"]
        assert all(ax.get_xscale() == "log" for ax in axes)
    finally:
        point_figure.plt.close(fig)


def test_ate_rmse_has_no_episode_band_but_other_metrics_do():
    fig = point_figure.build_figure(history())
    try:
        axes = fig.get_axes()
        assert len(axes[0].collections) == 0
        assert all(len(ax.collections) == 1 for ax in axes[1:])
    finally:
        point_figure.plt.close(fig)


def test_trajectories_are_log_binned_for_display():
    fig = point_figure.build_figure(history(), n_bins=2)
    try:
        axes = fig.get_axes()
        assert len(axes[0].lines[0].get_xdata()) == 2
        assert len(axes[1].lines[0].get_xdata()) == 2
    finally:
        point_figure.plt.close(fig)


def test_constant_ate_reference_lines_are_present():
    fig = point_figure.build_figure(history())
    try:
        axes = fig.get_axes()
        assert any(np.allclose(line.get_ydata(), 1.0) for line in axes[2].lines)
        assert any(np.allclose(line.get_ydata(), 0.0) for line in axes[3].lines)
    finally:
        point_figure.plt.close(fig)
