"""Tests for the two-context posterior-evolution figure."""

from __future__ import annotations

import numpy as np
import pytest

from analysis_training import posterior_evolution


def _posterior_grid():
    edges = np.linspace(-2.0, 2.0, 42)
    centres = 0.5 * (edges[:-1] + edges[1:])
    grid = []
    for context_scale in (1.0, 0.65):
        row = []
        for step_scale in (1.0, 0.7, 0.45):
            scale = context_scale * step_scale
            probs = np.exp(-0.5 * ((centres - 0.2) / scale) ** 2)
            row.append((edges, probs / probs.sum()))
        grid.append(row)
    return grid


def test_builds_two_panels_with_shared_limits_and_training_curves():
    fig = posterior_evolution.build_figure(
        _posterior_grid(),
        [16, 2048],
        [1, 500, 75_000],
        true_value=0.2,
        cdf_window=9,
        cdf_polyorder=2,
    )
    try:
        axes = fig.get_axes()
        assert len(axes) == 2
        assert axes[0].get_xlim() == axes[1].get_xlim()
        assert axes[0].get_ylim() == axes[1].get_ylim()
        assert all(len(ax.lines) == 4 for ax in axes)
        assert all(
            any(np.allclose(line.get_xdata(), 0.2) for line in ax.lines) for ax in axes
        )
        assert axes[0].get_title(loc="left").startswith("(a)")
        assert axes[1].get_title(loc="left").startswith("(b)")
        assert [text.get_text() for text in fig.legends[0].get_texts()] == [
            "1",
            "500",
            "75,000",
            "true CATE",
        ]
    finally:
        posterior_evolution.plt.close(fig)


def test_requires_exactly_two_contexts():
    with pytest.raises(ValueError, match="exactly two contexts"):
        posterior_evolution.build_figure(
            _posterior_grid()[:1], [16], [1, 500, 75_000], true_value=0.2
        )
