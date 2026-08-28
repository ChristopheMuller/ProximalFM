"""Tests for the single-posterior inspection figure."""

from __future__ import annotations

import numpy as np

from analysis_training import posterior_snapshot


def posterior():
    edges = np.linspace(-2.0, 2.0, 9)
    probs = np.array([0.01, 0.03, 0.08, 0.38, 0.28, 0.12, 0.06, 0.04])
    return edges, probs


def test_smoothing_preserves_cdf_endpoints_and_nonnegative_density():
    edges, probs = posterior()
    raw, smooth, density = posterior_snapshot.smooth_cdf(
        edges, probs, window=5, polyorder=2
    )
    assert raw[0] == 0.0
    assert raw[-1] == 1.0
    assert smooth[0] == 0.0
    assert smooth[-1] == 1.0
    assert np.all(np.diff(smooth) >= 0.0)
    assert np.all(density >= 0.0)


def test_builds_four_posterior_panels_with_true_lines():
    edges, probs = posterior()
    fig = posterior_snapshot.build_figure(
        edges, probs, true_value=0.2, cdf_window=5, cdf_polyorder=2
    )
    try:
        axes = fig.get_axes()
        assert len(axes) == 4
        assert all(len(ax.lines) >= 1 for ax in axes)
        assert all(
            any(np.allclose(line.get_xdata(), 0.2) for line in ax.lines) for ax in axes
        )
        assert [text.get_text() for text in axes[0].get_legend().get_texts()] == [
            "true CATE"
        ]
        assert axes[0].get_xlim()[1] - axes[0].get_xlim()[0] < 4.0
    finally:
        posterior_snapshot.plt.close(fig)


def test_zoom_limits_expand_to_include_outlying_true_value():
    edges, probs = posterior()
    lo, hi = posterior_snapshot.posterior_view_limits(
        edges, probs, true_value=3.0, n_sd=2.0
    )
    assert lo < 3.0 < hi
