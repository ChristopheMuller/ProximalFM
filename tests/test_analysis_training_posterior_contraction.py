"""Tests for the posterior-width contraction heatmap."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from analysis_training import posterior_contraction


def test_parse_int_list():
    assert posterior_contraction.parse_int_list("16, 64,2048", name="n-ctx") == [
        16,
        64,
        2048,
    ]
    with pytest.raises(ValueError):
        posterior_contraction.parse_int_list("16,16", name="n-ctx")


def test_builds_evenly_spaced_heatmap_with_shared_colourbar():
    contexts = [16, 64]
    steps = [1, 100]
    widths = np.array([[2.0, 0.8], [1.2, 0.4]])
    fig = posterior_contraction.build_figure(widths, contexts, steps)
    try:
        axes = fig.get_axes()
        assert len(axes) == 2
        assert axes[0].get_xscale() == "linear"
        assert axes[0].get_yscale() == "linear"
        np.testing.assert_array_equal(axes[0].get_xticks(), [0.0, 1.0])
        np.testing.assert_array_equal(axes[0].get_yticks(), [0.0, 1.0])
        assert axes[0].get_title() == ""
        assert axes[0].get_title(loc="left") == ""
        assert axes[0].collections
        assert axes[1].get_ylabel() == "mean predictive SD"
    finally:
        posterior_contraction.plt.close(fig)


def test_episode_cache_reuses_and_extends_contexts(tmp_path, monkeypatch):
    data_dir = tmp_path / "set_a"
    (data_dir / "val").mkdir(parents=True)
    checkpoint = tmp_path / "step-000001.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    monkeypatch.setattr(
        posterior_contraction,
        "iter_frozen_episodes",
        lambda *_args, **_kwargs: iter([0, 1]),
    )

    def fake_episode_predictive(_model, batch, _record, *, n_ctx, **_kwargs):
        calls.append((batch, n_ctx))
        return torch.tensor([[float(batch + n_ctx)]]), torch.tensor([0.0, 1.0])

    def fake_predictive_mean_var(probs, _edges):
        return probs, probs.square()

    monkeypatch.setattr(
        posterior_contraction, "episode_predictive", fake_episode_predictive
    )
    monkeypatch.setattr(
        posterior_contraction, "predictive_mean_var", fake_predictive_mean_var
    )

    kwargs = dict(
        model=object(),
        data_dir=data_dir,
        records=[{}, {}],
        n_max=16,
        n_datasets=2,
        progress=False,
        ckpt_path=checkpoint,
        cache_dir=tmp_path / "cache",
        dataset_digest="digest-a",
    )
    first = posterior_contraction.posterior_widths(n_ctx_list=[4], **kwargs)
    assert calls == [(0, 4), (1, 4)]

    calls.clear()
    second = posterior_contraction.posterior_widths(n_ctx_list=[4], **kwargs)
    np.testing.assert_array_equal(second, first)
    assert calls == []

    calls.clear()
    expanded = posterior_contraction.posterior_widths(n_ctx_list=[4, 8], **kwargs)
    np.testing.assert_array_equal(expanded[:1], first)
    assert calls == [(0, 8), (1, 8)]
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 2


def test_cache_identity_includes_dataset_folder(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    common = dict(
        ckpt_path=checkpoint,
        dataset_index=3,
        dataset_digest="same-content",
        n_max=128,
    )
    first = posterior_contraction._cache_identity(
        data_dir=tmp_path / "folder_a", **common
    )
    second = posterior_contraction._cache_identity(
        data_dir=tmp_path / "folder_b", **common
    )
    assert posterior_contraction._cache_path(first, tmp_path) != (
        posterior_contraction._cache_path(second, tmp_path)
    )
