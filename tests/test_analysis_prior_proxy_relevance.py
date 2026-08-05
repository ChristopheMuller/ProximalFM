"""Tests for the proxy-relevance audit (B.3.4), on the fast linear model.

The synthetic records below make ``W`` a strong image of ``U`` and ``Z`` a
deliberately weak one, so the audit's two readings are both checkable: a large
incremental ``dR2`` where the proxy carries the confounder, a near-zero one
where it does not -- and, critically, a *weak* proxy that nonetheless satisfies
the counting heuristic (``d_Z >= d_U``), which is the disagreement the audit
exists to quantify.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from analysis_prior import generate as gen
from analysis_prior.analyses.proxy_relevance import (
    PROXIES,
    WEAK_THRESHOLD,
    ProxyRelevanceAnalysis,
    RelevanceResults,
    _knobs,
    _slack,
    relevance_frame,
)
from analysis_prior.config import PriorSpec


def _record(index=0, n=400, seed=0, dim_U=2, z_signal=0.05, dim_Z=3):
    """A record where ``W`` carries ``U`` strongly and ``Z`` barely at all.

    ``Z`` is still wide enough to pass the counting heuristic, which is the
    point: dimension slack is satisfied while functional relevance is not.
    """
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, dim_U)).astype(np.float32)
    X = rng.normal(size=(n, 2)).astype(np.float32)

    def proxy(scale, width):
        base = U[:, : min(width, dim_U)] * scale
        pad = np.zeros((n, max(width - base.shape[1], 0)), dtype=np.float32)
        signal = np.hstack([base, pad])
        return (signal + rng.normal(scale=0.3, size=(n, width))).astype(np.float32)

    W = proxy(2.0, max(dim_U, 2))
    Z = proxy(z_signal, dim_Z)

    a_probs = np.full(n, 0.5, dtype=np.float32)
    A = (rng.uniform(size=n) < a_probs).astype(np.float32)
    y0 = (U[:, 0] if dim_U else np.zeros(n)) + X[:, 0]
    y1 = y0 + 1.0
    return gen.PriorDatasetRecord(
        index=index,
        seed=seed,
        U=U,
        X=X,
        W=W,
        Z=Z,
        A=A,
        A_probs=a_probs,
        Y_obs=np.where(A == 1, y1, y0).astype(np.float32),
        Y0=y0.astype(np.float32),
        Y1=y1.astype(np.float32),
        MU0=y0.astype(np.float32),
        MU1=y1.astype(np.float32),
        sampled_hp={
            "dim_U": dim_U,
            "dim_X": 2,
            "noise_share_of_input_in_W": 0.2,
            "covariate_share_of_X_in_W": 0.5,
            "noise_share_of_input_in_Z": 0.6,
            "covariate_share_of_X_in_Z": 0.5,
        },
    )


def _frame(records, tmp_path, **kwargs):
    return relevance_frame(
        records,
        cache_key="test",
        model="linear",
        n_splits=3,
        max_rows=None,
        regression_cache_dir=tmp_path,
        progress=False,
        **kwargs,
    )


def test_slack_uses_delivered_widths():
    rec = _record(dim_U=2, dim_Z=3)
    assert _slack(rec) == {"W": rec.W.shape[1] - 2, "Z": 1}


def test_knobs_derive_the_expected_u_share():
    """``u_share_expected = d_U (1 - s) / (d_U + p d_X)`` -- both dilution knobs
    in one number, and lower for the noisier proxy."""
    rec = _record(dim_U=2)
    w, z = _knobs(rec, "W"), _knobs(rec, "Z")
    assert w["noise_share"] == 0.2 and z["noise_share"] == 0.6
    # d_U=2, d_X=2, p=0.5 -> signal width 3.
    assert w["u_share_expected"] == pytest.approx(2 * 0.8 / 3)
    assert z["u_share_expected"] == pytest.approx(2 * 0.4 / 3)
    assert z["u_share_expected"] < w["u_share_expected"]


def test_knobs_are_nan_without_sampled_hp():
    """Caches written before ``sampled_hp`` existed still render: the knob
    panel drops out rather than raising."""
    rec = dataclasses.replace(_record(), sampled_hp=None)
    assert all(np.isnan(v) for v in _knobs(rec, "W").values())


def test_frame_has_one_row_per_u_column_and_proxy(tmp_path):
    records = [_record(index=i, seed=i, dim_U=2) for i in range(3)]
    frame = _frame(records, tmp_path)
    assert len(frame) == 3 * 2 * len(PROXIES)
    assert set(frame["proxy"]) == set(PROXIES)
    assert set(frame["u_column"]) == {0, 1}


def test_delta_is_the_increment_over_the_covariates(tmp_path):
    frame = _frame([_record()], tmp_path)
    np.testing.assert_allclose(
        frame["delta_r2"], frame["r2_full"] - frame["r2_covariates"], rtol=1e-9
    )


def test_strong_proxy_is_relevant_and_weak_one_is_not(tmp_path):
    records = [_record(index=i, seed=i) for i in range(4)]
    per = RelevanceResults(
        frame=_frame(records, tmp_path), model="linear", n_skipped_no_u=0
    ).per_dataset()

    w = per.loc[per["proxy"] == "W", "delta_r2"]
    z = per.loc[per["proxy"] == "Z", "delta_r2"]
    # W is a high-SNR image of U, so it adds a great deal over X alone...
    assert w.median() > 0.5
    # ...while Z, wide enough to pass the counting heuristic, adds almost
    # nothing: heuristic satisfied, functional relevance absent.
    assert z.median() < WEAK_THRESHOLD


def test_datasets_without_u_are_skipped(tmp_path):
    frame = _frame([_record(index=0), _record(index=1, seed=1, dim_U=0)], tmp_path)
    assert set(frame["index"]) == {0}


def test_covariate_only_fit_is_shared_between_the_two_proxies(tmp_path):
    """``R2(u_j | X)`` is one predictor set, not two: the second proxy's call
    must hit the same cache entry rather than refit it."""
    _frame([_record(dim_U=1)], tmp_path)
    # Per U column: X, X+W, W, X+Z, Z -- five, not six.
    assert len(list(tmp_path.glob("*.npz"))) == 5

    n_first = len(list(tmp_path.glob("*.npz")))
    _frame([_record(dim_U=1)], tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == n_first


def test_max_rows_truncates(tmp_path):
    frame = relevance_frame(
        [_record(n=500)],
        cache_key="test",
        model="linear",
        n_splits=2,
        max_rows=200,
        regression_cache_dir=tmp_path,
        progress=False,
    )
    assert np.isfinite(frame["delta_r2"]).all()


def test_proxy_frame_and_slack_min(tmp_path):
    records = [_record(index=i, seed=i) for i in range(3)]
    results = RelevanceResults(
        frame=_frame(records, tmp_path), model="linear", n_skipped_no_u=0
    )
    frame_w = results.proxy_frame("W")
    assert list(frame_w.index) == [0, 1, 2]
    np.testing.assert_allclose(frame_w["delta_r2"], results.series("W"))
    # Each proxy keeps its own slack; the joint statistic the counting heuristic
    # uses -- and the figure deliberately does not colour by -- is their minimum.
    np.testing.assert_array_equal(
        results.slack_min(),
        np.minimum(results.series("W", "slack"), results.series("Z", "slack")),
    )


def test_per_dataset_min_is_at_most_the_mean(tmp_path):
    records = [_record(index=i, seed=i, dim_U=3) for i in range(2)]
    per = RelevanceResults(
        frame=_frame(records, tmp_path), model="linear", n_skipped_no_u=0
    ).per_dataset()
    assert (per["delta_r2_min"] <= per["delta_r2"] + 1e-9).all()


def test_analysis_end_to_end(tmp_path):
    spec = PriorSpec(
        name="tiny-relevance",
        n_datasets=6,
        seq_len=200,
        mc_cate=1,
        min_dim_U=2,
        max_dim_U=2,
        min_dim_W=2,
        max_dim_W=3,
        min_dim_Z=2,
        max_dim_Z=3,
        min_dim_X=2,
        max_dim_X=3,
        seed=0,
    )
    cache = gen.generate(spec, n_jobs=1, progress=False)

    analysis = ProxyRelevanceAnalysis()
    analysis.model = "linear"
    analysis.n_splits = 3
    analysis.max_rows = None
    analysis.regression_cache_dir = tmp_path / "reg"
    analysis.frame_cache_dir = tmp_path / "frames"

    artifacts = analysis.run(cache, tmp_path / "out")
    for path in artifacts.figures + artifacts.tables:
        assert path.exists() and path.stat().st_size > 0
    assert "proxy_relevance.pdf" in {p.name for p in artifacts.figures}
    assert "proxy_relevance_table.tex" in {p.name for p in artifacts.tables}
    assert artifacts.summary["model"] == "linear"
    for key in (
        "median_delta_r2_mean_W",
        "median_delta_r2_mean_Z",
        "share_weak_mean_W",
        "share_positive_mean_W",
        "median_delta_r2_weakest_Z",
        "share_heuristic_passes",
        "share_both_weak",
    ):
        assert key in artifacts.summary
        assert np.isfinite(artifacts.summary[key])
    # The driver correlations are reported for every candidate explanation; they
    # are NaN-tolerant by design (a spec that pins a knob has no rank
    # correlation to report), so only their presence is asserted here.
    for key in (
        "spearman_dim_proxy_W",
        "spearman_dim_U_W",
        "spearman_slack_W",
        "spearman_slack_min_W",
        "spearman_noise_share_W",
        "spearman_u_share_expected_Z",
    ):
        assert key in artifacts.summary

    # Two aggregations per proxy.
    table = pd.read_csv(next(p for p in artifacts.tables if p.suffix == ".csv"))
    assert len(table) == 2 * len(PROXIES)


def test_registered_under_its_name():
    from analysis_prior.analyses import get_analysis

    assert get_analysis("proxy_relevance").name == "proxy_relevance"


@pytest.mark.parametrize("proxy", list(PROXIES))
def test_series_selects_one_proxy(tmp_path, proxy):
    records = [_record(index=i, seed=i) for i in range(2)]
    results = RelevanceResults(
        frame=_frame(records, tmp_path), model="linear", n_skipped_no_u=0
    )
    assert results.series(proxy).shape == (2,)
