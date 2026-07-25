"""Tests for the "visualize the prior" overview analysis (appendix B.2.x).

Two layers:

- the pure helpers -- categorical-column detection, the pooled summary
  statistics, and the seeded gallery samplers (feature marginals, within-block
  and cross-block pairs);
- ``PriorOverviewAnalysis`` end-to-end on a tiny generated cache: it writes the
  four promised figures and a self-describing summary.
"""

import numpy as np

from analysis_prior import generate as gen
from analysis_prior.analyses.overview import (
    CROSS_PAIRS,
    GALLERY_BLOCKS,
    PriorOverviewAnalysis,
    _categorical_mask,
    _feature_columns,
    sample_cross_pairs,
    sample_distributions,
    sample_within_pairs,
)
from analysis_prior.config import PriorSpec


def _record(index=0, n=400, seed=0):
    """A synthetic record: one continuous + one categorical column per block."""
    rng = np.random.default_rng(seed)

    def block(d):
        cont = rng.normal(size=(n, d - 1))
        cat = rng.integers(0, 3, size=(n, 1)).astype(np.float64)
        return np.column_stack([cont, cat]).astype(np.float32)

    U, X, W, Z = block(2), block(3), block(2), block(3)
    a_probs = np.clip(1.0 / (1.0 + np.exp(-U[:, 0])), 1e-3, 1 - 1e-3)
    A = (rng.random(n) < a_probs).astype(np.float32)
    y0 = rng.normal(size=n).astype(np.float32)
    y1 = (y0 + 1.5).astype(np.float32)
    return gen.PriorDatasetRecord(
        index=index,
        seed=seed,
        U=U,
        X=X,
        W=W,
        Z=Z,
        A=A,
        A_probs=a_probs.astype(np.float32),
        Y_obs=y0,
        Y0=y0,
        Y1=y1,
        MU0=y0,
        MU1=y1,
        sampled_hp={"max_categories": 20, "dim_U": 2},
        gamma_hetero=0.5,
    )


def test_categorical_mask_flags_low_cardinality_columns():
    rec = _record()
    # X has 2 continuous + 1 categorical column.
    assert _categorical_mask(rec.X, 20).tolist() == [False, False, True]
    assert _categorical_mask(np.empty((5, 0)), 20).shape == (0,)


def test_feature_columns_maps_scalar_blocks():
    rec = _record()
    np.testing.assert_array_equal(_feature_columns(rec, "A").ravel(), rec.A_probs)
    np.testing.assert_array_equal(_feature_columns(rec, "Y").ravel(), rec.Y_obs)
    assert _feature_columns(rec, "U").shape[1] == 2


def test_sample_distributions_shapes_and_scalar_blocks():
    records = [_record(index=i, seed=i) for i in range(6)]
    rng = np.random.default_rng(0)
    dist = sample_distributions(records, rng, k=4)
    assert set(dist) == set(GALLERY_BLOCKS)
    for block, panels in dist.items():
        assert len(panels) == 4
    # A's feature is the propensity; its label is pi and it has no second series.
    assert dist["A"][0].label.startswith(r"$\pi")
    assert dist["A"][0].values2 is None
    # Y overlays both potential outcomes.
    y = dist["Y"][0]
    assert y.values2 is not None
    np.testing.assert_array_equal(y.values, records[y.index].Y0)
    np.testing.assert_array_equal(y.values2, records[y.index].Y1)


def test_sample_within_pairs_special_cases():
    records = [_record(index=i, seed=i) for i in range(6)]
    rng = np.random.default_rng(0)
    within = sample_within_pairs(records, rng, k=3, scatter_rows=100)
    # A: propensity vs the (jittered) draw.
    a = within["A"][0]
    assert a.xlabel == r"$\pi$" and a.ylabel == "$A$"
    # Y: the two potential outcomes.
    y = within["Y"][0]
    assert y.xlabel == "$Y_0$" and y.ylabel == "$Y_1$"
    # Scatter rows are subsampled to the requested cap.
    assert len(a.x) == 100 and len(a.y) == 100


def test_sample_cross_pairs_keys_and_content():
    records = [_record(index=i, seed=i) for i in range(6)]
    rng = np.random.default_rng(0)
    cross = sample_cross_pairs(records, rng, k=3, scatter_rows=100)
    assert list(cross) == [f"{a}-{b}" for a, b in CROSS_PAIRS]
    # The U-A pair scatters a U feature against the propensity.
    ua = cross["U-A"][0]
    assert ua.ylabel == r"$\pi$"
    assert len(ua.x) == 100


def test_analysis_end_to_end(tmp_path):
    spec = PriorSpec(
        name="tiny-overview",
        n_datasets=6,
        seq_len=200,
        mcmc_cate=1,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=2,
        max_dim_W=3,
        min_dim_Z=2,
        max_dim_Z=3,
        min_dim_X=2,
        max_dim_X=3,
        seed=0,
    )
    cache = gen.generate(spec, n_jobs=1, progress=False)

    analysis = PriorOverviewAnalysis()
    analysis.n_datasets = 6
    analysis.k_rows = 3
    artifacts = analysis.run(cache, tmp_path / "out")

    names = {p.name for p in artifacts.figures}
    assert names == {
        "prior_feature_distributions.pdf",
        "prior_feature_distributions.png",
        "prior_feature_pairs.pdf",
        "prior_feature_pairs.png",
        "prior_feature_pairs_cross.pdf",
        "prior_feature_pairs_cross.png",
    }
    for path in artifacts.figures:
        assert path.exists() and path.stat().st_size > 0
    assert artifacts.summary["n_datasets"] == 6
