"""Tests for analysis_prior.generate."""


import numpy as np


import pytest


from analysis_prior import generate as gen


from analysis_prior.config import PriorSpec


@pytest.fixture(scope="module")
def tiny_spec():
    """Smallest spec that still exercises the full DAG. Kept module-scoped
    because generation dominates the runtime of this file."""
    return PriorSpec(
        name="tiny",
        n_datasets=2,
        seq_len=60,
        mc_cate=3,
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


@pytest.fixture(scope="module")
def tiny_cache(tiny_spec):
    return gen.generate(tiny_spec, n_jobs=1, progress=False)


def _as_arrays(cache):
    """Flatten a cache into a comparable dict of arrays."""
    out = {}
    for r in cache.records:
        for name, value in r.to_dict().items():
            out[f"{r.index}:{name}"] = value
    return out


def assert_caches_identical(a, b):
    da, db = _as_arrays(a), _as_arrays(b)
    assert da.keys() == db.keys()
    for key in da:
        np.testing.assert_array_equal(da[key], db[key], err_msg=f"mismatch at {key}")


def test_shapes_are_consistent(tiny_spec, tiny_cache):
    for r in tiny_cache:
        assert r.n_rows == tiny_spec.seq_len
        for name in ("U", "X", "W", "Z"):
            block = getattr(r, name)
            assert block.ndim == 2
            assert block.shape[0] == tiny_spec.seq_len
        for name in ("A", "A_probs", "Y_obs", "Y0", "Y1", "MU0", "MU1"):
            vec = getattr(r, name)
            assert vec.shape == (tiny_spec.seq_len,)


def test_observed_outcome_matches_potential_outcomes(tiny_cache):
    # Y_obs is the consistency assumption made concrete; if this breaks, every
    # downstream estimate is meaningless.
    for r in tiny_cache:
        expected = np.where(r.A == 1, r.Y1, r.Y0)
        np.testing.assert_array_equal(r.Y_obs, expected)


def test_generation_is_reproducible(tiny_spec, tiny_cache):
    assert_caches_identical(
        tiny_cache, gen.generate(tiny_spec, n_jobs=1, progress=False)
    )


def test_save_load_round_trip(tiny_spec, tiny_cache, tmp_path):
    gen.save(tiny_cache, tmp_path)
    assert_caches_identical(tiny_cache, gen.load(tiny_spec, tmp_path))


def _renamed(spec, name, **kwargs):
    return PriorSpec(**{**spec.to_dict(), "name": name, **kwargs})
