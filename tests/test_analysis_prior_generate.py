"""Tests for analysis_prior.generate."""

import numpy as np
import pytest

from analysis_prior import generate as gen
from analysis_prior.config import PriorSpec, pinned


@pytest.fixture(scope="module")
def tiny_spec():
    """Smallest spec that still exercises the full DAG. Kept module-scoped
    because generation dominates the runtime of this file."""
    return PriorSpec(
        name="tiny",
        n_datasets=2,
        seq_len=60,
        mcmc_cate=3,
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


# --------------------------------------------------------------------------
# Shapes and content
# --------------------------------------------------------------------------


def test_generate_returns_requested_number_of_datasets(tiny_spec, tiny_cache):
    assert len(tiny_cache) == tiny_spec.n_datasets
    assert [r.index for r in tiny_cache] == [0, 1]


def test_records_carry_their_generating_seed(tiny_spec, tiny_cache):
    for i, record in enumerate(tiny_cache):
        assert record.seed == tiny_spec.dataset_seed(i)


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


def test_blocks_are_trimmed_to_active_width(tiny_spec, tiny_cache):
    # The generator pads each block up to max_dim_*; the cache must strip that,
    # otherwise every analysis would silently regress on columns of zeros.
    for r in tiny_cache:
        dims = r.dims
        # The generator rejects and redraws any dataset with an empty X, W or Z
        # (_dataset.py: `if (d_X > 0).all() and (d_W > 0).all() and (d_Z > 0).all()`).
        assert 0 < dims["d_X"] <= tiny_spec.max_dim_X
        assert 0 < dims["d_W"] <= max(tiny_spec.max_dim_W, tiny_spec.max_dim_U)
        assert 0 < dims["d_Z"] <= max(tiny_spec.max_dim_Z, tiny_spec.max_dim_U)
        # d_U is deliberately absent from that check, so U may come back empty.
        # See test_confounder_may_be_empty for why we pass it through.
        assert 0 <= dims["d_U"] <= tiny_spec.max_dim_U


def test_confounder_may_be_empty():
    # The generator's validity check covers X, W and Z but not U, so a dataset
    # whose confounder columns are all constant is accepted -- it is simply an
    # unconfounded draw. Rare (0/40 at T=1000, ~2% at T=60), but the cache must
    # represent it faithfully rather than crash or silently drop it: the rate is
    # itself an audit result. This test pins the shape contract that makes that
    # possible; it does not assert the case occurs.
    spec = PriorSpec(
        name="emptyU",
        n_datasets=1,
        seq_len=60,
        mcmc_cate=2,
        min_dim_U=2,
        max_dim_U=2,
        min_dim_W=2,
        max_dim_W=2,
        min_dim_Z=2,
        max_dim_Z=2,
        min_dim_X=2,
        max_dim_X=2,
        seed=0,
    )
    record = gen.generate(spec, n_jobs=1, progress=False)[0]
    assert record.U.shape[0] == spec.seq_len
    assert record.U.shape[1] == record.dims["d_U"]


def test_no_constant_columns_survive(tiny_cache):
    # filter_constant_features_and_pad drops zero-variance columns before
    # padding, so every retained column must actually vary.
    for r in tiny_cache:
        for name in ("X", "W", "Z"):
            block = getattr(r, name)
            assert (block.std(axis=0) > 0).all(), f"constant column in {name}"


def test_all_values_finite(tiny_cache):
    for r in tiny_cache:
        for name, value in r.to_dict().items():
            if isinstance(value, np.ndarray):
                assert np.isfinite(value).all(), f"non-finite values in {name}"


def test_treatment_is_binary(tiny_cache):
    for r in tiny_cache:
        assert set(np.unique(r.A)).issubset({0.0, 1.0})


def test_propensity_respects_clamping(tiny_spec, tiny_cache):
    eps = tiny_spec.resolved_hp()[0]["propensity_epsilon"]
    for r in tiny_cache:
        assert (r.A_probs >= eps).all()
        assert (r.A_probs <= 1.0 - eps).all()


def test_observed_outcome_matches_potential_outcomes(tiny_cache):
    # Y_obs is the consistency assumption made concrete; if this breaks, every
    # downstream estimate is meaningless.
    for r in tiny_cache:
        expected = np.where(r.A == 1, r.Y1, r.Y0)
        np.testing.assert_array_equal(r.Y_obs, expected)


def test_derived_properties(tiny_cache):
    for r in tiny_cache:
        np.testing.assert_array_equal(r.ite, r.Y1 - r.Y0)
        np.testing.assert_array_equal(r.cate, r.MU1 - r.MU0)
        assert r.true_ate == pytest.approx(float((r.MU1 - r.MU0).mean()))


def test_datasets_are_independent_draws(tiny_cache):
    # batch_size_per_gp=1 plus one get_batch call per dataset means no shared
    # mechanisms; identical blocks would indicate the seeding collapsed.
    a, b = tiny_cache[0], tiny_cache[1]
    assert not np.array_equal(a.X[:, 0], b.X[:, 0])


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def test_generation_is_reproducible(tiny_spec, tiny_cache):
    assert_caches_identical(
        tiny_cache, gen.generate(tiny_spec, n_jobs=1, progress=False)
    )


def test_generation_is_independent_of_n_jobs(tiny_spec, tiny_cache):
    # The whole reason generation seeds per dataset: PriorDataset.get_batch reads
    # global RNGs, so a joblib worker pool would otherwise produce different data
    # from the same seed.
    parallel = gen.generate(tiny_spec, n_jobs=2, progress=False)
    assert_caches_identical(tiny_cache, parallel)


def test_generation_is_unaffected_by_ambient_rng_state(tiny_spec, tiny_cache):
    import random

    import torch

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    assert_caches_identical(
        tiny_cache, gen.generate(tiny_spec, n_jobs=1, progress=False)
    )


def test_different_seed_gives_different_data(tiny_spec, tiny_cache):
    other = gen.generate(
        PriorSpec(**{**tiny_spec.to_dict(), "seed": 999}), n_jobs=1, progress=False
    )
    assert not np.array_equal(tiny_cache[0].X, other[0].X)


def test_growing_n_datasets_preserves_existing_records(tiny_spec, tiny_cache):
    # Cheap insurance for a real workflow: bumping n_datasets should extend the
    # sample, not resample it.
    grown = gen.generate(
        PriorSpec(**{**tiny_spec.to_dict(), "n_datasets": 3}), n_jobs=1, progress=False
    )
    assert len(grown) == 3
    for i in range(2):
        np.testing.assert_array_equal(tiny_cache[i].X, grown[i].X)
        np.testing.assert_array_equal(tiny_cache[i].MU1, grown[i].MU1)


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


def test_save_load_round_trip(tiny_spec, tiny_cache, tmp_path):
    gen.save(tiny_cache, tmp_path)
    assert_caches_identical(tiny_cache, gen.load(tiny_spec, tmp_path))


def test_save_writes_json_sidecar(tiny_spec, tiny_cache, tmp_path):
    path = gen.save(tiny_cache, tmp_path)
    sidecar = path.with_suffix(".json")
    assert sidecar.exists()

    import json

    with open(sidecar) as f:
        meta = json.load(f)
    # The file is shared by every spec with this generative config, so it stores
    # the generative fields and the record count -- not a name or n_datasets.
    assert meta["generative_spec"] == tiny_spec.generative_dict()
    assert meta["n_stored"] == len(tiny_cache)
    assert "name" not in meta["generative_spec"]


def test_save_leaves_no_temporary_file(tiny_cache, tmp_path):
    gen.save(tiny_cache, tmp_path)
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_creates_missing_cache_dir(tiny_cache, tmp_path):
    target = tmp_path / "does" / "not" / "exist"
    path = gen.save(tiny_cache, target)
    assert path.exists()


def test_cache_path_uses_spec_key(tiny_spec, tmp_path):
    assert gen.cache_path(tiny_spec, tmp_path).name == f"{tiny_spec.cache_key}.pt"


def test_load_missing_cache_raises(tiny_spec, tmp_path):
    with pytest.raises(FileNotFoundError):
        gen.load(tiny_spec, tmp_path)


def test_load_rejects_version_mismatch(tiny_spec, tiny_cache, tmp_path):
    import torch

    path = gen.save(tiny_cache, tmp_path)
    payload = torch.load(path, weights_only=False)
    payload["version"] = gen.CACHE_FORMAT_VERSION + 1
    torch.save(payload, path)

    with pytest.raises(ValueError, match="format version"):
        gen.load(tiny_spec, tmp_path)


def test_load_rejects_generative_spec_mismatch(tiny_spec, tiny_cache, tmp_path):
    import torch

    # Simulate a hash collision: right filename, wrong contents.
    path = gen.save(tiny_cache, tmp_path)
    payload = torch.load(path, weights_only=False)
    payload["generative_spec"]["seed"] = tiny_spec.seed + 1
    torch.save(payload, path)

    with pytest.raises(ValueError, match="does not match"):
        gen.load(tiny_spec, tmp_path)


def test_load_or_generate_populates_then_reuses_cache(tiny_spec, tmp_path, monkeypatch):
    first = gen.load_or_generate(tiny_spec, tmp_path, n_jobs=1, progress=False)
    assert gen.cache_path(tiny_spec, tmp_path).exists()

    # A second call must hit the cache; if it regenerated, this would raise.
    def fail(*args, **kwargs):
        raise AssertionError("load_or_generate regenerated instead of using the cache")

    monkeypatch.setattr(gen, "generate", fail)
    assert_caches_identical(
        first, gen.load_or_generate(tiny_spec, tmp_path, progress=False)
    )


def test_load_or_generate_force_regenerates(tiny_spec, tmp_path, monkeypatch):
    gen.load_or_generate(tiny_spec, tmp_path, n_jobs=1, progress=False)

    called = {"n": 0}
    real_generate = gen.generate

    def counting(*args, **kwargs):
        called["n"] += 1
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(gen, "generate", counting)
    gen.load_or_generate(tiny_spec, tmp_path, force=True, progress=False)
    assert called["n"] == 1


def test_specs_differing_in_overrides_do_not_share_a_cache(
    tiny_spec, tiny_cache, tmp_path
):
    gen.save(tiny_cache, tmp_path)
    ablated = PriorSpec(
        **{
            **tiny_spec.to_dict(),
            "sampled_hp_overrides": {"propensity_shift": pinned(0.0)},
        }
    )
    # Different configuration => different key => must miss rather than serve the
    # baseline's data.
    with pytest.raises(FileNotFoundError):
        gen.load(ablated, tmp_path)


def test_list_caches(tiny_spec, tiny_cache, tmp_path):
    assert gen.list_caches(tmp_path) == []

    gen.save(tiny_cache, tmp_path)
    entries = gen.list_caches(tmp_path)
    assert len(entries) == 1
    assert entries[0]["generative_spec"] == tiny_spec.generative_dict()
    assert entries[0]["n_stored"] == len(tiny_cache)


def test_list_caches_missing_dir(tmp_path):
    assert gen.list_caches(tmp_path / "nope") == []


def test_list_caches_ignores_sidecar_without_data(tiny_spec, tiny_cache, tmp_path):
    path = gen.save(tiny_cache, tmp_path)
    path.unlink()  # leave the sidecar orphaned
    assert gen.list_caches(tmp_path) == []


# --------------------------------------------------------------------------
# Content-addressed caching: name/n_datasets do not affect the cache
# --------------------------------------------------------------------------


def _renamed(spec, name, **kwargs):
    return PriorSpec(**{**spec.to_dict(), "name": name, **kwargs})


def test_renaming_a_spec_hits_the_same_cache(
    tiny_spec, tiny_cache, tmp_path, monkeypatch
):
    # The whole point of content addressing: a rename must not regenerate a byte.
    gen.save(tiny_cache, tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("regenerated after a rename")

    monkeypatch.setattr(gen, "generate", fail)
    renamed = _renamed(tiny_spec, "totally_different_name")
    loaded = gen.load_or_generate(renamed, tmp_path, progress=False)

    assert len(loaded) == len(tiny_cache)
    np.testing.assert_array_equal(loaded[0].X, tiny_cache[0].X)
    # The caller's own name is preserved on the returned cache.
    assert loaded.spec.name == "totally_different_name"


def test_cache_serves_a_prefix_when_it_holds_more_than_requested(
    tiny_spec, tiny_cache, tmp_path, monkeypatch
):
    gen.save(tiny_cache, tmp_path)  # 2 datasets

    def fail(*args, **kwargs):
        raise AssertionError("regenerated instead of slicing the cache")

    monkeypatch.setattr(gen, "generate", fail)
    smaller = _renamed(tiny_spec, "smaller", n_datasets=1)
    loaded = gen.load_or_generate(smaller, tmp_path, progress=False)

    assert len(loaded) == 1
    np.testing.assert_array_equal(loaded[0].X, tiny_cache[0].X)


def test_cache_extends_when_it_holds_fewer_than_requested(
    tiny_spec, tiny_cache, tmp_path
):
    gen.save(tiny_cache, tmp_path)  # 2 datasets

    bigger = _renamed(tiny_spec, "bigger", n_datasets=4)
    grown = gen.load_or_generate(bigger, tmp_path, n_jobs=1, progress=False)

    assert len(grown) == 4
    # The already-cached datasets must be reused untouched...
    for i in range(2):
        np.testing.assert_array_equal(grown[i].X, tiny_cache[i].X)
    # ...and the new ones must match a from-scratch generation of those indices.
    fresh = gen.generate(bigger, n_jobs=1, progress=False, indices=[2, 3])
    for got, want in zip(grown.records[2:], fresh.records):
        np.testing.assert_array_equal(got.X, want.X)
        assert got.index == want.index

    # The extension is persisted, so the next request is a plain hit.
    assert len(gen.load(bigger, tmp_path)) == 4


def test_extend_only_generates_the_missing_indices(
    tiny_spec, tiny_cache, tmp_path, monkeypatch
):
    gen.save(tiny_cache, tmp_path)  # 2 datasets

    seen = {}
    real = gen.generate

    def spy(spec, **kwargs):
        seen["indices"] = list(kwargs.get("indices"))
        return real(spec, **kwargs)

    monkeypatch.setattr(gen, "generate", spy)
    gen.load_or_generate(
        _renamed(tiny_spec, "bigger", n_datasets=3), tmp_path, n_jobs=1, progress=False
    )
    assert seen["indices"] == [2]  # not [0, 1, 2]


def test_generate_honours_explicit_indices(tiny_spec):
    out = gen.generate(tiny_spec, n_jobs=1, progress=False, indices=[1])
    assert len(out) == 1
    assert out[0].index == 1
    assert out[0].seed == tiny_spec.dataset_seed(1)
