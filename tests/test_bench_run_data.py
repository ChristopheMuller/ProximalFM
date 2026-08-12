"""Tests for run.py's data step (get_data / _generate_data wiring)."""

import pytest
import torch

from benchmarks import run
from test_bench_data import (
    assert_dicts_equal,
    build_from_pool,
    make_pool,
    write_val_file,
)


# ─────────────────────────────────────────────────────────────────────────────
# get_data: val-format directory loading
# ─────────────────────────────────────────────────────────────────────────────


def _load_cfg(val_dir):
    return {"data": {"source": "load", "val_dir": str(val_dir)}, "run": {}}


def test_get_data_builds_from_val_dir_and_caches(tmp_path):
    pool = make_pool()
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    write_val_file(val_dir / "batch_000000.pt", pool)
    output_dir = tmp_path / "exp"

    data = run.get_data(_load_cfg(val_dir), output_dir)
    assert_dicts_equal(data, build_from_pool(pool))

    # The resolved dict is cached as data.pt (plots.py reads it there)…
    cached = torch.load(output_dir / "data.pt", weights_only=False)
    assert_dicts_equal(cached, data)

    # …and later runs use the cache: loading works even without the val dir.
    for f in val_dir.iterdir():
        f.unlink()
    val_dir.rmdir()
    assert_dicts_equal(run.get_data(_load_cfg(val_dir), output_dir), data)


def test_get_data_force_regen_rebuilds_from_val_dir(tmp_path):
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    write_val_file(val_dir / "batch_000000.pt", make_pool(seed=0))
    output_dir = tmp_path / "exp"

    stale = run.get_data(_load_cfg(val_dir), output_dir)

    # Replace the val data; without --force the stale cache wins, with it the
    # dict is rebuilt from the directory.
    new_pool = make_pool(seed=1)
    write_val_file(val_dir / "batch_000000.pt", new_pool)
    assert_dicts_equal(run.get_data(_load_cfg(val_dir), output_dir), stale)
    data = run.get_data(_load_cfg(val_dir), output_dir, force_regen=True)
    assert_dicts_equal(data, build_from_pool(new_pool))


def test_get_data_load_without_any_source_raises(tmp_path):
    cfg = {"data": {"source": "load"}, "run": {}}
    with pytest.raises(FileNotFoundError):
        run.get_data(cfg, tmp_path / "exp")


# ─────────────────────────────────────────────────────────────────────────────
# get_data: single-CSV loading (source: csv)
# ─────────────────────────────────────────────────────────────────────────────


def _csv_cfg(csv_path):
    return {
        "data": {
            "source": "csv",
            "csv_path": str(csv_path),
            "columns": {
                "X": ["red"],
                "W": ["pol_1"],
                "Z": ["led_2_uv"],
                "A": "led_3_uv",
                "Y": "ir_3",
                "U": ["green"],
                "Y0": "Y0",
                "Y1": "Y1",
            },
            "shuffle": False,
        },
        "run": {"seed": 0},
    }


def _make_chamber_csv(tmp_path, T=20):
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "green": np.arange(T, dtype=np.float64),
            "red": np.arange(T, dtype=np.float64) * 2,
            "pol_1": np.arange(T, dtype=np.float64) * 3,
            "led_2_uv": np.arange(T, dtype=np.float64) * 5,
            "led_3_uv": np.tile([0.0, 2047.0], T // 2),
            "ir_3": np.arange(T, dtype=np.float64) - 7,
            "Y0": np.arange(T, dtype=np.float64) + 100,
            "Y1": np.arange(T, dtype=np.float64) + 200,
        }
    )
    path = tmp_path / "chamber.csv"
    df.to_csv(path)
    return path


def test_get_data_builds_from_csv_and_caches(tmp_path):
    from benchmarks.data_csv import load_benchmark_from_csv

    csv_path = _make_chamber_csv(tmp_path)
    output_dir = tmp_path / "exp"
    cfg = _csv_cfg(csv_path)

    data = run.get_data(cfg, output_dir)
    expected = load_benchmark_from_csv(csv_path, cfg["data"]["columns"], shuffle=False)
    assert_dicts_equal(data, expected)

    cached = torch.load(output_dir / "data.pt", weights_only=False)
    assert_dicts_equal(cached, data)

    # Later runs reuse the cache even if the CSV disappears.
    csv_path.unlink()
    assert_dicts_equal(run.get_data(cfg, output_dir), data)


# ─────────────────────────────────────────────────────────────────────────────
# _generate_data uses the shared builder (schema identical to the load path)
# ─────────────────────────────────────────────────────────────────────────────


class _FakePriorDataset:
    """Stands in for PriorDataset: yields one fixed 19-field batch."""

    def __init__(self, batch, **kwargs):
        self.batch = batch
        self.kwargs = kwargs

    def __iter__(self):
        return iter([self.batch])


def _pool_batch(pool):
    """The 19-field tuple SCMPrior.get_batch yields, from a test pool."""
    return tuple(
        pool[k]
        for k in (
            "U", "X", "W", "Z", "A", "A_probs", "Y_obs", "Y0", "Y1",
            "MU0", "MU1", "d_U", "d_X", "d_W", "d_Z",
            "seq_lens", "train_sizes", "S2", "K_eff",
        )
    )  # fmt: skip


def test_generate_data_matches_shared_builder(tmp_path, monkeypatch):
    B, T, train_size = 6, 32, 24
    pool = make_pool(B=B, T=T, train_size=train_size)
    batch = _pool_batch(pool)
    monkeypatch.setattr(run, "_PROXFM_AVAILABLE", True)
    monkeypatch.setattr(
        run, "PriorDataset", lambda **kw: _FakePriorDataset(batch, **kw)
    )

    data_cfg = {"n_context": train_size, "n_query": T - train_size, "n_batch": B}
    data = run._generate_data(data_cfg, {"seed": 0}, tmp_path)

    assert_dicts_equal(data, build_from_pool(pool))
    assert (tmp_path / "data.pt").exists()
    assert (tmp_path / "prior_config.yaml").exists()


def test_generate_data_passes_the_split_to_the_prior_verbatim(tmp_path, monkeypatch):
    # n_context/n_query are the prior's own spelling, so they must arrive
    # untranslated: max_seq_len bounds the context and the prior composes the
    # total. A config saying 768/256 that produced anything else would be a
    # silent size change no output records.
    seen = {}

    class _Capture(_FakePriorDataset):
        def __init__(self, batch, **kwargs):
            super().__init__(batch, **kwargs)
            seen.update(kwargs)

    monkeypatch.setattr(run, "_PROXFM_AVAILABLE", True)
    batch = _pool_batch(make_pool(B=2, T=8, train_size=6))
    monkeypatch.setattr(run, "PriorDataset", lambda **kw: _Capture(batch, **kw))

    run._generate_data(
        {"n_context": 768, "n_query": 256, "n_batch": 2}, {"seed": 0}, tmp_path
    )

    assert seen["max_seq_len"] == 768
    assert seen["n_query"] == 256
    assert seen["min_seq_len"] is None  # every dataset shares the context size


@pytest.mark.parametrize(
    "legacy",
    [{"n_rows": 1024}, {"prop_train": 0.75}, {"n_rows": 1024, "prop_train": 0.75}],
)
def test_generate_data_refuses_the_old_row_spelling(tmp_path, legacy):
    # Silently ignoring n_rows/prop_train would generate datasets of a size
    # nobody asked for, and the config file would still look like it said 1024.
    with pytest.raises(KeyError, match="n_context"):
        run._generate_data({"n_batch": 2, **legacy}, {"seed": 0}, tmp_path)
