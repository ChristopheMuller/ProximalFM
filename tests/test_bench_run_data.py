"""Tests for run.py's data step (get_data)."""

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


def test_get_data_refuses_the_removed_generate_source(tmp_path):
    """`source: generate` is gone. A config still asking for it must say so —
    falling through to "no data file found" would read as a missing path."""
    cfg = {"data": {"source": "generate", "n_context": 768}, "run": {"seed": 0}}
    with pytest.raises(ValueError, match="has been removed"):
        run.get_data(cfg, tmp_path / "exp")


def test_source_defaults_to_load(tmp_path):
    """No `source:` at all resolves through the val_dir/cache path, not generate."""
    pool = make_pool()
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    write_val_file(val_dir / "batch_000000.pt", pool)

    data = run.get_data({"data": {"val_dir": str(val_dir)}, "run": {}}, tmp_path / "e")
    assert_dicts_equal(data, build_from_pool(pool))


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
    from benchmarks.data_csv import load_benchmark_from_csvs

    csv_path = _make_chamber_csv(tmp_path)
    output_dir = tmp_path / "exp"
    cfg = _csv_cfg(csv_path)

    data = run.get_data(cfg, output_dir)
    expected = load_benchmark_from_csvs(
        [csv_path], cfg["data"]["columns"], shuffle=False
    )
    assert_dicts_equal(data, expected)

    cached = torch.load(output_dir / "data.pt", weights_only=False)
    assert_dicts_equal(cached, data)

    # Later runs reuse the cache even if the CSV disappears.
    csv_path.unlink()
    assert_dicts_equal(run.get_data(cfg, output_dir), data)


# ─────────────────────────────────────────────────────────────────────────────
# build_batches
# ─────────────────────────────────────────────────────────────────────────────


def test_build_batches_trims_each_episode_to_its_own_dims():
    """The pool is padded to its widest episode; a batch is not.

    Episodes are fit independently, so the padding is nothing but zero columns
    that every method has to ignore — and an estimator that needs to know how
    many columns are real (ProximalFM) would otherwise have to be told
    separately from the array it describes.
    """
    data = build_from_pool(make_pool())
    batches = run.build_batches(data)

    n_train, n_test = data["N_TRAIN"], data["N_TEST"]
    for b, batch in enumerate(batches):
        for key, dim_key in (("X", "d_X"), ("W", "d_W"), ("Z", "d_Z")):
            d = int(data[dim_key][b])
            assert batch[f"{key}_ctx"].shape == (n_train, d)
            assert batch[f"{key}_q"].shape == (n_test, d)
            # Trimmed, not re-sliced: the surviving columns are the leading ones.
            expected = data[key][b, :, :d].numpy()
            assert (batch[f"{key}_ctx"] == expected[:n_train]).all()
            assert (batch[f"{key}_q"] == expected[n_train:]).all()
