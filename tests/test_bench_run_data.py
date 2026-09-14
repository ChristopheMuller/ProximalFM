"""Tests for run.py's data step (get_data)."""

from pathlib import Path

import numpy as np
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


def test_get_data_builds_from_val_dir_and_caches(tmp_path):
    pool = make_pool()
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    write_val_file(val_dir / "batch_000000.pt", pool)
    output_dir = tmp_path / "exp"

    data = run.get_data(val_dir, output_dir)
    assert_dicts_equal(data, build_from_pool(pool))

    # The resolved dict is cached as data.pt (plots.py reads it there)…
    cached = torch.load(output_dir / "data.pt", weights_only=False)
    assert_dicts_equal(cached, data)

    # …and later runs use the cache: loading works even without the val dir.
    for f in val_dir.iterdir():
        f.unlink()
    val_dir.rmdir()
    assert_dicts_equal(run.get_data(val_dir, output_dir), data)


def test_get_data_force_regen_rebuilds_from_val_dir(tmp_path):
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    write_val_file(val_dir / "batch_000000.pt", make_pool(seed=0))
    output_dir = tmp_path / "exp"

    stale = run.get_data(val_dir, output_dir)

    # Replace the val data; without --force the stale cache wins, with it the
    # dict is rebuilt from the directory.
    new_pool = make_pool(seed=1)
    write_val_file(val_dir / "batch_000000.pt", new_pool)
    assert_dicts_equal(run.get_data(val_dir, output_dir), stale)
    data = run.get_data(val_dir, output_dir, force_regen=True)
    assert_dicts_equal(data, build_from_pool(new_pool))


def test_get_data_refreshes_cache_when_val_directory_grows(tmp_path):
    pool = make_pool(B=2)
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    write_val_file(val_dir / "batch_000000.pt", pool, slice(0, 1))
    output_dir = tmp_path / "exp"

    assert run.get_data(val_dir, output_dir)["N_BATCH"] == 1
    write_val_file(val_dir / "batch_000001.pt", pool, slice(1, 2))
    refreshed = run.get_data(val_dir, output_dir)
    assert refreshed["N_BATCH"] == 2
    assert_dicts_equal(refreshed, build_from_pool(pool))


def test_default_output_dir_is_val_sibling():
    assert run.default_output_dir(Path("data/a/val")) == Path("data/a/bench")


def test_extended_diagnostics_preserve_episode_alignment_and_summary(tmp_path):
    path = tmp_path / "diagnostics.yaml"
    run.save_diagnostics_yaml(
        path, {"shared": np.array([1.0, 3.0]), "old_only": np.array([5.0, 7.0])}
    )
    merged = run.extend_diagnostics(
        path, {"shared": np.array([8.0]), "new_only": np.array([9.0])}, 2, 1
    )
    run.save_diagnostics_yaml(path, merged)
    loaded = run.load_diagnostics_yaml(path)
    np.testing.assert_equal(loaded["shared"], [1.0, 3.0, 8.0])
    np.testing.assert_equal(loaded["old_only"], [5.0, 7.0, np.nan])
    np.testing.assert_equal(loaded["new_only"], [np.nan, np.nan, 9.0])
    assert run.yaml.safe_load(path.read_text())["summary"]["shared"] == 4.0


def test_extended_diagnostics_pad_when_suffix_emits_nothing(tmp_path):
    path = tmp_path / "diagnostics.yaml"
    run.save_diagnostics_yaml(path, {"value": np.array([2.0])})
    merged = run.extend_diagnostics(path, {}, 1, 2)
    np.testing.assert_equal(merged["value"], [2.0, np.nan, np.nan])


def test_extended_diagnostics_pad_missing_historical_file(tmp_path):
    merged = run.extend_diagnostics(
        tmp_path / "missing.yaml", {"value": np.array([2.0])}, 2, 1
    )
    np.testing.assert_equal(merged["value"], [np.nan, np.nan, 2.0])


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


def test_build_batches_uses_context_prefix_without_moving_query_rows():
    data = build_from_pool(make_pool(train_size=24))
    full = run.build_batches(data)
    prefix = run.build_batches(data, n_ctx=7)

    for full_batch, prefix_batch in zip(full, prefix):
        for key in ("U", "X", "W", "Z"):
            np.testing.assert_array_equal(
                prefix_batch[f"{key}_ctx"], full_batch[f"{key}_ctx"][:7]
            )
            np.testing.assert_array_equal(
                prefix_batch[f"{key}_q"], full_batch[f"{key}_q"]
            )
        np.testing.assert_array_equal(prefix_batch["A_ctx"], full_batch["A_ctx"][:7])
        np.testing.assert_array_equal(prefix_batch["Y_ctx"], full_batch["Y_ctx"][:7])
        np.testing.assert_array_equal(
            prefix_batch["true_ITE_std"], full_batch["true_ITE_std"]
        )


def test_context_sizes_are_sorted_and_bounded_by_dataset_context():
    assert run.resolve_context_sizes(None, 24) == [24]
    assert run.resolve_context_sizes([16, 4, 8], 24) == [4, 8, 16]

    with pytest.raises(ValueError, match="exceed the dataset context size 24"):
        run.resolve_context_sizes([8, 25], 24)
    with pytest.raises(ValueError, match="positive"):
        run.resolve_context_sizes([0, 8], 24)
    with pytest.raises(ValueError, match="unique"):
        run.resolve_context_sizes([8, 8], 24)


def test_context_sweep_scores_every_size_into_one_frame():
    target = np.arange(12, dtype=np.float32).reshape(2, 6)
    times = np.zeros(2, dtype=np.float32)
    accumulated = {
        4: {"method": (target + 1, times)},
        8: {"method": (target, times)},
    }

    results = run.score_context_sweep(
        accumulated, {"method": "cate(x)"}, {"cate": target, "ite": target}
    )

    assert set(results["n_ctx"]) == {4, 8}
    assert len(results[results["n_ctx"] == 4]) == len(results[results["n_ctx"] == 8])


def test_context_sweep_propagates_post_tuning_times():
    target = np.arange(12, dtype=np.float32).reshape(2, 6)
    total = np.array([3.0, 5.0], dtype=np.float32)
    post = np.array([1.0, 2.0], dtype=np.float32)
    accumulated = {8: {"method": (target, total, post)}}

    results = run.score_context_sweep(
        accumulated, {"method": "cate(x)"}, {"cate": target, "ite": target}
    )

    cate = results[results["target"].eq("cate")].sort_values("episode")
    np.testing.assert_array_equal(cate["time"], total)
    np.testing.assert_array_equal(cate["time_post_tune"], post)


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"cls": "S_Learner"}, False),
        ({"cls": "NMMR", "tune": {"grid": {"lr": [0.1]}}}, True),
        ({"cls": "PLearner", "base_cv": {"folds": 2}}, True),
        (
            {"cls": "PLearner", "bridge_params": {"lam_grid": [1e-3]}},
            True,
        ),
        (
            {
                "cls": "ProxyMarginalizer",
                "estimator": {"cls": "NMMR", "tune": {"sweep": {"n_epochs": [5]}}},
            },
            True,
        ),
    ],
)
def test_legacy_cache_tuning_detection(config, expected):
    assert run.method_has_explicit_tuning(config) is expected
