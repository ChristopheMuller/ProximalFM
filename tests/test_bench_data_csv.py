"""Tests for data_csv.py's single-CSV loader."""

import numpy as np
import pandas as pd
import pytest

from benchmarks.data_csv import (
    load_benchmark_from_csv,
    load_benchmark_from_csvs,
    resolve_csv_paths,
)


def make_chamber_csv(tmp_path, T=20):
    """Write a small chamber-style CSV (unnamed index col + known columns)."""
    df = pd.DataFrame(
        {
            "green": np.arange(T, dtype=np.float64),  # U
            "red": np.arange(T, dtype=np.float64) * 2,  # X
            "pol_1": np.arange(T, dtype=np.float64) * 3,  # W
            "led_2_uv": np.arange(T, dtype=np.float64) * 5,  # Z
            "led_3_uv": np.tile([0.0, 2047.0], T // 2),  # A (binary, raw units)
            "ir_3": np.arange(T, dtype=np.float64) - 7,  # Y (observed)
            "Y0": np.arange(T, dtype=np.float64) + 100,  # ground-truth control
            "Y1": np.arange(T, dtype=np.float64) + 200,  # ground-truth treated
        }
    )
    path = tmp_path / "chamber.csv"
    df.to_csv(path)  # writes the unnamed index column, matching the real files
    return path, df


COLUMNS = {
    "X": ["red"],
    "W": ["pol_1"],
    "Z": ["led_2_uv"],
    "A": "led_3_uv",
    "Y": "ir_3",
    "U": ["green"],
    "Y0": "Y0",
    "Y1": "Y1",
}


def test_schema_and_shapes(tmp_path):
    path, df = make_chamber_csv(tmp_path, T=20)
    data = load_benchmark_from_csv(path, COLUMNS, prop_train=0.75, shuffle=False)

    assert data["N_BATCH"] == 1
    assert data["N_TRAIN"] == 15
    assert data["N_TEST"] == 5
    assert data["X"].shape == (1, 20, 1)
    assert data["W"].shape == (1, 20, 1)
    assert data["Z"].shape == (1, 20, 1)
    assert data["U"].shape == (1, 20, 1)
    assert data["A"].shape == (1, 20)
    assert data["Y"].shape == (1, 20)
    assert data["true_ITEs"].shape == (1, 5)
    assert data["d_X"].tolist() == [1]
    assert data["d_W"].tolist() == [1]
    assert data["d_Z"].tolist() == [1]


def test_true_ites_match_manual_formula(tmp_path):
    path, df = make_chamber_csv(tmp_path, T=20)
    data = load_benchmark_from_csv(path, COLUMNS, prop_train=0.75, shuffle=False)

    y_obs = df["ir_3"].to_numpy(dtype=np.float32)
    y_ctx_std = y_obs[:15].std(ddof=0)
    expected = (df["Y1"].to_numpy()[15:] - df["Y0"].to_numpy()[15:]) / y_ctx_std
    np.testing.assert_allclose(data["true_ITEs"][0], expected, rtol=1e-5, atol=1e-6)


def test_normalize_treatment_rescales_binary_to_unit_interval(tmp_path):
    path, _ = make_chamber_csv(tmp_path)
    data = load_benchmark_from_csv(
        path, COLUMNS, shuffle=False, normalize_treatment=True
    )
    assert set(np.unique(data["A"].numpy())) == {0.0, 1.0}


def test_normalize_treatment_false_keeps_raw_values(tmp_path):
    path, _ = make_chamber_csv(tmp_path)
    data = load_benchmark_from_csv(
        path, COLUMNS, shuffle=False, normalize_treatment=False
    )
    assert set(np.unique(data["A"].numpy())) == {0.0, 2047.0}


def test_shuffle_is_deterministic_for_fixed_seed(tmp_path):
    path, _ = make_chamber_csv(tmp_path)
    a = load_benchmark_from_csv(path, COLUMNS, shuffle=True, seed=7)
    b = load_benchmark_from_csv(path, COLUMNS, shuffle=True, seed=7)
    np.testing.assert_array_equal(a["X"].numpy(), b["X"].numpy())
    np.testing.assert_array_equal(a["true_ITEs"], b["true_ITEs"])


def test_shuffle_reorders_rows(tmp_path):
    path, _ = make_chamber_csv(tmp_path)
    unshuffled = load_benchmark_from_csv(path, COLUMNS, shuffle=False)
    shuffled = load_benchmark_from_csv(path, COLUMNS, shuffle=True, seed=1)
    # X is a monotonically increasing sequence, so any permutation is detectable.
    assert not np.array_equal(unshuffled["X"].numpy(), shuffled["X"].numpy())


def test_u_optional_yields_zero_width(tmp_path):
    path, _ = make_chamber_csv(tmp_path)
    columns = {k: v for k, v in COLUMNS.items() if k != "U"}
    data = load_benchmark_from_csv(path, columns, shuffle=False)
    assert data["U"].shape[-1] == 0


def test_missing_required_column_keys_raise(tmp_path):
    path, _ = make_chamber_csv(tmp_path)
    for missing_key in ("X", "W", "Z", "A", "Y", "Y0", "Y1"):
        columns = {k: v for k, v in COLUMNS.items() if k != missing_key}
        with pytest.raises(ValueError, match=missing_key):
            load_benchmark_from_csv(path, columns, shuffle=False)


def test_loads_csv_written_without_index_column(tmp_path):
    # A hardcoded index_col=0 would swallow the first real column ("green"),
    # making the mapping fail on a column that is plainly present in the file.
    path, df = make_chamber_csv(tmp_path)
    no_index = tmp_path / "no_index.csv"
    df.to_csv(no_index, index=False)

    with_idx = load_benchmark_from_csv(path, COLUMNS, shuffle=False)
    without_idx = load_benchmark_from_csv(no_index, COLUMNS, shuffle=False)
    np.testing.assert_array_equal(with_idx["X"].numpy(), without_idx["X"].numpy())
    np.testing.assert_array_equal(with_idx["U"].numpy(), without_idx["U"].numpy())


def test_shuffle_keeps_rows_aligned_across_blocks(tmp_path):
    # Every block is a fixed affine function of the row index in the fixture, so
    # any per-block permutation mismatch breaks these identities. This is the
    # property the whole benchmark's validity rests on.
    path, _ = make_chamber_csv(tmp_path, T=20)
    d = load_benchmark_from_csv(path, COLUMNS, shuffle=True, seed=3, prop_train=1.0)
    u = d["U"].numpy()[0, :, 0]  # standardized, so recover order via argsort
    order = np.argsort(u)
    # U=green=i, X=red=2i, W=pol_1=3i, Z=led_2_uv=5i -> all share one ordering.
    for key in ("X", "W", "Z"):
        np.testing.assert_array_equal(np.argsort(d[key].numpy()[0, :, 0]), order)
    # Y0 = i + 100 and Y1 = i + 200 are monotone in i as well.
    y = d["Y"].numpy()[0]
    np.testing.assert_array_equal(np.argsort(y), order)


def test_unknown_csv_column_raises(tmp_path):
    path, _ = make_chamber_csv(tmp_path)
    columns = dict(COLUMNS, X=["not_a_real_column"])
    with pytest.raises(ValueError, match="not_a_real_column"):
        load_benchmark_from_csv(path, columns, shuffle=False)


def test_feature_column_pattern_expands_in_csv_header_order(tmp_path):
    path, df = make_chamber_csv(tmp_path)
    df.insert(3, "X_2", np.arange(len(df), dtype=np.float64) * 11)
    df.insert(2, "X_1", np.arange(len(df), dtype=np.float64) * 7)
    df.to_csv(path)
    columns = dict(COLUMNS, X="X_*")

    data = load_benchmark_from_csv(path, columns, shuffle=False)

    assert data["X"].shape == (1, len(df), 2)
    assert data["d_X"].tolist() == [2]
    # Context standardization preserves the source columns' relative scaling.
    np.testing.assert_allclose(data["X"][0, :, 1], data["X"][0, :, 0])


def test_unmatched_feature_column_pattern_raises(tmp_path):
    path, _ = make_chamber_csv(tmp_path)
    columns = dict(COLUMNS, X="X_*")
    with pytest.raises(ValueError, match=r"unmatched pattern.*X_\*"):
        load_benchmark_from_csv(path, columns, shuffle=False)


def test_multiple_csvs_are_independent_episodes(tmp_path):
    first, _ = make_chamber_csv(tmp_path, T=20)
    second = tmp_path / "second.csv"
    frame = pd.read_csv(first)
    frame["ir_3"] += 1000  # a distinct outcome scale for the second episode
    frame.to_csv(second, index=False)

    data = load_benchmark_from_csvs(
        [first, second], COLUMNS, prop_train=0.75, shuffle=False
    )

    assert data["N_BATCH"] == 2
    assert data["X"].shape == (2, 20, 1)
    assert data["true_ITEs"].shape == (2, 5)
    assert data["episode_files"] == [str(first), str(second)]
    # Adding a level shift to one file must disappear under that file's own
    # context-only outcome standardization.
    np.testing.assert_allclose(data["Y"][0], data["Y"][1])


def test_multiple_csvs_require_matching_row_counts(tmp_path):
    first, _ = make_chamber_csv(tmp_path, T=20)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    second, _ = make_chamber_csv(other_dir, T=24)
    with pytest.raises(ValueError, match="same train/test sizes"):
        load_benchmark_from_csvs([first, second], COLUMNS, shuffle=False)


def test_resolve_csv_directory_is_sorted_and_filters_pattern(tmp_path):
    (tmp_path / "b.csv").touch()
    (tmp_path / "a.csv").touch()
    (tmp_path / "ignore.txt").touch()
    assert resolve_csv_paths({"csv_dir": tmp_path}) == [
        tmp_path / "a.csv",
        tmp_path / "b.csv",
    ]


def test_resolve_csv_paths_refuses_ambiguous_inputs(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        resolve_csv_paths({"csv_path": tmp_path / "a.csv", "csv_dir": tmp_path})
