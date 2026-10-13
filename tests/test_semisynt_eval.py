"""Contracts for the semi-synthetic mechanism matrix."""


from __future__ import annotations


import json


from pathlib import Path


import pandas as pd


import torch


import yaml


from benchmarks.data import load_benchmark_from_val_dir


from eval_datasets.semisynt_eval._config import load_case_configs


from eval_datasets.semisynt_eval.build import (
    DEFAULT_CONFIG_PATH,
    build,
    parse_args,
)


from eval_datasets.semisynt_eval.source import load_csv


def _recipe() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())


def _configs():
    recipe = _recipe()
    return load_case_configs(recipe["common"], recipe["cases"])


def _csv(path: Path, delimiter: str = ",") -> None:
    rows = (
        ("a", 1, "z", ""),
        ("b", 2, "y", 3),
        ("c", 4, "x", ""),
        ("d", 8, "y", 9),
        ("e", 16, "z", ""),
        ("f", 32, "x", 12),
        ("g", 64, "y", 15),
        ("h", 128, "z", 18),
        ("i", 256, "x", 21),
        ("j", 512, "y", 24),
        ("k", 1024, "z", 27),
        ("l", 2048, "x", 30),
    )
    path.write_text(
        delimiter.join(("id", "numeric", "category", "missing", "constant"))
        + "\n"
        + "\n".join(delimiter.join(map(str, (*row, 7))) for row in rows)
        + "\n"
    )


def test_preprocessing_is_fitted_on_calibration_rows_only(tmp_path):
    path = tmp_path / "source.csv"
    _csv(path)
    first = load_csv(
        path, split_seed=23, n_features_x=2, n_features_u=2, n_calibration=4
    )
    frame = pd.read_csv(path)
    evaluation = list(first.evaluation_row_indices)
    frame.loc[evaluation, "numeric"] = 1e12
    frame.loc[evaluation, "missing"] = -1e12
    frame.loc[evaluation, "category"] = "evaluation-only"
    frame.to_csv(path, index=False)
    second = load_csv(
        path, split_seed=23, n_features_x=2, n_features_u=2, n_calibration=4
    )

    assert second.calibration_row_indices == first.calibration_row_indices
    assert second.evaluation_row_indices == first.evaluation_row_indices
    assert second.preprocessing == first.preprocessing
    torch.testing.assert_close(second.calibration_values, first.calibration_values)
    assert not torch.equal(second.evaluation_values, first.evaluation_values)
    torch.testing.assert_close(
        first.calibration_values.mean(0), torch.zeros(4), atol=2e-6, rtol=0
    )
    torch.testing.assert_close(
        first.calibration_values.std(0, unbiased=False),
        torch.ones(4),
        atol=2e-6,
        rtol=0,
    )


def _tiny_build_args(csv: Path, out_root: Path, name: str):
    return parse_args(
        [
            "--csv",
            str(csv),
            "--out_root",
            str(out_root),
            "--name",
            name,
            "--n_context",
            "4",
            "--n_query",
            "2",
            "--n_calibration",
            "4",
            "--replicates_per_csv",
            "1",
            "--datasets_per_file",
            "1",
            "--dimensions",
            "2",
            "--seed",
            "7",
        ]
    )


def _batch(root: Path, case: str) -> dict:
    return torch.load(
        root / "d-2" / case / "val" / "batch_000000.pt", weights_only=True
    )


def test_tiny_build_materializes_all_paired_cases(tmp_path):
    csv = tmp_path / "source.csv"
    _csv(csv)
    args = _tiny_build_args(csv, tmp_path, "matrix")
    root = build(args)

    assert root == tmp_path / "matrix"
    assert {path.name for path in root.iterdir() if path.is_dir()} == {"d-2"}
    assert {path.name for path in (root / "d-2").iterdir()} == set(_configs())
    manifests = {
        case: json.loads((root / "d-2" / case / "manifest.json").read_text())
        for case in _configs()
    }
    row_hashes = {
        manifest["episodes"][0]["source_row_indices_sha256"]
        for manifest in manifests.values()
    }
    assert len(row_hashes) == 1
    assert {
        (record["source_file"], record["replicate"], record["seed"])
        for manifest in manifests.values()
        for record in manifest["episodes"]
    } == {("source.csv", 1, 7)}
    benchmark = load_benchmark_from_val_dir(
        root / "d-2" / "linear_conf_low_proxy_low" / "val"
    )
    assert benchmark["true_CATEs"] is not None
    assert benchmark["true_CATEs"].shape == (1, 2)


def test_written_cases_preserve_factorial_pairing(tmp_path):
    csv = tmp_path / "source.csv"
    _csv(csv)
    root = build(_tiny_build_args(csv, tmp_path, "matrix"))

    low = _batch(root, "linear_conf_low_proxy_low")
    high_proxy = _batch(root, "linear_conf_low_proxy_high")
    high_conf = _batch(root, "linear_conf_high_proxy_low")
    for field in ("U", "X", "A", "A_probs", "Y_obs", "Y0", "Y1", "CATE"):
        assert torch.equal(low[field], high_proxy[field])
    assert not torch.equal(low["W"], high_proxy["W"])
    assert not torch.equal(low["Z"], high_proxy["Z"])
    for field in ("U", "X", "W", "Z", "CATE"):
        assert torch.equal(low[field], high_conf[field])


def test_dimension_matrix_matches_standalone_and_saved_recipe(tmp_path):
    csv = tmp_path / "source.csv"
    _csv(csv)
    args = _tiny_build_args(csv, tmp_path, "matrix")
    args.dimensions = [1, 2]
    matrix = build(args)
    standalone_args = _tiny_build_args(csv, tmp_path, "standalone")
    standalone_args.dimensions = [1]
    standalone = build(standalone_args)
    for case in _configs():
        small_root = matrix / "d-1" / case
        manifest = json.loads((small_root / "manifest.json").read_text())
        large_manifest = json.loads(
            (matrix / "d-2" / case / "manifest.json").read_text()
        )
        solo_manifest = json.loads(
            (standalone / "d-1" / case / "manifest.json").read_text()
        )
        assert manifest["content_digest"] == solo_manifest["content_digest"]
        assert manifest["dimension"] == 1
        assert (
            manifest["episodes"][0]["source_row_indices_sha256"]
            == large_manifest["episodes"][0]["source_row_indices_sha256"]
        )
        for role in ("x_columns", "u_columns"):
            assert (
                manifest["sources"][0][role] == large_manifest["sources"][0][role][:1]
            )
        first = torch.load(small_root / "val" / "batch_000000.pt", weights_only=True)
        second = torch.load(
            standalone / "d-1" / case / "val" / "batch_000000.pt", weights_only=True
        )
        for key, value in first.items():
            assert (
                torch.equal(value, second[key])
                if isinstance(value, torch.Tensor)
                else value == second[key]
            )
        for field in ("d_X", "d_U", "d_W", "d_Z"):
            assert first[field].item() == 1
        assert torch.allclose(first["Y1"] - first["Y0"], first["CATE"], atol=1e-6)
        loaded = load_benchmark_from_val_dir(small_root / "val")
        assert loaded["true_CATEs"].shape == (1, 2)
    cell = matrix / "d-1" / "linear_conf_low_proxy_low"
    rebuilt = build(
        parse_args(
            [
                "--config",
                str(cell / "config.yaml"),
                "--name",
                "rebuilt",
                "--case",
                cell.name,
            ]
        )
    )
    assert (
        json.loads((rebuilt / "d-1" / cell.name / "manifest.json").read_text())[
            "content_digest"
        ]
        == json.loads((cell / "manifest.json").read_text())["content_digest"]
    )
