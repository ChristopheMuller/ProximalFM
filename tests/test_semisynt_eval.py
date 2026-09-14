"""Contracts for the semi-synthetic mechanism matrix."""

from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import yaml

import semisynt_eval.mechanisms.base as mechanism_base
from benchmarks.data import load_benchmark_from_val_dir
from semisynt_eval._config import NONLINEAR_TRANSFORMS, load_case_configs
from semisynt_eval.build import (
    DEFAULT_CONFIG_PATH,
    _case_payload,
    _content_digest,
    _resolved_recipe,
    build,
    parse_args,
)
from semisynt_eval.source import discover_csvs, load_csv, resolve_csv


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


def test_default_recipe_resolves_complete_calibrated_latent_proxy_matrix():
    configs = _configs()
    assert len(configs) == 8
    assert {case.mechanism for case in configs.values()} == {"calibrated_latent_proxy"}
    assert {
        (case.family, case.confounding, case.proxy) for case in configs.values()
    } == {
        (family, confounding, proxy)
        for family in ("linear", "nonlinear")
        for confounding in ("low", "high")
        for proxy in ("low", "high")
    }
    assert {
        (case.confounding, case.alpha, case.gamma) for case in configs.values()
    } == {("low", 0.35, 0.5), ("high", 1.25, 1.5)}
    assert {(case.proxy, case.reliability) for case in configs.values()} == {
        ("low", 0.2),
        ("high", 0.8),
    }


def test_default_recipe_carries_all_fixed_mechanism_constants():
    case = _configs()["nonlinear_conf_high_proxy_low"]
    assert case.latent.innovation_sd == 0.5
    assert case.latent.whitening_ridge == 0.05
    assert case.latent.direction_sd == 1.0
    assert case.basis.nonlinear_transforms == NONLINEAR_TRANSFORMS
    assert case.basis.max_interactions == 8
    assert case.basis.coefficient_distribution == "normal"
    assert case.basis.coefficient_scale == 1.0
    assert case.basis.normalize_by_basis_dim is True
    assert case.basis.min_scale == 1e-8
    assert case.proxies.x_scale == 0.5
    assert case.treatment.x_scale == 0.5
    assert (case.treatment.intercept_low, case.treatment.intercept_high) == (
        -0.25,
        0.25,
    )
    assert (case.treatment.propensity_lower, case.treatment.propensity_upper) == (
        0.02,
        0.98,
    )
    assert case.outcome.noise_sd == 1.0
    assert (case.cate.intercept, case.cate.scale) == (0.5, 1.0)


def test_yaml_values_are_resolved_without_hidden_python_defaults():
    recipe = _recipe()
    recipe["common"]["latent"]["innovation_sd"] = 0.4
    recipe["common"]["levels"]["confounding"]["low"]["alpha"] = 0.7
    recipe["common"]["levels"]["proxy"]["high"]["reliability"] = 0.9
    configs = load_case_configs(recipe["common"], recipe["cases"])

    assert configs["linear_conf_low_proxy_low"].latent.innovation_sd == 0.4
    assert configs["linear_conf_low_proxy_low"].alpha == 0.7
    assert configs["linear_conf_low_proxy_high"].reliability == 0.9


def test_resolved_settings_are_serialized_in_generated_case_configs():
    args = parse_args(["--list"])
    case = args._case_configs["nonlinear_conf_high_proxy_low"]
    payload = _case_payload(case)
    recipe = _resolved_recipe(args, [])

    assert payload["mechanism"] == "calibrated_latent_proxy"
    assert payload["family"] == "nonlinear"
    assert payload["alpha"] == 1.25
    assert payload["gamma"] == 1.5
    assert payload["reliability"] == 0.2
    assert payload["latent"]["whitening_ridge"] == 0.05
    assert payload["treatment"]["propensity_lower"] == 0.02
    assert recipe["resolved_cases"][case.name] == payload


def test_content_digest_includes_resolved_settings():
    args = parse_args(["--list"])
    case = args._case_configs["linear_conf_low_proxy_low"]
    datasets = [SimpleNamespace(options={"source": "fixture"})]
    baseline = _content_digest(args, case, datasets)

    changed_ridge = replace(
        case,
        latent=replace(case.latent, whitening_ridge=case.latent.whitening_ridge + 0.01),
    )
    assert _content_digest(args, changed_ridge, datasets) != baseline


def test_generated_recipe_rejects_stale_resolved_case_snapshot(tmp_path):
    args = parse_args(["--list"])
    recipe = _resolved_recipe(args, [])
    recipe["resolved_cases"]["linear_conf_low_proxy_low"]["alpha"] = 99
    path = tmp_path / "stale.yaml"
    path.write_text(yaml.safe_dump(recipe))

    with pytest.raises(ValueError, match="resolved_cases does not match"):
        parse_args(["--config", str(path), "--list"])


def test_resolved_cases_do_not_expose_random_basis_fields():
    case = _configs()["linear_conf_low_proxy_low"]
    for field in (
        "coefficients",
        "nonlinear",
        "w",
        "z",
        "a",
        "y",
        "tau",
        "a_u_variance_share",
        "y_u_variance_share",
        "w_u_reliability",
        "z_u_reliability",
    ):
        assert not hasattr(case, field)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda r: r["common"].update(mechanism="unknown_mechanism"),
            "common.mechanism",
        ),
        (
            lambda r: r["common"]["latent"].update(innovation_sd=1.0),
            r"innovation_sd must lie in \[0, 1\)",
        ),
        (
            lambda r: r["common"]["latent"].update(whitening_ridge=0),
            "whitening_ridge must be positive",
        ),
        (
            lambda r: r["common"]["basis"].update(max_interactions=-1),
            "max_interactions must be non-negative",
        ),
        (
            lambda r: r["common"]["basis"].update(
                nonlinear_transforms=["identity", "sine"]
            ),
            "nonlinear_transforms must be exactly",
        ),
        (
            lambda r: r["common"]["treatment"].update(propensity_lower=0.99),
            "propensity bounds",
        ),
        (
            lambda r: r["common"]["levels"]["proxy"]["low"].update(reliability=0),
            r"reliability must lie in \(0, 1\)",
        ),
        (
            lambda r: r["cases"]["linear_conf_low_proxy_low"].update(
                confounding="missing"
            ),
            "confounding must name one of",
        ),
        (
            lambda r: r["cases"]["linear_conf_low_proxy_low"].update(extra=1),
            "unknown",
        ),
        (
            lambda r: r["cases"]["linear_conf_low_proxy_low"].update(family="other"),
            "family must be linear or nonlinear",
        ),
    ],
)
def test_schema_rejects_unknown_or_invalid_values(edit, message):
    recipe = _recipe()
    edit(recipe)
    with pytest.raises(ValueError, match=message):
        load_case_configs(recipe["common"], recipe["cases"])


@pytest.mark.parametrize("level", ["confounding", "proxy"])
def test_schema_requires_exact_low_high_level_definitions(level):
    recipe = _recipe()
    recipe["common"]["levels"][level]["medium"] = recipe["common"]["levels"][level].pop(
        "high"
    )

    with pytest.raises(ValueError, match=f"levels.{level}.*exactly low and high"):
        load_case_configs(recipe["common"], recipe["cases"])


def test_schema_rejects_missing_factorial_cell():
    recipe = _recipe()
    recipe["cases"].pop("nonlinear_conf_high_proxy_high")

    with pytest.raises(
        ValueError,
        match=r"each family x confounding x proxy cell exactly once: missing=.*nonlinear",
    ):
        load_case_configs(recipe["common"], recipe["cases"])


def test_schema_rejects_duplicate_factorial_cell():
    recipe = _recipe()
    recipe["cases"]["duplicate_name"] = dict(
        recipe["cases"]["linear_conf_low_proxy_low"]
    )

    with pytest.raises(
        ValueError,
        match=r"each family x confounding x proxy cell exactly once: duplicates=.*linear",
    ):
        load_case_configs(recipe["common"], recipe["cases"])


def test_mechanism_identity_is_separate_from_function_family():
    configs = _configs()
    linear = configs["linear_conf_low_proxy_low"]
    nonlinear = configs["nonlinear_conf_low_proxy_low"]

    assert linear.mechanism == nonlinear.mechanism == "calibrated_latent_proxy"
    assert linear.family == "linear"
    assert nonlinear.family == "nonlinear"


def test_shared_mechanism_base_contains_no_random_basis_machinery():
    for name in (
        "CoefficientConfig",
        "NonlinearConfig",
        "RandomBasisRegression",
        "apply_transform",
        "combine_variance_share",
        "normalize_columns",
    ):
        assert not hasattr(mechanism_base, name)


def test_row_split_has_only_context_and_query_spelling():
    args = parse_args(["--list"])
    assert args.n_context == 2048
    assert args.n_query == 256
    assert args.n_calibration == 1000
    assert args.row_sampling == "without_replacement"
    for removed in ("seq_len", "prop_train", "n_max", "n_q"):
        assert not hasattr(args, removed)
    with pytest.raises(SystemExit):
        parse_args(["--list", "--seq_len", "100"])


def test_csv_discovery_encoding_and_split_are_deterministic(tmp_path):
    path = tmp_path / "arbitrary.csv"
    _csv(path, "\t")
    assert discover_csvs(tmp_path) == [path]
    assert resolve_csv("arbitrary", tmp_path) == path.resolve()
    first = load_csv(
        path, split_seed=19, n_features_x=2, n_features_u=2, n_calibration=4
    )
    second = load_csv(
        path, split_seed=19, n_features_x=2, n_features_u=2, n_calibration=4
    )
    assert first.columns == second.columns
    assert torch.equal(first.values, second.values)
    assert first.calibration_values.shape == (4, 4)
    assert first.evaluation_values.shape == (8, 4)
    assert torch.isfinite(first.values).all()
    assert set(first.calibration_row_indices).isdisjoint(first.evaluation_row_indices)
    assert sorted(
        (*first.calibration_row_indices, *first.evaluation_row_indices)
    ) == list(range(12))


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


def test_single_case_build_matches_full_matrix(tmp_path):
    csv = tmp_path / "source.csv"
    _csv(csv)
    case = "nonlinear_conf_high_proxy_high"
    full = build(_tiny_build_args(csv, tmp_path, "full"))
    selected_args = _tiny_build_args(csv, tmp_path, "selected")
    selected_args.case_names = [case]
    selected_args._selected_case_configs = {case: selected_args._case_configs[case]}
    selected = build(selected_args)

    full_manifest = json.loads((full / "d-2" / case / "manifest.json").read_text())
    selected_manifest = json.loads(
        (selected / "d-2" / case / "manifest.json").read_text()
    )
    assert full_manifest["content_digest"] == selected_manifest["content_digest"]
    full_batch, selected_batch = _batch(full, case), _batch(selected, case)
    assert full_batch.keys() == selected_batch.keys()
    for key in full_batch:
        value = full_batch[key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, selected_batch[key])
        else:
            assert value == selected_batch[key]
    assert {path.name for path in (selected / "d-2").iterdir()} == {case}


@pytest.mark.parametrize(
    "selection",
    [
        ["--case", "missing"],
        [
            "--case",
            "linear_conf_low_proxy_low",
            "--case",
            "linear_conf_low_proxy_low",
        ],
    ],
)
def test_case_selection_rejects_unknown_and_duplicate_names(selection):
    with pytest.raises(SystemExit):
        parse_args(["--list", *selection])


def test_repeatable_case_selection_uses_canonical_recipe_order():
    first = "nonlinear_conf_high_proxy_high"
    second = "linear_conf_low_proxy_low"
    args = parse_args(["--list", "--case", first, "--case", second])
    assert list(args._selected_case_configs) == [second, first]


def test_builder_fits_exactly_two_families_per_source_replicate(tmp_path, monkeypatch):
    csv = tmp_path / "source.csv"
    _csv(csv)
    build_module = importlib.import_module("semisynt_eval.build")
    original = build_module.fit_mechanism_with_identity
    calls = []

    def recording_fit(*args, **kwargs):
        calls.append(
            (kwargs["source_id"], kwargs["replicate"], kwargs["config"].family)
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(build_module, "fit_mechanism_with_identity", recording_fit)
    build(_tiny_build_args(csv, tmp_path, "matrix"))
    assert calls == [("source.csv", 1, "linear"), ("source.csv", 1, "nonlinear")]


def test_case_traversal_order_does_not_change_case_outputs(tmp_path):
    csv = tmp_path / "source.csv"
    _csv(csv)
    normal_args = _tiny_build_args(csv, tmp_path, "normal")
    normal = build(normal_args)

    reversed_args = _tiny_build_args(csv, tmp_path, "reversed")
    reversed_args._case_configs = dict(reversed(reversed_args._case_configs.items()))
    reversed_args._selected_case_configs = dict(reversed_args._case_configs)
    reversed_root = build(reversed_args)

    for case in _configs():
        normal_batch, reversed_batch = _batch(normal, case), _batch(reversed_root, case)
        for key, value in normal_batch.items():
            other = reversed_batch[key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other)
            else:
                assert value == other


@pytest.mark.parametrize("dimensions", [[], [0], [-1], [True], [1, 1], [1.5], "5"])
def test_invalid_dimension_recipes(tmp_path, dimensions):
    recipe = _recipe()
    recipe["build"]["dimensions"] = dimensions
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(SystemExit):
        parse_args(["--config", str(path), "--list"])


def test_dimension_cli_is_sorted_and_replaces_old_flags():
    assert parse_args(["--list", "--dimensions", "5", "1", "3"]).dimensions == [1, 3, 5]
    with pytest.raises(SystemExit):
        parse_args(["--list", "--n_features_x", "3"])


def test_source_roles_are_nested_even_when_loaded_independently(tmp_path):
    from semisynt_eval.source import dimension_view

    csv = tmp_path / "source.csv"
    _csv(csv)
    large = load_csv(csv, n_features_x=2, n_features_u=2, n_calibration=4)
    small = load_csv(csv, n_features_x=1, n_features_u=1, n_calibration=4)
    view = dimension_view(large, 1)
    assert small.x_columns == large.x_columns[:1] == view.x_columns
    assert small.u_columns == large.u_columns[:1] == view.u_columns
    assert set(large.x_columns).isdisjoint(large.u_columns)
    assert torch.equal(small.X_cal, view.X_cal)
    assert torch.equal(small.U_eval, view.U_eval)


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


def test_explicit_feature_lists_use_prefixes(tmp_path):
    csv = tmp_path / "source.csv"
    _csv(csv)
    args = _tiny_build_args(csv, tmp_path, "explicit")
    args.dimensions = [1, 2]
    args.x_columns = ["numeric", "category"]
    args.u_columns = ["id", "missing"]
    root = build(args)
    cell = root / "d-1" / "linear_conf_low_proxy_low"
    recipe = yaml.safe_load((cell / "config.yaml").read_text())
    assert recipe["build"]["x_columns"] == ["numeric"]
    assert recipe["build"]["u_columns"] == ["id"]
    build(
        parse_args(
            ["--config", str(cell / "config.yaml"), "--name", "explicit-rebuilt"]
        )
    )


def test_force_preserves_benchmark_files_and_validates_first(tmp_path):
    csv = tmp_path / "source.csv"
    _csv(csv)
    args = _tiny_build_args(csv, tmp_path, "matrix")
    root = build(args)
    bench = root / "d-2" / "linear_conf_low_proxy_low" / "bench"
    bench.mkdir()
    keep = bench / "keep.md"
    keep.write_text("keep")
    with pytest.raises(FileExistsError):
        build(args)
    args.force = True
    old = (root / "manifest.json").read_bytes()
    args.dimensions = [3]
    with pytest.raises(ValueError, match="usable columns"):
        build(args)
    assert (root / "manifest.json").read_bytes() == old
    args.dimensions = [2]
    args.n_context = 20
    with pytest.raises(ValueError, match="source rows"):
        build(args)
    assert (root / "manifest.json").read_bytes() == old
    args.n_context = 4
    args.dimensions = [1]
    build(args)
    assert keep.read_text() == "keep"
    assert not list((root / "d-2").glob("*/val/batch_*.pt"))


def test_legacy_root_is_rejected_even_with_force(tmp_path):
    csv = tmp_path / "source.csv"
    _csv(csv)
    args = _tiny_build_args(csv, tmp_path, "legacy")
    (tmp_path / "legacy" / "linear_conf_low_proxy_low").mkdir(parents=True)
    args.force = True
    with pytest.raises(ValueError, match="legacy build layout"):
        build(args)
