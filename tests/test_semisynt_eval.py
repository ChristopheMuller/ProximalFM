"""Contracts for the format-v5 semi-synthetic mechanism matrix."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

import semisynt_eval.mechanisms.base as mechanism_base
from semisynt_eval._config import NONLINEAR_TRANSFORMS, load_case_configs
from semisynt_eval.build import DEFAULT_CONFIG_PATH, build, parse_args
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
    assert {case.mechanism_version for case in configs.values()} == {"1.0.0"}
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
        (lambda r: r["common"].update(mechanism="ayu"), "common.mechanism"),
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
    ],
)
def test_schema_rejects_unknown_or_invalid_values(edit, message):
    recipe = _recipe()
    edit(recipe)
    with pytest.raises(ValueError, match=message):
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


def test_format_v3_is_rejected_with_migration_message(tmp_path):
    recipe = _recipe()
    recipe["format_version"] = 3
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match="migrate.*common/cases"):
        parse_args(["--config", str(path), "--list"])


def test_format_v4_is_rejected_with_scientific_migration_message(tmp_path):
    recipe = _recipe()
    recipe["format_version"] = 4
    path = tmp_path / "legacy_random_basis.yaml"
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(
        ValueError,
        match=(
            r"format_version 4.*legacy random-basis causal semantics.*"
            r"native calibrated_latent_proxy schema.*"
            r"changing only the format_version number"
        ),
    ):
        parse_args(["--config", str(path), "--list"])


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
            "--n_features_x",
            "2",
            "--n_features_u",
            "2",
            "--seed",
            "7",
        ]
    )


def test_build_refuses_unimplemented_mechanism_without_writing(tmp_path):
    csv = tmp_path / "source.csv"
    _csv(csv)
    args = _tiny_build_args(csv, tmp_path, "matrix")

    with pytest.raises(
        NotImplementedError,
        match="not implemented yet: calibrated_latent_proxy",
    ):
        build(args)
    assert not (tmp_path / "matrix").exists()
