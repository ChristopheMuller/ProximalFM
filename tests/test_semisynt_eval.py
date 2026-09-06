"""Contracts for the format-v5 semi-synthetic mechanism matrix."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from benchmarks.data import load_benchmark_from_val_dir
from semisynt_eval._config import (
    CALIBRATED_LATENT_PROXY,
    TRANSFORMS,
    load_case_configs,
)
from semisynt_eval.build import DEFAULT_CONFIG_PATH, build, parse_args
from semisynt_eval.generator import CSVProximalDataset
from semisynt_eval.mechanisms import available_mechanisms, fit_mechanism
from semisynt_eval.mechanisms.base import (
    RandomBasisRegression,
    apply_transform,
    combine_variance_share,
)
from semisynt_eval.source import discover_csvs, load_csv, resolve_csv


def _recipe() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())


def _configs():
    recipe = _recipe()
    return load_case_configs(recipe["common"], recipe["cases"])


def _fit_mechanism(case, d_x, d_u, seed):
    calibration = torch.Generator().manual_seed(seed + 10_000)
    x_cal = torch.randn(64, d_x, generator=calibration)
    u_cal = torch.randn(64, d_u, generator=calibration)
    return fit_mechanism(
        case.mechanism,
        x_cal,
        u_cal,
        torch.Generator().manual_seed(seed),
        config=case,
    )


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


def test_default_recipe_is_the_complete_two_by_two_by_two_matrix():
    configs = _configs()
    assert len(configs) == 8
    assert available_mechanisms() == ["random_basis"]
    assert {case.mechanism for case in configs.values()} == {"random_basis"}
    observed = {
        (
            case.family,
            case.a_u_variance_share,
            case.w_u_reliability,
        )
        for case in configs.values()
    }
    assert observed == {
        (family, confounding, information)
        for family in ("linear", "nonlinear")
        for confounding in (0.25, 0.75)
        for information in (0.25, 0.75)
    }
    assert all(
        case.a_u_variance_share == case.y_u_variance_share for case in configs.values()
    )
    assert all(
        case.w_u_reliability == case.z_u_reliability for case in configs.values()
    )
    assert all(
        not getattr(case, equation).interactions
        for case in configs.values()
        for equation in ("w", "z", "a", "y", "tau")
    )


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda r: r["common"].update(mechanism="unknown"),
            "common.mechanism",
        ),
        (lambda r: r["common"]["coefficients"].update(extra=1), "unknown"),
        (
            lambda r: r["common"]["nonlinear"]["probabilities"].update(identity=0.3),
            "sum to 1",
        ),
        (
            lambda r: r["cases"]["linear_conf_low_proxy_low"].update(
                a_u_variance_share=1.1
            ),
            r"lie in \[0, 1\]",
        ),
        (
            lambda r: r["cases"]["linear_conf_low_proxy_low"].update(
                w_u_reliability=-0.1
            ),
            r"lie in \[0, 1\]",
        ),
    ],
)
def test_schema_rejects_unknown_and_invalid_values(edit, message):
    recipe = _recipe()
    edit(recipe)
    with pytest.raises(ValueError, match=message):
        load_case_configs(recipe["common"], recipe["cases"])


def test_mechanism_identity_is_separate_from_function_family():
    recipe = _recipe()
    recipe["common"]["mechanism"] = CALIBRATED_LATENT_PROXY
    configs = load_case_configs(recipe["common"], recipe["cases"])

    assert {case.mechanism for case in configs.values()} == {CALIBRATED_LATENT_PROXY}
    assert {case.family for case in configs.values()} == {"linear", "nonlinear"}


def test_unimplemented_calibrated_latent_proxy_is_rejected_by_registry():
    recipe = _recipe()
    recipe["common"]["mechanism"] = CALIBRATED_LATENT_PROXY
    configs = load_case_configs(recipe["common"], recipe["cases"])
    case = configs["linear_conf_low_proxy_low"]

    with pytest.raises(KeyError, match="unknown mechanism.*random_basis"):
        fit_mechanism(
            case.mechanism,
            torch.zeros(4, 2),
            torch.zeros(4, 2),
            torch.Generator().manual_seed(1),
            config=case,
        )


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


def test_interaction_default_and_per_equation_override_resolve_to_booleans():
    recipe = _recipe()
    recipe["common"]["interactions_default"] = True
    case = recipe["cases"]["linear_conf_low_proxy_low"]
    case["equations"] = {"w": {"interactions": False}}
    configs = load_case_configs(recipe["common"], recipe["cases"])
    selected = configs["linear_conf_low_proxy_low"]
    assert selected.w.interactions is False
    assert selected.z.interactions is True
    assert selected.a.interactions is True


def test_nonlinear_transforms_are_finite_on_extreme_inputs():
    values = torch.tensor([-1e30, -100.0, -1.0, 0.0, 1.0, 100.0, 1e30])
    for transform in TRANSFORMS:
        result = apply_transform(values, transform, exp_clip=3.0)
        assert torch.isfinite(result).all(), transform


@pytest.mark.parametrize("family", ["linear", "nonlinear"])
def test_random_basis_components_are_normalized_and_deterministic(family):
    case = _configs()[f"{family}_conf_low_proxy_low"]
    kwargs = dict(
        block_names=("x", "u"),
        block_widths=(3, 2),
        n_outputs=2,
        family=family,
        interactions=True,
        coefficients=case.coefficients,
        nonlinear=case.nonlinear,
    )
    first = RandomBasisRegression(**kwargs, generator=torch.Generator().manual_seed(13))
    second = RandomBasisRegression(
        **kwargs, generator=torch.Generator().manual_seed(13)
    )
    assert first.pairs
    torch.testing.assert_close(first.coefficients, second.coefficients)
    torch.testing.assert_close(first.transforms, second.transforms)
    generator = torch.Generator().manual_seed(91)
    x = torch.randn(4096, 3, generator=generator)
    u = torch.randn(4096, 2, generator=generator)
    latent, observed = first.contributions(x=x, u=u)
    for component in (latent, observed):
        torch.testing.assert_close(component.mean(0), torch.zeros(2), atol=2e-6, rtol=0)
        torch.testing.assert_close(
            component.std(0, unbiased=False), torch.ones(2), atol=2e-6, rtol=0
        )


def test_linear_without_interactions_has_identity_main_terms_only():
    case = _configs()["linear_conf_low_proxy_low"]
    regression = RandomBasisRegression(
        block_names=("x",),
        block_widths=(4,),
        n_outputs=1,
        family="linear",
        interactions=False,
        coefficients=case.coefficients,
        nonlinear=case.nonlinear,
        generator=torch.Generator().manual_seed(2),
    )
    assert not regression.pairs
    assert torch.equal(regression.transforms, torch.zeros_like(regression.transforms))


def test_fitted_mechanism_rejects_invalid_calibration_rows():
    case = _configs()["linear_conf_low_proxy_low"]
    with pytest.raises(ValueError, match="matching non-empty rows"):
        fit_mechanism(
            case.mechanism,
            torch.zeros(3, 2),
            torch.zeros(2, 2),
            torch.Generator().manual_seed(1),
            config=case,
        )
    with pytest.raises(ValueError, match="must be finite"):
        fit_mechanism(
            case.mechanism,
            torch.tensor([[0.0, float("nan")]]),
            torch.zeros(1, 2),
            torch.Generator().manual_seed(1),
            config=case,
        )


@pytest.mark.parametrize("family", ["linear", "nonlinear"])
def test_calibrated_components_and_proxy_reliability(family):
    base = _configs()[f"{family}_conf_low_proxy_low"]
    case = replace(
        base,
        w=replace(base.w, use_x=False, normalize_output=False),
        w_u_reliability=0.25,
    )
    mechanism = _fit_mechanism(case, 3, 2, 17)
    generator = torch.Generator().manual_seed(18)
    x = torch.randn(20000, 3, generator=generator)
    u = torch.randn(20000, 2, generator=generator)
    latent, observed = mechanism.w_regression.contributions(u=u)
    assert observed is None
    state = mechanism.generator.get_state()
    w = mechanism._proxy(mechanism.w_regression, case.w, 0.25, x, u)
    mechanism.generator.set_state(state)
    expected_noise = (3.0**0.5) * torch.randn(20000, 2, generator=mechanism.generator)
    torch.testing.assert_close(w, latent + expected_noise)
    measured = latent.var(0, unbiased=False) / (
        latent.var(0, unbiased=False) + expected_noise.var(0, unbiased=False)
    )
    torch.testing.assert_close(measured, torch.full((2,), 0.25), atol=0.01, rtol=0)

    latent = torch.randn(20000, 1, generator=generator)
    observed = torch.randn(20000, 1, generator=generator)
    latent = (latent - latent.mean()) / latent.std(unbiased=False)
    observed = (observed - observed.mean()) / observed.std(unbiased=False)
    mixed = combine_variance_share(latent, observed, 0.25)
    torch.testing.assert_close(
        (0.25**0.5 * latent).var(unbiased=False),
        torch.tensor(0.25),
        atol=1e-5,
        rtol=0,
    )
    assert torch.isfinite(mixed).all()


def test_disabled_parent_blocks_do_not_enter_equations():
    base = _configs()["linear_conf_low_proxy_low"]
    case = replace(
        base,
        a=replace(base.a, use_x=False),
        y=replace(base.y, use_x=False),
    )
    mechanism = _fit_mechanism(case, 3, 2, 8)
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(1000, 3, generator=generator)
    changed_x = x + 100 * torch.randn(1000, 3, generator=generator)
    u = torch.randn(1000, 2, generator=generator)
    w = torch.randn(1000, 2, generator=generator)
    z = torch.randn(1000, 2, generator=generator)
    torch.testing.assert_close(
        mechanism.treatment_probability(x, u, z),
        mechanism.treatment_probability(changed_x, u, z),
    )
    state = mechanism.generator.get_state()
    original = mechanism.potential_outcomes(x, u, w)
    mechanism.generator.set_state(state)
    changed = mechanism.potential_outcomes(changed_x, u, w)
    torch.testing.assert_close(original[0], changed[0])


@pytest.mark.parametrize("family", ["linear", "nonlinear"])
def test_exact_cate_shared_noise_and_propensity_bounds(family):
    case = _configs()[f"{family}_conf_low_proxy_low"]
    mechanism = _fit_mechanism(case, 4, 3, 25)
    x = torch.randn(2048, 4, generator=torch.Generator().manual_seed(26))
    u = torch.randn(2048, 3, generator=torch.Generator().manual_seed(27))
    w, z = mechanism.proxies(x, u)
    propensity = mechanism.treatment_probability(x, u, z)
    y0, y1 = mechanism.potential_outcomes(x, u, w)
    assert torch.isfinite(propensity).all()
    assert torch.all((propensity > 0) & (propensity < 1))
    torch.testing.assert_close(y1 - y0, mechanism.cate(x), atol=2e-6, rtol=1e-6)
    for proxy in (w, z):
        torch.testing.assert_close(
            proxy.std(0, unbiased=False), torch.ones(3), atol=2e-6, rtol=0
        )


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


def _tiny_build_args(csv: Path, out_root: Path, name: str, config: Path | None = None):
    values = [] if config is None else ["--config", str(config)]
    return parse_args(
        [
            *values,
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


def test_matrix_build_writes_eight_paired_benchmark_sets_and_replays(tmp_path):
    csv = tmp_path / "source.csv"
    _csv(csv)
    root = build(_tiny_build_args(csv, tmp_path, "matrix"))
    matrix = json.loads((root / "manifest.json").read_text())
    assert len(matrix["cases"]) == 8
    assert {item["mechanism"] for item in matrix["cases"]} == {"random_basis"}
    case_roots = [root / item["case"] for item in matrix["cases"]]
    batches = []
    for case_root in case_roots:
        manifest = json.loads((case_root / "manifest.json").read_text())
        data = load_benchmark_from_val_dir(case_root / "val")
        assert data["true_CATEs"].shape == (1, 2)
        assert manifest["episodes"][0]["pair_id"] == "source.csv:0"
        batch_path = next((case_root / "val").glob("batch_*.pt"))
        batches.append(torch.load(batch_path, weights_only=True))
    assert all(torch.equal(batches[0]["X"], batch["X"]) for batch in batches[1:])
    assert all(torch.equal(batches[0]["U"], batch["U"]) for batch in batches[1:])

    source_partition = matrix["sources"][0]
    calibration_rows = set(source_partition["calibration_row_indices"])
    evaluation_rows = set(source_partition["evaluation_row_indices"])
    assert calibration_rows.isdisjoint(evaluation_rows)
    assert calibration_rows | evaluation_rows == set(range(12))
    assert len(calibration_rows) == 4
    assert all(
        json.loads((case_root / "manifest.json").read_text())["episodes"][0][
            "source_row_indices_sha256"
        ]
        == json.loads((case_roots[0] / "manifest.json").read_text())["episodes"][0][
            "source_row_indices_sha256"
        ]
        for case_root in case_roots
    )

    original = case_roots[0]
    replay = build(_tiny_build_args(csv, tmp_path, "replay", original / "config.yaml"))
    replay_batch = torch.load(
        next((replay / original.name / "val").glob("batch_*.pt")), weights_only=True
    )
    assert replay_batch.keys() == batches[0].keys()
    for key, value in batches[0].items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, replay_batch[key]), key


def test_generated_dataset_applies_final_overlap_clamp(tmp_path):
    csv = tmp_path / "source.csv"
    _csv(csv)
    source = load_csv(csv, n_features_x=2, n_features_u=2, n_calibration=4)
    case = _configs()["nonlinear_conf_high_proxy_low"]
    dataset = CSVProximalDataset(
        source, mechanism=case.mechanism, mechanism_config=case
    )
    episode = dataset.generate(seed=3, n_rows=8)
    epsilon = case.a.overlap_epsilon
    assert torch.all(episode.A_probs >= epsilon)
    assert torch.all(episode.A_probs <= 1 - epsilon)
    torch.testing.assert_close(episode.Y1 - episode.Y0, episode.CATE)
    assert set(episode.source_row_indices.tolist()).isdisjoint(
        source.calibration_row_indices
    )


def test_build_rejects_source_too_small_for_fixed_partitions(tmp_path):
    csv = tmp_path / "source.csv"
    _csv(csv)
    args = _tiny_build_args(csv, tmp_path, "undersized")
    args.n_calibration = 8
    with pytest.raises(ValueError, match=r"require at least 14 source rows; found 12"):
        build(args)
