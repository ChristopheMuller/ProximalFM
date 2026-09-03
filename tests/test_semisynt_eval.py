"""Focused contracts for generic empirical CSV proximal datasets."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
import yaml

from benchmarks.data import load_benchmark_from_val_dir
from semisynt_eval._config import load_mechanism_configs
from semisynt_eval.build import (
    DEFAULT_CONFIG_PATH,
    build,
    parse_args,
    save_validation,
)
from semisynt_eval.generator import CSVProximalDataset
from semisynt_eval.mechanisms.base import Mechanism
from semisynt_eval.mechanisms import available_mechanisms, build_mechanism
from semisynt_eval.source import discover_csvs, load_csv, resolve_csv


def _mechanism_configs():
    recipe = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    return load_mechanism_configs(
        recipe["mechanisms"], available=set(available_mechanisms())
    )


def test_yaml_mechanism_config_is_complete_and_strict():
    recipe = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    incomplete = dict(recipe["mechanisms"])
    incomplete["linear"] = dict(incomplete["linear"])
    incomplete["linear"]["proxies"] = dict(incomplete["linear"]["proxies"])
    incomplete["linear"]["proxies"].pop("noise_std")
    with pytest.raises(ValueError, match="missing.*noise_std"):
        load_mechanism_configs(incomplete, available=set(available_mechanisms()))


def _csv(path, delimiter=","):
    path.write_text(
        delimiter.join(("id", "numeric", "category", "missing", "constant"))
        + "\n"
        + "\n".join(
            delimiter.join(map(str, row))
            for row in (
                ("a", 1, "z", "", 7),
                ("b", 2, "y", 3, 7),
                ("c", 4, "x", "", 7),
                ("d", 8, "y", 9, 7),
                ("e", 16, "z", "", 7),
                ("f", 32, "x", 12, 7),
            )
        )
        + "\n"
    )


def test_csv_discovery_delimiter_encoding_missing_and_deterministic_split(tmp_path):
    # Extension controls discovery; delimiter is intentionally non-comma.
    path = tmp_path / "arbitrary.csv"
    _csv(path, "\t")
    assert discover_csvs(tmp_path) == [path]
    assert resolve_csv("arbitrary", tmp_path) == path.resolve()

    first = load_csv(path, split_seed=19, n_features_x=2, n_features_u=2)
    second = load_csv(path, split_seed=19, n_features_x=2, n_features_u=2)
    assert first.columns == second.columns
    assert first.x_columns == second.x_columns
    assert first.u_columns == second.u_columns
    assert torch.equal(first.values, second.values)
    assert "constant" not in first.columns
    assert first.values.shape == (6, 4)
    assert torch.isfinite(first.values).all()
    assert first.X.shape[1] == 2 and first.U.shape[1] == 2


def test_csv_rejects_duplicate_headers_and_explicit_columns(tmp_path):
    duplicate_header = tmp_path / "duplicate.csv"
    duplicate_header.write_text("x,x,u\n1,2,3\n4,5,6\n")
    with pytest.raises(ValueError, match="duplicate column names"):
        load_csv(duplicate_header)

    source = tmp_path / "source.csv"
    _csv(source)
    with pytest.raises(ValueError, match="must not contain duplicates"):
        load_csv(
            source,
            x_columns="numeric,numeric",
            u_columns="category",
            n_features_x=2,
            n_features_u=1,
        )


def test_csv_requires_exact_feature_counts(tmp_path):
    source = tmp_path / "source.csv"
    _csv(source)
    with pytest.raises(ValueError, match="only 4 .* columns are available"):
        load_csv(source, n_features_x=3, n_features_u=2)

    with pytest.raises(ValueError, match="explicit lists must contain exactly"):
        load_csv(
            source,
            x_columns="numeric",
            u_columns="category",
            n_features_x=2,
            n_features_u=1,
        )


def test_mechanism_causal_interface_has_exact_cate_and_shared_noise():
    generator = torch.Generator().manual_seed(3)
    default = _mechanism_configs()["linear"]
    config = replace(
        default,
        outcome=replace(default.outcome, noise_std=0.8),
    )
    mechanism = build_mechanism("linear", 2, 1, generator, config=config)
    x = torch.randn(9, 2, generator=generator)
    u = torch.randn(9, 1, generator=generator)
    w, z = mechanism.proxies(x, u)
    p = mechanism.treatment_probability(x, u, z)
    y0, y1 = mechanism.potential_outcomes(x, u, w)
    assert p.shape == (9, 1) and torch.all((p > 0) & (p < 1))
    torch.testing.assert_close(y1 - y0, mechanism.cate(x))
    assert y0.shape == y1.shape == (9, 1)


def test_mechanisms_are_separate_closed_form_families():
    assert available_mechanisms() == ["linear", "linear_quadratic"]
    configs = _mechanism_configs()
    linear = build_mechanism(
        "linear", 7, 5, torch.Generator().manual_seed(12), config=configs["linear"]
    )
    mixed = build_mechanism(
        "linear_quadratic",
        7,
        5,
        torch.Generator().manual_seed(12),
        config=configs["linear_quadratic"],
    )
    regressions = (
        "w_regression",
        "z_regression",
        "treatment_regression",
        "outcome_regression",
        "effect_regression",
    )
    assert all(torch.all(getattr(linear, name).powers == 1) for name in regressions)
    mixed_powers = torch.cat(
        [getattr(mixed, name).powers.flatten() for name in regressions]
    )
    assert set(mixed_powers.tolist()) == {1, 2}


@pytest.mark.parametrize("mechanism_name", available_mechanisms())
def test_generated_proxies_and_outcome_baseline_are_renormalized(mechanism_name):
    generator = torch.Generator().manual_seed(25)
    mechanism = build_mechanism(
        mechanism_name,
        4,
        3,
        generator,
        config=_mechanism_configs()[mechanism_name],
    )
    x = torch.randn(512, 4, generator=generator)
    u = torch.randn(512, 3, generator=generator)
    w, z = mechanism.proxies(x, u)
    y0, y1 = mechanism.potential_outcomes(x, u, w)
    for value in (w, z, y0):
        torch.testing.assert_close(
            value.mean(0), torch.zeros(value.shape[1]), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            value.std(0, unbiased=False), torch.ones(value.shape[1]), atol=1e-6, rtol=0
        )
    torch.testing.assert_close(y1 - y0, mechanism.cate(x))


def test_tiny_build_round_trip_uses_explicit_cate_without_replicates(tmp_path):
    csv = tmp_path / "source.csv"
    _csv(csv)
    common = [
        "--csv",
        str(csv),
        "--out_root",
        str(tmp_path),
        "--n_max",
        "8",
        "--n_q",
        "3",
        "--n_datasets",
        "2",
        "--datasets_per_file",
        "1",
        "--n_features_x",
        "2",
        "--n_features_u",
        "2",
        "--seed",
        "7",
    ]
    build(parse_args([*common, "--name", "cate"]))
    cate = load_benchmark_from_val_dir(tmp_path / "cate" / "val")
    assert cate["true_CATEs"] is not None
    assert cate["true_CATEs"].shape == (2, 3)
    assert cate["true_ITEs"].shape == (2, 3)
    assert cate["cate_K_eff"] is None
    assert not cate["cate_s2"].any()

    payloads = [
        torch.load(path, weights_only=True)
        for path in sorted((tmp_path / "cate" / "val").glob("batch_*.pt"))
    ]
    assert all("CATE" in payload for payload in payloads)
    assert all(
        torch.equal(payload["K_eff"], torch.ones(1, dtype=torch.long))
        for payload in payloads
    )
    manifest = json.loads((tmp_path / "cate" / "manifest.json").read_text())
    assert manifest["mechanism_config"]["outcome"]["use_w"] is True
    assert manifest["mechanism_config"]["powers"]["effect_quadratic_probability"] == 0

    resolved = yaml.safe_load((tmp_path / "cate" / "config.yaml").read_text())
    assert resolved["build"]["csv"] == str(csv)
    assert resolved["build"]["n_max"] == 8
    assert resolved["source"]["sha256"] == manifest["source_sha256"]
    assert set(resolved["mechanisms"]) == {"linear", "linear_quadratic"}

    # The saved recipe is directly reusable. Only the output name changes;
    # every data-affecting value, including the source digest, is inherited.
    build(
        parse_args(
            [
                "--config",
                str(tmp_path / "cate" / "config.yaml"),
                "--name",
                "replica",
            ]
        )
    )
    original_paths = sorted((tmp_path / "cate" / "val").glob("batch_*.pt"))
    replica_paths = sorted((tmp_path / "replica" / "val").glob("batch_*.pt"))
    assert len(original_paths) == len(replica_paths)
    for original_path, replica_path in zip(original_paths, replica_paths):
        original = torch.load(original_path, weights_only=True)
        replica = torch.load(replica_path, weights_only=True)
        assert original.keys() == replica.keys()
        for key in original:
            if isinstance(original[key], torch.Tensor):
                assert torch.equal(original[key], replica[key]), key


def test_semisynt_eval_package_has_no_lit_eval_dependency():
    from pathlib import Path

    import semisynt_eval

    root = Path(semisynt_eval.__file__).parent
    sources = "\n".join(path.read_text() for path in root.rglob("*.py"))
    assert "from lit_eval" not in sources
    assert "import lit_eval" not in sources


def test_future_ite_only_mechanism_omits_exact_target(tmp_path, monkeypatch):
    class ITEOnly(Mechanism):
        target_capability = "ite_only"

        def proxies(self, x, u):
            return u.clone(), u.clone()

        def treatment_probability(self, x, u, z):
            return torch.sigmoid(x[:, :1] + u[:, :1] + z[:, :1])

        def potential_outcomes(self, x, u, w):
            y0 = x[:, :1] + u[:, :1] + w[:, :1]
            return y0, y0 + u[:, :1]

    monkeypatch.setattr(
        "semisynt_eval.generator.build_mechanism",
        lambda *args, **kwargs: ITEOnly(),
    )
    csv = tmp_path / "source.csv"
    _csv(csv)
    dataset = CSVProximalDataset(
        load_csv(csv, n_features_x=2, n_features_u=2),
        mechanism="linear",
        mechanism_config=_mechanism_configs()["linear"],
    )
    episode = dataset.generate(seed=3, seq_len=10)
    assert dataset.target_capability == "ite_only"
    assert episode.CATE is None

    val = tmp_path / "val"
    save_validation(
        [episode],
        out_dir=val,
        train_size=7,
        datasets_per_file=None,
        dataset=dataset,
        cli_command="test",
    )
    data = load_benchmark_from_val_dir(val)
    assert data["true_CATEs"] is None
    assert data["true_ITEs"].shape == (1, 3)
