"""Focused contracts for generic empirical CSV proximal datasets."""
from __future__ import annotations

import pytest
import torch

from benchmarks.data import load_benchmark_from_val_dir
from semi_synt_eval.build import build, parse_args, save_validation
from semi_synt_eval.generator import CSVProximalDataset
from semi_synt_eval.mechanisms.base import Mechanism
from semi_synt_eval.mechanisms import MechanismConfig, build_mechanism
from semi_synt_eval.source import discover_csvs, load_csv, resolve_csv


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
    mechanism = build_mechanism(
        "linear", 2, 1, MechanismConfig(outcome_noise=0.8), generator
    )
    x = torch.randn(9, 2, generator=generator)
    u = torch.randn(9, 1, generator=generator)
    w, z = mechanism.proxies(x, u)
    p = mechanism.treatment_probability(x, u, z)
    y0, y1 = mechanism.potential_outcomes(x, u, w)
    assert p.shape == (9, 1) and torch.all((p > 0) & (p < 1))
    torch.testing.assert_close(y1 - y0, mechanism.cate(x))
    assert y0.shape == y1.shape == (9, 1)


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


def test_semi_synt_eval_package_has_no_lit_eval_dependency():
    from pathlib import Path

    import semi_synt_eval

    root = Path(semi_synt_eval.__file__).parent
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
        "semi_synt_eval.generator.build_mechanism",
        lambda *args, **kwargs: ITEOnly(),
    )
    csv = tmp_path / "source.csv"
    _csv(csv)
    dataset = CSVProximalDataset(
        load_csv(csv, n_features_x=2, n_features_u=2), mechanism="linear"
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
