"""Tests for the analysis_prior.main CLI.

The CLI is factored so the pieces (spec building, analysis resolution, full run)
are callable in-process; no subprocess is needed.
"""

import json

import pytest

from analysis_prior import main as cli
from analysis_prior.config import PRESETS


def parse(argv):
    return cli.build_parser().parse_args(argv)


# --------------------------------------------------------------------------
# build_spec
# --------------------------------------------------------------------------


def test_build_spec_uses_preset_by_default():
    spec = cli.build_spec(parse(["--spec", "smoke"]))
    assert spec == PRESETS["smoke"]


def test_build_spec_applies_overrides():
    spec = cli.build_spec(
        parse(
            ["--spec", "smoke", "--n-datasets", "9", "--seq-len", "123", "--seed", "7"]
        )
    )
    assert spec.n_datasets == 9
    assert spec.seq_len == 123
    assert spec.seed == 7
    # Untouched fields keep the preset value.
    assert spec.mcmc_cate == PRESETS["smoke"].mcmc_cate


def test_build_spec_rename():
    spec = cli.build_spec(parse(["--spec", "smoke", "--name", "renamed"]))
    assert spec.name == "renamed"


def test_build_spec_unknown_preset_exits():
    with pytest.raises(SystemExit, match="Unknown --spec"):
        cli.build_spec(parse(["--spec", "does_not_exist"]))


def test_build_spec_invalid_override_is_rejected():
    # dataclasses.replace re-runs validation, so a bad override fails loudly.
    with pytest.raises(ValueError, match="seq_len"):
        cli.build_spec(parse(["--spec", "smoke", "--seq-len", "1"]))


# --------------------------------------------------------------------------
# resolve_analyses
# --------------------------------------------------------------------------


def test_resolve_all_expands_to_registry():
    from analysis_prior.analyses import all_analyses

    assert cli.resolve_analyses(["all"]) == sorted(all_analyses())


def test_resolve_named_analysis():
    assert cli.resolve_analyses(["dimensions"]) == ["dimensions"]


def test_resolve_unknown_analysis_raises():
    with pytest.raises(KeyError, match="Unknown analysis"):
        cli.resolve_analyses(["nope"])


def test_resolve_all_with_extra_names_is_rejected():
    with pytest.raises(SystemExit, match="cannot be combined"):
        cli.resolve_analyses(["all", "dimensions"])


# --------------------------------------------------------------------------
# list commands
# --------------------------------------------------------------------------


def test_list_analyses_returns_zero(capsys):
    assert cli.main(["--list-analyses"]) == 0
    assert "dimensions" in capsys.readouterr().out


def test_list_caches_empty(capsys, tmp_path):
    assert cli.main(["--list-caches", "--cache-dir", str(tmp_path)]) == 0
    assert "No caches" in capsys.readouterr().out


# --------------------------------------------------------------------------
# full run
# --------------------------------------------------------------------------


def test_full_run_generates_cache_and_writes_results(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "results"

    rc = cli.main(
        [
            "--spec",
            "smoke",
            "--analysis",
            "dimensions",
            "--cache-dir",
            str(cache_dir),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    # Cache was created.
    assert list(cache_dir.glob("*.pt"))

    # Results are grouped under the human-readable spec name (caches are
    # content-addressed, but output directories stay browsable).
    spec = cli.build_spec(parse(["--spec", "smoke"]))
    analysis_dir = out_dir / spec.name / "dimensions"
    assert (analysis_dir / "summary.json").exists()
    assert (analysis_dir / "spec.json").exists()

    with open(analysis_dir / "summary.json") as f:
        summary = json.load(f)
    assert summary["n_datasets"] == PRESETS["smoke"].n_datasets

    # The summary was echoed to stdout.
    assert "completeness_overall_fraction" in capsys.readouterr().out


def test_full_run_second_call_reuses_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "results"
    argv = [
        "--spec",
        "smoke",
        "--analysis",
        "dimensions",
        "--cache-dir",
        str(cache_dir),
        "--out-dir",
        str(out_dir),
    ]
    assert cli.main(argv) == 0

    # Second call must not regenerate.
    from analysis_prior import generate

    def fail(*args, **kwargs):
        raise AssertionError("regenerated instead of using the cache")

    monkeypatch.setattr(generate, "generate", fail)
    assert cli.main(argv) == 0


# --------------------------------------------------------------------------
# families
# --------------------------------------------------------------------------


def test_build_family_unknown_name_exits():
    with pytest.raises(SystemExit, match="Unknown --family"):
        cli.build_family(parse(["--family", "does_not_exist"]))


def test_build_family_applies_cli_overrides_to_base():
    family = cli.build_family(parse(["--family", "smoke_family", "--n-datasets", "2"]))
    assert family.base.n_datasets == 2
    # Every resolved variant inherits the override.
    assert all(spec.n_datasets == 2 for spec in family.resolve().values())


def test_list_families(capsys):
    assert cli.main(["--list-families"]) == 0
    out = capsys.readouterr().out
    assert "dim_ablation" in out
    assert "smoke_family" in out


@pytest.fixture
def dummy_comparison(monkeypatch):
    """Register a throwaway ComparisonAnalysis in the real registry."""
    from analysis_prior.analyses import base as analyses_base
    from analysis_prior.analyses.base import AnalysisArtifacts, ComparisonAnalysis

    class _Cmp(ComparisonAnalysis):
        name = "cmp_test"
        description = "test comparison"
        calls = []

        def compute(self, caches):
            _Cmp.calls.append(list(caches))
            return {"labels": list(caches)}

        def render(self, results, outdir):
            return AnalysisArtifacts(summary={"n_variants": len(results["labels"])})

    monkeypatch.setitem(analyses_base._REGISTRY, "cmp_test", _Cmp())
    return _Cmp


def test_family_run_dispatches_per_variant_and_once_for_comparisons(
    tmp_path, dummy_comparison
):
    argv = [
        "--family",
        "smoke_family",
        "--n-datasets",
        "2",
        "--analysis",
        "dimensions",
        "cmp_test",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    assert cli.main(argv) == 0

    out = tmp_path / "out" / "smoke_family"

    # Family-level provenance: label -> resolved spec.
    with open(out / "specs.json") as f:
        provenance = json.load(f)
    assert list(provenance["variants"]) == ["triangular", "uniform"]

    # Plain analysis: once per variant, under <family>/<label>/<analysis>/.
    for label in ("triangular", "uniform"):
        assert (out / label / "dimensions" / "summary.json").exists()

    # Comparison analysis: exactly once, with all labels, under <family>/<analysis>/.
    assert dummy_comparison.calls == [["triangular", "uniform"]]
    assert (out / "cmp_test" / "specs.json").exists()
    assert (out / "cmp_test" / "summary.json").exists()


def test_single_spec_rejects_explicit_comparison_analysis(tmp_path, dummy_comparison):
    with pytest.raises(SystemExit, match="--family"):
        cli.main(
            [
                "--spec",
                "smoke",
                "--analysis",
                "cmp_test",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )


def test_single_spec_all_skips_comparison_analyses(tmp_path, dummy_comparison, capsys):
    # 'all' includes cmp_test via the registry, but a single spec cannot feed a
    # comparison -- it must be skipped with a note, not crash the run.
    argv = [
        "--spec",
        "smoke",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "Skipping comparison analyses" in out
    assert dummy_comparison.calls == []
