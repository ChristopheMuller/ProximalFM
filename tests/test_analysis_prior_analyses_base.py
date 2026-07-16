"""Tests for analysis_prior.analyses.base (the Analysis contract and registry)."""

import json

import pytest

from analysis_prior.analyses import base
from analysis_prior.analyses.base import (
    Analysis,
    AnalysisArtifacts,
    all_analyses,
    get_analysis,
    register,
)


@pytest.fixture
def isolated_registry(monkeypatch):
    """Give each test a clean registry so registrations don't leak."""
    monkeypatch.setattr(base, "_REGISTRY", {})
    return base._REGISTRY


class _Dummy(Analysis):
    name = "dummy"
    description = "dummy analysis"

    def compute(self, cache):
        return {"value": 42}

    def render(self, results, outdir):
        (outdir / "artifact.txt").write_text("hello")
        return AnalysisArtifacts(
            summary={"value": results["value"]},
            figures=[outdir / "artifact.txt"],
        )


class _FakeCache:
    """Stand-in for PriorCache; run() only touches cache.spec."""

    class _Spec:
        def to_dict(self):
            return {"name": "fake", "n_datasets": 1}

    spec = _Spec()


def test_register_and_get(isolated_registry):
    register(_Dummy)
    got = get_analysis("dummy")
    assert isinstance(got, _Dummy)
    assert "dummy" in all_analyses()


def test_register_rejects_blank_name(isolated_registry):
    class Blank(Analysis):
        name = ""

    with pytest.raises(ValueError, match="non-empty"):
        register(Blank)


def test_register_rejects_duplicate(isolated_registry):
    register(_Dummy)
    with pytest.raises(ValueError, match="Duplicate"):
        register(_Dummy)


def test_get_unknown_analysis_lists_available(isolated_registry):
    register(_Dummy)
    with pytest.raises(KeyError, match="Unknown analysis"):
        get_analysis("nope")


def test_all_analyses_returns_a_copy(isolated_registry):
    register(_Dummy)
    snapshot = all_analyses()
    snapshot["injected"] = object()
    # Mutating the returned dict must not corrupt the real registry.
    assert "injected" not in all_analyses()


def test_run_writes_provenance_and_returns_artifacts(isolated_registry, tmp_path):
    artifacts = _Dummy().run(_FakeCache(), tmp_path)

    assert artifacts.summary == {"value": 42}
    assert (tmp_path / "artifact.txt").exists()

    # Every output dir is self-describing: the spec it came from and the findings.
    with open(tmp_path / "spec.json") as f:
        assert json.load(f)["name"] == "fake"
    with open(tmp_path / "summary.json") as f:
        assert json.load(f) == {"value": 42}


def test_run_creates_missing_output_dir(isolated_registry, tmp_path):
    target = tmp_path / "nested" / "out"
    _Dummy().run(_FakeCache(), target)
    assert (target / "summary.json").exists()


def test_base_methods_are_abstract():
    with pytest.raises(NotImplementedError):
        Analysis().compute(None)
    with pytest.raises(NotImplementedError):
        Analysis().render(None, None)


def test_real_registry_contains_dimensions():
    # Importing the package should have registered the shipped analyses.
    import analysis_prior.analyses  # noqa: F401

    assert "dimensions" in all_analyses()
