from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from analysis_eval import posterior_calibration as calibration


def _write_analytic_root(root: Path, *, include_cate: bool = True) -> Path:
    root.mkdir(parents=True)
    val = root / "val"
    val.mkdir()
    n_ctx, n_query = 4, 2
    y = torch.tensor([[1.0, 2.0, 3.0, 4.0, 99.0, 98.0], [2.0, 2.0, 2.0, 2.0, 7.0, 8.0]])
    batch = {
        "Y_obs": y,
        "train_sizes": torch.full((2,), n_ctx),
        "seq_lens": torch.full((2,), n_ctx + n_query),
    }
    if include_cate:
        batch["CATE"] = torch.tensor(
            [[5.0, 6.0, 10.0, 11.0, 12.0, 13.0], [3.0, 3.0, 3.0, 3.0, 21.0, 22.0]]
        )
    torch.save(batch, val / "batch_000000.pt")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "target_kind": "exact_cate",
                "n_context": n_ctx,
                "n_query": n_query,
                "content_digest": "abc123",
                "episodes": [
                    {
                        "episode_id": "source.csv:7",
                        "source_file": "source.csv",
                        "replicate": 7,
                    },
                    {
                        "episode_id": "other.csv:2",
                        "source_file": "other.csv",
                        "replicate": 2,
                    },
                ],
            }
        )
    )
    return root


def test_resolve_analytic_records_freeze_scale_target_and_manifest_order(tmp_path):
    spec = calibration.resolve_dataset(_write_analytic_root(tmp_path / "cell"))

    assert spec.kind == "semisynthetic_exact_cate"
    assert spec.content_digest == "abc123"
    assert spec.n_max == 4
    assert [r["episode_id"] for r in spec.records] == ["source.csv:7", "other.csv:2"]
    first = spec.records[0]
    assert first["source_file"] == "source.csv"
    assert first["replicate"] == 7
    assert first["query_index"].tolist() == [4, 5]
    assert torch.equal(first["tau_star"], torch.tensor([12.0, 13.0]))
    expected_scale = torch.tensor([1.0, 2.0, 3.0, 4.0]).std(unbiased=False)
    assert first["y_context_std"] == pytest.approx(expected_scale)
    assert torch.equal(first["tau_star_std"], first["tau_star"] / expected_scale)
    assert torch.equal(first["s2_std"], torch.zeros(2))
    # Degenerate context uses the established 1e-3 floor.
    assert spec.records[1]["y_context_std"].item() == pytest.approx(1e-3)


def test_resolve_sidecar_backed_input_checks_episode_alignment(tmp_path, monkeypatch):
    root = tmp_path / "frozen"
    val = root / "val"
    val.mkdir(parents=True)
    torch.save(
        {"train_sizes": torch.tensor([4]), "seq_lens": torch.tensor([6])},
        val / "batch_000000.pt",
    )
    (root / "frozen_meta.pt").touch()
    meta = {"global": {"n_max": 4, "set_kind": "prior"}, "episodes": [{"index": 0}]}
    monkeypatch.setattr(calibration, "load_frozen_meta", lambda _: meta)
    monkeypatch.setattr(calibration, "frozen_content_digest", lambda _: "digest")

    spec = calibration.resolve_dataset(root)
    assert spec.kind == "prior"
    assert spec.content_digest == "digest"
    assert spec.records == meta["episodes"]
    assert spec.n_episodes == 1

    meta["episodes"].append({"index": 1})
    with pytest.raises(ValueError, match="sidecar has 2 episodes"):
        calibration.resolve_dataset(root)


def test_context_and_level_normalization_and_validation():
    assert calibration.default_contexts(700) == [128, 256, 512, 700]
    assert calibration.normalize_contexts([512, 128], 700) == [128, 512]
    assert calibration.normalize_levels([90, 50, 1]) == (0.01, 0.5, 0.9)
    with pytest.raises(ValueError, match="outside"):
        calibration.normalize_contexts([0], 700)
    with pytest.raises(ValueError, match="unique"):
        calibration.normalize_contexts([128, 128], 700)
    with pytest.raises(ValueError, match="strictly between"):
        calibration.normalize_levels([0, 90])


def test_run_calibration_produces_deterministic_coverage_pit_and_nested_calls(
    monkeypatch, tmp_path
):
    records = [
        {
            "index": 0,
            "episode_id": "e0",
            "query_index": torch.tensor([8, 9]),
            "tau_star_std": torch.tensor([0.0, 0.0]),
            "s2_std": torch.zeros(2),
        },
        {
            "index": 1,
            "episode_id": "e1",
            "query_index": torch.tensor([8, 9]),
            "tau_star_std": torch.tensor([0.0, 0.0]),
            "s2_std": torch.zeros(2),
        },
    ]
    dataset = calibration.DatasetSpec(tmp_path, tmp_path, "test", "digest", 8, records)
    model = SimpleNamespace(
        loss_type="cate_single", cate_bin_edges=torch.tensor([-1.0, 0.0, 1.0])
    )
    monkeypatch.setattr(
        calibration, "iter_frozen_episodes", lambda *_args: iter([["ep0"], ["ep1"]])
    )
    called = []

    def fake_predictive(_model, batch, _record, *, n_ctx, n_max, device):
        called.append((batch[0], n_ctx, n_max, device))
        # A uniform density on [-1, 1] gives PIT 0.5 at target zero; its
        # central intervals contain zero at each requested level.
        return torch.full((2, 2), 0.5), torch.tensor([-1.0, 0.0, 1.0])

    monkeypatch.setattr(calibration, "episode_predictive", fake_predictive)
    result = calibration.run_calibration(
        model,
        dataset,
        contexts=[4, 8],
        levels=[0.5, 0.9],
        progress=False,
    )

    assert called == [
        ("ep0", 4, 8, "cpu"),
        ("ep0", 8, 8, "cpu"),
        ("ep1", 4, 8, "cpu"),
        ("ep1", 8, 8, "cpu"),
    ]
    assert len(result.points) == 8
    assert result.points.groupby("episode")["query_index"].apply(list).to_dict() == {
        0: [8, 9, 8, 9],
        1: [8, 9, 8, 9],
    }
    assert np.allclose(result.points["pit"], 0.5)
    assert np.allclose(result.metrics["cov50_pooled"], 1.0)
    assert np.allclose(result.metrics["cov90_pooled"], 1.0)
    assert np.allclose(result.coverage["calibration_error"], [0.5, 0.1] * 2)
    ecdf = result.pit_ecdf
    assert np.allclose(ecdf[ecdf["pit"] < 0.5]["ecdf"], 0.0)
    assert np.allclose(ecdf[ecdf["pit"] >= 0.5]["ecdf"], 1.0)


def test_invalid_roots_missing_cate_and_bad_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="contain val"):
        calibration.resolve_dataset(tmp_path / "missing")
    root = _write_analytic_root(tmp_path / "no-cate", include_cate=False)
    with pytest.raises(ValueError, match="no analytic CATE"):
        calibration.resolve_dataset(root)
    with pytest.raises(FileNotFoundError, match="neither a file nor a directory"):
        calibration.resolve_checkpoint(tmp_path / "missing.ckpt")


def test_run_rejects_two_potential_outcome_checkpoint(tmp_path):
    from proximalfm.train._dist_eval import UnsupportedLossTypeError

    dataset = calibration.DatasetSpec(tmp_path, tmp_path, "test", None, 8, [])
    model = SimpleNamespace(loss_type="bins_po")
    with pytest.raises(UnsupportedLossTypeError, match="two per-arm histograms"):
        calibration.run_calibration(
            model, dataset, contexts=[8], levels=[0.9], progress=False
        )


def test_checkpoint_directory_resolution_and_figure_formats(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "step-000002.ckpt").touch()
    latest = ckpt_dir / "step-000010.ckpt"
    latest.touch()
    assert calibration.resolve_checkpoint(ckpt_dir) == latest.resolve()

    coverage = pd.DataFrame(
        {"n_ctx": [4, 4], "nominal": [0.5, 0.9], "empirical": [0.5, 1.0]}
    )
    pit = pd.DataFrame({"n_ctx": [4, 4], "pit": [0.0, 1.0], "ecdf": [0.5, 1.0]})
    # Fontconfig probes the host with ``fc-list`` during Matplotlib's first
    # import. Return an empty response in this headless test so the figure uses
    # Matplotlib's own fallback fonts without depending on host utilities.
    import subprocess

    monkeypatch.setattr(subprocess, "check_output", lambda *_args, **_kwargs: b"")
    result = calibration.CalibrationResult(
        points=pd.DataFrame(
            {
                "episode": [0],
                "query_index": [3],
                "n_ctx": [4],
                "target": [0.0],
                "pit": [0.5],
            }
        ),
        metrics=pd.DataFrame({"n_ctx": [4], "crps": [0.25]}),
        coverage=coverage,
        pit_ecdf=pit,
    )
    paths = calibration.write_outputs(
        result, tmp_path / "outputs", provenance={"dataset_kind": "test"}
    )
    assert {path.name for path in paths} == set(calibration.OUTPUT_FILES)
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert all(
        (tmp_path / "outputs" / f"posterior_calibration{suffix}").stat().st_size > 100
        for suffix in (".pdf", ".png")
    )
    assert pd.read_parquet(tmp_path / "outputs" / "points.parquet").equals(
        result.points
    )
    assert pd.read_csv(tmp_path / "outputs" / "metrics.csv").equals(result.metrics)
    assert pd.read_csv(tmp_path / "outputs" / "coverage.csv").equals(coverage)
    assert pd.read_csv(tmp_path / "outputs" / "pit_ecdf.csv").equals(pit)
    assert json.loads((tmp_path / "outputs" / "run.json").read_text()) == {
        "dataset_kind": "test"
    }
