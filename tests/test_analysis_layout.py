"""The analysis namespace keeps repository inputs and relocated outputs reachable."""

import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = REPO_ROOT / "analysis"


@pytest.mark.parametrize(
    "module_name",
    [
        "analysis.analysis_eval.speed_context",
        "analysis.analysis_eval.method_table",
        "analysis.analysis_eval.main_results_figure",
        "analysis.analysis_training.hparam_table",
        "analysis.analysis_training.posterior_contraction",
    ],
)
def test_repository_inputs_still_resolve_from_repository_root(module_name):
    module = importlib.import_module(module_name)
    root = getattr(module, "REPO_ROOT", getattr(module, "ROOT", None))
    assert root == REPO_ROOT
    if hasattr(module, "DEFAULT_SCRIPT"):
        assert module.DEFAULT_SCRIPT == REPO_ROOT / "scripts" / "pretrain_prod.sh"
        assert module.DEFAULT_SCRIPT.is_file()
    if hasattr(module, "DEFAULT_CKPT_DIR"):
        assert module.DEFAULT_CKPT_DIR == REPO_ROOT / "whole_1"
    if hasattr(module, "DEFAULT_CONFIG"):
        assert module.DEFAULT_CONFIG == REPO_ROOT / "benchmarks" / "config_methods.yaml"
        assert module.DEFAULT_CONFIG.is_file()
    if hasattr(module, "DEFAULT_BENCH_DIR"):
        assert module.DEFAULT_BENCH_DIR.is_relative_to(REPO_ROOT / "data")
    if hasattr(module, "DEFAULT_PRIOR_COVERAGE"):
        assert module.DEFAULT_PRIOR_COVERAGE.is_relative_to(
            ANALYSIS_ROOT / "analysis_eval" / "results"
        )


@pytest.mark.parametrize(
    ("module_name", "attribute", "relative_path"),
    [
        (
            "analysis.analysis_prior.generate",
            "DEFAULT_CACHE_DIR",
            "analysis_prior/cache",
        ),
        (
            "analysis.analysis_prior.main",
            "DEFAULT_OUTPUT_DIR",
            "analysis_prior/results",
        ),
        (
            "analysis.analysis_eval.exhaustive_tables",
            "DEFAULT_OUT",
            "analysis_eval/results/semisynt_exhaustive_npehe.tex",
        ),
        (
            "analysis.analysis_eval.method_table",
            "DEFAULT_METADATA",
            "analysis_eval/methods.json",
        ),
        (
            "analysis.analysis_training.loss_figure",
            "DEFAULT_OUTDIR",
            "analysis_training/results",
        ),
        (
            "analysis.analysis_training.wandb_history",
            "CACHE_DIR",
            "analysis_training/results/history",
        ),
        (
            "analysis.analysis_training.posterior_contraction",
            "DEFAULT_CACHE_DIR",
            "analysis_training/cache/posterior_contraction",
        ),
    ],
)
def test_outputs_and_caches_use_relocated_directories(
    module_name, attribute, relative_path
):
    module = importlib.import_module(module_name)
    assert getattr(module, attribute) == ANALYSIS_ROOT / relative_path


def test_external_consumers_use_the_same_prior_record_class():
    from analysis.analysis_prior.generate import PriorDatasetRecord
    from prior_eval._frozen_records import PriorDatasetRecord as FrozenRecord
    from prior_eval import strata
    from analysis.analysis_prior.analyses.confounding import delivered_shares

    assert FrozenRecord is PriorDatasetRecord
    assert strata.delivered_shares is delivered_shares


def test_sweep_defaults_keep_the_checkpoint_at_repository_root(monkeypatch):
    from analysis.analysis_synthetic_proxy_sweep_linear import run_ablation

    captured = {}

    def capture(args):
        captured.update(vars(args))
        return 0

    monkeypatch.setattr(run_ablation, "run", capture)
    monkeypatch.setattr(
        sys, "argv", ["run_ablation.py", "--n-query", "8", "--backdoor-methods"]
    )
    assert run_ablation.main() == 0
    assert (
        captured["proximalfm_checkpoint"] == REPO_ROOT / "whole_1" / "best_model.ckpt"
    )
    assert (
        captured["data_dir"]
        == ANALYSIS_ROOT / "analysis_synthetic_proxy_sweep_linear" / "data"
    )
    assert (
        captured["output_dir"]
        == ANALYSIS_ROOT / "analysis_synthetic_proxy_sweep_linear" / "results"
    )


def test_sweep_runs_from_relocated_folder_with_spawned_workers(tmp_path):
    """Exercise the direct-script import bootstrap, worker imports, and inference."""
    from analysis.analysis_synthetic_proxy_sweep_linear.generate_synthetic_sweep import (
        StudyConfig,
        generate_sweep,
    )
    from proximalfm._model import ProximalFM

    config = dict(
        embed_dim=16,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=4,
        icl_num_blocks=1,
        icl_nhead=2,
        nbins=32,
    )
    torch.manual_seed(0)
    model = ProximalFM(**config).eval()
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {"config": config, "state_dict": model.state_dict(), "curr_step": 1}, checkpoint
    )
    data_dir = tmp_path / "data"
    generate_sweep(data_dir, StudyConfig(n_rows=200, n_levels=2, dimensions=(1,)))
    output_dir = tmp_path / "results"
    command = [
        sys.executable,
        "run_ablation.py",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--proximalfm-checkpoint",
        str(checkpoint),
        "--context-sizes",
        "16",
        "--n-query",
        "8",
        "--workers",
        "2",
        "--backdoor-methods",
        "--fail-fast",
    ]
    result = subprocess.run(
        command,
        cwd=ANALYSIS_ROOT / "analysis_synthetic_proxy_sweep_linear",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    scores = pd.read_csv(output_dir / "method_results.csv")
    assert len(scores) == 8  # Two datasets, three linear baselines, and ProximalFM.
    assert set(scores["status"]) == {"ok"}
    assert np.isfinite(scores["pehe"]).all()
    assert "proximalfm" in set(scores["method"])
    # Repeat from the relocated folder to verify the persisted fit cache is usable.
    cached = subprocess.run(
        command,
        cwd=ANALYSIS_ROOT / "analysis_synthetic_proxy_sweep_linear",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert cached.returncode == 0, cached.stdout + cached.stderr
    reused = pd.read_csv(output_dir / "method_results.csv")
    assert reused["cached"].eq(1).all()
    np.testing.assert_array_equal(scores["pehe"], reused["pehe"])
