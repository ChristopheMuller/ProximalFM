"""Relocated builders remain callable and their saved datasets interoperable."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from eval_datasets.prior_eval.build_prior_frozen_set import (
    BuildConfig,
    build,
    frozen_content_digest,
    load_frozen_meta,
)
from eval_datasets.semisynt_eval.build import DEFAULT_CONFIG_PATH, parse_args
from eval_datasets.semisynt_eval.source import DEFAULT_DATA_DIR
from eval_datasets.synt_eval.build_proximal_datasets import _REPO_ROOT
from eval_datasets.synt_eval.realistic_sidecar import RealisticBuildConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module",
    [
        "prior_eval.build",
        "prior_eval.build_prior_frozen_set",
        "prior_eval.report",
        "prior_eval.verify",
        "prior_eval.score",
        "prior_eval.strata",
        "prior_eval.score_strata",
        "prior_eval.subsample",
        "prior_eval.sweep",
        "semisynt_eval.build",
        "synt_eval.build_proximal_datasets",
    ],
)
def test_module_entry_points(module):
    result = subprocess.run(
        [sys.executable, "-m", f"eval_datasets.{module}", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_semisynthetic_defaults_find_relocated_inputs():
    recipe = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    args = parse_args(["--name", "layout_check"])
    assert DEFAULT_CONFIG_PATH == (
        REPO_ROOT / "eval_datasets" / "semisynt_eval" / "config.yaml"
    )
    assert REPO_ROOT / recipe["build"]["data_dir"] == DEFAULT_DATA_DIR
    assert REPO_ROOT / args.data_dir == DEFAULT_DATA_DIR
    assert args.out_root == Path("data/semisynt_eval")


def test_direct_synthetic_builder_works_outside_repository(tmp_path):
    assert _REPO_ROOT == REPO_ROOT
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "eval_datasets/synt_eval/build_proximal_datasets.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_realistic_digest_fallback_resolves_relocated_builder():
    cfg = RealisticBuildConfig(
        n_datasets=2,
        seq_len=24,
        prop_train=0.75,
        mc_cate=4,
        seed=1234,
        mechanisms=[1, 9],
    )
    assert frozen_content_digest({"global": cfg.to_dict()}) == cfg.content_digest()


def test_parallel_prior_workers_preserve_saved_batches(tmp_path):
    cfg = BuildConfig(
        n_datasets=2,
        n_max=16,
        n_q=4,
        mc_cate=4,
        seed=7,
        prior_mc_mem_budget=2_000_000_000,
    )
    serial = build(cfg, tmp_path / "serial", n_jobs=1, progress=False)
    parallel = build(cfg, tmp_path / "parallel", n_jobs=2, progress=False)
    serial_files = sorted(serial.glob("batch_*.pt"))
    parallel_files = sorted(parallel.glob("batch_*.pt"))
    assert len(serial_files) == len(parallel_files) > 0
    for a, b in zip(serial_files, parallel_files):
        serial_batch = torch.load(a, weights_only=False)
        parallel_batch = torch.load(b, weights_only=False)
        assert serial_batch.keys() == parallel_batch.keys()
        for key, value in serial_batch.items():
            if isinstance(value, torch.Tensor):
                # Separate BLAS processes already round slightly differently
                # before the relocation; compare tensors, not pickle bytes.
                torch.testing.assert_close(
                    value, parallel_batch[key], rtol=1e-5, atol=1e-5, equal_nan=True
                )
            else:
                assert value == parallel_batch[key]
    assert frozen_content_digest(load_frozen_meta(parallel.parent)) == (
        cfg.content_digest()
    )
