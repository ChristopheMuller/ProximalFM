"""Tests for the deliberately minimal ``inspect_eval`` SCM."""

import json

import torch

from benchmarks.data import load_benchmark_from_val_dir
from inspect_eval.build_proximal_datasets import build, parse_args
from inspect_eval.linear_gaussian_mechanism import (
    GeneratorConfig,
    LinearGaussianGenerator,
)
from prior_eval.build_prior_frozen_set import frozen_content_digest


def _generator(tmp_path, *, mc_cate=1, seed=31):
    cfg = GeneratorConfig(
        save_dir=tmp_path / "scratch",
        K=3,
        seq_len=64,
        train_size=48,
        min_dim_U=1,
        max_dim_U=1,
        min_dim_X=1,
        max_dim_X=1,
        min_dim_W=1,
        max_dim_W=1,
        min_dim_Z=1,
        max_dim_Z=1,
        extra_observation_noise=0.0,
        mc_cate=mc_cate,
        seed=seed,
    )
    return LinearGaussianGenerator(cfg)


def test_scalar_linear_structure_and_random_coefficients(tmp_path):
    gen = _generator(tmp_path)
    batch = gen.get_batch(mechanism_id=1, batch_size=3)
    U, X, _W, _Z, _A, p, _Y, Y0, Y1 = batch[:9]
    d_U, d_X, d_W, d_Z = batch[9:13]

    assert torch.equal(d_U, torch.ones_like(d_U))
    assert torch.equal(d_X, torch.ones_like(d_X))
    assert torch.equal(d_W, torch.ones_like(d_W))
    assert torch.equal(d_Z, torch.ones_like(d_Z))
    assert len(gen.sampled_coefficients) == 3
    assert len({c["b_au"] for c in gen.sampled_coefficients}) == 3

    for i, c in enumerate(gen.sampled_coefficients):
        logits = c["b_a0"] + c["b_ax"] * X[i] + c["b_au"] * U[i]
        eps = gen.cfg.overlap_eps
        expected_p = eps + (1 - 2 * eps) * torch.sigmoid(
            logits.clamp(-gen.cfg.logit_clip, gen.cfg.logit_clip)
        )
        torch.testing.assert_close(p[i], expected_p)

        # The arm difference is exactly tau(X): neither W nor any noise enters.
        expected_tau = c["tau_0"] + c["tau_x"] * X[i]
        torch.testing.assert_close(Y1[i] - Y0[i], expected_tau, atol=2e-6, rtol=2e-6)


def test_replicate_target_is_analytic_cate(tmp_path):
    gen = _generator(tmp_path, mc_cate=8)
    batch = gen.get_batch(mechanism_id=1, batch_size=1)
    X, MU0, MU1, S2, K_eff = batch[1], batch[15], batch[16], batch[17], batch[18]
    c = gen.sampled_coefficients[0]
    n_ctx = 48

    assert torch.isnan(MU0[0, :n_ctx]).all()
    expected = c["tau_0"] + c["tau_x"] * X[0, n_ctx:]
    torch.testing.assert_close(
        MU1[0, n_ctx:] - MU0[0, n_ctx:], expected, atol=2e-6, rtol=2e-6
    )
    assert S2[0, n_ctx:].max() < 1e-12
    assert K_eff.item() == 8


def test_build_round_trips_and_writes_inspect_sidecar(tmp_path):
    args = parse_args(
        [
            "--name",
            "tiny",
            "--out_root",
            str(tmp_path),
            "--n_datasets",
            "3",
            "--n_max",
            "24",
            "--n_q",
            "8",
            "--mc_cate",
            "4",
            "--seed",
            "9",
        ]
    )
    build(args)
    root = tmp_path / "tiny"

    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["dimensions"] == {"U": 1, "X": 1, "W": 1, "Z": 1}
    data = load_benchmark_from_val_dir(root / "val")
    assert data["N_BATCH"] == 3
    assert data["true_CATEs"] is not None

    meta = torch.load(root / "frozen_meta.pt", weights_only=False)
    assert meta["global"]["set_kind"] == "inspect"
    assert frozen_content_digest(meta) == meta["global"]["content_digest"]


def test_sidecar_identity_includes_generative_options():
    from inspect_eval.sidecar import InspectBuildConfig

    base = dict(n_datasets=2, seq_len=8, prop_train=0.5, mc_cate=4, seed=1)
    digest = InspectBuildConfig(**base).content_digest()
    assert InspectBuildConfig(**base, overlap_eps=0.1).content_digest() != digest
    assert (
        InspectBuildConfig(**base, extra_observation_noise=0.2).content_digest()
        != digest
    )
