"""Contract tests for Wu et al., Appendix H.2 (high-dimensional DGP)."""
from __future__ import annotations

import pytest
import torch
import json
import yaml

from lit_eval.base import EPISODE_FIELDS, EpisodeSpec
from lit_eval.build import build, parse_args
from lit_eval.datasets.wu_apph2 import quadratic_decay, truncated_logistic
from lit_eval.registry import available_datasets, build_dataset
from benchmarks.data import load_benchmark_from_val_dir
from prior_eval.verify import verify


def _spec(seed: int = 17, T: int = 2048, n_ctx: int = 1024, K: int = 1):
    return EpisodeSpec(index=0, seed=seed, seq_len=T, train_size=n_ctx, mc_cate=K)


def _draw(**kwargs):
    options = {k: kwargs.pop(k) for k in tuple(kwargs) if k.startswith("dim_")}
    tensors, info = build_dataset("wu_appH2", **options).generate(_spec(**kwargs))
    return dict(zip(EPISODE_FIELDS, tensors)), info


def test_wu_adapter_is_registered_with_default_dimensions():
    assert "wu_appH2" in available_datasets()
    ds = build_dataset("wu_appH2")
    assert (ds.max_dims.d_X, ds.max_dims.d_Z, ds.max_dims.d_W, ds.max_dims.d_U) == (
        100,
        10,
        10,
        2,
    )


def test_wu_dimensions_are_configurable_and_shape_all_outputs():
    ds = build_dataset("wu_appH2", dim_x=7, dim_z=3, dim_w=4)
    assert (ds.max_dims.d_X, ds.max_dims.d_Z, ds.max_dims.d_W) == (7, 3, 4)
    d, _ = _draw(seed=19, T=80, n_ctx=48, K=1, dim_x=7, dim_z=3, dim_w=4)
    assert d["X"].shape == (80, 7)
    assert d["Z"].shape == (80, 3)
    assert d["W"].shape == (80, 4)


def test_wu_small_dimension_coefficients_and_covariance():
    d, _ = _draw(seed=23, T=4096, n_ctx=2048, dim_x=4, dim_z=2, dim_w=3)
    cov = torch.cov(d["X"].float().T)
    torch.testing.assert_close(torch.diag(cov), torch.ones(4), atol=0.09, rtol=0)
    torch.testing.assert_close(
        torch.diag(cov, 1), torch.full((3,), 0.5), atol=0.10, rtol=0
    )
    bx, bz, bw = (quadratic_decay(n) for n in (4, 2, 3))
    uz, uw = d["U"][:, 0], d["U"][:, 1]
    expected_p = truncated_logistic(
        3 * (d["X"] @ bx) + 3 * (d["Z"] @ bz) + 0.25 * uw
    ).unsqueeze(1)
    torch.testing.assert_close(d["A_probs"], expected_p, rtol=0, atol=1e-6)
    expected_y0 = 1.2 * (d["X"] @ bx + d["W"] @ bw) + 0.25 * uz
    torch.testing.assert_close(d["Y0"][:, 0], expected_y0, rtol=0, atol=1e-6)


@pytest.mark.parametrize("option", ["dim_x", "dim_z", "dim_w"])
def test_wu_rejects_nonpositive_dimensions(option):
    with pytest.raises((TypeError, ValueError)):
        build_dataset("wu_appH2", **{option: 0})


@pytest.mark.parametrize("option", ["dim_z", "dim_w"])
def test_wu_requires_two_proxy_columns_for_two_latent_diagnostics(option):
    with pytest.raises(ValueError, match="must both be >= 2"):
        build_dataset("wu_appH2", **{option: 1})


def test_wu_dimensions_must_be_integers():
    with pytest.raises(TypeError, match="dim_x must be an integer"):
        build_dataset("wu_appH2", dim_x=3.5)


def test_covariates_have_appendix_h2_covariance():
    d, _ = _draw(seed=23, T=4096, n_ctx=2048)
    x = d["X"].float()
    cov = torch.cov(x.T)
    assert torch.allclose(torch.diag(cov), torch.ones(100), atol=0.09)
    assert torch.allclose(torch.diag(cov, 1), torch.full((99,), 0.5), atol=0.10)
    # Non-neighbouring coordinates are independent in the stated design.
    off = cov[torch.arange(98), torch.arange(2, 100)]
    assert float(off.abs().mean()) < 0.08


def test_proxy_uniform_noise_and_loading_directions():
    d, _ = _draw(seed=29, T=4096, n_ctx=2048)
    uz, uw = d["U"][:, :1], d["U"][:, 1:2]
    u_cov = torch.cov(torch.stack([uz[:, 0], uw[:, 0]]))
    torch.testing.assert_close(
        u_cov, torch.tensor([[2.0, 1.0], [1.0, 2.0]]), rtol=0.10, atol=0.10
    )

    # Removing the printed 0.25 latent loading must recover independent
    # Uniform(-1, 1) proxy noise.
    assert d["Z"].shape[-1] == d["W"].shape[-1] == 10
    for noise in (d["Z"] - 0.25 * uz, d["W"] - 0.25 * uw):
        assert float(noise.min()) >= -1.0
        assert float(noise.max()) <= 1.0
        assert torch.all(noise.mean(0).abs() < 0.04)
        assert torch.all((noise.var(0) - 1.0 / 3.0).abs() < 0.04)


def test_binary_adaptation_and_outcome_equation_are_explicit():
    d, _ = _draw(seed=37, T=512, n_ctx=256)
    bx, bz, bw = (quadratic_decay(dim) for dim in (100, 10, 10))
    uz, uw = d["U"][:, 0], d["U"][:, 1]
    expected_p = truncated_logistic(
        3.0 * (d["X"] @ bx) + 3.0 * (d["Z"] @ bz) + 0.25 * uw
    ).unsqueeze(1)
    torch.testing.assert_close(d["A_probs"], expected_p, rtol=0, atol=1e-6)
    assert float(d["A_probs"].min()) >= 0.1
    assert float(d["A_probs"].max()) <= 0.9
    assert abs(float((d["A"] - d["A_probs"]).mean())) < 0.06

    expected_y0 = 1.2 * (d["X"] @ bx + d["W"] @ bw) + 0.25 * uz
    torch.testing.assert_close(d["Y0"][:, 0], expected_y0, rtol=0, atol=1e-6)
    torch.testing.assert_close(d["Y_obs"], d["A"] * d["Y1"] + (1.0 - d["A"]) * d["Y0"])


def test_exact_cate_and_zero_replicate_noise():
    d, info = _draw(seed=31, T=96, n_ctx=64, K=3)
    expected = 2.2 + d["X"][64:, 0]
    actual = (d["MU1"] - d["MU0"])[64:, 0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert float(d["S2"][64:].max()) < 1e-10
    torch.testing.assert_close(info["tau_star_exact"][64:], expected, rtol=0, atol=0)


def test_same_seed_is_byte_deterministic():
    a, ia = _draw(seed=41, T=80, n_ctx=48, K=2)
    b, ib = _draw(seed=41, T=80, n_ctx=48, K=2)
    for x, y in zip(a.values(), b.values()):
        torch.testing.assert_close(x, y, rtol=0, atol=0, equal_nan=True)
    assert ia["sampled_hp"] == ib["sampled_hp"]
    torch.testing.assert_close(
        ia["tau_star_exact"], ib["tau_star_exact"], rtol=0, atol=0, equal_nan=True
    )


def test_unknown_options_are_rejected():
    with pytest.raises(TypeError, match="unknown option"):
        build_dataset("wu_appH2", does_not_exist=True)


def test_wu_set_round_trips_to_the_benchmark(tmp_path):
    args = parse_args(
        [
            "--dataset",
            "wu_appH2",
            "--name",
            "wu",
            "--out_root",
            str(tmp_path),
            "--n_datasets",
            "2",
            "--n_max",
            "32",
            "--n_q",
            "8",
            "--mc_cate",
            "2",
            "--seed",
            "71",
            "--option",
            "dim_x=7",
            "--option",
            "dim_z=3",
            "--option",
            "dim_w=4",
        ]
    )
    root = build(args)
    data = load_benchmark_from_val_dir(root / "val")
    assert data["X"].shape == (2, 40, 7)
    assert data["Z"].shape == (2, 40, 3)
    assert data["W"].shape == (2, 40, 4)
    assert data["true_CATEs"].shape == (2, 8)
    assert verify(root, cate_vmin=-100.0, cate_vmax=100.0)["ok"]
    manifest = json.loads((root / "manifest.json").read_text())
    config = yaml.safe_load((root / "config.yaml").read_text())
    assert manifest["options"] == config["options"]
    assert manifest["options"] == {"dim_x": 7, "dim_z": 3, "dim_w": 4}
