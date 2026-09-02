"""Tests for the scalar linear-Gaussian adjustment inspection set."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from inspect_eval_linear.analyze_noise_sweep import analyze, summarize
from inspect_eval_linear.build import POOL_FIELDS, build, parse_args
from inspect_eval_linear.export_ladder_predictions import (
    LADDER_OUTPUTS,
    export_ladder_predictions,
)
from inspect_eval_linear.generator import (
    LinearGeneratorConfig,
    LinearProximalGenerator,
    proxy_reliability,
)
from inspect_eval_linear.sidecar import (
    LinearBuildConfig,
    LinearSetConflictError,
    load_linear_meta,
)
from prior_eval.build_prior_frozen_set import frozen_content_digest
from prior_eval.verify import verify


def gen(*, k=64):
    return LinearProximalGenerator(LinearGeneratorConfig(mc_cate=k))


def test_generator_shape_nan_cate_and_zero_s2():
    g = gen(k=32)
    batch = g.generate_dataset(seq_len=12, train_size=8, proxy_noise_std=0.5, seed=7)
    p = dict(zip(POOL_FIELDS, batch))
    assert len(batch) == len(POOL_FIELDS) == 24
    for name in ("Q_naive_X", "Q_adj_W", "Q_adj_Z", "Q_adj_WZ", "Q_true"):
        assert p[name].shape == (12, 1)
        assert torch.isnan(p[name][:8]).all() and torch.isfinite(p[name][8:]).all()
    expected = g.cfg.tau_0 + g.cfg.tau_x * p["X"][8:]
    torch.testing.assert_close(p["Q_true"][8:], expected)
    torch.testing.assert_close(
        p["Q_true"][8:], (p["MU1"] - p["MU0"])[8:], atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(p["S2"][8:], torch.zeros_like(p["S2"][8:]))


@pytest.mark.parametrize("n_proxies", [0, 1, 2])
def test_observed_propensity_matches_gaussian_posterior_formula(n_proxies):
    g = gen()
    x = torch.tensor([[0.2], [-1.0], [1.5]], dtype=torch.float64)
    w = torch.tensor([[0.3], [-0.4], [2.0]], dtype=torch.float64)
    z = torch.tensor([[-0.2], [0.1], [1.0]], dtype=torch.float64)
    noise = 0.7
    measurements = torch.cat([w, z], 1)[:, :n_proxies]
    got = g.observed_propensity(x, measurements, noise)
    c = g.cfg
    v = 1 / (1 / c.sigma_u**2 + n_proxies * c.proxy_loading**2 / noise**2)
    m = v * (
        c.beta_ux * x / c.sigma_u**2
        + c.proxy_loading * measurements.sum(1, keepdim=True) / noise**2
    )
    standardized = (c.alpha_0 + c.alpha_x * x + c.alpha_u * m) / (
        c.sigma_a**2 + c.alpha_u**2 * v
    ) ** 0.5
    expected = torch.distributions.Normal(0.0, 1.0).cdf(standardized)
    torch.testing.assert_close(got, expected, atol=1e-10, rtol=1e-10)


def test_zero_noise_adjustment_equals_truth():
    p = dict(
        zip(
            POOL_FIELDS,
            gen(k=64).generate_dataset(
                seq_len=10, train_size=6, proxy_noise_std=0, seed=11
            ),
        )
    )
    truth = p["Q_true"][6:]
    for name in ("Q_adj_W", "Q_adj_Z", "Q_adj_WZ"):
        torch.testing.assert_close(p[name][6:], truth, atol=1e-6, rtol=1e-6)


def test_reliability_and_posterior_variance_monotonic():
    g = gen()
    rel = [proxy_reliability(n) for n in (0.25, 1, 4)]
    assert rel[0] > rel[1] > rel[2]
    assert (
        g.posterior_u_variance(1, 2)
        < g.posterior_u_variance(1, 1)
        < g.posterior_u_variance(1, 0)
    )


def test_proxy_information_moves_wz_adjustment_between_truth_and_naive():
    g = gen(k=4096)
    low_noise = dict(
        zip(
            POOL_FIELDS,
            g.generate_dataset(
                seq_len=16, train_size=8, proxy_noise_std=0.25, seed=314
            ),
        )
    )
    high_noise = dict(
        zip(
            POOL_FIELDS,
            g.generate_dataset(seq_len=16, train_size=8, proxy_noise_std=4.0, seed=314),
        )
    )

    def rmse(left, right):
        return (left[8:].sub(right[8:]).square().mean()).sqrt()

    assert rmse(low_noise["Q_adj_WZ"], low_noise["Q_true"]) < rmse(
        high_noise["Q_adj_WZ"], high_noise["Q_true"]
    )
    assert rmse(high_noise["Q_adj_WZ"], high_noise["Q_naive_X"]) < rmse(
        low_noise["Q_adj_WZ"], low_noise["Q_naive_X"]
    )


def test_build_sidecar_val_export_analysis_and_conflict(tmp_path):
    args = parse_args(
        [
            "--name",
            "tiny",
            "--out_root",
            str(tmp_path),
            "--n_datasets",
            "6",
            "--n_max",
            "6",
            "--n_q",
            "3",
            "--mc_cate",
            "16",
            "--proxy_noise_stds",
            "0",
            "1",
            "4",
            "--datasets_per_file",
            "6",
        ]
    )
    build(args)
    root = tmp_path / "tiny"
    meta = load_linear_meta(root)
    assert meta["global"]["set_kind"] == "inspect_linear"
    assert frozen_content_digest(meta) == meta["global"]["content_digest"]
    assert len(meta["episodes"]) == 6
    assert verify(root)["ok"]
    raw = torch.load(root / "val" / "batch_000000.pt", weights_only=True)
    assert "Q_adj_WZ" not in raw
    paths = export_ladder_predictions(root)
    for method, path in paths.items():
        with np.load(path) as cached:
            assert cached["preds"].shape == (6, 3)
            np.testing.assert_allclose(
                cached["preds"],
                torch.stack(
                    [r[LADDER_OUTPUTS[method]] for r in meta["episodes"]]
                ).numpy(),
            )
    csv_path, plot_path = analyze(root)
    assert csv_path.exists() and plot_path.exists()
    assert len(csv_path.read_text().splitlines()) == 1 + 5 * 3

    # Fitted benchmark predictions are stored in standardized outcome units;
    # the analysis discovers and converts them back to the raw ladder's units.
    y_std = np.array([float(record["y_context_std"]) for record in meta["episodes"]])
    truth_raw = np.stack([record["Q_true"].numpy() for record in meta["episodes"]])
    np.savez(
        root / "bench" / "preds" / "Exact-fitted_preds.npz",
        preds=truth_raw / y_std[:, None],
        times=np.zeros(6),
    )
    fitted_rows = [row for row in summarize(root) if row["method"] == "Exact-fitted"]
    assert len(fitted_rows) == 3
    assert all(row["source"] == "benchmark" for row in fitted_rows)
    assert all(row["rmse"] == pytest.approx(0, abs=1e-6) for row in fitted_rows)
    changed = parse_args(
        [
            "--name",
            "tiny",
            "--out_root",
            str(tmp_path),
            "--n_datasets",
            "6",
            "--n_max",
            "6",
            "--n_q",
            "3",
            "--mc_cate",
            "16",
            "--proxy_noise_stds",
            "0",
            "1",
            "2",
        ]
    )
    with pytest.raises(LinearSetConflictError):
        build(changed)


def test_digest_changes_with_noise_grid_or_coefficient():
    base = dict(
        n_datasets=6,
        seq_len=9,
        prop_train=2 / 3,
        mc_cate=8,
        seed=1,
        proxy_noise_stds=[0, 1],
    )
    digest = LinearBuildConfig(**base).content_digest()
    assert (
        LinearBuildConfig(**{**base, "proxy_noise_stds": [0, 2]}).content_digest()
        != digest
    )
    assert LinearBuildConfig(**{**base, "tau_x": 0.9}).content_digest() != digest
