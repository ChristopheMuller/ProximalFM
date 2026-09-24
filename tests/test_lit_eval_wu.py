"""Contract tests for Wu et al., Appendix H.2 (high-dimensional DGP)."""


from __future__ import annotations


import torch


import json


import yaml


from lit_eval.base import EPISODE_FIELDS, EpisodeSpec


from lit_eval.build import build, parse_args


from lit_eval.datasets.wu_apph2 import quadratic_decay, truncated_logistic


from lit_eval.registry import build_dataset


from benchmarks.data import load_benchmark_from_val_dir


from prior_eval.verify import verify


def _spec(seed: int = 17, T: int = 2048, n_ctx: int = 1024, K: int = 1):
    return EpisodeSpec(index=0, seed=seed, seq_len=T, train_size=n_ctx, mc_cate=K)


def _draw(**kwargs):
    options = {k: kwargs.pop(k) for k in tuple(kwargs) if k.startswith("dim_")}
    tensors, info = build_dataset("wu_appH2", **options).generate(_spec(**kwargs))
    return dict(zip(EPISODE_FIELDS, tensors)), info


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
