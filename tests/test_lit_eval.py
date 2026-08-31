"""Contract and Appendix-B tests for the Sverdrup literature adapter."""
from __future__ import annotations

import pytest
import torch

from lit_eval.base import EPISODE_FIELDS, EpisodeSpec
from lit_eval.build import build, parse_args
from lit_eval.datasets.sverdrup_appb import cate, expected_w_given_u_x
from lit_eval.registry import available_datasets, build_dataset
from lit_eval.sidecar import LitSetConflictError
from prior_eval.verify import verify
from benchmarks.data import load_benchmark_from_val_dir
from proximalfm.prior._genload import LoadPriorDataset


def make_spec(seed=11, T=96, n_ctx=64, K=8):
    return EpisodeSpec(index=0, seed=seed, seq_len=T, train_size=n_ctx, mc_cate=K)


def test_only_sverdrup_dataset_is_registered():
    assert available_datasets() == ["sverdrup_appB"]
    assert build_dataset("sverdrup_appB").name == "sverdrup_appB"


def test_appendix_b_equations_and_tau():
    tensors, info = build_dataset("sverdrup_appB").generate(make_spec(K=1))
    d = dict(zip(EPISODE_FIELDS, tensors))
    X, A, p = d["X"].float(), d["A"].float(), d["A_probs"].float()
    assert X.shape[-1] == 5
    assert torch.allclose(
        d["MU1"] - d["MU0"],
        torch.exp(X[..., :1]) - 3 * X[..., 1:2],
        atol=1e-5,
    )
    assert abs(float(X.mean())) < 0.12 and abs(float(X.var()) - 0.25) < 0.08
    expected = torch.sigmoid(-(0.125 * X[..., :2].sum(-1, keepdim=True)))
    assert torch.allclose(p, expected, atol=1e-6)
    assert set(torch.unique(A).tolist()) <= {0.0, 1.0}
    assert info["sampled_hp"]["mechanism_version"] == 1.0


def test_replicated_target_is_the_exact_appendix_b_cate():
    spec = make_spec(seed=7, T=80, n_ctx=64, K=3)
    tensors, info = build_dataset("sverdrup_appB").generate(spec)
    d = dict(zip(EPISODE_FIELDS, tensors))
    expected = cate(d["X"])[spec.train_size :]
    actual = (d["MU1"] - d["MU0"])[spec.train_size :, 0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        info["tau_star_exact"][spec.train_size :], expected, rtol=0, atol=0
    )
    assert float(d["S2"][spec.train_size :].max()) < 1e-10


def test_appendix_b_proxy_moment_laws():
    tensors, _ = build_dataset("sverdrup_appB").generate(
        make_spec(seed=19, T=4096, n_ctx=2048, K=1)
    )
    d = dict(zip(EPISODE_FIELDS, tensors))
    X, A = d["X"].float(), d["A"].float()
    A, x = A.reshape(-1), (0.25 * X[..., :2].sum(-1)).reshape(-1)
    for name, coef in (("U", 0.25), ("W", 0.125), ("Z", 0.25)):
        r = d[name].float().reshape(-1) - (0.25 + coef * A + x)
        assert abs(float(r.mean())) < 0.05
        assert abs(float(r.std()) - 1) < 0.08

    residuals = torch.stack(
        [
            d["Z"].reshape(-1) - (0.25 + 0.25 * A + x),
            d["W"].reshape(-1) - (0.25 + 0.125 * A + x),
            d["U"].reshape(-1) - (0.25 + 0.25 * A + x),
        ]
    )
    expected_cov = torch.tensor([[1.0, 0.25, 0.5], [0.25, 1.0, 0.5], [0.5, 0.5, 1.0]])
    torch.testing.assert_close(torch.cov(residuals), expected_cov, rtol=0.12, atol=0.06)

    outcome_mean = (
        2.0
        + cate(X) * A
        + x
        + 2.0 * expected_w_given_u_x(d["U"], X)
        + 2.0 * d["W"].reshape(-1)
    )
    outcome_residual = d["Y_obs"].reshape(-1) - outcome_mean
    assert abs(float(outcome_residual.mean())) < 0.02
    assert abs(float(outcome_residual.std()) - 0.25) < 0.02


def test_same_seed_is_deterministic():
    ds = build_dataset("sverdrup_appB")
    a, _ = ds.generate(make_spec(seed=3, K=4))
    b, _ = ds.generate(make_spec(seed=3, K=4))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0, equal_nan=True)


def test_build_row_split_val_round_trip_and_benchmark_loader(tmp_path):
    args = parse_args(
        [
            "--dataset",
            "sverdrup_appB",
            "--name",
            "x",
            "--out_root",
            str(tmp_path),
            "--n_datasets",
            "2",
            "--n_max",
            "32",
            "--n_q",
            "8",
            "--mc_cate",
            "3",
            "--seed",
            "4",
        ]
    )
    assert args.n_max == 32 and args.seq_len == 40
    build(args)
    val = tmp_path / "x" / "val"
    batch = next(iter(LoadPriorDataset(data_dir=str(val), batch_size=2, device="cpu")))
    assert torch.equal(batch[16], torch.tensor([32, 32]))
    assert batch[17].shape == (2, 40)
    data = load_benchmark_from_val_dir(val)
    assert data["true_CATEs"] is not None
    assert data["true_CATEs"].shape == (2, 8)
    assert verify(tmp_path / "x", cate_vmin=-100.0, cate_vmax=100.0)["ok"]


def test_force_rebuild_removes_stale_batches_and_sidecar(tmp_path):
    common = [
        "--dataset",
        "sverdrup_appB",
        "--name",
        "replace",
        "--out_root",
        str(tmp_path),
        "--n_max",
        "16",
        "--n_q",
        "4",
        "--datasets_per_file",
        "1",
    ]
    build(parse_args([*common, "--n_datasets", "3", "--mc_cate", "2"]))
    root = tmp_path / "replace"
    assert len(list((root / "val").glob("batch_*.pt"))) == 3
    assert (root / "frozen_meta.pt").exists()

    point_args = parse_args([*common, "--n_datasets", "1", "--mc_cate", "1"])
    with pytest.raises(LitSetConflictError):
        build(point_args)

    build(
        parse_args(
            [
                *common,
                "--n_datasets",
                "1",
                "--mc_cate",
                "1",
                "--force",
            ]
        )
    )
    assert len(list((root / "val").glob("batch_*.pt"))) == 1
    assert not (root / "frozen_meta.pt").exists()
    assert load_benchmark_from_val_dir(root / "val")["N_BATCH"] == 1
