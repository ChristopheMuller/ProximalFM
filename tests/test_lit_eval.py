"""Contract and Appendix-B tests for the Sverdrup literature adapter."""


from __future__ import annotations


import torch


from lit_eval.base import EPISODE_FIELDS, EpisodeSpec


from lit_eval.build import build, parse_args


from lit_eval.datasets.sverdrup_appb import cate


from lit_eval.registry import build_dataset


from prior_eval.verify import verify


from benchmarks.data import load_benchmark_from_val_dir


from proximalfm.prior._genload import LoadPriorDataset


def make_spec(seed=11, T=96, n_ctx=64, K=8):
    return EpisodeSpec(index=0, seed=seed, seq_len=T, train_size=n_ctx, mc_cate=K)


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
