"""Small checks for the benchmark's data contract."""

import numpy as np
import torch

from benchmarks.data import build_benchmark_dict, load_benchmark_from_val_dir
from proximalfm.prior._genload import dense2sparse


def _padded(batch, rows, dims, width, generator):
    return torch.randn(batch, rows, width, generator=generator) * (
        torch.arange(width).unsqueeze(0) < dims.unsqueeze(1)
    ).unsqueeze(1)


def make_pool(B=4, T=12, train_size=8, max_dim=3, seed=0, K=4):
    g = torch.Generator().manual_seed(seed)
    dims = {
        key: torch.randint(1, max_dim + 1, (B,), generator=g)
        for key in ("d_U", "d_X", "d_W", "d_Z")
    }
    blocks = {key[2:]: _padded(B, T, dim, max_dim, g) for key, dim in dims.items()}
    y0, y1 = torch.randn(B, T, generator=g), torch.randn(B, T, generator=g)
    a = torch.randint(0, 2, (B, T), generator=g).float()
    return {
        **blocks,
        **dims,
        "A": a,
        "Y0": y0,
        "Y1": y1,
        "Y_obs": torch.where(a > 0, y1, y0),
        "MU0": torch.randn(B, T, generator=g),
        "MU1": torch.randn(B, T, generator=g),
        "S2": torch.full((B, T), 0.1),
        "K_eff": torch.full((B,), K),
        "seq_lens": torch.full((B,), T),
        "train_sizes": torch.full((B,), train_size),
    }


def build_from_pool(pool):
    return build_benchmark_dict(
        U=pool["U"],
        X=pool["X"],
        W=pool["W"],
        Z=pool["Z"],
        A=pool["A"],
        Y_obs=pool["Y_obs"],
        Y0=pool["Y0"],
        Y1=pool["Y1"],
        d_X=pool["d_X"],
        d_W=pool["d_W"],
        d_Z=pool["d_Z"],
        train_size=int(pool["train_sizes"][0]),
        MU0=pool["MU0"],
        MU1=pool["MU1"],
        S2=pool["S2"],
        K_eff=pool["K_eff"],
        CATE=pool.get("CATE"),
    )


def write_val_file(path, pool, sl=slice(None)):
    B, T, _ = pool["X"][sl].shape
    payload = {
        key: pool[key][sl]
        for key in (
            "A",
            "Y_obs",
            "Y0",
            "Y1",
            "MU0",
            "MU1",
            "S2",
            "K_eff",
            "d_U",
            "d_X",
            "d_W",
            "d_Z",
            "seq_lens",
            "train_sizes",
        )
    }
    for key in ("U", "X", "W", "Z"):
        dim, value = pool[f"d_{key}"][sl], pool[key][sl]
        payload[key] = dense2sparse(
            value.reshape(-1, value.shape[-1]), dim.repeat_interleave(T)
        )
    payload["batch_size"] = B
    if "CATE" in pool:
        payload["CATE"] = pool["CATE"][sl]
    torch.save(payload, path)


def test_mc_cate_target_uses_mu_not_realized_ite_and_context_scaling():
    pool = make_pool()
    data = build_from_pool(pool)
    n_ctx = data["N_TRAIN"]
    scale = pool["Y_obs"][:, :n_ctx].std(dim=1, unbiased=False, keepdim=True)
    np.testing.assert_allclose(
        data["true_CATEs"], ((pool["MU1"] - pool["MU0"])[:, n_ctx:] / scale).numpy()
    )
    np.testing.assert_allclose(
        data["true_ITEs"], ((pool["Y1"] - pool["Y0"])[:, n_ctx:] / scale).numpy()
    )
    np.testing.assert_allclose(
        data["cate_s2"], (pool["S2"][:, n_ctx:] / scale.square()).numpy()
    )


def test_single_replicates_do_not_claim_a_cate_target():
    data = build_from_pool(make_pool(K=1))
    assert data["true_CATEs"] is None
    assert data["cate_s2"] is None


def test_val_loader_preserves_the_mc_cate_contract(tmp_path):
    pool = make_pool()
    write_val_file(tmp_path / "batch_000000.pt", pool)
    loaded, direct = load_benchmark_from_val_dir(tmp_path), build_from_pool(pool)
    np.testing.assert_allclose(loaded["true_CATEs"], direct["true_CATEs"])
    np.testing.assert_allclose(loaded["true_ITEs"], direct["true_ITEs"])
