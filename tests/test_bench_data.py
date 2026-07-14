import numpy as np
import pytest
import torch

from benchmarks.bench_data import (
    build_benchmark_dict,
    load_benchmark_from_val_dir,
    sparse2dense,
)
from proximalfm.prior._genload import dense2sparse
from proximalfm.prior._genload import sparse2dense as pfm_sparse2dense


# ─────────────────────────────────────────────────────────────────────────────
# Fixture helpers: build a small raw pool + write it in the val format
# ─────────────────────────────────────────────────────────────────────────────


def _rand_padded(B, T, dims, max_dim, g):
    """Dense [B, T, max_dim] with zeros beyond each dataset's active dims."""
    x = torch.randn(B, T, max_dim, generator=g)
    mask = torch.arange(max_dim).unsqueeze(0) < dims.unsqueeze(1)  # [B, max_dim]
    return x * mask.unsqueeze(1)


def make_pool(B=6, T=32, train_size=24, max_dim=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    dims = {
        k: torch.randint(1, max_dim + 1, (B,), generator=g)
        for k in ("d_U", "d_X", "d_W", "d_Z")
    }
    # Ensure the max width is attained so dense pool and sparse round-trip
    # reconstruct to the same feature width.
    for k in dims:
        dims[k][0] = max_dim

    pool = {
        "U": _rand_padded(B, T, dims["d_U"], max_dim, g),
        "X": _rand_padded(B, T, dims["d_X"], max_dim, g),
        "W": _rand_padded(B, T, dims["d_W"], max_dim, g),
        "Z": _rand_padded(B, T, dims["d_Z"], max_dim, g),
        "A": torch.randint(0, 2, (B, T), generator=g).float(),
        "A_probs": torch.rand(B, T, generator=g),
        "Y0": torch.randn(B, T, generator=g),
        "Y1": torch.randn(B, T, generator=g) + 1.0,
        **dims,
        "seq_lens": torch.full((B,), T, dtype=torch.long),
        "train_sizes": torch.full((B,), train_size, dtype=torch.long),
    }
    pool["Y_obs"] = torch.where(pool["A"] > 0.5, pool["Y1"], pool["Y0"])
    return pool


def write_val_file(path, pool, sl=slice(None)):
    """Sparse-encode pool[sl] into one batch_*.pt, mirroring the build script."""
    d_U, d_X = pool["d_U"][sl], pool["d_X"][sl]
    d_W, d_Z = pool["d_W"][sl], pool["d_Z"][sl]
    U, X, W, Z = pool["U"][sl], pool["X"][sl], pool["W"][sl], pool["Z"][sl]
    B, T, _ = X.shape
    payload = {
        "U": dense2sparse(U.reshape(-1, U.shape[-1]), d_U.repeat_interleave(T)),
        "X": dense2sparse(X.reshape(-1, X.shape[-1]), d_X.repeat_interleave(T)),
        "W": dense2sparse(W.reshape(-1, W.shape[-1]), d_W.repeat_interleave(T)),
        "Z": dense2sparse(Z.reshape(-1, Z.shape[-1]), d_Z.repeat_interleave(T)),
        "A": pool["A"][sl],
        "A_probs": pool["A_probs"][sl],
        "Y_obs": pool["Y_obs"][sl],
        "Y0": pool["Y0"][sl],
        "Y1": pool["Y1"][sl],
        "d_U": d_U,
        "d_X": d_X,
        "d_W": d_W,
        "d_Z": d_Z,
        "seq_lens": pool["seq_lens"][sl],
        "train_sizes": pool["train_sizes"][sl],
        "batch_size": B,
    }
    torch.save(payload, path)


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
    )


def assert_dicts_equal(a, b):
    assert set(a.keys()) == set(b.keys())
    for k in a:
        if isinstance(a[k], torch.Tensor):
            torch.testing.assert_close(a[k], b[k])
        elif isinstance(a[k], np.ndarray):
            np.testing.assert_allclose(a[k], b[k], rtol=1e-5, atol=1e-6)
        else:
            assert a[k] == b[k], k


# ─────────────────────────────────────────────────────────────────────────────
# Vendored sparse2dense stays in sync with proximalfm
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("max_len", [None, 7])
def test_vendored_sparse2dense_matches_proximalfm(max_len):
    g = torch.Generator().manual_seed(1)
    row_lengths = torch.randint(0, 6, (20,), generator=g)
    sparse = torch.randn(int(row_lengths.sum()), generator=g)
    torch.testing.assert_close(
        sparse2dense(sparse, row_lengths, max_len=max_len),
        pfm_sparse2dense(sparse, row_lengths, max_len=max_len),
    )


def test_sparse_roundtrip():
    g = torch.Generator().manual_seed(2)
    max_dim = 5
    dims = torch.randint(1, max_dim + 1, (10,), generator=g)
    dense = _rand_padded(10, 1, dims, max_dim, g).squeeze(1)
    sparse = dense2sparse(dense, dims)
    torch.testing.assert_close(sparse2dense(sparse, dims, max_len=max_dim), dense)


# ─────────────────────────────────────────────────────────────────────────────
# build_benchmark_dict
# ─────────────────────────────────────────────────────────────────────────────


def test_build_benchmark_dict_schema_and_stats():
    B, T, train_size = 6, 32, 24
    pool = make_pool(B=B, T=T, train_size=train_size)
    data = build_from_pool(pool)

    assert data["N_TRAIN"] == train_size
    assert data["N_TEST"] == T - train_size
    assert data["N_BATCH"] == B
    assert data["X"].shape == pool["X"].shape
    assert data["A"].shape == (B, T)
    assert data["Y"].shape == (B, T)
    assert data["true_ITEs"].shape == (B, T - train_size)
    torch.testing.assert_close(data["A"], pool["A"])  # not standardized
    torch.testing.assert_close(data["d_X"], pool["d_X"])

    # Context-only standardization: mean ~0 / std ~1 on the first train_size
    # rows, for the active (non-padding) feature columns.
    y_ctx = data["Y"][:, :train_size]
    torch.testing.assert_close(y_ctx.mean(dim=1), torch.zeros(B), atol=1e-5, rtol=0)
    torch.testing.assert_close(
        y_ctx.std(dim=1, unbiased=False), torch.ones(B), atol=1e-4, rtol=0
    )
    for b in range(B):
        x_ctx = data["X"][b, :train_size, : int(pool["d_X"][b])]
        torch.testing.assert_close(
            x_ctx.mean(dim=0), torch.zeros(x_ctx.shape[1]), atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            x_ctx.std(dim=0), torch.ones(x_ctx.shape[1]), atol=1e-4, rtol=0
        )
    # Padding columns stay exactly zero.
    assert (data["X"][0, :, int(pool["d_X"][0]) :] == 0).all()

    # true_ITEs = (Y1 - Y0) on test rows, scaled by the context Y std.
    y_std = pool["Y_obs"][:, :train_size].std(dim=1, unbiased=False, keepdim=True)
    expected = (pool["Y1"][:, train_size:] - pool["Y0"][:, train_size:]) / y_std
    np.testing.assert_allclose(
        data["true_ITEs"], expected.numpy(), rtol=1e-5, atol=1e-6
    )


def test_build_benchmark_dict_squeezes_trailing_singleton():
    pool = make_pool()
    pool3d = dict(pool)
    for k in ("A", "Y_obs", "Y0", "Y1"):
        pool3d[k] = pool[k].unsqueeze(-1)
    assert_dicts_equal(build_from_pool(pool3d), build_from_pool(pool))


# ─────────────────────────────────────────────────────────────────────────────
# load_benchmark_from_val_dir
# ─────────────────────────────────────────────────────────────────────────────


def test_load_single_file_matches_direct_build(tmp_path):
    pool = make_pool()
    write_val_file(tmp_path / "batch_000000.pt", pool)
    assert_dicts_equal(load_benchmark_from_val_dir(tmp_path), build_from_pool(pool))


def test_load_multi_file_pads_to_global_max(tmp_path):
    # File 0 holds datasets whose max active width is smaller than file 1's,
    # so the loader must pad file 0 up to the directory-wide max.
    pool = make_pool(B=6, max_dim=4)
    order = torch.argsort(pool["d_X"])  # narrow datasets first
    pool = {
        k: (v[order] if isinstance(v, torch.Tensor) and v.shape[0] == 6 else v)
        for k, v in pool.items()
    }
    assert int(pool["d_X"][:3].max()) < int(pool["d_X"][3:].max())

    write_val_file(tmp_path / "batch_000000.pt", pool, sl=slice(0, 3))
    write_val_file(tmp_path / "batch_000001.pt", pool, sl=slice(3, 6))
    assert_dicts_equal(load_benchmark_from_val_dir(tmp_path), build_from_pool(pool))


def test_load_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_benchmark_from_val_dir(tmp_path)


def test_load_varying_train_sizes_raises(tmp_path):
    pool = make_pool()
    pool["train_sizes"][-1] += 1
    write_val_file(tmp_path / "batch_000000.pt", pool)
    with pytest.raises(ValueError, match="train_size"):
        load_benchmark_from_val_dir(tmp_path)
