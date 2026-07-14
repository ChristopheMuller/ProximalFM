"""Equivalence gate for dropping the duplicate benchmark/ output.

Proves that, for the same generated pool, the benchmark dict rebuilt from the
val format (``save_validation`` → ``load_benchmark_from_val_dir``) is
numerically identical to the ``data.pt`` written by the legacy
``save_benchmark``. Once this holds, the ``benchmark/`` copy on disk is pure
redundancy and ``save_benchmark`` can be deleted.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "synthetic"))

import build_proximal_datasets as bpd  # noqa: E402
from realistic_proximal_mechanisms import (  # noqa: E402
    RealisticGeneratorConfig,
    RealisticProximalGenerator,
)

from benchmarks.bench_data import load_benchmark_from_val_dir  # noqa: E402


N_DATASETS = 6
SEQ_LEN = 64
PROP_TRAIN = 0.75
MECHANISMS = [1, 2, 3]


@pytest.fixture(scope="module")
def pool_and_cfg(tmp_path_factory):
    """One small realistic pool, generated once for all tests in this module."""
    cfg = RealisticGeneratorConfig(
        save_dir=tmp_path_factory.mktemp("gen_scratch"),
        K=N_DATASETS,
        seq_len=SEQ_LEN,
        train_size=PROP_TRAIN,
        max_dim_U=3,
        max_dim_X=3,
        max_dim_W=3,
        max_dim_Z=3,
        seed=1234,
        device="cpu",
    )
    gen = RealisticProximalGenerator(cfg)
    pool, _ = bpd.generate_pool(
        gen,
        n_datasets=N_DATASETS,
        seq_len=SEQ_LEN,
        train_size=PROP_TRAIN,
        mechanisms=MECHANISMS,
    )
    return pool, cfg


def assert_equivalent(old: dict, new: dict):
    """Old data.pt vs val-format rebuild: equal up to two inert shape details.

    1. The rebuild drops trailing singleton dims on A/Y/true_ITEs ([B,T,1] →
       [B,T]); both consumers (bench_run.build_batches, bench_plots) accept
       either shape.
    2. The rebuild pads U/X/W/Z to the max *attained* feature dim, while the
       old path kept the generator's full padded width; any trimmed columns
       must be all-zero padding.
    """
    assert set(old.keys()) == set(new.keys())
    for k in old:
        o, n = old[k], new[k]
        if isinstance(o, torch.Tensor):
            if o.dim() == n.dim() + 1 and o.shape[-1] == 1:
                o = o.squeeze(-1)
            if o.dim() == 3 and o.shape[-1] > n.shape[-1]:
                assert (o[..., n.shape[-1] :] == 0).all(), f"{k}: non-zero padding"
                o = o[..., : n.shape[-1]]
            torch.testing.assert_close(
                n.double(), o.double(), atol=1e-5, rtol=1e-5, msg=lambda m: f"{k}: {m}"
            )
        elif isinstance(o, np.ndarray):
            if o.ndim == n.ndim + 1 and o.shape[-1] == 1:
                o = np.squeeze(o, -1)
            np.testing.assert_allclose(n, o, atol=1e-5, rtol=1e-5, err_msg=k)
        else:
            assert o == n, f"{k}: {o} != {n}"


def test_val_roundtrip_equals_save_benchmark(pool_and_cfg, tmp_path):
    pool, cfg = pool_and_cfg

    bench_path = bpd.save_benchmark(pool, tmp_path / "benchmark")
    old = torch.load(bench_path, weights_only=False)

    n_files, _ = bpd.save_validation(
        pool, tmp_path / "val", cfg, datasets_per_file=None, cli_command="test"
    )
    assert n_files == 1
    new = load_benchmark_from_val_dir(tmp_path / "val")

    assert_equivalent(old, new)


def test_val_roundtrip_equals_save_benchmark_multifile(pool_and_cfg, tmp_path):
    """Splitting the val dir into several files must not change the rebuild."""
    pool, cfg = pool_and_cfg

    old = torch.load(
        bpd.save_benchmark(pool, tmp_path / "benchmark"), weights_only=False
    )

    n_files, dpf = bpd.save_validation(
        pool, tmp_path / "val", cfg, datasets_per_file=2, cli_command="test"
    )
    assert (n_files, dpf) == (N_DATASETS // 2, 2)
    assert_equivalent(old, load_benchmark_from_val_dir(tmp_path / "val"))
