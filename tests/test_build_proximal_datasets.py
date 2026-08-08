"""Tests for synt_eval/build_proximal_datasets.py.

The build script writes datasets to disk ONLY in the raw, sparse val format;
the benchmark rebuilds its standardized dict from those files at load time.
These tests pin the round-trip: ``save_validation`` →
``load_benchmark_from_val_dir`` must equal ``build_benchmark_dict`` applied
directly to the in-memory pool.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "synt_eval"))

import build_proximal_datasets as bpd  # noqa: E402
from realistic_proximal_mechanisms import (  # noqa: E402
    RealisticGeneratorConfig,
    RealisticProximalGenerator,
)

from benchmarks.bench_data import (  # noqa: E402
    build_benchmark_dict,
    load_benchmark_from_val_dir,
)

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
    pool, _mech_ids, _seeds = bpd.generate_pool(
        gen,
        n_datasets=N_DATASETS,
        seq_len=SEQ_LEN,
        train_size=PROP_TRAIN,
        mechanisms=MECHANISMS,
    )
    return pool, cfg


def direct_build(pool):
    """Standardized benchmark dict straight from the in-memory pool."""
    # Indices 0..14 are the original 15 fields; MU0/MU1/S2/K_eff follow and are
    # not part of the benchmark view.
    (
        U,
        X,
        W,
        Z,
        A,
        A_probs,
        Y_obs,
        Y0,
        Y1,
        d_U,
        d_X,
        d_W,
        d_Z,
        seq_lens,
        train_sizes,
    ) = pool[:15]
    return build_benchmark_dict(
        U=U,
        X=X,
        W=W,
        Z=Z,
        A=A,
        Y_obs=Y_obs,
        Y0=Y0,
        Y1=Y1,
        d_X=d_X,
        d_W=d_W,
        d_Z=d_Z,
        train_size=int(train_sizes[0].item()),
    )


def assert_equivalent(direct: dict, rebuilt: dict):
    """Direct build vs val-format rebuild: equal up to one inert shape detail.

    The rebuild pads U/X/W/Z to the max *attained* feature dim, while the
    direct build keeps the generator's full padded width; any trimmed columns
    must be all-zero padding.
    """
    assert set(direct.keys()) == set(rebuilt.keys())
    for k in direct:
        o, n = direct[k], rebuilt[k]
        if isinstance(o, torch.Tensor):
            if o.dim() == 3 and o.shape[-1] > n.shape[-1]:
                assert (o[..., n.shape[-1] :] == 0).all(), f"{k}: non-zero padding"
                o = o[..., : n.shape[-1]]
            torch.testing.assert_close(
                n.double(), o.double(), atol=1e-5, rtol=1e-5, msg=lambda m: f"{k}: {m}"
            )
        elif isinstance(o, np.ndarray):
            np.testing.assert_allclose(n, o, atol=1e-5, rtol=1e-5, err_msg=k)
        else:
            assert o == n, f"{k}: {o} != {n}"


def test_val_roundtrip_equals_direct_build(pool_and_cfg, tmp_path):
    pool, cfg = pool_and_cfg

    n_files, _ = bpd.save_validation(
        pool, tmp_path / "val", cfg, datasets_per_file=None, cli_command="test"
    )
    assert n_files == 1
    assert_equivalent(direct_build(pool), load_benchmark_from_val_dir(tmp_path / "val"))


def test_val_roundtrip_equals_direct_build_multifile(pool_and_cfg, tmp_path):
    """Splitting the val dir into several files must not change the rebuild."""
    pool, cfg = pool_and_cfg

    n_files, dpf = bpd.save_validation(
        pool, tmp_path / "val", cfg, datasets_per_file=2, cli_command="test"
    )
    assert (n_files, dpf) == (N_DATASETS // 2, 2)
    assert_equivalent(direct_build(pool), load_benchmark_from_val_dir(tmp_path / "val"))


def test_build_writes_only_val_format(tmp_path):
    """End-to-end build: val/ + manifest + config, and no benchmark/ copy."""
    args = bpd.parse_args(
        [
            "--name",
            "tiny",
            "--out_root",
            str(tmp_path),
            "--n_datasets",
            "4",
            "--seq_len",
            str(SEQ_LEN),
            "--prop_train",
            str(PROP_TRAIN),
            "--seed",
            "7",
            "--mechanisms",
            "1",
            "2",
            "--max_dim_U",
            "2",
            "--max_dim_X",
            "2",
            "--max_dim_W",
            "2",
            "--max_dim_Z",
            "2",
        ]
    )
    bpd.build(args)

    root = tmp_path / "tiny"
    assert sorted(p.name for p in root.iterdir()) == [
        "config.yaml",
        "manifest.json",
        "val",
    ]

    manifest = json.loads((root / "manifest.json").read_text())
    assert "benchmark_dir" not in manifest
    assert manifest["val_dir"] == str((root / "val").resolve())
    assert manifest["n_datasets"] == 4

    # The single val/ output feeds the benchmark loader directly.
    data = load_benchmark_from_val_dir(root / "val")
    assert data["N_BATCH"] == 4
    assert data["N_TRAIN"] == int(SEQ_LEN * PROP_TRAIN)
    assert data["N_TEST"] == SEQ_LEN - data["N_TRAIN"]


# -----------------------------------------------------------------------------
# --n_max/--n_q: a front-end spelling of the row split, not a second one
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_max,n_q", [(1536, 512), (2048, 128), (1000, 1), (3, 4093), (1, 1)]
)
def test_n_max_n_q_resolve_to_that_exact_context_size(n_max, n_q):
    """The float round-trip must land on n_max exactly, not near it.

    ``--n_max``/``--n_q`` resolve to ``prop_train = n_max/seq_len``, which the
    generator turns back into an integer via ``round(prop_train * seq_len)``. If
    that ever came back off by one, the set would still build and still look
    self-consistent -- so pin it.
    """
    args = bpd.parse_args(["--name", "x", "--n_max", str(n_max), "--n_q", str(n_q)])
    assert args.seq_len == n_max + n_q
    assert (
        RealisticProximalGenerator._resolve_train_size(
            None, args.prop_train, args.seq_len
        )
        == n_max
    )


def test_seq_len_prop_train_still_report_the_split_both_ways():
    """The legacy spelling gains n_max/n_q, so the record is spelling-agnostic."""
    args = bpd.parse_args(["--name", "x", "--seq_len", "2048", "--prop_train", "0.75"])
    assert (args.n_max, args.n_q) == (1536, 512)


@pytest.mark.parametrize(
    "bad",
    [
        ["--n_max", "10"],  # n_q missing
        ["--n_q", "10"],  # n_max missing
        ["--n_max", "10", "--n_q", "5", "--seq_len", "20"],  # mixed spellings
        ["--n_max", "10", "--n_q", "5", "--prop_train", "0.5"],  # mixed spellings
        ["--n_max", "0", "--n_q", "5"],  # empty context
        ["--n_max", "5", "--n_q", "0"],  # no query rows to replicate
    ],
)
def test_row_split_flag_misuse_is_refused(bad):
    with pytest.raises(SystemExit):
        bpd.parse_args(["--name", "x", *bad])
