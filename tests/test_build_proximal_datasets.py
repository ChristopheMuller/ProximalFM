"""Tests for eval_datasets/synt_eval/build_proximal_datasets.py.

The build script writes datasets to disk ONLY in the raw, sparse val format;
the benchmark rebuilds its standardized dict from those files at load time.
These tests pin the round-trip: ``save_validation`` →
``load_benchmark_from_val_dir`` must equal ``build_benchmark_dict`` applied
directly to the in-memory pool.
"""


import numpy as np


import pytest


import torch


import eval_datasets.synt_eval.build_proximal_datasets as bpd


from eval_datasets.synt_eval.realistic_proximal_mechanisms import (
    RealisticGeneratorConfig,
    RealisticProximalGenerator,
)


from benchmarks.data import (  # noqa: E402
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
    # Indices 0..14 are the original 15 fields; MU0/MU1/S2/K_eff follow and
    # carry the Monte-Carlo CATE target, which the benchmark now scores.
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
        MU0,
        MU1,
        S2,
        K_eff,
    ) = pool
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
        MU0=MU0,
        MU1=MU1,
        S2=S2,
        K_eff=K_eff,
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
        if o is None or n is None:
            assert o is None and n is None, k
        elif isinstance(o, torch.Tensor):
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


def test_val_roundtrip_carries_the_cate_target(tmp_path):
    """A replicate build must round-trip MU0/MU1/S2/K_eff, not just Y0/Y1.

    The module fixture uses the default ``mc_cate=1``, where the benchmark is
    ITE-only by design — so it cannot catch a CATE target lost in the writer.
    """
    cfg = RealisticGeneratorConfig(
        save_dir=tmp_path / "gen_scratch",
        K=N_DATASETS,
        seq_len=SEQ_LEN,
        train_size=PROP_TRAIN,
        max_dim_U=3,
        max_dim_X=3,
        max_dim_W=3,
        max_dim_Z=3,
        mc_cate=4,
        seed=1234,
        device="cpu",
    )
    pool, _, _ = bpd.generate_pool(
        RealisticProximalGenerator(cfg),
        n_datasets=N_DATASETS,
        seq_len=SEQ_LEN,
        train_size=PROP_TRAIN,
        mechanisms=MECHANISMS,
    )
    bpd.save_validation(
        pool, tmp_path / "val", cfg, datasets_per_file=None, cli_command="test"
    )

    direct = direct_build(pool)
    rebuilt = load_benchmark_from_val_dir(tmp_path / "val")
    assert direct["true_CATEs"] is not None
    assert_equivalent(direct, rebuilt)
    # The whole point: the CATE target is a different object from the ITE one.
    assert not np.allclose(rebuilt["true_CATEs"], rebuilt["true_ITEs"])


def test_n_max_n_q_resolve_to_that_exact_context_size():
    """The float round-trip must land on n_max exactly, not near it.

    ``--n_max``/``--n_q`` resolve to ``prop_train = n_max/seq_len``, which the
    generator turns back into an integer via ``round(prop_train * seq_len)``. If
    that ever came back off by one, the set would still build and still look
    self-consistent -- so pin it.
    """
    for n_max, n_q in ((1536, 512), (2048, 128), (1000, 1), (3, 4093), (1, 1)):
        args = bpd.parse_args(["--name", "x", "--n_max", str(n_max), "--n_q", str(n_q)])
        assert args.seq_len == n_max + n_q
        assert (
            RealisticProximalGenerator._resolve_train_size(
                None, args.prop_train, args.seq_len
            )
            == n_max
        )
