"""The frozen posterior-eval set must be reproducible and readable as a val set.

M1 of ``POSTERIOR_EVAL_NOTES.md``. Three invariants are load-bearing and all
three fail silently if broken: the build is a pure function of its seed, the
realized ``K`` equals the requested one (``prior_mc_mem_budget`` caps it behind
a once-per-process warning), and the sidecar's frozen comparison space matches
the trainer's standardization exactly.
"""


import pytest


import torch


from eval_datasets.prior_eval.build_prior_frozen_set import (
    BuildConfig,
    MonteCarloBudgetError,
    build,
    load_frozen_meta,
)


from proximalfm.prior._genload import LoadPriorDataset


def _cfg(**overrides):
    """A tiny but structurally faithful build config."""
    base = dict(
        n_datasets=4,
        n_max=32,
        n_q=8,
        mc_cate=16,
        seed=20260805,
        prior_mc_mem_budget=2_000_000_000,
    )
    base.update(overrides)
    return BuildConfig(**base)


def _load_episodes(val_dir):
    n_files = len([p for p in val_dir.iterdir() if p.name.startswith("batch_")])
    loader = LoadPriorDataset(data_dir=str(val_dir), batch_size=1, max_batches=n_files)
    return list(loader)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    out = tmp_path_factory.mktemp("frozen")
    cfg = _cfg()
    val_dir = build(cfg, out, datasets_per_file=2, progress=False)
    return cfg, out, val_dir


def test_rebuild_with_the_same_seed_is_bit_identical(tmp_path):
    cfg = _cfg(n_datasets=2)
    a = build(cfg, tmp_path / "a", datasets_per_file=2, progress=False)
    b = build(cfg, tmp_path / "b", datasets_per_file=2, progress=False)

    files_a = sorted(p for p in a.iterdir() if p.name.startswith("batch_"))
    files_b = sorted(p for p in b.iterdir() if p.name.startswith("batch_"))
    assert [p.name for p in files_a] == [p.name for p in files_b]
    for pa, pb in zip(files_a, files_b):
        assert pa.read_bytes() == pb.read_bytes(), pa.name


def test_round_trips_through_the_standard_val_loader(frozen):
    cfg, _out, val_dir = frozen
    episodes = _load_episodes(val_dir)
    assert len(episodes) == cfg.n_datasets

    for batch in episodes:
        X, Y_obs, MU0, MU1 = batch[1], batch[6], batch[9], batch[10]
        seq_lens, train_sizes, S2, K_eff = batch[15], batch[16], batch[17], batch[18]

        assert X.shape[1] == cfg.seq_len
        assert int(seq_lens[0]) == cfg.seq_len
        # train_size is pinned so every episode shares the same prefix geometry.
        assert int(train_sizes[0]) == cfg.n_max
        assert int(K_eff[0]) == cfg.mc_cate
        assert Y_obs.reshape(1, -1).shape[1] == cfg.seq_len

        # mc_targets_on_context=False: the context prefix was never replicated,
        # so its targets are NaN by convention -- and the query block must not be.
        for t in (MU0, MU1, S2):
            flat = t.reshape(-1)
            assert flat[: cfg.n_max].isnan().all()
            assert flat[cfg.n_max :].isfinite().all()


def test_sidecar_targets_match_the_data_on_disk(frozen):
    cfg, out, val_dir = frozen
    meta = load_frozen_meta(out)
    episodes = _load_episodes(val_dir)

    assert meta["global"]["n_grid"] == cfg.n_grid()
    assert len(meta["episodes"]) == cfg.n_datasets

    for record, batch in zip(meta["episodes"], episodes):
        Y_obs, MU0, MU1, S2 = batch[6], batch[9], batch[10], batch[17]

        # The frozen comparison space is the *trainer's* formula: biased std
        # with the 1e-3 floor, not predict()'s 1e-5 clamp.
        ctx = Y_obs.reshape(-1)[: cfg.n_max].float()
        assert torch.allclose(record["y_context_mean"], ctx.mean())
        assert torch.allclose(
            record["y_context_std"], ctx.std(unbiased=False).clamp(min=1e-3)
        )

        tau = (MU1.reshape(-1)[cfg.n_max :] - MU0.reshape(-1)[cfg.n_max :]).float()
        assert torch.allclose(record["tau_star"], tau, atol=1e-6)
        assert torch.allclose(
            record["tau_star_std"], tau / record["y_context_std"], atol=1e-6
        )

        # s2 must be pre-rescaled into head units by the *variance*, matching
        # how the noise-aware loss consumes it (POSTERIOR_EVAL_NOTES 2.2 item 2).
        s2 = S2.reshape(-1)[cfg.n_max :].float()
        assert torch.allclose(record["s2"], s2, atol=1e-9)
        assert torch.allclose(
            record["s2_std"], s2 / record["y_context_std"] ** 2, atol=1e-9
        )

        assert record["query_index"].tolist() == list(range(cfg.n_max, cfg.seq_len))
        assert record["k_eff"] == cfg.mc_cate


def test_a_capped_k_fails_the_build_loudly(tmp_path):
    # prior_mc_mem_budget caps mc_cate behind a once-per-process warning, so a
    # long build would report it for the first episode and stay silent after.
    cfg = _cfg(n_datasets=1, mc_cate=64, prior_mc_mem_budget=10_000)
    with pytest.raises(MonteCarloBudgetError, match="capped mc_cate"):
        build(cfg, tmp_path / "capped", progress=False)


def test_verify_recomputes_the_stored_digest(tmp_path):
    # verify.py is what runs after a set is copied between machines, so it is
    # where a sidecar whose digest no longer matches its own config surfaces.
    from eval_datasets.prior_eval.verify import _recompute_from_config

    out = tmp_path / "set"
    build(_cfg(), out, datasets_per_file=2, progress=False)
    g = load_frozen_meta(out)["global"]
    assert _recompute_from_config(g) == g["content_digest"]
