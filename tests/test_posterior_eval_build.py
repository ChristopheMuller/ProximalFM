"""The frozen posterior-eval set must be reproducible and readable as a val set.

M1 of ``POSTERIOR_EVAL_NOTES.md``. Three invariants are load-bearing and all
three fail silently if broken: the build is a pure function of its seed, the
realized ``K`` equals the requested one (``prior_mc_mem_budget`` caps it behind
a once-per-process warning), and the sidecar's frozen comparison space matches
the trainer's standardization exactly.
"""

import json

import pytest
import torch

from posterior_eval.build_prior_frozen_set import (
    BuildConfig,
    FrozenSetConflictError,
    MonteCarloBudgetError,
    build,
    episode_seed,
    frozen_content_digest,
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


def test_episode_seed_is_stable_and_index_local():
    # Appending episodes must leave the existing ones untouched, and nearby
    # master seeds must not produce overlapping streams.
    assert episode_seed(7, 3) == episode_seed(7, 3)
    assert episode_seed(7, 3) != episode_seed(8, 3)
    assert episode_seed(7, 3) != episode_seed(7, 4)


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


def test_metadata_json_matches_the_prior_format(frozen):
    _cfg_, _out, val_dir = frozen
    with open(val_dir / "metadata.json") as f:
        meta = json.load(f)
    # The keys SavePriorDataset writes, so the directory reads like any val set.
    for key in ("prior_type", "max_seq_len", "mc_cate", "mc_targets_on_context"):
        assert key in meta
    assert meta["mc_targets_on_context"] is False


def test_batch_files_carry_every_field_the_loader_expects(frozen):
    _cfg_, _out, val_dir = frozen
    payload = torch.load(val_dir / "batch_000000.pt", weights_only=True)
    # MU0/MU1/S2/K_eff are the fields synthetic/build_proximal_datasets.py omits
    # and LoadPriorDataset silently substitutes for. A frozen set that lost them
    # would train and score as noise-free with no error.
    for key in ("U", "X", "W", "Z", "A", "A_probs", "Y_obs", "Y0", "Y1"):
        assert key in payload
    for key in ("MU0", "MU1", "S2", "K_eff"):
        assert key in payload
    for key in ("d_U", "d_X", "d_W", "d_Z", "seq_lens", "train_sizes", "batch_size"):
        assert key in payload


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


def test_sidecar_carries_per_episode_hyperparameters(frozen):
    _cfg_, out, _val_dir = frozen
    record = load_frozen_meta(out)["episodes"][0]
    # The section-4 stratification inputs. Impossible to recover later without
    # regenerating the whole set.
    assert record["sampled_hp"]
    assert "confounding_share_of_U_in_Y" in record["sampled_hp"]
    assert set(record["mechanisms"]) == {"A", "Y"}
    assert record["gamma_hetero"] is not None
    assert record["a_logit_std"] is not None


def test_a_capped_k_fails_the_build_loudly(tmp_path):
    # prior_mc_mem_budget caps mc_cate behind a once-per-process warning, so a
    # long build would report it for the first episode and stay silent after.
    cfg = _cfg(n_datasets=1, mc_cate=64, prior_mc_mem_budget=10_000)
    with pytest.raises(MonteCarloBudgetError, match="capped mc_cate"):
        build(cfg, tmp_path / "capped", progress=False)


def test_the_set_has_a_uniform_and_usable_noise_estimate(frozen):
    # The set's whole point is a known, uniform target-noise level: every query
    # row needs a finite positive s2 and every episode the same K, or the
    # section-2.1 s/sigma_post read-off is comparing episodes at different
    # noise floors.
    #
    # The 1/K scaling of s2 itself is pinned at the generator, in
    # tests/test_prior.py::test_s2_scales_as_one_over_k, where episodes can be
    # held fixed. It is *not* checkable here: changing K changes N and so the
    # RNG stream, so episodes are unpaired across K and the between-episode
    # spread of s2 (orders of magnitude, heavy-tailed) swamps the 1/K signal.
    cfg, out, _val_dir = frozen
    meta = load_frozen_meta(out)

    assert {r["k_eff"] for r in meta["episodes"]} == {cfg.mc_cate}
    s2 = torch.cat([r["s2"] for r in meta["episodes"]])
    assert s2.isfinite().all()
    assert (s2 > 0).all()
    assert s2.numel() == cfg.n_datasets * cfg.n_q


# --- content digest / silent-mutation guard (ROADMAP A4) --------------------
#
# The digest is the identity a wandb curve is pinned to, so what matters is not
# that it exists but that it moves exactly when episode *content* moves --
# never on a cosmetic change, always on a generative one -- and that the build
# refuses to reuse a directory whose digest disagrees.


def test_digest_is_stable_and_ignores_the_appendable_axis():
    cfg = _cfg()
    assert cfg.content_digest() == _cfg().content_digest()
    # n_datasets is appendable: episodes 0..N-1 are bit-identical, so a longer
    # set is the same set. If this ever fails, extending a set stops being
    # possible without orphaning every run that scored the shorter one.
    assert _cfg(n_datasets=64).content_digest() == cfg.content_digest()


@pytest.mark.parametrize(
    "override",
    [
        {"n_q": 16},
        {"mc_cate": 32},
        {"n_max": 64},
        {"seed": 1},
        {"max_dim_X": 21},
        # Determinism is per-device, so a cuda set is not a cpu set even at an
        # identical config -- it must not inherit the same identity.
        {"gen_device": "cuda:0"},
    ],
)
def test_digest_moves_when_episode_content_moves(override):
    assert _cfg(**override).content_digest() != _cfg().content_digest()


def test_appending_episodes_to_an_existing_set_is_allowed(tmp_path):
    out = tmp_path / "set"
    build(_cfg(n_datasets=2), out, datasets_per_file=2, progress=False)
    build(_cfg(n_datasets=4), out, datasets_per_file=2, progress=False)
    assert load_frozen_meta(out)["global"]["n_datasets"] == 4


def test_rebuilding_a_set_with_a_different_config_is_refused(tmp_path):
    # The failure this guards: same --out, different K, so every run that
    # logged the path keeps claiming a set that no longer exists. Nothing on
    # disk would reveal it afterwards -- the rewritten sidecar is perfectly
    # self-consistent -- so refusing up front is the only defence.
    out = tmp_path / "set"
    build(_cfg(), out, datasets_per_file=2, progress=False)
    with pytest.raises(FrozenSetConflictError, match="already holds a"):
        build(_cfg(mc_cate=32), out, datasets_per_file=2, progress=False)
    # The original survives untouched.
    assert load_frozen_meta(out)["global"]["mc_cate"] == _cfg().mc_cate


def test_shrinking_a_set_is_refused_because_it_orphans_batch_files(tmp_path):
    out = tmp_path / "set"
    build(_cfg(n_datasets=4), out, datasets_per_file=2, progress=False)
    with pytest.raises(FrozenSetConflictError, match="surplus batch files"):
        build(_cfg(n_datasets=2), out, datasets_per_file=2, progress=False)


def test_force_overrides_the_guard(tmp_path):
    out = tmp_path / "set"
    build(_cfg(), out, datasets_per_file=2, progress=False)
    build(_cfg(mc_cate=32), out, datasets_per_file=2, progress=False, force=True)
    g = load_frozen_meta(out)["global"]
    assert g["mc_cate"] == 32
    assert g["content_digest"] == _cfg(mc_cate=32).content_digest()


def test_verify_recomputes_the_stored_digest(tmp_path):
    # verify.py is what runs after a set is copied between machines, so it is
    # where a sidecar whose digest no longer matches its own config surfaces.
    from posterior_eval.verify import _recompute_from_config

    out = tmp_path / "set"
    build(_cfg(), out, datasets_per_file=2, progress=False)
    g = load_frozen_meta(out)["global"]
    assert _recompute_from_config(g) == g["content_digest"]


def test_a_legacy_sidecar_still_has_an_identity(tmp_path):
    # The stored field is a cache, not the source of truth: sets built before
    # it existed (frozen_prod_v1 among them) must still resolve to the same
    # digest, or adopting this would have meant rewriting production sidecars.
    out = tmp_path / "set"
    build(_cfg(), out, datasets_per_file=2, progress=False)
    meta = load_frozen_meta(out)
    expected = meta["global"].pop("content_digest")
    assert frozen_content_digest(meta) == expected
