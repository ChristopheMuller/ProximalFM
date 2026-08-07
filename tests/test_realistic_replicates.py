"""Replicate mode in the realistic generator (ROADMAP B1/B2/B3).

The load-bearing test here is ``test_legacy_path_is_byte_identical``: replicate
mode had to be threaded through twelve hand-written mechanisms without moving a
single RNG draw on the ``mc_cate=1`` path, or the committed
``synthetic/validation_data`` set -- and every ``val/point`` number ever logged
against it -- would silently stop being reproducible.

The other sharp checks are the two mechanisms whose Monte Carlo target has a
known closed form: m9 (``tau`` is a function of X alone, so the replicate
variance is zero) and m12 (``E[tau|X] = (1-null_frac)*c*tanh(lin(X))``). They
are the only ground truth available for the averaging itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "synthetic"))

from build_proximal_datasets import POOL_FIELDS, build, parse_args  # noqa: E402
from realistic_proximal_mechanisms import (  # noqa: E402
    N_MECHANISMS,
    MonteCarloBudgetError,
    RealisticGeneratorConfig,
    RealisticProximalGenerator,
)
from realistic_sidecar import (  # noqa: E402
    RealisticBuildConfig,
    RealisticSetConflictError,
    load_realistic_meta,
    realistic_content_digest,
)

from proximalfm.prior._genload import LoadPriorDataset  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "realistic_legacy_ref.pt"

T = 64
TRAIN = 48
N_Q = T - TRAIN


def make_gen(mc_cate=1, seq_len=T, seed=1234, **kw):
    cfg = RealisticGeneratorConfig(
        save_dir=kw.pop("save_dir", "/tmp/_rpm_test_scratch"),
        K=2,
        seq_len=seq_len,
        train_size=0.75,
        min_dim_U=1,
        max_dim_U=3,
        min_dim_X=1,
        max_dim_X=8,
        min_dim_W=1,
        max_dim_W=8,
        min_dim_Z=1,
        max_dim_Z=8,
        mc_cate=mc_cate,
        seed=seed,
        device="cpu",
        **kw,
    )
    return RealisticProximalGenerator(cfg)


def batch_for(mech, mc_cate, batch_size=2, seq_len=T, train_size=TRAIN, seed=1234):
    gen = make_gen(mc_cate=mc_cate, seq_len=seq_len, seed=seed)
    b = gen.get_batch(
        mechanism_id=mech,
        batch_size=batch_size,
        seq_len=seq_len,
        train_size=train_size,
    )
    return gen, dict(zip(POOL_FIELDS, b))


# ---------------------------------------------------------------------------
# 1. Legacy identity
# ---------------------------------------------------------------------------


def test_legacy_path_is_byte_identical():
    """mc_cate=1 must reproduce the pre-replicate-mode generator exactly.

    The fixture was captured from the commit before replicate mode landed. Any
    difference here -- even one float32 ulp -- means the committed 480-dataset
    validation set can no longer be regenerated from its recorded seed.
    """
    ref = torch.load(FIXTURE, weights_only=False)
    fields = ref["config"]["fields"]
    gen = make_gen(mc_cate=1)

    for m in range(1, N_MECHANISMS + 1):
        got = gen.get_batch(mechanism_id=m, batch_size=2, seq_len=64, train_size=0.75)
        assert len(got) == 19, "MU0/MU1/S2/K_eff must be appended, not inserted"
        for name, actual in zip(fields, got[:15]):
            expected = ref["mechanisms"][m][name]
            assert torch.equal(expected, actual), f"mechanism {m}, field {name}"


def test_legacy_path_reports_the_realized_draw_as_the_target():
    """Without replicates MU is Y, matching what LoadPriorDataset substitutes.

    Writing NaN instead would be more honest about there being no CATE target,
    but ``_genload`` only falls back to Y0/Y1 when the *keys are absent* -- so
    NaN values would flow into ``true_cate`` and silently destroy
    ``val/point/cate/npehe``, the default ``--best_metric``.
    """
    _, p = batch_for(mech=1, mc_cate=1)
    assert torch.equal(p["MU0"], p["Y0"])
    assert torch.equal(p["MU1"], p["Y1"])
    assert (p["S2"] == 0).all()
    assert (p["K_eff"] == 1).all()


# ---------------------------------------------------------------------------
# 2. Replicate layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mech", range(1, N_MECHANISMS + 1))
def test_replicate_layout_and_prefix_convention(mech):
    gen, p = batch_for(mech, mc_cate=16)
    assert (p["K_eff"] == 16).all()
    for name in ("MU0", "MU1", "S2"):
        t = p[name]
        assert torch.isnan(t[:, :TRAIN]).all(), f"{name} must be NaN on the context"
        assert torch.isfinite(t[:, TRAIN:]).all(), f"{name} must be finite on queries"
    assert (p["S2"][:, TRAIN:] >= 0).all()
    # Observed rows still carry exactly one realized world each.
    assert p["Y_obs"].shape[1] == T


def test_replicate_holds_the_covariate_row_fixed():
    """_replicate(X)[obs_idx] == X is the invariant the whole target rests on."""
    gen = make_gen(mc_cate=8)
    gen._rep = gen._make_rep(T, TRAIN)
    x = torch.randn(T, 3)
    rep = gen._replicate(x)
    assert rep.shape == (gen._rep.N, 3)
    assert torch.equal(gen._obs(rep), x)
    # Every replicate of a query row is the same row.
    block = rep[TRAIN:].view(N_Q, 8, 3)
    assert torch.equal(block, x[TRAIN:, None, :].expand_as(block))


def test_mc_targets_are_not_the_realized_ite():
    """The whole point of B1: MU1-MU0 must differ from the sampled Y1-Y0.

    If these agreed, replicate mode would be doing nothing and a distributional
    score would still be measuring the ITE predictive.
    """
    _, p = batch_for(mech=2, mc_cate=64)
    tau = (p["MU1"] - p["MU0"])[:, TRAIN:, 0]
    ite = (p["Y1"] - p["Y0"])[:, TRAIN:, 0]
    assert (tau - ite).abs().mean() > 0.05 * ite.std()


def test_mc_reduce_is_the_paired_variance_of_the_mean():
    """S2 = Var_k(y1-y0)/K, differenced *before* the variance.

    Taking ``[Var(y1) + Var(y0)]/K`` instead is the classic version of this bug:
    it still produces plausible small numbers, but inflates the reported target
    noise by roughly twice the outcome variance. The construction below makes
    the two answers maximally different -- both arms are noisy, their difference
    is exactly constant -- so the paired form must report zero.

    Driven directly rather than through a mechanism because changing ``mc_cate``
    changes the RNG stream, so two runs at different K are different datasets
    and cannot be compared row-wise.
    """
    gen = make_gen(mc_cate=4)
    n_ctx, n_q, K = 3, 5, 4
    gen._rep = gen._make_rep(n_ctx + n_q, n_ctx)
    assert (gen._rep.n_ctx, gen._rep.n_q, gen._rep.K) == (n_ctx, n_q, K)

    shared = torch.randn(n_q * K, 1)
    gap = 0.75
    y0 = torch.cat([torch.randn(n_ctx, 1), shared])
    y1 = torch.cat([torch.randn(n_ctx, 1), shared + gap])

    MU0, MU1, S2, k_eff = gen._mc_reduce(y0, y1)
    assert k_eff == K
    assert torch.allclose(MU1[n_ctx:] - MU0[n_ctx:], torch.full((n_q, 1), gap))
    # Paired: the shared draw cancels, so there is no sampling noise at all.
    assert S2[n_ctx:].abs().max() < 1e-6
    # And the arm means really are per-row means over the replicate block.
    assert torch.allclose(MU0[n_ctx:], shared.view(n_q, K).mean(1, keepdim=True))

    # Unpaired arms: S2 must now be Var_k(difference)/K exactly.
    y1_indep = torch.cat([torch.randn(n_ctx, 1), torch.randn(n_q * K, 1)])
    _, _, S2b, _ = gen._mc_reduce(y0, y1_indep)
    diff = (y1_indep - y0)[n_ctx:, 0].view(n_q, K)
    assert torch.allclose(S2b[n_ctx:, 0], diff.var(dim=1, unbiased=True) / K, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. The two mechanisms with ground truth
# ---------------------------------------------------------------------------


def test_m09_target_variance_is_zero():
    """m9's tau is a function of X alone, so replicates cannot disagree.

    Not *exactly* zero in fp32: tau is recovered as (f0+tau+eps)-(f0+eps), and
    that cancellation rounds. But it lands ~1e-16, some seven orders of
    magnitude below any mechanism that has genuine latent spread, so the bound
    still fails loudly if X is not held fixed across replicates -- or if the
    view(n_q, K) reduction were K-major rather than row-major.
    """
    _, p = batch_for(mech=9, mc_cate=32)
    assert p["S2"][:, TRAIN:].max() < 1e-12

    _, other = batch_for(mech=1, mc_cate=32)
    assert other["S2"][:, TRAIN:].mean() > 1e-8, "control: m1 must have real spread"


def test_m12_monte_carlo_converges_to_its_closed_form():
    """E[tau|X] = (1-null_frac)*c*tanh(lin(X)) -- the averaging's only ground truth."""
    gen, p = batch_for(mech=12, mc_cate=256, batch_size=1)
    exact = gen._obs(gen._debug["m12_tau_star"])[TRAIN:, 0]
    mc = (p["MU1"] - p["MU0"])[0, TRAIN:, 0]
    s = p["S2"][0, TRAIN:, 0].sqrt()
    # Within 4 sampling sd on every row; S2 is the estimate of exactly this error.
    assert (mc - exact).abs().max() <= 4 * s.max() + 1e-6


def test_m12_error_shrinks_with_more_replicates():
    """Convergence at the Monte Carlo rate, not merely 'close at one K'."""

    def mean_abs_err(k):
        gen, p = batch_for(mech=12, mc_cate=k, batch_size=1)
        exact = gen._obs(gen._debug["m12_tau_star"])[TRAIN:, 0]
        return ((p["MU1"] - p["MU0"])[0, TRAIN:, 0] - exact).abs().mean().item()

    assert mean_abs_err(16) > 1.5 * mean_abs_err(256)


# ---------------------------------------------------------------------------
# 4. Budget
# ---------------------------------------------------------------------------


def test_an_unaffordable_k_raises_rather_than_capping():
    """A silent cap would vary K_eff across datasets and machines, invisibly."""
    gen = make_gen(mc_cate=4096, mc_mem_budget=10_000)
    with pytest.raises(MonteCarloBudgetError, match="mc_mem_budget"):
        gen.get_batch(mechanism_id=1, batch_size=1, seq_len=T, train_size=TRAIN)


# ---------------------------------------------------------------------------
# 5. Round trip through the on-disk format and the sidecar
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_set(tmp_path_factory):
    root = tmp_path_factory.mktemp("realistic_dist")
    args = parse_args(
        [
            "--name",
            "toy",
            "--out_root",
            str(root),
            "--n_datasets",
            "12",
            "--seq_len",
            "128",
            "--prop_train",
            "0.75",
            "--mc_cate",
            "32",
            "--seed",
            "20260807",
        ]
    )
    build(args)
    return root / "toy", args


def test_batch_files_carry_every_field_the_loader_expects(toy_set):
    out, _ = toy_set
    raw = torch.load(out / "val" / "batch_000000.pt", weights_only=True)
    # MU0/MU1/S2/K_eff are the four fields the hand-rolled writer used to drop,
    # which is how realistic sets silently lost their CATE target.
    for key in ("MU0", "MU1", "S2", "K_eff"):
        assert key in raw, key
    meta = json.loads((out / "val" / "metadata.json").read_text())
    assert meta["mc_cate"] == 32
    assert meta["mc_targets_on_context"] is False


def test_round_trips_through_the_standard_val_loader(toy_set):
    out, args = toy_set
    loader = LoadPriorDataset(
        data_dir=str(out / "val"), batch_size=args.n_datasets, device="cpu"
    )
    batch = next(iter(loader))
    MU0, MU1, S2, K_eff = batch[9], batch[10], batch[17], batch[18]
    n_max = int(batch[16][0])
    assert (K_eff == args.mc_cate).all()
    assert torch.isnan(MU0[:, :n_max]).all()
    assert torch.isfinite(MU1[:, n_max:]).all()
    assert (S2[:, n_max:] >= 0).all()


def test_sidecar_targets_match_the_data_on_disk(toy_set):
    out, args = toy_set
    meta = load_realistic_meta(out)
    g = meta["global"]
    assert g["set_kind"] == "realistic"
    assert len(meta["episodes"]) == args.n_datasets

    raw = torch.load(out / "val" / "batch_000000.pt", weights_only=True)
    n_max = g["n_max"]
    for i, rec in enumerate(meta["episodes"]):
        ctx = raw["Y_obs"][i].reshape(-1)[:n_max].float()
        y_std = ctx.std(unbiased=False).clamp(min=1e-3)
        assert torch.allclose(rec["y_context_mean"], ctx.mean())
        assert torch.allclose(rec["y_context_std"], y_std)

        tau = (raw["MU1"][i] - raw["MU0"][i]).reshape(-1)[n_max:].float()
        assert torch.allclose(rec["tau_star"], tau, atol=1e-6)
        assert torch.allclose(rec["tau_star_std"], tau / y_std, atol=1e-6)
        # s2 is divided by the VARIANCE, not the sd -- the invisible scale bug.
        s2 = raw["S2"][i].reshape(-1)[n_max:].float()
        assert torch.allclose(rec["s2_std"], s2 / (y_std**2), atol=1e-9)
        assert rec["index"] == i
        assert 1 <= rec["mechanism_id"] <= N_MECHANISMS


def test_score_frozen_set_reads_a_realistic_set_unchanged(toy_set):
    """The point of reusing the schema: no realistic-specific branch downstream."""
    from proximalfm.train._dist_eval import score_frozen_set

    from test_posterior_eval import _FakeCateModel  # noqa: F401

    out, args = toy_set
    metrics = score_frozen_set(_FakeCateModel(), out)
    n_q = args.seq_len - int(round(args.prop_train * args.seq_len))
    assert metrics["prior_dist/eval_rows"] == args.n_datasets * n_q
    assert metrics["prior_dist/grid_scale"] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 6. Set identity
# ---------------------------------------------------------------------------


def test_digest_is_stable_and_moves_with_generative_fields():
    base = dict(
        n_datasets=12,
        seq_len=128,
        prop_train=0.75,
        mc_cate=32,
        seed=1,
        mechanisms=[1, 2, 3],
    )
    d = RealisticBuildConfig(**base).content_digest()
    assert d == RealisticBuildConfig(**base).content_digest()
    for field, value in [
        ("mc_cate", 64),
        ("seed", 2),
        ("seq_len", 256),
        ("prop_train", 0.5),
        ("mechanisms", [1, 2]),
        # Unlike the prior sets, n_datasets IS generative here: the pool is
        # shuffled by one permutation whose length it sets, so adding episodes
        # reorders every existing one rather than extending the set.
        ("n_datasets", 24),
    ]:
        assert RealisticBuildConfig(**{**base, field: value}).content_digest() != d


def test_a_sidecar_without_a_stored_digest_still_has_an_identity(toy_set):
    out, _ = toy_set
    meta = load_realistic_meta(out)
    stored = meta["global"].pop("content_digest")
    assert realistic_content_digest(meta) == stored


def test_rebuilding_a_set_with_a_different_config_is_refused(toy_set, tmp_path):
    out, args = toy_set
    from realistic_sidecar import check_no_silent_mutation

    cfg = RealisticBuildConfig(
        n_datasets=args.n_datasets,
        seq_len=args.seq_len,
        prop_train=args.prop_train,
        mc_cate=args.mc_cate * 2,  # different content, same path
        seed=args.seed,
        mechanisms=list(range(1, N_MECHANISMS + 1)),
    )
    with pytest.raises(RealisticSetConflictError, match="already holds"):
        check_no_silent_mutation(out, cfg)
    # --force is the documented escape hatch.
    check_no_silent_mutation(out, cfg, force=True)


def test_no_sidecar_is_written_without_replicates(tmp_path):
    """The gate for val/dist: a point-only set has no identity to score against."""
    args = parse_args(
        [
            "--name",
            "point_only",
            "--out_root",
            str(tmp_path),
            "--n_datasets",
            "3",
            "--seq_len",
            "64",
            "--mechanisms",
            "1",
            "--seed",
            "7",
        ]
    )
    build(args)
    assert not (tmp_path / "point_only" / "frozen_meta.pt").exists()
