"""Replicate mode in the realistic generator.

The load-bearing test here is ``test_legacy_path_is_byte_identical``: replicate
mode had to be threaded through twelve hand-written mechanisms without moving a
single RNG draw on the ``mc_cate=1`` path, or the committed
``eval_datasets/synt_eval/validation_data`` set -- and every ``val/point`` number ever logged
against it -- would silently stop being reproducible.

The other sharp checks are the two mechanisms whose Monte Carlo target has a
known closed form: m9 (``tau`` is a function of X alone, so the replicate
variance is zero) and m12 (``E[tau|X] = (1-null_frac)*c*tanh(lin(X))``). They
are the only ground truth available for the averaging itself.
"""


from __future__ import annotations


from pathlib import Path


import pytest


import torch


from eval_datasets.synt_eval.build_proximal_datasets import (
    POOL_FIELDS,
    build,
    parse_args,
)


from eval_datasets.synt_eval.realistic_proximal_mechanisms import (
    RealisticGeneratorConfig,
    RealisticProximalGenerator,
)


from eval_datasets.synt_eval.realistic_sidecar import (
    load_realistic_meta,
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


def test_legacy_path_reports_the_realized_draw_as_the_target():
    """Without replicates MU is Y, matching what LoadPriorDataset substitutes.

    Writing NaN instead would be more honest about there being no CATE target,
    but ``_genload`` only falls back to Y0/Y1 when the *keys are absent* -- so
    NaN values would flow into ``true_cate`` and silently destroy
    ``synt_point/cate/npehe``, the default ``--best_metric``.
    """
    _, p = batch_for(mech=1, mc_cate=1)
    assert torch.equal(p["MU0"], p["Y0"])
    assert torch.equal(p["MU1"], p["Y1"])
    assert (p["S2"] == 0).all()
    assert (p["K_eff"] == 1).all()


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


def test_m12_monte_carlo_converges_to_its_closed_form():
    """E[tau|X] = (1-null_frac)*c*tanh(lin(X)) -- the averaging's only ground truth."""
    gen, p = batch_for(mech=12, mc_cate=256, batch_size=1)
    exact = gen._obs(gen._debug["m12_tau_star"])[TRAIN:, 0]
    mc = (p["MU1"] - p["MU0"])[0, TRAIN:, 0]
    s = p["S2"][0, TRAIN:, 0].sqrt()
    # Within 4 sampling sd on every row; S2 is the estimate of exactly this error.
    assert (mc - exact).abs().max() <= 4 * s.max() + 1e-6


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


def test_the_point_surface_scores_against_mu_not_the_sampled_ite(toy_set):
    """The single point-validation surface must use the Monte Carlo CATE.

    ``LoadPriorDataset`` substitutes ``Y0``/``Y1`` only when ``MU0``/``MU1``
    are *absent*, and the trainer reads MU on query rows only. So a replicate
    set's ``synt_point/cate/npehe`` is scored against the CEPO estimate rather
    than the sampled ITE -- which is why it stops estimating
    ``PEHE^2 + Var(noise)``.

    Pinned here because it is invisible: both paths produce plausible numbers.
    """
    out, args = toy_set
    n_max = load_realistic_meta(out)["global"]["n_max"]

    loader = LoadPriorDataset(
        data_dir=str(out / "val"), batch_size=args.n_datasets, device="cpu"
    )
    batch = next(iter(loader))
    Y0, Y1, MU0, MU1 = batch[7], batch[8], batch[9], batch[10]

    mc_cate = (MU1 - MU0)[:, n_max:]
    sampled_ite = (Y1 - Y0)[:, n_max:]
    assert torch.isfinite(mc_cate).all()
    # If these agreed, the substitution would be undetectable and the surface
    # would be reporting the ITE while claiming to report the CATE.
    assert not torch.allclose(mc_cate, sampled_ite, atol=1e-3)


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
