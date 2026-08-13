"""Tests for benchmarks/metrics.py — the scoring layer.

Two things are pinned here. First the arithmetic of the metrics themselves,
which had no test at all and is about to carry a second target. Second, and
more valuable, that this implementation and the trainer's
``CausalEvaluator`` agree number-for-number: after the CATE-target rework both
score the *same* estimand on the *same* sets, so ``synt_point/cate/npehe`` in
wandb and the benchmark's cate-track nPEHE are directly comparable — and will be
compared. Nothing shares code between them (the benchmark deliberately imports
no proximalfm), so only a test keeps them from drifting.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from benchmarks.metrics import (
    TARGET_VAR_THRESHOLD,
    build_summary_table,
    canonical_estimand,
    compute_all_metrics,
    compute_episode_metrics,
    estimand_in_track,
    format_estimand,
    parse_estimand,
    select_track,
)


def _episode(pred, target, **kw):
    return compute_episode_metrics(
        "m", 0, np.asarray(pred, float), np.asarray(target, float), 0.0, **kw
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode arithmetic
# ─────────────────────────────────────────────────────────────────────────────


def test_perfect_prediction_scores_zero():
    tau = np.array([1.0, -2.0, 0.5, 3.0])
    m = _episode(tau, tau)
    assert m.npehe == pytest.approx(0.0)
    assert m.cate_r2 == pytest.approx(1.0)
    assert m.shape_r2 == pytest.approx(1.0)
    assert m.ate_bias_norm == pytest.approx(0.0)


def test_constant_ate_prediction_is_the_npehe_1_anchor():
    """Predicting the true episode mean sits at exactly nPEHE 1 / Shape R² 0."""
    tau = np.array([1.0, -2.0, 0.5, 3.0])
    m = _episode(np.full_like(tau, tau.mean()), tau)
    assert m.npehe == pytest.approx(1.0)
    assert m.cate_r2 == pytest.approx(0.0)
    assert m.shape_r2 == pytest.approx(0.0)


def test_npehe_and_r2_are_the_same_number():
    rng = np.random.default_rng(0)
    tau = rng.normal(size=50)
    m = _episode(tau + rng.normal(scale=0.4, size=50), tau)
    assert m.cate_r2 == pytest.approx(1.0 - m.npehe**2)


def test_shape_r2_ignores_a_pure_level_shift():
    tau = np.array([1.0, -2.0, 0.5, 3.0])
    m = _episode(tau + 5.0, tau)
    assert m.shape_r2 == pytest.approx(1.0)  # shape is perfect...
    assert m.cate_r2 < 0  # ...but the level is not
    assert m.ate_bias == pytest.approx(5.0)


def test_homogeneous_episode_yields_no_scale_free_metrics():
    tau = np.full(8, 2.0)
    m = _episode(np.zeros(8), tau)
    assert m.homogeneous
    assert m.npehe is None and m.shape_r2 is None and m.ate_bias_norm is None
    assert m.target_var < TARGET_VAR_THRESHOLD


def test_partial_nans_fall_back_to_the_episode_mean():
    tau = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, np.nan, 3.0, 4.0])
    m = _episode(pred, tau)
    assert not m.failed
    # The NaN row is scored against the mean of the surviving predictions.
    filled = np.where(np.isnan(pred), np.nanmean(pred), pred)
    assert m.npehe == pytest.approx(np.sqrt(((filled - tau) ** 2).mean() / tau.var()))


def test_all_nan_episode_is_a_failure():
    m = _episode(np.full(4, np.nan), np.array([1.0, 2.0, 3.0, 4.0]))
    assert m.failed
    assert m.npehe is None


# ─────────────────────────────────────────────────────────────────────────────
# The Monte-Carlo noise share, reported and never subtracted
# ─────────────────────────────────────────────────────────────────────────────


def test_s2_over_var_is_the_noise_share_of_the_target():
    tau = np.array([1.0, -2.0, 0.5, 3.0])
    s2 = np.full(4, 0.1)
    m = _episode(tau, tau, s2=s2)
    assert m.s2_over_var == pytest.approx(0.1 / tau.var())


def test_s2_does_not_change_any_score():
    """It is a diagnostic: the metrics score MU1-MU0 exactly as given."""
    rng = np.random.default_rng(1)
    tau, pred = rng.normal(size=30), rng.normal(size=30)
    without = _episode(pred, tau)
    with_s2 = _episode(pred, tau, s2=np.full(30, 0.5))
    assert with_s2.npehe == without.npehe
    assert with_s2.shape_r2 == without.shape_r2


def test_s2_absent_leaves_the_column_empty():
    tau = np.array([1.0, -2.0, 0.5, 3.0])
    assert _episode(tau, tau).s2_over_var is None


# ─────────────────────────────────────────────────────────────────────────────
# Tracks: which rows a table shows, and who may be ranked
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def frame():
    rng = np.random.default_rng(2)
    tau = rng.normal(size=(4, 25))
    ite = tau + rng.normal(size=(4, 25))
    good = tau + rng.normal(scale=0.3, size=(4, 25))
    const = np.repeat(tau.mean(axis=1, keepdims=True), 25, axis=1)
    times = np.zeros(4)

    blocks = []
    for kind, target in (("cate", tau), ("ite", ite)):
        blocks += [
            compute_all_metrics(
                good, target, times, "S-OLS", target_kind=kind, estimand="cate(x)"
            ),
            compute_all_metrics(
                good, target, times, "PMMR-XW", target_kind=kind, estimand="cate(x,w)"
            ),
            compute_all_metrics(
                const, target, times, "P2SLS", target_kind=kind, estimand="ate(x)"
            ),
            compute_all_metrics(
                tau,
                target,
                times,
                "Oracle-CATE(X)",
                target_kind=kind,
                estimand="cate(x)",
                is_reference=True,
            ),
        ]
    return pd.concat(blocks, ignore_index=True)


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("cate(w,x)", "cate(x,w)"),  # canonical order, not the order given
        ("cate(u,z,w,x)", "cate(x,w,z,u)"),
        ("ate( x , w )", "ate(x,w)"),
        ("x", "cate(x)"),  # the legacy spellings still parse
        ("xw", "cate(x,w)"),
        ("ate", "ate(x)"),
    ],
)
def test_estimand_tags_round_trip_to_a_canonical_spelling(tag, expected):
    assert canonical_estimand(tag) == expected
    kind, inputs = parse_estimand(tag)
    assert format_estimand(kind, inputs) == expected


@pytest.mark.parametrize("tag", ["cate", "cate(x", "cate(x,q)", "tau(x)", ""])
def test_malformed_estimand_tags_are_refused(tag):
    with pytest.raises(ValueError):
        parse_estimand(tag)


@pytest.mark.parametrize(
    "tag,cate,ite,ate",
    [
        ("cate(x)", True, True, True),
        ("cate(x,w)", False, True, True),
        ("cate(x,z)", False, True, True),
        ("cate(x,w,z)", False, True, True),
        # An oracle's u is not disqualifying: those rows are upper bounds, and
        # dropping them from the headline table is exactly what they are for.
        ("cate(x,u)", True, True, True),
        ("cate(x,w,u)", False, True, True),
        ("ate(x)", False, False, True),
    ],
)
def test_track_membership_is_kind_plus_a_proxy_check(tag, cate, ite, ate):
    assert estimand_in_track(tag, "cate") is cate
    assert estimand_in_track(tag, "ite") is ite
    assert estimand_in_track(tag, "ate") is ate


def test_legacy_tags_are_stored_in_the_new_spelling(frame):
    """A pre-existing config's `estimand: xw` lands in the frame as cate(x,w)."""
    df = compute_all_metrics(
        np.zeros((2, 5)), np.ones((2, 5)), np.zeros(2), "m", estimand="xw"
    )
    assert set(df["estimand"]) == {"cate(x,w)"}


def test_cate_track_excludes_proxy_conditioned_and_ate_methods(frame):
    methods = set(select_track(frame, "cate")["method"])
    assert methods == {"S-OLS", "Oracle-CATE(X)"}


def test_ite_track_admits_the_proxy_conditioned_methods(frame):
    methods = set(select_track(frame, "ite")["method"])
    assert methods == {"S-OLS", "PMMR-XW", "Oracle-CATE(X)"}


def test_ate_track_shows_everyone_and_reads_the_cate_rows(frame):
    rows = select_track(frame, "ate")
    assert set(rows["method"]) == {
        "S-OLS",
        "PMMR-XW",
        "P2SLS",
        "Oracle-CATE(X)",
    }
    assert (rows["target"] == "cate").all()


def test_ate_track_falls_back_to_ite_without_a_cate_target(frame):
    ite_only = frame[frame["target"] == "ite"]
    assert (select_track(ite_only, "ate")["target"] == "ite").all()


def test_unknown_track_is_refused(frame):
    with pytest.raises(ValueError, match="unknown track"):
        select_track(frame, "nonsense")


def test_reference_rows_are_shown_but_never_ranked(frame):
    summary = build_summary_table(frame, "cate")
    assert "Oracle-CATE(X)" in summary.index
    assert pd.isna(summary.loc["Oracle-CATE(X)", "Mean Rank"])
    assert summary.loc["Oracle-CATE(X)", "Reference"]
    # The one real method still ranks 1st — the oracle did not displace it.
    assert summary.loc["S-OLS", "Mean Rank"] == pytest.approx(1.0)
    assert not summary.loc["S-OLS", "Reference"]


def test_oracle_is_exact_on_cate_and_noise_bound_on_ite(frame):
    """The invariant the whole rework rests on.

    tau*(x) scores 0 against itself — if it does not, the target plumbing is
    wrong — and strictly worse than 0 against the realized ITE, by exactly the
    irreducible E[Var(Y1-Y0 | X)] that no tau(x) method can escape.
    """
    on_cate = build_summary_table(frame, "cate").loc["Oracle-CATE(X)", "nPEHE"]
    on_ite = build_summary_table(frame, "ite").loc["Oracle-CATE(X)", "nPEHE"]
    assert on_cate == pytest.approx(0.0, abs=1e-12)
    assert on_ite > 0.5


def test_ate_track_ranks_on_the_level_not_the_shape(frame):
    """P2SLS predicts the true episode mean, so it must win the ATE table
    outright while being last on nPEHE."""
    summary = build_summary_table(frame, "ate")
    assert summary.index[0] == "P2SLS"
    assert summary.loc["P2SLS", "ATE RMSE"] == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Agreement with the trainer's evaluator
# ─────────────────────────────────────────────────────────────────────────────


def test_matches_causal_evaluator():
    """Same predictions, same targets, same four numbers.

    The two implementations share no code by design. They must still agree, or
    the wandb curve and the benchmark table quietly mean different things.
    """
    from proximalfm.train._eval import CausalEvaluator

    rng = np.random.default_rng(7)
    true_cate = rng.normal(size=(6, 40))
    pred_cate = true_cate + rng.normal(scale=0.6, size=(6, 40))

    ev = CausalEvaluator()
    ev.update(
        pred_cate=torch.from_numpy(pred_cate),
        true_cate=torch.from_numpy(true_cate),
    )
    theirs = {
        k: torch.cat(v).numpy()
        for k, v in ev.metric_values.items()
        if k in ("cate/npehe", "cate/r2", "cate/r2_shape", "ate/bias_norm")
    }

    ours = compute_all_metrics(
        pred_cate.astype(np.float32),
        true_cate.astype(np.float32),
        np.zeros(6),
        "m",
    )

    for mine, trainer in (
        ("npehe", "cate/npehe"),
        ("cate_r2", "cate/r2"),
        ("shape_r2", "cate/r2_shape"),
        ("ate_bias_norm", "ate/bias_norm"),
    ):
        np.testing.assert_allclose(
            ours[mine].to_numpy(dtype=float), theirs[trainer], rtol=1e-5, atol=1e-6
        )


def test_homogeneity_cutoff_matches_causal_evaluator():
    """Both must drop the same episodes, or their medians are taken over
    different populations."""
    from proximalfm.train._eval import CausalEvaluator

    assert CausalEvaluator().homogeneous_var_threshold == TARGET_VAR_THRESHOLD
