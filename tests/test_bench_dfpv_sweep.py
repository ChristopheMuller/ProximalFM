"""DFPV's `tune.sweep` block — the axes that are free once a candidate is trained.

`sweep` is not "a second grid that runs faster". Per :class:`TuningConfig` it is
a *declaration* that a parameter is nearly free off a part-built candidate, and
the whole value of the block rests on a claim the tuner cannot check: that a
swept value gives exactly what a `grid` over it would. If that fails, the search
still returns a confident winner — one selected on scores no fresh fit would
reproduce. So the first two tests here are equivalence tests, and they are the
point of the file.

DFPV admits exactly three axes, for two different reasons:

- ``n_epochs`` — one run paused at each milestone, whose state is bit-identical
  to a fit that stopped there (`DFPVEstimator.fit_trajectory`).
- ``final_lam1`` / ``final_lam2`` — they enter only `_finalise`, so a new value
  is two Cholesky factorisations (`DFPVEstimator.refinalise`).

``lam1``/``lam2`` are the trap. They look like the same knob as ``final_lam*``
and are not: they enter the feature *training*, so no re-solve can produce them
and a swept value would select a model nobody trained. That refusal is pinned
below, because being wrong about it is silent.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from benchmarks.methods import _DFPV_SWEEPABLE, build_method
from benchmarks.methods.dfpv import TUNABLE_PARAMS, DFPVConfig, DFPVEstimator

from test_bench_dfpv_tune import _FAST, _dfpv, make_data


# ─────────────────────────────────────────────────────────────────────────────
# The equivalence claims the sweep rests on
# ─────────────────────────────────────────────────────────────────────────────


def test_milestone_state_is_bit_identical_to_a_fit_that_stopped_there():
    """A yielded milestone equals a fresh `fit` at that `n_epochs`, exactly.

    Not "close": bit-identical. The seeds are set once at the top of `_fit_to`
    and every net is built before the first epoch, so neither the
    initialisation nor the RNG stream depends on where the run stops — and
    `_finalise` at a milestone consumes no randomness. Anything that broke that
    (resampling the split per milestone, a dropout layer, scoring that draws)
    would leave the sweep silently selecting on scores no refit reproduces.
    """
    X, A, Y, W, Z = make_data()
    milestones = [1, 2, 3]

    traj = {
        epoch: (
            m.oos_score_,
            m._stage2_weight.clone(),
            m._mean_W_feat.clone(),
        )
        for epoch, m in DFPVEstimator(
            replace(DFPVConfig(**_FAST), n_epochs=max(milestones))
        ).fit_trajectory(X, A, Y, W=W, Z=Z, milestones=milestones)
    }

    for epoch in milestones:
        fresh = DFPVEstimator(replace(DFPVConfig(**_FAST), n_epochs=epoch))
        fresh.fit(X, A, Y, W=W, Z=Z)
        score, w2, mw = traj[epoch]
        assert score == fresh.oos_score_, f"score drifted at epoch {epoch}"
        assert torch_equal(w2, fresh._stage2_weight), f"weights drifted at {epoch}"
        assert torch_equal(mw, fresh._mean_W_feat)


def torch_equal(a, b) -> bool:
    import torch

    return bool(torch.equal(a, b))


@pytest.mark.parametrize("lam1,lam2", [(0.01, 1.0), (1.0, 0.01), (0.1, 0.1)])
def test_refinalise_equals_a_fresh_fit_at_those_final_lams(lam1, lam2):
    """Re-solving on trained features == fitting with those `final_lam*` set.

    This is the other half of the sweep's equivalence claim, and it holds only
    because `final_lam*` are genuinely absent from the training path: `_two_stage`
    takes the *training* penalties during `_stage2_update` and the effective
    final ones during `_finalise`. If a `final_lam` ever leaked into the loss,
    this test is what catches it.
    """
    X, A, Y, W, Z = make_data()

    trained = DFPVEstimator(DFPVConfig(**_FAST)).fit(X, A, Y, W=W, Z=Z)
    swept = trained.refinalise(lam1, lam2)

    fresh = DFPVEstimator(
        replace(DFPVConfig(**_FAST), final_lam1=lam1, final_lam2=lam2)
    ).fit(X, A, Y, W=W, Z=Z)

    assert swept == pytest.approx(fresh.oos_score_, rel=1e-9)
    assert torch_equal(trained._stage2_weight, fresh._stage2_weight)


def test_final_lams_default_to_the_training_penalties():
    """`final_lamK=None` is the pre-existing estimator, unchanged.

    The fields were added with a `None` default precisely so every config
    written before they existed keeps meaning what it did.
    """
    X, A, Y, W, Z = make_data()
    implicit = DFPVEstimator(DFPVConfig(**_FAST, lam1=0.3, lam2=0.7))
    implicit.fit(X, A, Y, W=W, Z=Z)
    explicit = DFPVEstimator(
        DFPVConfig(**_FAST, lam1=0.3, lam2=0.7, final_lam1=0.3, final_lam2=0.7)
    )
    explicit.fit(X, A, Y, W=W, Z=Z)

    assert implicit.oos_score_ == explicit.oos_score_
    assert torch_equal(implicit._stage2_weight, explicit._stage2_weight)


def test_final_lam_does_not_leak_into_feature_training():
    """Two configs differing only in `final_lam*` train the *same* features.

    The distinction the whole design rests on. Pinned on `_g_W`'s parameters
    rather than on a score, so it fails for the right reason if a penalty ever
    reaches the optimiser.
    """
    X, A, Y, W, Z = make_data()
    a = DFPVEstimator(DFPVConfig(**_FAST)).fit(X, A, Y, W=W, Z=Z)
    b = DFPVEstimator(replace(DFPVConfig(**_FAST), final_lam2=100.0)).fit(
        X, A, Y, W=W, Z=Z
    )

    for pa, pb in zip(a._g_W.parameters(), b._g_W.parameters()):
        assert torch_equal(pa, pb), "final_lam2 changed the trained features"
    # ... but it did change the solve, or the field would be inert.
    assert not torch_equal(a._stage2_weight, b._stage2_weight)


# ─────────────────────────────────────────────────────────────────────────────
# The search
# ─────────────────────────────────────────────────────────────────────────────


def test_swept_epochs_select_the_same_winner_as_a_grid_over_them():
    """`sweep: {n_epochs: …}` and `grid: {n_epochs: …}` pick the same value.

    The re-spelling claim, end to end through the wrapper: same candidate set,
    same scores, same winner — one training run instead of three.
    """
    X, A, Y, W, Z = make_data()
    epochs = [1, 2, 3]

    swept = _dfpv(tune={"enabled": True, "sweep": {"n_epochs": epochs}})
    swept.fit(X, A, Y, W=W, Z=Z)

    gridded = _dfpv(
        tune={"enabled": True, "n_trials": 10, "grid": {"n_epochs": epochs}}
    )
    gridded.fit(X, A, Y, W=W, Z=Z)

    assert swept.model.cfg.n_epochs == gridded.model.cfg.n_epochs
    assert swept.diagnostics()["tune_best_score"] == pytest.approx(
        gridded.diagnostics()["tune_best_score"], rel=1e-9
    )
    # The saving is real: one prepared run against one per value.
    assert swept.diagnostics()["tune_n_prepared"] == 1
    assert gridded.diagnostics()["tune_n_prepared"] == 3


def test_swept_final_lams_select_the_same_winner_as_a_grid_over_them():
    X, A, Y, W, Z = make_data()
    lams = [0.01, 0.1, 1.0]

    swept = _dfpv(tune={"enabled": True, "sweep": {"final_lam2": lams}})
    swept.fit(X, A, Y, W=W, Z=Z)

    gridded = _dfpv(
        tune={"enabled": True, "n_trials": 10, "grid": {"final_lam2": lams}}
    )
    gridded.fit(X, A, Y, W=W, Z=Z)

    assert swept.model.cfg.final_lam2 == gridded.model.cfg.final_lam2
    assert swept.diagnostics()["tune_best_score"] == pytest.approx(
        gridded.diagnostics()["tune_best_score"], rel=1e-9
    )
    assert swept.diagnostics()["tune_n_prepared"] == 1


def test_grid_and_sweep_compose():
    """`n_trials` runs, each scored at every sweep point.

    The cost model the block advertises: candidates = n_trials * |sweep|,
    training runs = n_trials.
    """
    X, A, Y, W, Z = make_data()
    est = _dfpv(
        tune={
            "enabled": True,
            "n_trials": 2,
            "seed": 0,
            "grid": {"lam2": [0.1, 1.0]},
            "sweep": {"n_epochs": [1, 2], "final_lam2": [0.05, 0.5]},
        }
    )
    est.fit(X, A, Y, W=W, Z=Z)

    diag = est.diagnostics()
    assert diag["tune_n_prepared"] == 2  # two training runs
    assert diag["tune_n_candidates"] == 2 * 2 * 2  # eight scored points
    # The winner is a full spec across both blocks.
    assert est.model.cfg.n_epochs in (1, 2)
    assert est.model.cfg.final_lam2 in (0.05, 0.5)
    assert est.model.cfg.lam2 in (0.1, 1.0)


def test_swept_winner_is_refit_and_backs_every_output():
    X, A, Y, W, Z = make_data()
    est = _dfpv(
        tune={"enabled": True, "sweep": {"n_epochs": [1, 2], "final_lam2": [0.1, 1.0]}},
        outputs=[
            {"name": "DFPV-ATE", "strategy": "marginal"},
            {"name": "DFPV-XW", "strategy": "paired"},
        ],
    )
    est.fit(X, A, Y, W=W, Z=Z)
    model = est.model

    # The kept model really is a fit at the selected spec, not the baseline.
    fresh = DFPVEstimator(
        replace(
            DFPVConfig(**_FAST),
            n_epochs=model.cfg.n_epochs,
            final_lam2=model.cfg.final_lam2,
        )
    ).fit(X, A, Y, W=W, Z=Z)
    assert model.oos_score_ == pytest.approx(fresh.oos_score_, rel=1e-9)

    out = est.predict_variants(X, W=W, Z=Z)
    assert set(out) == {"DFPV-ATE", "DFPV-XW"}
    assert est.model is model, "a read-out refitted the model"


# ─────────────────────────────────────────────────────────────────────────────
# The whitelist
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("param", ["lam1", "lam2"])
def test_training_penalties_are_refused_in_sweep_and_point_at_the_alternative(param):
    """The trap this whitelist exists for.

    `lam1`/`lam2` are legal in `grid` and illegal in `sweep`, because they enter
    the feature training — no re-solve produces them, so a swept value would
    price a full retrain as a Cholesky and select a model nobody trained. The
    message has to name `final_lam*`, or the refusal just looks arbitrary and
    invites `--force`-style workarounds.
    """
    with pytest.raises(ValueError, match="may not appear in `tune.sweep`") as exc:
        _dfpv(tune={"enabled": True, "sweep": {param: [0.1, 1.0]}})
    assert "final_lam1" in str(exc.value)

    # ... and the same field is accepted in `grid`.
    _dfpv(tune={"enabled": True, "grid": {param: [0.1, 1.0]}})


@pytest.mark.parametrize("param", ["feature_dim", "lr", "split_ratio", "hidden_dim"])
def test_other_tunables_are_refused_in_sweep(param):
    """Searchable is not the same as sweepable: every one of these needs a retrain."""
    with pytest.raises(ValueError, match="may not appear in `tune.sweep`"):
        _dfpv(tune={"enabled": True, "sweep": {param: [1, 2]}})


def test_sweepable_is_a_subset_of_tunable():
    """A field cannot be sweepable without being searchable at all."""
    assert _DFPV_SWEEPABLE <= TUNABLE_PARAMS


def test_milestones_must_be_valid():
    X, A, Y, W, Z = make_data(n=60)
    est = DFPVEstimator(DFPVConfig(**_FAST))
    with pytest.raises(ValueError, match="non-empty"):
        list(est.fit_trajectory(X, A, Y, W=W, Z=Z, milestones=[]))
    with pytest.raises(ValueError, match=">= 1"):
        list(est.fit_trajectory(X, A, Y, W=W, Z=Z, milestones=[0, 2]))


def test_refinalise_before_fit_raises():
    with pytest.raises(RuntimeError, match="before refinalise"):
        DFPVEstimator(DFPVConfig(**_FAST)).refinalise(0.1, 0.1)
