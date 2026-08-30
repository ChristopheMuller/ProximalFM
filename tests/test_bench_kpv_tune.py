"""KPV's kernel-scale search, now owned by the generic `tune:` scaffolding.

The search used to live inside ``KPVEstimator.fit`` as a hand-rolled loop over
``np.logspace(kernel_scale_min, kernel_scale_max, n_kernel_scale_search)``.
Moving it into :mod:`benchmarks.tuning` has to be a *relocation*, not a change
of estimator, so the first test pins the selected model against the loop the
wrapper replaced.

The rest are the refusals. KPV's selection score is the Stage-2 LOOCV loss, and
the whole point of :data:`benchmarks.methods.kpv.TUNABLE_PARAMS` is that this
score cannot rank most of ``KPVConfig``. A search over a parameter it cannot
rank does not fail — it returns a confident, meaningless winner — so the guards
are the only thing standing between a config typo and a silently wrong table.
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from benchmarks.methods import build_method
from benchmarks.methods.kpv import TUNABLE_PARAMS, KPVConfig, KPVEstimator

from test_bench_kpv_blocks import make_data

#: A narrow multiplicative grid around the per-dimension median heuristic.
_SCALES = [0.5, 0.7071067811865476, 1.0, 1.4142135623730951, 2.0]


def _kpv(**kwargs):
    """The wrapper as `run.py` builds it — through the config bridge, not by
    importing the class, so a refusal is pinned where a user would hit it."""
    _, est = build_method({"name": "KPV", "cls": "KernelPV", **kwargs})
    return est


def test_tuned_fit_matches_the_hand_rolled_scale_loop():
    """The winner is the argmin of `loocv_score_`, and is fitted at that scale.

    Two claims in one, because a search that selected correctly and then refit
    at the wrong settings would pass either half alone.
    """
    X, A, Y, W, Z = make_data(n=180, seed=3)

    # What `KPVEstimator.fit` used to do internally.
    base = KPVConfig()
    fits = {
        s: KPVEstimator(replace(base, kernel_scale=s)).fit(X, A, Y, W=W, Z=Z)
        for s in _SCALES
    }
    expected_scale = min(_SCALES, key=lambda s: fits[s].loocv_score_)

    est = _kpv(
        tune={"enabled": True, "n_trials": 5, "grid": {"kernel_scale": _SCALES}},
    )
    est.fit(X, A, Y, W=W, Z=Z)

    assert est.model.cfg.kernel_scale == pytest.approx(expected_scale, rel=1e-12)
    # Same λ's and the same bridge as a direct fit at the winning scale — the
    # refit in `fit` must reproduce the candidate, not merely resemble it.
    ref = fits[expected_scale]
    assert est.model.lam1_ == pytest.approx(ref.lam1_, rel=1e-12)
    assert est.model.lam2_ == pytest.approx(ref.lam2_, rel=1e-12)
    np.testing.assert_allclose(est.model._alpha, ref._alpha, rtol=1e-10)


def test_no_tune_block_leaves_the_configured_scale_alone():
    """Without `tune:`, one fit at `kernel_scale` and no diagnostics."""
    X, A, Y, W, Z = make_data(n=140, seed=4)

    est = _kpv(kernel_scale=1.4142135623730951)
    est.fit(X, A, Y, W=W, Z=Z)

    ref = KPVEstimator(KPVConfig(kernel_scale=1.4142135623730951)).fit(
        X, A, Y, W=W, Z=Z
    )
    np.testing.assert_allclose(est.model._alpha, ref._alpha, rtol=1e-12)
    assert est.diagnostics() == {}


def test_diagnostics_report_the_search():
    X, A, Y, W, Z = make_data(n=140, seed=5)
    est = _kpv(tune={"enabled": True, "n_trials": 3, "grid": {"kernel_scale": _SCALES}})
    est.fit(X, A, Y, W=W, Z=Z)

    diag = est.diagnostics()
    # `n_trials` below the grid size means the search saw a *subset*, and
    # `tune_n_candidates` is what says so.
    assert diag["tune_n_candidates"] == 3.0
    assert diag["tune_n_failed"] == 0.0
    assert np.isfinite(diag["tune_best_score"])
    # Exactly one offered value wins; values never drawn are absent, not zero.
    won = {k: v for k, v in diag.items() if k.startswith("tune_kernel_scale_frac@")}
    assert len(won) == 3 and sum(won.values()) == 1.0


def test_one_fit_backs_every_output():
    """`outputs:` reuses the tuned fit rather than re-searching per strategy."""
    X, A, Y, W, Z = make_data(n=140, seed=6)
    est = _kpv(
        tune={"enabled": True, "n_trials": 2, "grid": {"kernel_scale": _SCALES}},
        outputs=[
            {"name": "KPV-XW", "strategy": "paired"},
            {"name": "KPV-ATE", "strategy": "marginal"},
        ],
    )
    est.fit(X, A, Y, W=W, Z=Z)
    out = est.predict_variants(X, W=W)
    assert set(out) == {"KPV-XW", "KPV-ATE"}
    # The ATE head is constant by construction; the paired head is not. If the
    # two had come from different fits this would still hold, so the real check
    # is that diagnostics describe one search.
    assert est.diagnostics()["tune_n_candidates"] == 2.0


def test_one_shared_scale_is_reported_for_all_continuous_kernels():
    """The public KPV scale matches the reference's single multiplier."""
    X, A, Y, W, Z = make_data(n=140, seed=7)
    est = KPVEstimator(KPVConfig(kernel_scale=2.0)).fit(X, A, Y, W=W, Z=Z)
    assert est.kernel_scale_ == pytest.approx(2.0)


def test_scales_default_to_the_plain_median_heuristic():
    """All-1.0 is the paper's initialisation, and must stay the default."""
    X, A, Y, W, Z = make_data(n=120, seed=8)
    est = KPVEstimator(KPVConfig()).fit(X, A, Y, W=W, Z=Z)
    assert est.kernel_scale_ == pytest.approx(1.0)


def test_shared_scale_is_used_without_covariates():
    """The same scale applies when the empty X product is omitted."""
    X, A, Y, W, Z = make_data(n=120, seed=9, d_x=0)
    est = KPVEstimator(KPVConfig(kernel_scale=4.0)).fit(X, A, Y, W=W, Z=Z)
    assert est.kernel_scale_ == pytest.approx(4.0)


def test_wrapper_forwards_all_supported_config_fields():
    est = _kpv(
        kernel_scale=1.7,
        lam1=1e-3,
        lam1_min=1e-6,
        lam1_max=2.0,
        n_lam1_search=7,
        lam2=2e-3,
        lam2_min=2e-6,
        lam2_max=3.0,
        n_lam2_search=8,
    )
    assert est._base_cfg == KPVConfig(
        kernel_scale=1.7,
        lam1=1e-3,
        lam1_min=1e-6,
        lam1_max=2.0,
        n_lam1_search=7,
        lam2=2e-3,
        lam2_min=2e-6,
        lam2_max=3.0,
        n_lam2_search=8,
    )


# ── Refusals ─────────────────────────────────────────────────────────────────


def test_retired_scale_search_kwargs_raise_with_the_translation():
    for key in [
        "tune_kernel_scale",
        "kernel_scale_min",
        "kernel_scale_max",
        "n_kernel_scale_search",
        "kernel_scale_z",
        "kernel_scale_w",
        "kernel_scale_x",
    ]:
        with pytest.raises(ValueError, match="no longer method-level keys"):
            _kpv(**{key: 0.5})


def test_retired_per_variable_scales_raise_in_tune_grid():
    for key in ("kernel_scale_z", "kernel_scale_w", "kernel_scale_x"):
        with pytest.raises(ValueError, match="one shared continuous-kernel multiplier"):
            _kpv(tune={"enabled": True, "grid": {key: [0.5, 1.0]}})


def test_tunable_params_are_accepted():
    for param in sorted(TUNABLE_PARAMS):
        _kpv(tune={"enabled": True, "grid": {param: [0.5, 1.0]}})


def test_unscoreable_params_are_refused():
    """Fields the Stage-2 LOOCV cannot rank, for the reasons in TUNABLE_PARAMS."""
    for param in ("lam1_min", "lam2_max", "n_lam1_search"):
        with pytest.raises(ValueError, match="may not appear in `tune.grid`"):
            _kpv(tune={"enabled": True, "grid": {param: [0.4, 0.6]}})


def test_unknown_tune_key_is_refused():
    with pytest.raises(ValueError, match="unknown `tune.grid` key"):
        _kpv(tune={"enabled": True, "grid": {"bandwidth": [1.0]}})


def test_sweep_is_refused():
    """Nothing in KPV is free to re-evaluate off a part-built system."""
    with pytest.raises(ValueError, match="`tune.sweep` is not supported"):
        _kpv(tune={"enabled": True, "sweep": {"kernel_scale": [0.5, 1.0]}})


def test_val_frac_is_refused():
    """LOOCV is already held out; a second split would score on fewer rows."""
    with pytest.raises(ValueError, match="`tune.val_frac` has no effect"):
        _kpv(
            tune={
                "enabled": True,
                "val_frac": 0.2,
                "grid": {"kernel_scale": [0.5, 1.0]},
            }
        )
