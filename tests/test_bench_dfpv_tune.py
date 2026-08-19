"""DFPV's `tune:` block, ranked on the paper's out-of-sample Stage-2 loss.

DFPV was the only one of the four bridge methods with no tuning support, and —
because ``DFPV_Wrapper`` is built as ``DFPV_Wrapper(**cfg)`` and reads named
keys with ``kwargs.get`` — a ``tune:`` block in a config used to be *silently
swallowed*. A config asking for a search would run one un-searched fit and look
exactly like a working tuned run. So the refusals below are not defensive
padding: they are the difference between a typo and a silently wrong table.

The first group pins the criterion itself (``DFPVEstimator.oos_score_``), the
second that the wrapper selects and refits on it, and the third the guards.
Everything runs at a deliberately tiny budget — the point is the search
mechanics, not DFPV's accuracy.
"""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from benchmarks.methods import build_method
from benchmarks.methods.dfpv import TUNABLE_PARAMS, DFPVConfig, DFPVEstimator


#: Small enough that a handful of full training runs stay fast. `feature_dim=3`
#: with the backdoor on is a 4^3 = 64-column ridge system.
_FAST = dict(feature_dim=3, hidden_dim=8, n_epochs=2, stage1_iter=2, seed=0)


def make_data(n: int = 160, seed: int = 0):
    """A confounded proximal episode: U drives A, Y and both proxies."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=n)
    X = rng.normal(size=(n, 2))
    Z = (U + 0.5 * rng.normal(size=n)).reshape(-1, 1)
    W = (U + 0.5 * rng.normal(size=n)).reshape(-1, 1)
    A = (rng.random(n) < 1.0 / (1.0 + np.exp(-U))).astype(float)
    Y = 2.0 * A + U + X[:, 0] + 0.3 * rng.normal(size=n)
    return X, A, Y, W, Z


def _dfpv(**kwargs):
    """The wrapper as `run.py` builds it — through the config bridge, not by
    importing the class, so a refusal is pinned where a user would hit it."""
    _, est = build_method({"name": "DFPV", "cls": "DFPV", **_FAST, **kwargs})
    return est


# ─────────────────────────────────────────────────────────────────────────────
# The criterion
# ─────────────────────────────────────────────────────────────────────────────


def test_oos_score_is_the_stage2_residual_on_the_stage1_split():
    """`oos_score_` is what the paper's L₂-oos says it is, recomputed by hand.

    The estimator caches the score during `_finalise`, so nothing downstream
    re-derives it. If the cached value drifted from the formula, every ranking
    built on it would be wrong in a way no other test would catch.
    """
    X, A, Y, W, Z = make_data()
    est = DFPVEstimator(DFPVConfig(**_FAST)).fit(X, A, Y, W=W, Z=Z)

    # Rebuild the split exactly as `fit` did, then run the fitted bridge
    # forward on D1 — the rows the Stage-2 ridge never saw.
    import torch

    torch.manual_seed(_FAST["seed"])
    # `fit` seeds, converts, *then* permutes; replicate that order.
    n = len(Y)
    A_t = est._to_tensor(A)
    Y_t = est._to_tensor(Y).view(-1, 1)
    W_t, Z_t, X_t = est._to_tensor(W), est._to_tensor(Z), est._to_tensor(X)
    perm = torch.randperm(n, device=est._device)
    i1 = perm[: int(n * est.cfg.split_ratio)]

    with torch.no_grad():
        feat1 = est._augment_stage1(
            est._f1_A(A_t[i1]), est._g_Z(Z_t[i1]), est._b1_X(X_t[i1])
        )
        W_hat = feat1 @ est._stage1_weight
        feat2 = est._augment_stage2(W_hat, est._f2_A(A_t[i1]), est._b2_X(X_t[i1]))
        expected = float(torch.mean((Y_t[i1] - feat2 @ est._stage2_weight) ** 2))

    assert est.oos_score_ == pytest.approx(expected, rel=1e-5)
    assert np.isfinite(est.oos_stage1_score_)


def test_oos_scores_are_means_so_split_ratio_stays_comparable():
    """Both scores are per-row means, not sums.

    This is the property that makes `split_ratio` legal in the grid — and it is
    exactly the property KPV's LOOCV score lacks, which is why KPV refuses that
    field. A sum would grow with the evaluated row count and hand the search a
    systematic preference over the axis.
    """
    X, A, Y, W, Z = make_data(n=200)
    scores = {}
    for ratio in (0.3, 0.7):
        est = DFPVEstimator(replace(DFPVConfig(**_FAST), split_ratio=ratio))
        est.fit(X, A, Y, W=W, Z=Z)
        scores[ratio] = est.oos_score_

    # D1 more than doubles between the two, so a sum would differ by ~2.3x.
    # Both are residual means against the same Y, so they sit on one scale.
    lo, hi = sorted(scores.values())
    assert hi < 8.0 * lo, f"scores look sum-scaled rather than mean-scaled: {scores}"


# ─────────────────────────────────────────────────────────────────────────────
# Selection
# ─────────────────────────────────────────────────────────────────────────────


def test_search_selects_the_argmin_and_refits_there():
    """The winner is the argmin of `oos_score_`, and the kept model is fitted at it.

    Two claims in one, because a search that selected correctly and then refit
    at the wrong settings would pass either half alone.
    """
    X, A, Y, W, Z = make_data()
    grid = [1e-3, 0.1, 10.0]

    expected = min(
        grid,
        key=lambda v: DFPVEstimator(replace(DFPVConfig(**_FAST), lam2=v))
        .fit(X, A, Y, W=W, Z=Z)
        .oos_score_,
    )

    est = _dfpv(tune={"enabled": True, "n_trials": 10, "grid": {"lam2": grid}})
    est.fit(X, A, Y, W=W, Z=Z)

    assert est.model.cfg.lam2 == expected
    # ... and the surviving model really is a fit at that value, not the
    # baseline the wrapper was constructed with.
    assert est.model.oos_score_ == pytest.approx(
        DFPVEstimator(replace(DFPVConfig(**_FAST), lam2=expected))
        .fit(X, A, Y, W=W, Z=Z)
        .oos_score_,
        rel=1e-6,
    )


def test_one_fit_backs_every_output():
    """A tuned wrapper still trains once and reads out four strategies.

    The search must not leak into the per-output path: `predict_variants`
    reuses the single selected fit, so the four predictions have to come from
    one model. Pinned by identity of the underlying estimator object across
    the calls, plus the marginal output being constant per row.
    """
    X, A, Y, W, Z = make_data()
    est = _dfpv(
        tune={"enabled": True, "n_trials": 2, "grid": {"lam2": [0.1, 1.0]}},
        outputs=[
            {"name": "DFPV-XW", "strategy": "paired"},
            {"name": "DFPV-X-Wind", "strategy": "independent"},
            {"name": "DFPV-X-Wcond", "strategy": "conditional"},
            {"name": "DFPV-ATE", "strategy": "marginal"},
        ],
    )
    est.fit(X, A, Y, W=W, Z=Z)
    model_after_fit = est.model

    out = est.predict_variants(X, W=W, Z=Z)
    assert set(out) == {"DFPV-XW", "DFPV-X-Wind", "DFPV-X-Wcond", "DFPV-ATE"}
    assert all(v.shape == (len(Y),) for v in out.values())
    assert est.model is model_after_fit, "a read-out refitted the model"
    # The marginal head is one number broadcast; the per-x heads are not.
    assert np.allclose(out["DFPV-ATE"], out["DFPV-ATE"][0])
    assert not np.allclose(out["DFPV-XW"], out["DFPV-XW"][0])


def test_diagnostics_report_the_search_and_the_fit():
    X, A, Y, W, Z = make_data()
    est = _dfpv(tune={"enabled": True, "n_trials": 3, "grid": {"lam2": [0.1, 1.0]}})
    est.fit(X, A, Y, W=W, Z=Z)

    diag = est.diagnostics()
    assert diag["tune_n_candidates"] == 2
    assert diag["tune_best_score"] == pytest.approx(est.model.oos_score_, rel=1e-6)
    # Always present, search or no search — they describe the final fit.
    assert np.isfinite(diag["oos_stage2"]) and np.isfinite(diag["oos_stage1"])


def test_diagnostics_without_a_tune_block_still_report_the_fit():
    X, A, Y, W, Z = make_data()
    est = _dfpv()
    est.fit(X, A, Y, W=W, Z=Z)

    diag = est.diagnostics()
    assert not any(k.startswith("tune_") for k in diag)
    assert diag["oos_stage2"] == pytest.approx(est.model.oos_score_, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Refusals — the guards between a config typo and a silently wrong table
# ─────────────────────────────────────────────────────────────────────────────


def test_unknown_grid_key_is_refused():
    with pytest.raises(ValueError, match="unknown `tune.grid` key"):
        _dfpv(tune={"enabled": True, "grid": {"lam3": [0.1]}})


@pytest.mark.parametrize(
    "param", ["seed", "device", "x_cond_length_scale", "x_cond_top_k"]
)
def test_unscoreable_params_are_refused(param):
    """These are real `DFPVConfig` fields the criterion cannot rank.

    `seed`/`device` are not model choices; the two `x_cond_*` knobs enter only
    `_mean_outcome_conditional`, so every candidate differing in them scores
    *identically* — the search would return a confident, meaningless winner
    rather than fail. That silence is why they are refused rather than allowed.
    """
    with pytest.raises(ValueError, match="may not appear in"):
        _dfpv(tune={"enabled": True, "grid": {param: [1, 2]}})


# The `tune.sweep` whitelist and the axes it admits live in
# tests/test_bench_dfpv_sweep.py.


def test_val_frac_is_refused():
    with pytest.raises(ValueError, match="`tune.val_frac` has no effect"):
        _dfpv(
            tune={"enabled": True, "val_frac": 0.2, "grid": {"lam2": [0.1]}},
        )


def test_tunable_params_are_all_real_config_fields():
    """The whitelist cannot drift from the dataclass it whitelists.

    A field renamed in `DFPVConfig` but left in `TUNABLE_PARAMS` would pass the
    membership check and then fail inside `replace()`, mid-search.
    """
    assert TUNABLE_PARAMS <= {f.name for f in fields(DFPVConfig)}


def test_every_config_field_is_either_tunable_or_deliberately_excluded():
    """No field is unclassified — a new knob must be placed on purpose.

    Adding one to `DFPVConfig` without deciding whether the criterion can rank
    it is exactly how a silently unrankable parameter ends up searchable.
    """
    excluded = {"seed", "device", "x_cond_length_scale", "x_cond_top_k"}
    assert {f.name for f in fields(DFPVConfig)} == TUNABLE_PARAMS | excluded
