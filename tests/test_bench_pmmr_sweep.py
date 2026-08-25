"""The prepared-system split and the two-level search must change nothing.

`PMMREstimator.prepare` + `solve` exists purely to avoid recomputing the grams
across lambda candidates, and `tune.sweep` exists purely to make the search
exploit that. Neither is allowed to move a number: the claim under test
throughout is that a swept search evaluates the same candidates, produces the
same scores, and picks the same winner as the flat search it replaces.

Speed is verified separately (and is not asserted here — a timing assertion on
CI hardware is a flake, not a test).
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.methods import build_method
from benchmarks.methods import pmmr as pmmr_module
from benchmarks.methods.pmmr import SOLVE_ONLY_PARAMS, PMMRConfig, PMMREstimator
from benchmarks.tuning import TuningConfig, search, search_nested


def make_data(n=300, seed=0, d_x=2):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, 1))
    X = rng.normal(size=(n, d_x)) if d_x else np.zeros((n, 0))
    W = U + 0.5 * rng.normal(size=(n, 2))
    Z = U + 0.5 * rng.normal(size=(n, 2))
    A = (rng.uniform(size=n) < 1 / (1 + np.exp(-1.5 * U.ravel()))).astype(float)
    Y = 2 * A + 2 * U.ravel() + 0.3 * rng.normal(size=n)
    return X, A.reshape(-1, 1), Y, W, Z


# ─────────────────────────────────────────────────────────────────────────────
# prepare() / solve() against fit()
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("d_x", [0, 2])
def test_prepare_then_solve_is_bit_identical_to_fit(d_x):
    """Not merely close: `fit` *is* prepare+solve, so the bits must match."""
    X, A, Y, W, Z = make_data(d_x=d_x)
    cfg = PMMRConfig(lam1=0.1)

    direct = PMMREstimator(cfg).fit(X, A, Y, W=W, Z=Z)
    swept = PMMREstimator(cfg).prepare(X, A, Y, W=W, Z=Z).solve(lam1=0.1)

    assert np.array_equal(direct._alpha, swept._alpha)


def test_one_prepare_reproduces_every_lambda():
    """The point of the split: N solves off one prepare == N independent fits."""
    X, A, Y, W, Z = make_data()
    base = PMMRConfig()
    prepared = PMMREstimator(base).prepare(X, A, Y, W=W, Z=Z)

    for lam1 in (4.9e-6, 1e-3, 0.25):
        from dataclasses import replace

        want = PMMREstimator(replace(base, lam1=lam1))
        want.fit(X, A, Y, W=W, Z=Z)
        got = prepared.solve(lam1=lam1)
        assert np.array_equal(got._alpha, want._alpha)
        assert got.cfg.lam1 == lam1


def test_lam1_is_objective_lambda_at_the_current_fit_size(monkeypatch):
    """The split and refit must both assemble lambda * s^2 * L."""
    X, A, Y, W, Z = make_data(n=120)
    prepared = PMMREstimator(PMMRConfig()).prepare(X, A, Y, W=W, Z=Z)
    captured = []

    def capture(lhs, rhs, jitter, context=""):
        captured.append(lhs.copy())
        return np.zeros_like(rhs)

    monkeypatch.setattr(pmmr_module, "_solve_pd", capture)
    lam1 = 0.0123
    prepared.solve(lam1=lam1)

    assert len(captured) == len(prepared.blocks)
    for lhs, block in zip(captured, prepared.blocks):
        expected = block.LPhiL + lam1 * prepared.n**2 * block.L
        assert np.array_equal(lhs, expected)


def test_solve_omitting_lambda_keeps_the_prepared_value():
    X, A, Y, W, Z = make_data()
    cfg = PMMRConfig(lam1=0.25)
    prepared = PMMREstimator(cfg).prepare(X, A, Y, W=W, Z=Z)

    got = prepared.solve()
    assert got.cfg.lam1 == 0.25


def test_cached_validation_moment_kernel_preserves_score():
    """Caching fixed validation grams must change no candidate ranking."""
    X, A, Y, W, Z = make_data(n=120)
    prepared = PMMREstimator(PMMRConfig()).prepare(
        X[:80], A[:80], Y[:80], W=W[:80], Z=Z[:80]
    )
    est = prepared.solve()
    Xva, Ava, Yva, Wva, Zva = X[80:], A[80:], Y[80:], W[80:], Z[80:]

    direct = est.score(Xva, Ava, Yva, W=Wva, Z=Zva)
    cached_kernel = PMMREstimator.validation_moment_kernel(Xva, Ava, Zva)
    cached_bridge = prepared.bridge_gram(Ava, Wva, Xva)
    cached = est.score(
        Xva,
        Ava,
        Yva,
        W=Wva,
        Z=Zva,
        moment_kernel=cached_kernel,
        bridge_kernel=cached_bridge,
    )

    assert cached == direct


def test_solve_refuses_a_parameter_that_would_need_new_grams():
    """The silent-wrong-answer guard: cfg must never disagree with the grams."""
    X, A, Y, W, Z = make_data()
    prepared = PMMREstimator(PMMRConfig()).prepare(X, A, Y, W=W, Z=Z)

    for bad in ("kernel_scale", "kernel_scale_z"):
        with pytest.raises(ValueError, match="fresh prepare"):
            prepared.solve(**{bad: 2.0})

    assert SOLVE_ONLY_PARAMS == {"lam1"}


def test_estimators_from_one_prepare_do_not_share_alpha():
    """State is shared deliberately; the *fit* must not be."""
    X, A, Y, W, Z = make_data()
    prepared = PMMREstimator(PMMRConfig()).prepare(X, A, Y, W=W, Z=Z)

    a = prepared.solve(lam1=1e-3)
    b = prepared.solve(lam1=0.25)
    assert not np.array_equal(a._alpha, b._alpha)
    # ...while the read-only grams genuinely are the same objects.
    assert a._K_W_train is b._K_W_train
    # And a second solve has not disturbed the first.
    assert np.array_equal(a._alpha, prepared.solve(lam1=1e-3)._alpha)


# ─────────────────────────────────────────────────────────────────────────────
# TuningConfig
# ─────────────────────────────────────────────────────────────────────────────


def test_sweep_absent_is_the_old_behaviour():
    tc = TuningConfig.from_dict({"grid": {"lam1": [1.0, 2.0]}, "n_trials": 5})
    assert tc.sweep == {}
    assert tc.search_space == {"lam1": [1.0, 2.0]}


def test_overlapping_grid_and_sweep_is_refused():
    with pytest.raises(ValueError, match="both `grid` and `sweep`"):
        TuningConfig.from_dict({"grid": {"lam1": [1.0]}, "sweep": {"lam1": [2.0]}})


def test_search_space_merges_both_blocks():
    tc = TuningConfig.from_dict(
        {"grid": {"kernel_scale": [1.0]}, "sweep": {"lam1": [1.0, 2.0]}}
    )
    assert set(tc.search_space) == {"kernel_scale", "lam1"}


# ─────────────────────────────────────────────────────────────────────────────
# search_nested
# ─────────────────────────────────────────────────────────────────────────────


def test_search_nested_matches_flat_search_on_the_same_space():
    """The whole correctness claim, on a toy objective with no PMMR involved."""
    outer = [{"a": 1}, {"a": 2}]
    inner = [{"b": 10}, {"b": 20}, {"b": 30}]
    flat = [{**o, **i} for o in outer for i in inner]

    def objective(hp):
        return (hp["a"] - 2) ** 2 + (hp["b"] - 20) ** 2

    want = search(flat, objective)
    got = search_nested(
        outer, inner, prepare=lambda o: o, score=lambda o, i: objective({**o, **i})
    )

    assert got[0] == want[0]
    assert got[1] == want[1]
    assert got[2] == want[2]


def test_search_nested_prepares_once_per_outer_candidate():
    calls = []
    search_nested(
        [{"a": 1}, {"a": 2}],
        [{"b": i} for i in range(4)],
        prepare=lambda o: calls.append(o) or o,
        score=lambda o, i: 0.0,
    )
    assert len(calls) == 2  # not 8


def test_failed_prepare_sinks_only_its_own_sweep():
    """A dead outer candidate must not take the search down with it."""
    inner = [{"b": 1}, {"b": 2}]

    def prepare(o):
        return None if o["a"] == 1 else o

    best, best_score, results = search_nested(
        [{"a": 1}, {"a": 2}], inner, prepare, score=lambda o, i: float(i["b"])
    )
    assert best == {"a": 2, "b": 1}
    assert best_score == 1.0
    assert sum(1 for _, s in results if not np.isfinite(s)) == 2


# ─────────────────────────────────────────────────────────────────────────────
# End to end through the wrapper
# ─────────────────────────────────────────────────────────────────────────────


def _tune_block(**over):
    block = {"enabled": True, "val_frac": 0.25, "seed": 0}
    block.update(over)
    return block


def test_swept_search_selects_what_the_flat_search_would():
    """The config refactor is a re-spelling, not a different search.

    Same union of parameters, same candidates, same split (the split is seeded
    off `tune.seed`, not off the candidate list) — so the winner must agree.
    """
    X, A, Y, W, Z = make_data(n=400)
    grid = {"kernel_scale_z": [1.0, 2.7778]}
    lam1 = [4.9e-6, 1e-3, 0.25]

    _, flat = build_method(
        {
            "name": "PMMR",
            "cls": "PMMR",
            "strategy": "marginal",
            "tune": _tune_block(n_trials=99, grid={**grid, "lam1": lam1}),
        }
    )
    _, swept = build_method(
        {
            "name": "PMMR",
            "cls": "PMMR",
            "strategy": "marginal",
            "tune": _tune_block(n_trials=99, grid=grid, sweep={"lam1": lam1}),
        }
    )
    flat.fit(X, A, Y, W=W, Z=Z)
    swept.fit(X, A, Y, W=W, Z=Z)

    assert swept.model.cfg.lam1 == flat.model.cfg.lam1
    assert swept.model.cfg.kernel_scale_z == flat.model.cfg.kernel_scale_z
    assert np.array_equal(swept.predict(X, W=W), flat.predict(X, W=W))
    # Same candidate count and same winning score, not merely the same choice.
    assert (
        swept.diagnostics()["tune_n_candidates"]
        == flat.diagnostics()["tune_n_candidates"]
        == 6
    )
    assert swept.diagnostics()["tune_best_score"] == pytest.approx(
        flat.diagnostics()["tune_best_score"]
    )


def test_n_trials_budgets_the_grid_only():
    X, A, Y, W, Z = make_data(n=300)
    _, m = build_method(
        {
            "name": "PMMR",
            "cls": "PMMR",
            "tune": _tune_block(
                n_trials=2,
                grid={"kernel_scale": [0.5, 1.0, 2.0]},
                sweep={"lam1": [4.9e-6, 1e-3, 0.01, 0.25]},
            ),
        }
    )
    m.fit(X, A, Y, W=W, Z=Z)
    diag = m.diagnostics()
    assert diag["tune_n_prepared"] == 2  # n_trials, not 2 * 4
    assert diag["tune_n_candidates"] == 8  # n_trials * |sweep|


def test_diagnostics_report_swept_parameters_like_grid_ones():
    X, A, Y, W, Z = make_data(n=300)
    _, m = build_method(
        {
            "name": "PMMR",
            "cls": "PMMR",
            "tune": _tune_block(
                n_trials=99,
                grid={"kernel_scale": [1.0]},
                sweep={"lam1": [1e-3, 0.25]},
            ),
        }
    )
    m.fit(X, A, Y, W=W, Z=Z)
    diag = m.diagnostics()
    fracs = {k: v for k, v in diag.items() if k.startswith("tune_lam1_frac@")}
    assert set(fracs) == {"tune_lam1_frac@0.001", "tune_lam1_frac@0.25"}
    assert sum(fracs.values()) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Config validation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("block", ["grid", "sweep"])
def test_jitter_refused_in_either_block(block):
    with pytest.raises(ValueError, match="jitter"):
        build_method(
            {
                "name": "PMMR",
                "cls": "PMMR",
                "tune": _tune_block(**{block: {"jitter": [1e-10, 1e-8]}}),
            }
        )


@pytest.mark.parametrize("block", ["grid", "sweep"])
def test_unknown_key_refused_in_either_block(block):
    with pytest.raises(ValueError, match="unknown"):
        build_method(
            {
                "name": "PMMR",
                "cls": "PMMR",
                "tune": _tune_block(**{block: {"lam_typo": [1.0]}}),
            }
        )


def test_removed_lam2_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        build_method(
            {
                "name": "PMMR",
                "cls": "PMMR",
                "tune": _tune_block(sweep={"lam2": [1e-4]}),
            }
        )


@pytest.mark.parametrize("param", ["kernel_scale", "kernel_scale_z"])
def test_bandwidth_in_sweep_is_refused_with_the_fix(param):
    """It would be correct but slow, and the config would be claiming a saving."""
    with pytest.raises(ValueError, match=r"Move them to\s+`tune.grid`"):
        build_method(
            {
                "name": "PMMR",
                "cls": "PMMR",
                "tune": _tune_block(sweep={param: [1.0, 2.0]}),
            }
        )
