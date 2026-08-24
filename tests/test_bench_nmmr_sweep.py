"""Pausing a training run must be indistinguishable from stopping it.

`NMMREstimator.fit_trajectory` exists purely so a search over `n_epochs` costs
one training run instead of one run per value, and `tune.sweep` exists purely to
make the search exploit that. Neither is allowed to move a number, and the claim
is stronger than "close": the weights at milestone m are the weights a fresh
`fit` with `n_epochs=m` would have produced, bit for bit, because nothing in the
loop depends on where it stops.

That exactness is what the first group tests. The second group tests the search
built on it — same candidates, same scores, same winner as the flat grid it
replaces.

Speed is verified separately (and is not asserted here — a timing assertion on
CI hardware is a flake, not a test).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from benchmarks.methods import build_method
from benchmarks.methods.nmmr import SWEEP_ONLY_PARAMS, NMMRConfig, NMMREstimator


def make_data(n=200, seed=0, d_x=2):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, 1))
    X = rng.normal(size=(n, d_x)) if d_x else np.zeros((n, 0))
    W = U + 0.5 * rng.normal(size=(n, 2))
    Z = U + 0.5 * rng.normal(size=(n, 2))
    A = (rng.uniform(size=n) < 1 / (1 + np.exp(-1.5 * U.ravel()))).astype(float)
    Y = 2 * A + 2 * U.ravel() + 0.3 * rng.normal(size=n)
    return X, A.reshape(-1, 1), Y, W, Z


def _cfg(**over) -> NMMRConfig:
    # Small and fast; the claims under test are exactness claims, not fit
    # quality, so the network only has to be deterministic.
    base = NMMRConfig(
        hidden_dim=8, n_layers=1, n_epochs=4, batch_size=64, seed=7, device="cpu"
    )
    return replace(base, **over)


def _weights(est: NMMREstimator) -> list[torch.Tensor]:
    return [p.detach().clone() for p in est._model.parameters()]


def _same_weights(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


# ─────────────────────────────────────────────────────────────────────────────
# fit_trajectory against fit
# ─────────────────────────────────────────────────────────────────────────────


def test_milestone_weights_are_bit_identical_to_a_fit_that_stopped_there():
    """The whole correctness claim: a pause is a stop, exactly."""
    X, A, Y, W, Z = make_data()
    milestones = [1, 2, 5]

    got = {}
    run = NMMREstimator(_cfg())
    for epoch, m in run.fit_trajectory(X, A, Y, W=W, Z=Z, milestones=milestones):
        got[epoch] = _weights(m)

    assert sorted(got) == milestones
    for e in milestones:
        want = NMMREstimator(_cfg(n_epochs=e)).fit(X, A, Y, W=W, Z=Z)
        assert _same_weights(got[e], _weights(want)), f"diverged at epoch {e}"


def test_scoring_at_a_milestone_does_not_perturb_later_epochs():
    """The RNG-quiet requirement, tested rather than merely asserted.

    A score between yields runs a forward pass and flips the module out of
    train mode. If either consumed randomness or leaked into the optimiser, the
    tail of the run would drift away from an unscored one.
    """
    X, A, Y, W, Z = make_data()
    milestones = [1, 2, 6]

    scored = {}
    est = NMMREstimator(_cfg())
    for epoch, m in est.fit_trajectory(X, A, Y, W=W, Z=Z, milestones=milestones):
        m.score(X, A, Y, W=W, Z=Z)  # the perturbation under test
        scored[epoch] = _weights(m)

    quiet = {}
    est2 = NMMREstimator(_cfg())
    for epoch, m in est2.fit_trajectory(X, A, Y, W=W, Z=Z, milestones=milestones):
        quiet[epoch] = _weights(m)

    for e in milestones:
        assert _same_weights(scored[e], quiet[e])


def test_fit_is_a_fully_consumed_trajectory():
    """`fit` must not become a second training implementation."""
    X, A, Y, W, Z = make_data()
    cfg = _cfg(n_epochs=3)

    direct = NMMREstimator(cfg).fit(X, A, Y, W=W, Z=Z)

    via = NMMREstimator(cfg)
    last = None
    for _, m in via.fit_trajectory(X, A, Y, W=W, Z=Z, milestones=(3,)):
        last = _weights(m)

    assert _same_weights(_weights(direct), last)
    assert np.array_equal(direct.predict(X, W=W), via.predict(X, W=W))


def test_milestones_are_deduplicated_and_sorted():
    X, A, Y, W, Z = make_data()
    seen = [
        e
        for e, _ in NMMREstimator(_cfg()).fit_trajectory(
            X, A, Y, W=W, Z=Z, milestones=(5, 1, 5, 2)
        )
    ]
    assert seen == [1, 2, 5]


@pytest.mark.parametrize("bad", [(), (0,), (2, 0), (-1,)])
def test_degenerate_milestones_are_refused(bad):
    X, A, Y, W, Z = make_data()
    with pytest.raises(ValueError, match="epoch counts"):
        list(NMMREstimator(_cfg()).fit_trajectory(X, A, Y, W=W, Z=Z, milestones=bad))


# ─────────────────────────────────────────────────────────────────────────────
# End to end through the wrapper
# ─────────────────────────────────────────────────────────────────────────────


def _tune_block(**over):
    block = {"enabled": True, "val_frac": 0.25, "seed": 0}
    block.update(over)
    return block


def _method(**over):
    entry = {
        "name": "NMMR",
        "cls": "NMMR",
        "strategy": "paired",
        "hidden_dim": 8,
        "n_layers": 1,
        "batch_size": 64,
        "seed": 7,
        "device": "cpu",
    }
    entry.update(over)
    return build_method(entry)[1]


@pytest.mark.parametrize("kernel_width", ["fixed", "median"])
def test_swept_search_selects_what_the_flat_search_would(kernel_width):
    """The config refactor is a re-spelling, not a different search.

    Same union of parameters, same candidates, same split (the split is seeded
    off `tune.seed`, not off the candidate list) — so the winner must agree.
    """
    X, A, Y, W, Z = make_data(n=240)
    grid = {"lr": [1e-3, 5e-3]}
    epochs = [1, 3, 6]

    flat = _method(
        kernel_width=kernel_width,
        tune=_tune_block(n_trials=99, grid={**grid, "n_epochs": epochs}),
    )
    swept = _method(
        kernel_width=kernel_width,
        tune=_tune_block(n_trials=99, grid=grid, sweep={"n_epochs": epochs}),
    )

    flat.fit(X, A, Y, W=W, Z=Z)
    swept.fit(X, A, Y, W=W, Z=Z)

    assert swept.model.cfg.n_epochs == flat.model.cfg.n_epochs
    assert swept.model.cfg.lr == flat.model.cfg.lr
    assert np.array_equal(swept.predict(X, W=W), flat.predict(X, W=W))
    # Same candidate count and the same winning score, not merely the same
    # choice — the scores themselves have to match, since they are the reason
    # the choice is what it is.
    assert (
        swept.diagnostics()["tune_n_candidates"]
        == flat.diagnostics()["tune_n_candidates"]
        == 6
    )
    assert swept.diagnostics()["tune_best_score"] == pytest.approx(
        flat.diagnostics()["tune_best_score"]
    )


def test_n_trials_budgets_the_grid_only():
    X, A, Y, W, Z = make_data(n=200)
    m = _method(
        tune=_tune_block(
            n_trials=2,
            grid={"lr": [1e-3, 5e-3, 1e-2]},
            sweep={"n_epochs": [1, 2, 3, 4]},
        )
    )
    m.fit(X, A, Y, W=W, Z=Z)
    diag = m.diagnostics()
    assert diag["tune_n_prepared"] == 2  # training runs: n_trials, not 2 * 4
    assert diag["tune_n_candidates"] == 8  # scores: n_trials * |sweep|


def test_a_sweep_only_block_still_tunes():
    """The gate is `search_space`, not `grid` — a sweep alone is a search."""
    X, A, Y, W, Z = make_data(n=200)
    m = _method(n_epochs=1, tune=_tune_block(sweep={"n_epochs": [2, 5]}))
    m.fit(X, A, Y, W=W, Z=Z)

    assert m.model.cfg.n_epochs in (2, 5)  # not the configured default of 1
    assert m.diagnostics()["tune_n_candidates"] == 2


def test_diagnostics_report_swept_parameters_like_grid_ones():
    X, A, Y, W, Z = make_data(n=200)
    m = _method(
        tune=_tune_block(n_trials=99, grid={"lr": [1e-3]}, sweep={"n_epochs": [1, 4]})
    )
    m.fit(X, A, Y, W=W, Z=Z)

    fracs = {k: v for k, v in m.diagnostics().items() if k.startswith("tune_n_epochs_")}
    assert set(fracs) == {"tune_n_epochs_frac@1", "tune_n_epochs_frac@4"}
    assert sum(fracs.values()) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Config validation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("block", ["grid", "sweep"])
def test_loss_refused_in_either_block(block):
    """U- and V-statistic scores are not on a common scale, swept or gridded."""
    with pytest.raises(ValueError, match="not comparable"):
        _method(tune=_tune_block(**{block: {"loss": ["u_stat", "v_stat"]}}))


@pytest.mark.parametrize("block", ["grid", "sweep"])
def test_unknown_key_refused_in_either_block(block):
    with pytest.raises(ValueError, match="unknown"):
        _method(tune=_tune_block(**{block: {"hiden_dim": [8]}}))


@pytest.mark.parametrize("param", ["hidden_dim", "lr", "weight_decay"])
def test_trajectory_changing_parameter_in_sweep_is_refused_with_the_fix(param):
    """It would be correct but slow, and the config would be claiming a saving."""
    with pytest.raises(ValueError, match=r"Move\s+them to `tune.grid`"):
        _method(tune=_tune_block(sweep={param: [1.0, 2.0]}))

    assert SWEEP_ONLY_PARAMS == {"n_epochs"}


def test_prediction_side_knobs_are_not_sweepable():
    """Free to re-evaluate, but the selection criterion cannot see them.

    `score` is a moment loss on the bridge h(A,W,X); the W-marginalisation
    knobs never enter it, so a sweep would score every value identically and
    "select" the first. Refused for the same reason as the expensive ones,
    though the failure mode is the opposite one.
    """
    for param in ("w_strategy", "x_cond_kernel_scale", "x_cond_top_k"):
        with pytest.raises(ValueError, match="Move"):
            _method(tune=_tune_block(sweep={param: [1.0, 2.0]}))


@pytest.mark.parametrize("block", ["grid", "sweep"])
def test_kernel_width_is_method_level_not_searchable(block):
    with pytest.raises(ValueError, match="method-level"):
        _method(tune=_tune_block(**{block: {"kernel_width": ["fixed", "median"]}}))


@pytest.mark.parametrize("block", ["grid", "sweep"])
def test_kernel_scale_is_removed(block):
    with pytest.raises(ValueError, match="unknown"):
        _method(tune=_tune_block(**{block: {"kernel_scale": [1.0]}}))


# ─────────────────────────────────────────────────────────────────────────────
# kernel precomputation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("loss", ["u_stat", "v_stat"])
@pytest.mark.parametrize("kernel_width", ["fixed", "median"])
def test_precomputed_kernel_is_bit_identical_to_per_batch(loss, kernel_width):
    """The one-kernel-per-fit path may not move a number, only a clock.

    `K` depends on the data alone, so each minibatch's kernel is a submatrix of
    one n x n matrix; `kernel_precompute_max_n=0` forces the old per-batch
    recomputation. Bit-identical weights are the claim, and they also cover the
    subtler failure: the U-statistic zeroes K's diagonal, so a path that
    mutated the shared matrix in place would corrupt every later minibatch and
    show up here rather than as a silently worse fit.
    """
    X, A, Y, W, Z = make_data(n=200)

    fast = NMMREstimator(_cfg(loss=loss, kernel_width=kernel_width)).fit(
        X, A, Y, W=W, Z=Z
    )
    slow = NMMREstimator(
        _cfg(loss=loss, kernel_width=kernel_width, kernel_precompute_max_n=0)
    ).fit(X, A, Y, W=W, Z=Z)

    assert _same_weights(_weights(fast), _weights(slow))
    assert np.array_equal(fast.predict(X, W=W), slow.predict(X, W=W))


def test_the_cap_switches_paths_without_switching_results():
    """Crossing the memory cap is a time/memory trade, never a numerical one."""
    X, A, Y, W, Z = make_data(n=200)

    under = NMMREstimator(_cfg(kernel_precompute_max_n=200)).fit(X, A, Y, W=W, Z=Z)
    over = NMMREstimator(_cfg(kernel_precompute_max_n=199)).fit(X, A, Y, W=W, Z=Z)

    assert _same_weights(_weights(under), _weights(over))


def test_precomputation_does_not_perturb_the_minibatch_stream():
    """Nothing in the precompute draws, so milestones stay where they were."""
    X, A, Y, W, Z = make_data(n=200)

    ref = NMMREstimator(_cfg(n_epochs=3)).fit(X, A, Y, W=W, Z=Z)

    est = NMMREstimator(_cfg(n_epochs=5))
    seen = {}
    for epoch, model in est.fit_trajectory(X, A, Y, W=W, Z=Z, milestones=(3, 5)):
        seen[epoch] = _weights(model)

    assert _same_weights(seen[3], _weights(ref))


# ─────────────────────────────────────────────────────────────────────────────
# bandwidth: fixed reference width versus the median heuristic
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("d_x", [0, 2, 10, 30])
def test_median_heuristic_holds_the_kernel_open_in_every_dimension(d_x):
    """The point of the heuristic: one grid of multipliers, all dimensions.

    Under an absolute width the median off-diagonal decays as exp(-d/sigma²) —
    1e-3 by d = 8, 1e-10 by d = 23 — so K becomes the identity and the moment
    condition is silently switched off. The heuristic pins that median at
    exp(-1/2) instead, by construction, whatever d is.
    """
    X, A, Y, W, Z = make_data(n=200, d_x=d_x)
    est = NMMREstimator(_cfg()).fit(X, A, Y, W=W, Z=Z)

    assert est.diagnostics_["kernel_dim"] == 1 + 2 + d_x
    assert est.diagnostics_["kernel_offdiag_median"] == pytest.approx(
        np.exp(-0.5), rel=0.05
    )


def test_fixed_width_is_exactly_one():
    X, A, Y, W, Z = make_data(n=200)

    est = NMMREstimator(_cfg(kernel_width="fixed")).fit(X, A, Y, W=W, Z=Z)

    assert est.diagnostics_["kernel_length_scale"] == 1.0


@pytest.mark.parametrize("kernel_width", ["fixed", "median"])
def test_kernel_diagnostics_describe_diagonal_and_offdiagonal(kernel_width):
    X, A, Y, W, Z = make_data(n=200)

    est = NMMREstimator(_cfg(kernel_width=kernel_width)).fit(X, A, Y, W=W, Z=Z)
    diag = est.diagnostics_

    assert diag["kernel_diag_mean"] == pytest.approx(1.0)
    assert diag["kernel_diag_std"] == pytest.approx(0.0)
    assert diag["kernel_offdiag_min"] <= diag["kernel_offdiag_p25"]
    assert diag["kernel_offdiag_p25"] <= diag["kernel_offdiag_median"]
    assert diag["kernel_offdiag_median"] <= diag["kernel_offdiag_p75"]
    assert diag["kernel_offdiag_p75"] <= diag["kernel_offdiag_max"]

    if kernel_width == "fixed":
        assert diag["kernel_length_scale"] == 1.0
    else:
        assert diag["kernel_length_scale"] > 0.0


def test_median_width_matches_the_fit_kernel_inputs():
    X, A, Y, W, Z = make_data(n=200)
    est = NMMREstimator(_cfg(kernel_width="median")).fit(X, A, Y, W=W, Z=Z)

    k_input = est._kernel_input(
        est._as_2d_tensor(A, name="A"),
        est._as_2d_tensor(Z, name="Z"),
        est._as_2d_tensor(X, name="X"),
    )
    sq_dist = (k_input.T.unsqueeze(2) - k_input.T.unsqueeze(1)).pow(2).sum(dim=0)
    iu = torch.triu_indices(sq_dist.shape[0], sq_dist.shape[1], offset=1)
    expected = float(torch.sqrt(sq_dist[iu[0], iu[1]].median()))

    assert est.diagnostics_["kernel_length_scale"] == pytest.approx(expected)


def test_bandwidth_does_not_depend_on_the_batch_size():
    """sigma is a property of the data; resolving it per minibatch would make
    the objective a function of an optimisation knob."""
    X, A, Y, W, Z = make_data(n=200)

    wide = NMMREstimator(_cfg(batch_size=200)).fit(X, A, Y, W=W, Z=Z)
    narrow = NMMREstimator(_cfg(batch_size=32)).fit(X, A, Y, W=W, Z=Z)

    assert wide.diagnostics_["kernel_length_scale"] == pytest.approx(
        narrow.diagnostics_["kernel_length_scale"]
    )


def test_score_uses_the_selected_width_variant():
    """Validation scoring follows the method-level width choice."""
    X, A, Y, W, Z = make_data(n=200)
    Xv, Av, Yv, Wv, Zv = make_data(n=120, seed=11)

    est = NMMREstimator(_cfg()).fit(X, A, Y, W=W, Z=Z)
    baseline = est.score(Xv, Av, Yv, W=Wv, Z=Zv)

    est.cfg = replace(est.cfg, kernel_width="fixed")
    fixed_score = est.score(Xv, Av, Yv, W=Wv, Z=Zv)

    assert fixed_score != baseline
