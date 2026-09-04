"""Tests for the rewritten P-learner (`benchmarks/methods/plearner.py`).

The staged rewrite means each step should land with the check that pins *its*
contribution. Step 1 is the driver — cross-fitting, arm indexing, the score's
sign, and the one-fit-many-final-models contract — so these tests are about
plumbing, not about statistical quality. There is nothing here that a real
bridge would pass and a broken one would not; that arrives with the bridges.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

import benchmarks.methods as benchmark_methods
from benchmarks.methods import build_method
from benchmarks.methods.plearner import (
    BRIDGES,
    KernelBridge,
    PLearner,
    _median_bandwidth,
    _MomentSystem,
    _stratified_split,
    _NystromFeatures,
    build_bridge,
    proximal_scores,
)


def _toy(n=400, seed=0, dx=3):
    """A confounded proximal draw. Deliberately not identifiable by design —
    these tests never ask for the right answer, only for the right arithmetic."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=n)
    X = rng.normal(size=(n, dx))
    Z = U + 0.5 * rng.normal(size=n)
    W = U + 0.5 * rng.normal(size=n)
    tau = 1.0 + X[:, 0]
    p = 1.0 / (1.0 + np.exp(-(1.5 * U + 0.3 * X[:, 0])))
    A = (rng.random(n) < p).astype(float)
    Y = X[:, 0] + 2.0 * U + 0.5 * rng.normal(size=n) + A * tau
    return X, A, Y, W[:, None], Z[:, None], tau


# ── the score ────────────────────────────────────────────────────────────────


def test_proximal_scores_matches_the_paper_formula_by_hand():
    # Two rows, one per arm, with h and q chosen so the answer is checkable.
    A = np.array([1.0, 0.0])
    Y = np.array([10.0, 4.0])
    h = np.array([[1.0, 3.0], [2.0, 5.0]])  # columns: arm 0, arm 1
    q = np.array([[7.0, 2.0], [4.0, 9.0]])

    got = proximal_scores(A, Y, h, q)

    # row 0 (A=1): +1 * q[0,1] * (10 - h[0,1]) + h[0,1] - h[0,0]
    #            = 2 * (10 - 3) + 3 - 1 = 16
    # row 1 (A=0): -1 * q[1,0] * (4  - h[1,0]) + h[1,1] - h[1,0]
    #            = -4 * (4 - 2) + 5 - 2 = -5
    assert got == pytest.approx([16.0, -5.0])


def test_q_clip_caps_magnitude_not_a_symmetric_window():
    A = np.array([1.0])
    Y = np.array([1.0])
    h = np.zeros((1, 2))
    q = np.array([[0.0, -50.0]])  # a large *negative* q on the observed arm

    # A symmetric clip to [-2, 2] and a magnitude clip at 2 agree here; the point
    # is that the sign survives and the magnitude does not.
    assert proximal_scores(A, Y, h, q, q_clip=2.0) == pytest.approx([-2.0])


def test_cross_fitting_is_out_of_fold():
    """Every row must be scored by a bridge that did not see it."""
    X, A, Y, W, Z, _ = _toy(n=300)

    pl = PLearner(
        bridge="kernel",
        bridge_params={"rank": 32},
        final_models={"m": DummyRegressor()},
        n_splits=3,
    ).fit(X, A, Y, W=W, Z=Z)

    assert len(pl.bridges_) == 3
    assert pl.scores_.shape == (300,)
    assert np.isfinite(pl.scores_).all()
    # Distinct fold means: a single bridge reused everywhere would give one value.
    assert len({round(float(b.h_scaler_.mean_[0]), 9) for b in pl.bridges_}) == 3


def test_folds_are_stratified_so_no_arm_is_lost():
    """An unstratified split drops an arm on imbalanced episodes and crashes."""
    rng = np.random.default_rng(1)
    n = 200
    X = rng.normal(size=(n, 2))
    A = np.zeros(n)
    A[:12] = 1.0  # 6% treated
    Y = rng.normal(size=n)
    W = rng.normal(size=(n, 1))
    Z = rng.normal(size=(n, 1))

    pl = PLearner(
        bridge="kernel",
        bridge_params={"rank": 32},
        final_models={"m": DummyRegressor()},
        n_splits=5,
    ).fit(X, A, Y, W=W, Z=Z)
    assert np.isfinite(pl.scores_).all()


def test_one_fit_serves_many_final_models():
    """The point of the rewrite: N learners, one cross-fit of the bridges."""
    X, A, Y, W, Z, _ = _toy()

    pl = PLearner(
        bridge="kernel",
        bridge_params={"rank": 64},
        final_models={"rf": DummyRegressor(), "ols": LinearRegression()},
        n_splits=3,
    ).fit(X, A, Y, W=W, Z=Z)

    out = pl.predict_variants(X)
    assert set(out) == {"rf", "ols"}
    assert all(v.shape == (len(X),) for v in out.values())
    # Both were trained on the same scores, so they are the same problem.
    assert len(pl.bridges_) == 3
    # And they are genuinely different fits, not one aliased twice.
    assert not np.allclose(out["rf"], out["ols"])


def test_prediction_cv_tunes_ridge_and_rf_from_one_plearner_fit():
    """Both final regressions search their config grids after one bridge fit."""
    X, A, Y, W, Z, _ = _toy(n=240)
    cfg = {
        "name": "P-Kernel",
        "cls": "PLearner",
        "bridge": "kernel",
        "bridge_params": {"rank": 24},
        "n_splits": 2,
        "base_cv": {
            "folds": 3,
            "seed": 7,
            "grids": {
                "linear": {"alpha": [0.01, 1.0]},
                "rf": {
                    "n_estimators": [10],
                    "max_depth": [2, 4],
                    "min_samples_leaf": [1, 5],
                },
            },
        },
        "outputs": [
            {"name": "ridge", "base": "linear"},
            {"name": "forest", "base": "rf"},
        ],
    }

    _, wrapped = build_method(cfg)
    wrapped.fit(X, A, Y, W=W, Z=Z)
    models = wrapped.learner.final_models_

    assert set(models) == {"ridge", "forest"}
    assert all(isinstance(model, GridSearchCV) for model in models.values())
    assert isinstance(models["ridge"].best_estimator_, Pipeline)
    assert isinstance(models["ridge"].best_estimator_["model"], Ridge)
    assert models["ridge"].best_params_["model__alpha"] in {0.01, 1.0}
    assert models["forest"].best_params_["max_depth"] in {2, 4}
    assert set(wrapped.predict_variants(X[:11])) == {"ridge", "forest"}
    diagnostics = wrapped.diagnostics()
    assert diagnostics["base_cv/ridge/best_index"] in {0.0, 1.0}
    assert diagnostics["base_cv/ridge/alpha"] in {0.01, 1.0}
    assert diagnostics["base_cv/ridge/mse"] >= 0.0
    assert diagnostics["base_cv/forest/max_depth"] in {2.0, 4.0}
    assert diagnostics["base_cv/forest/mse"] >= 0.0
    assert wrapped.tuning_time() > 0.0


def test_base_cv_is_reused_by_s_learner():
    """The CV layer is not P-learner-specific."""
    X, A, Y, W, Z, _ = _toy(n=120)
    cfg = {
        "name": "S-Ridge",
        "cls": "S_Learner",
        "base": "linear",
        "use_W": False,
        "use_Z": False,
        "base_cv": {
            "folds": 2,
            "grids": {"linear": {"alpha": [0.1, 1.0]}},
        },
    }

    _, learner = build_method(cfg)
    learner.fit(X, A, Y, W=W, Z=Z)

    assert isinstance(learner.model, GridSearchCV)
    assert learner.model.best_params_["model__alpha"] in {0.1, 1.0}
    assert learner.tuning_time() > 0.0


@pytest.mark.parametrize("cls", ["S_Learner", "T_Learner", "X_Learner"])
def test_meta_learners_accumulate_prediction_cv_tuning_time(cls):
    X, A, Y, W, Z, _ = _toy(n=120)
    cfg = {
        "name": cls,
        "cls": cls,
        "base": "linear",
        "use_W": False,
        "use_Z": False,
        "base_cv": {
            "folds": 2,
            "grids": {"linear": {"alpha": [0.1, 1.0]}},
        },
    }

    _, learner = build_method(cfg)
    learner.fit(X, A, Y, W=W, Z=Z)

    assert learner.tuning_time() > 0.0


@pytest.mark.parametrize("cls", ["S_Learner", "T_Learner", "X_Learner"])
def test_untuned_meta_learners_report_zero_tuning_time(cls):
    X, A, Y, W, Z, _ = _toy(n=120)
    cfg = {
        "name": cls,
        "cls": cls,
        "base": "linear",
        "use_W": False,
        "use_Z": False,
    }

    _, learner = build_method(cfg)
    learner.fit(X, A, Y, W=W, Z=Z)

    assert learner.tuning_time() == 0.0


@pytest.mark.parametrize(
    "cls, expected_fits",
    [("S_Learner", 1), ("T_Learner", 2), ("X_Learner", 4)],
)
def test_meta_learner_tuning_time_sums_every_searched_component(
    cls, expected_fits, monkeypatch
):
    X, A, Y, W, Z, _ = _toy(n=120)

    def fit_with_fixed_search_cost(estimator, features, target):
        estimator.fit(features, target)
        return 0.25

    monkeypatch.setattr(
        benchmark_methods, "fit_and_measure_tuning", fit_with_fixed_search_cost
    )
    _, learner = build_method(
        {
            "name": cls,
            "cls": cls,
            "base": "linear",
            "use_W": False,
            "use_Z": False,
        }
    )
    learner.fit(X, A, Y, W=W, Z=Z)

    assert learner.tuning_time() == 0.25 * expected_fits


def test_base_cv_rejects_unknown_parameters_early():
    cfg = {
        "name": "bad",
        "cls": "S_Learner",
        "base": "linear",
        "base_cv": {"grids": {"linear": {"not_a_ridge_parameter": [1]}}},
    }
    with pytest.raises(ValueError, match="unknown linear base_cv parameters"):
        build_method(cfg)


def test_predict_ate_broadcasts_the_scalar():
    X, A, Y, W, Z, _ = _toy()
    pl = PLearner(
        bridge="kernel",
        bridge_params={"rank": 64},
        final_models={"m": DummyRegressor()},
    ).fit(X, A, Y, W=W, Z=Z)

    got = pl.predict_ate(X[:7])
    assert got.shape == (7,)
    assert np.allclose(got, pl.ate_)


# ── interface guarantees the bridges rely on ─────────────────────────────────


def test_bridges_return_both_arms():
    X, A, Y, W, Z, _ = _toy(n=120)
    for name in BRIDGES:
        b = build_bridge(name).fit(X, A, Y, W=W, Z=Z)
        assert b.predict_h(X, W).shape == (120, 2), name
        assert b.predict_q(X, Z).shape == (120, 2), name


def test_only_kernel_bridge_is_registered():
    assert set(BRIDGES) == {"kernel"}


def test_binary_treatment_is_enforced():
    X, A, Y, W, Z, _ = _toy(n=80)
    pl = PLearner(
        bridge="kernel",
        bridge_params={"rank": 32},
        final_models={"m": DummyRegressor()},
    )
    with pytest.raises(ValueError, match="binary"):
        pl.fit(X, A + 1.0, Y, W=W, Z=Z)


def test_proxies_are_required():
    X, A, Y, W, Z, _ = _toy(n=80)
    pl = PLearner(
        bridge="kernel",
        bridge_params={"rank": 32},
        final_models={"m": DummyRegressor()},
    )
    with pytest.raises(ValueError, match="proxies"):
        pl.fit(X, A, Y, W=W, Z=None)


def test_unknown_bridge_names_the_registry():
    with pytest.raises(ValueError, match="unknown bridge"):
        build_bridge("does-not-exist")


# ── toy data for kernel-bridge tests ─────────────────────────────────────────


def _linear_toy(n=4000, seed=0, dx=2, dp=2):
    """A proximal draw whose *outcome* mechanism is exactly linear in (U, X).

    The outcome mechanism is deliberately linear while the treatment mechanism is
    logistic. This gives the kernel bridge a simple reference problem without
    introducing a separate linear bridge implementation.

    Proxy widths are equal (`dp`) so both moment systems are square, and the two
    invariants below hold to solver precision rather than approximately.

    Note `dp=2` against a *scalar* U: the proxies are deliberately over-complete,
    which makes the moment matrix rank-deficient. That is the common case in the
    prior (d_U, d_W, d_Z are drawn independently), and it is what makes the
    bridge's damping load-bearing rather than cosmetic.
    """
    rng = np.random.default_rng(seed)
    U = rng.normal(size=n)
    X = rng.normal(size=(n, dx))

    Z = 0.7 * U[:, None] + rng.normal(size=(n, dp))
    W = 1.3 * U[:, None] + rng.normal(size=(n, dp))

    p = 1.0 / (1.0 + np.exp(-(1.5 * U + 0.3 * X[:, 0])))
    A = (rng.random(n) < p).astype(float)

    tau = 1.0 + X[:, 0]  # E[tau(X)] = 1
    Y = 0.5 + X @ np.array([0.4, -0.8])[:dx] + 2.0 * U + A * tau
    Y = Y + 0.3 * rng.normal(size=n)
    return X, A, Y, W, Z, tau


# ── the kernel bridge ────────────────────────────────────────────────────────


def _rbf(P, Q, sigma):
    d2 = (
        np.sum(P**2, axis=1)[:, None]
        + np.sum(Q**2, axis=1)[None, :]
        - 2.0 * P @ Q.T
    )
    return np.exp(-np.maximum(d2, 0.0) / (2.0 * sigma**2))


def test_nystrom_with_every_row_as_landmark_reproduces_the_kernel_exactly():
    """`rank=None` must be the textbook bridge, not an approximation of it.

    That is what lets one implementation serve as both the fast path and its own
    reference; if this drifts, every comparison against `rank=None` is void.
    """
    rng = np.random.default_rng(0)
    P = rng.normal(size=(60, 4))

    feats = _NystromFeatures(sigma=1.7).fit(P, np.arange(60))
    Phi = feats.transform(P)

    assert np.allclose(Phi @ Phi.T, _rbf(P, P, 1.7), atol=1e-8)


def test_feature_transform_is_independent_of_chunking():
    """Chunking is a memory device; it must not be a numerical one."""
    rng = np.random.default_rng(1)
    P = rng.normal(size=(200, 3))
    feats = _NystromFeatures(sigma=1.0, chunk=7).fit(P, np.arange(0, 200, 4))

    small = feats.transform(P)
    feats.chunk = 10_000
    assert np.allclose(small, feats.transform(P), atol=1e-12)


def test_nystrom_truncates_a_rank_deficient_landmark_set():
    """Duplicated landmarks make K_rr singular; the whitening must truncate
    rather than divide by a zero eigenvalue."""
    rng = np.random.default_rng(2)
    P = np.repeat(rng.normal(size=(5, 2)), 4, axis=0)  # 20 rows, 5 distinct

    feats = _NystromFeatures(sigma=1.0).fit(P, np.arange(20))

    assert feats.rank_ <= 5
    assert np.isfinite(feats.transform(P)).all()


def test_median_bandwidth_tracks_the_true_median():
    """Subsampled because the exact median needs an n x n distance matrix — the
    one object this bridge exists to avoid. It must still be the right number."""
    rng = np.random.default_rng(3)
    P = rng.normal(size=(400, 5))

    exact = np.median(
        np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)[~np.eye(400, dtype=bool)]
    )
    got = _median_bandwidth(P, np.random.default_rng(4), max_pairs=4000)

    assert got == pytest.approx(exact, rel=0.05)


def test_kernel_bridge_recovers_the_ate_under_linear_mechanisms():
    """The RKHS contains the linear functions, so it should handle this
    linear-mechanism reference problem.

    `lam` is set for this draw's size rather than left at the default: the
    optimum moves with n, which is precisely the defect tuning has to fix.
    """
    X, A, Y, W, Z, _ = _linear_toy(n=6000)

    pl = PLearner(
        bridge="kernel",
        bridge_params={"rank": 256, "lam_h": 1e-6, "lam_q": 1e-6},
        final_models={"ols": LinearRegression()},
        n_splits=3,
        q_clip=20.0,
    ).fit(X, A, Y, W=W, Z=Z)

    assert pl.ate_ == pytest.approx(1.0, abs=0.15)
    assert pl.final_models_["ols"].coef_[0] == pytest.approx(1.0, abs=0.20)


def test_kernel_bridge_landmarks_are_reproducible():
    X, A, Y, W, Z, _ = _linear_toy(n=600)
    kw = dict(rank=64, random_state=7)

    a = KernelBridge(**kw).fit(X, A, Y, W=W, Z=Z)
    b = KernelBridge(**kw).fit(X, A, Y, W=W, Z=Z)

    assert np.allclose(a.predict_h(X, W), b.predict_h(X, W))
    assert np.allclose(a.predict_q(X, Z), b.predict_q(X, Z))


def test_kernel_bridge_uses_a_separate_bandwidth_per_block():
    """The two blocks have different widths, so they need different sigmas.

    Raw scale is not the reason — standardization removes that. Dimension is:
    the median pairwise distance grows like sqrt(d), and d_W and d_Z are drawn
    independently in 1..10, so one shared sigma over-smooths the wider block.
    """
    X, A, Y, W, Z, _ = _linear_toy(n=600, dp=8)
    Z = Z[:, :1]  # (W, X) is 10 columns wide, (Z, X) is 3

    b = KernelBridge(rank=64).fit(X, A, Y, W=W, Z=Z)

    assert b.h_feats_.sigma > 1.5 * b.q_feats_.sigma


def test_kernel_bridge_stays_within_its_rank_budget():
    """The memory contract. Every intermediate in `fit` is bounded by the
    feature widths and the landmark count, so pinning those pins the claim that
    nothing is n x n."""
    X, A, Y, W, Z, _ = _linear_toy(n=2000)
    rank = 64

    b = KernelBridge(rank=rank, chunk=256).fit(X, A, Y, W=W, Z=Z)

    assert b.h_feats_.rank_ <= rank
    assert b.q_feats_.rank_ <= rank
    assert b.h_feats_.landmarks_.shape[0] == rank
    assert b.q_feats_.landmarks_.shape[0] == rank
    assert b.theta_.shape == (2, b.h_feats_.rank_)
    assert b.gamma_.shape == (2, b.q_feats_.rank_)
    assert b.predict_h(X, W).shape == (2000, 2)


# ── step 4: tuning the regularizers on held-out moment loss ──────────────────


def test_moment_loss_is_the_training_objective_without_the_penalty():
    """The load-bearing identity behind the whole tuner.

    `solve_h` minimises `loss_h(theta) + lam_h ||theta||^2` over theta, so the
    selection score is the fit's own objective with only the penalty dropped.
    Checked by perturbing the solution in random directions: the *penalised*
    value must be a minimum, while the unpenalised `loss_h` alone must not be
    (its minimiser is a different, less damped, theta).
    """
    X, A, Y, W, Z, _ = _linear_toy(n=1500)
    b = KernelBridge(rank=32).fit(X, A, Y, W=W, Z=Z)
    sys_ = _MomentSystem(
        b.h_feats_.transform(b.h_scaler_.transform(np.column_stack([W, X]))),
        b.q_feats_.transform(b.q_scaler_.transform(np.column_stack([Z, X]))),
        A,
        Y,
        b.lam_g,
    )
    lam = 1e-3
    theta = sys_.solve_h(0, lam)
    obj = lambda t: sys_.loss_h(0, t) + lam * float(t @ t)

    rng = np.random.default_rng(0)
    for _ in range(20):
        d = rng.normal(size=theta.shape)
        d *= 1e-3 * np.linalg.norm(theta) / np.linalg.norm(d)
        assert obj(theta + d) > obj(theta)


def test_residual_mse_and_moment_loss_rank_the_bridge_oppositely():
    """The trap the criterion exists to avoid, stated as sharply as it can be.

    Tuning `h` on held-out `mean (Y - h)^2` does not merely pick a different
    damping — it targets a different function: the *regression* of Y on the
    outcome proxy, which under confounding is not the bridge. So take the two
    candidates directly. The regression beats the bridge on MSE (it is the least
    squares fit, so this is by construction) and the bridge beats the regression
    on moment loss. Every ordering an MSE-based tuner would produce is therefore
    the wrong way round, at every grid point.
    """
    X, A, Y, W, Z, _ = _linear_toy(n=2000)
    b = KernelBridge(rank=64).fit(X, A, Y, W=W, Z=Z)

    Phi_H = b.h_feats_.transform(b.h_scaler_.transform(np.column_stack([W, X])))
    sys_ = _MomentSystem(
        Phi_H,
        b.q_feats_.transform(b.q_scaler_.transform(np.column_stack([Z, X]))),
        A,
        Y,
        b.lam_g,
    )
    arm = A == 0
    theta_bridge = sys_.solve_h(0, b.lam_h)
    theta_reg = np.linalg.lstsq(Phi_H[arm], Y[arm], rcond=None)[0]
    mse = lambda t: float(np.mean((Y[arm] - Phi_H[arm] @ t) ** 2))

    assert mse(theta_reg) < mse(theta_bridge)
    assert sys_.loss_h(0, theta_bridge) < sys_.loss_h(0, theta_reg)


def test_the_two_lambda_searches_are_independent():
    """`theta` sees only `lam_h` and `gamma` only `lam_q`, which is what turns a
    grid of size |g|^2 into two 1-D searches of total size 2|g|. If this ever
    couples, `_select_lambdas` is silently searching the wrong space."""
    X, A, Y, W, Z, _ = _linear_toy(n=1200)
    base = dict(rank=32, random_state=0)

    a = KernelBridge(lam_h=1e-3, lam_q=1e-2, **base).fit(X, A, Y, W=W, Z=Z)
    b = KernelBridge(lam_h=1e-3, lam_q=1e-6, **base).fit(X, A, Y, W=W, Z=Z)
    c = KernelBridge(lam_h=1e-6, lam_q=1e-2, **base).fit(X, A, Y, W=W, Z=Z)

    assert np.allclose(a.theta_, b.theta_)  # lam_q does not move h
    assert np.allclose(a.gamma_, c.gamma_)  # lam_h does not move q
    assert not np.allclose(a.theta_, c.theta_)


def test_tuning_selects_from_the_grid_and_records_both_curves():
    X, A, Y, W, Z, _ = _linear_toy(n=2000)
    grid = [1e-7, 1e-5, 1e-3, 1e-1]

    b = KernelBridge(rank=64, lam_grid=grid).fit(X, A, Y, W=W, Z=Z)

    assert b.lam_h_ in grid and b.lam_q_ in grid
    assert b.tuning_time_ > 0.0
    for key in ("lam_h", "lam_q"):
        curve = b.lam_scores_[key]
        assert [lam for lam, _ in curve] == sorted(grid)
        assert all(s >= 0.0 for _, s in curve)  # a squared norm
        assert min(curve, key=lambda p: p[1])[0] == getattr(b, key + "_")


def test_tuning_off_by_default_and_bit_identical_to_the_fixed_lambdas():
    """`lam_grid=None` must not perturb a single digit — the refactor that
    introduced `_MomentSystem` has to be invisible to existing configs."""
    X, A, Y, W, Z, _ = _linear_toy(n=1500)

    b = KernelBridge(rank=64, lam_h=3e-4, lam_q=7e-3).fit(X, A, Y, W=W, Z=Z)

    assert b.lam_h_ == 3e-4 and b.lam_q_ == 7e-3
    assert b.tuning_time_ == 0.0
    assert not hasattr(b, "lam_scores_")


def test_tuning_split_keeps_both_arms_on_a_lopsided_episode():
    """A plain permutation loses an arm here and `_MomentSystem` then raises
    mid-search; the split is stratified so it cannot."""
    rng = np.random.default_rng(0)
    A = np.zeros(300)
    A[:12] = 1.0  # 4% treated
    tr, va = _stratified_split(A, 0.25, rng)

    assert sorted(np.concatenate([tr, va])) == list(range(300))
    for idx in (tr, va):
        assert set(np.unique(A[idx])) == {0.0, 1.0}


def test_tuner_damps_harder_when_given_less_data():
    """The behaviour that justifies tuning at all: a single constant cannot be
    right at every n, and the criterion moves in the statistically correct
    direction — fewer rows, more damping."""
    grid = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    picks = {}
    for n in (400, 6000):
        X, A, Y, W, Z, _ = _linear_toy(n=n)
        picks[n] = KernelBridge(rank=64, lam_grid=grid).fit(X, A, Y, W=W, Z=Z).lam_h_

    assert picks[400] > picks[6000]
