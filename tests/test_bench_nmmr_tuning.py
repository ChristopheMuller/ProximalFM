"""NMMR's selection criterion: each variant is validated on its own statistic.

The bug this pins was silent. ``score`` used to default to the V-statistic
whatever the model was trained on, so a ``NMMR-U`` entry trained on the
U-statistic was ranked by a criterion carrying V's diagonal term — exactly
``MSE/n``, i.e. an ordinary-regression preference over ``(A, W, X)``, the
confounded fit the method exists to avoid. Nothing in the loss curve or the
tuning diagnostics showed it.
"""

import numpy as np
import pytest

from benchmarks.methods import NMMR_Wrapper
from benchmarks.methods.nmmr import NMMRConfig, NMMREstimator


def _episode(n=96, d_X=3, d_Z=2, d_W=2, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, 1))
    X = rng.normal(size=(n, d_X))
    Z = U + rng.normal(size=(n, d_Z))
    W = U + rng.normal(size=(n, d_W))
    A = (rng.random((n, 1)) < 0.5).astype(np.float64)
    Y = (A[:, 0] + U[:, 0] + X[:, 0] + rng.normal(size=n) * 0.1).astype(np.float64)
    return X, A, Y, W, Z


def _fit(loss):
    X, A, Y, W, Z = _episode()
    cfg = NMMRConfig(
        n_epochs=1, hidden_dim=8, n_layers=1, device="cpu", loss=loss, seed=0
    )
    return NMMREstimator(cfg).fit(X, A, Y, W=W, Z=Z), (X, A, Y, W, Z)


@pytest.mark.parametrize("loss", ["u_stat", "v_stat"])
def test_score_defaults_to_the_models_own_statistic(loss):
    est, (X, A, Y, W, Z) = _fit(loss)

    assert est.score(X, A, Y, W=W, Z=Z) == est.score(X, A, Y, W=W, Z=Z, loss=loss)


def test_u_and_v_scores_differ_so_they_could_not_be_ranked_together():
    """Why `loss` is refused in the grid: the two scores are not on one scale."""
    est, (X, A, Y, W, Z) = _fit("u_stat")

    u = est.score(X, A, Y, W=W, Z=Z, loss="u_stat")
    v = est.score(X, A, Y, W=W, Z=Z, loss="v_stat")
    assert u != pytest.approx(v)


def test_loss_in_the_tune_grid_is_refused():
    with pytest.raises(ValueError, match=r"`loss` may not appear in `tune.grid`"):
        NMMR_Wrapper(
            device="cpu",
            tune={"enabled": True, "grid": {"loss": ["u_stat", "v_stat"]}},
        )


def test_loss_is_settable_as_a_method_level_key():
    est = NMMR_Wrapper(device="cpu", loss="v_stat")
    assert est._base_cfg.loss == "v_stat"


def test_tuning_ranks_candidates_on_the_configured_statistic(monkeypatch):
    """The tuned model keeps its variant's loss, and every candidate is scored
    with it — not with the old hardcoded V-statistic."""
    X, A, Y, W, Z = _episode()

    seen: list[str] = []
    real_score = NMMREstimator.score

    def spy(self, *args, loss=None, **kwargs):
        seen.append(loss if loss is not None else self.cfg.loss)
        return real_score(self, *args, loss=loss, **kwargs)

    monkeypatch.setattr(NMMREstimator, "score", spy)

    est = NMMR_Wrapper(
        device="cpu",
        loss="u_stat",
        n_epochs=1,
        hidden_dim=8,
        n_layers=1,
        tune={"enabled": True, "n_trials": 2, "grid": {"hidden_dim": [4, 8]}},
    )
    est.fit(X, A, Y, W=W, Z=Z)

    assert seen and set(seen) == {"u_stat"}
    assert est.model.cfg.loss == "u_stat"
