"""Tests for the benchmark figures — ``benchmarks/plot_frames.py`` and ``plots.py``.

Pictures are not asserted on. What is pinned is the layer *underneath* them,
where a silent mistake is invisible afterwards:

* the paired statistics run on a **complete block** — an episode any method
  missed is dropped for everyone, or a mean rank compares methods scored on
  different sets of episodes;
* **reference rows never rank** — an oracle taking rank 1 would shift every
  competitor by one;
* the level/shape decomposition is the **exact** identity
  ``nPEHE² = b² + (1 − shape R²)``, trimming included, since two figures are
  drawn as though it holds;
* every figure survives a degenerate run (one episode, two methods, a failed
  method) rather than taking the whole sheet down with it.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch

matplotlib.use("Agg")  # before benchmarks.plots pulls in pyplot

from benchmarks import plots
from benchmarks.metrics import TASKS, score_method
from benchmarks.plot_frames import BenchmarkRun, select_methods, terciles
from benchmarks.plots import _decomposition

CATE, ITE, ATE = TASKS["cate"], TASKS["ite"], TASKS["ate"]

# ─────────────────────────────────────────────────────────────────────────────
# A small synthetic run
# ─────────────────────────────────────────────────────────────────────────────


def _write_run(tmp_path, n_episodes=12, n_rows=16, seed=0, fail_method=True):
    """A benchmark output directory scored by the real scoring layer.

    Predictions are deliberately different per method (one shrunken, one noisy,
    one constant, one that fails on an episode) so ranks, win rates and the
    decomposition all have something to say.
    """
    rng = np.random.default_rng(seed)
    root = tmp_path / "bench"
    (root / "results").mkdir(parents=True)
    (root / "preds").mkdir()

    tau = rng.normal(size=(n_episodes, n_rows))
    ite = tau + rng.normal(scale=0.5, size=tau.shape)
    preds = {
        "S-OLS": 0.7 * tau + rng.normal(scale=0.2, size=tau.shape),
        "S-RF": tau + rng.normal(scale=0.6, size=tau.shape),
        "ProximalFM": tau + rng.normal(scale=0.1, size=tau.shape),
    }
    if fail_method:
        broken = tau + rng.normal(scale=0.3, size=tau.shape)
        broken[0] = np.nan  # all-NaN episode → `failed`
        preds["T-RF"] = broken

    references = {
        "Oracle-CATE(X)": tau,
        "Oracle-ATE": np.repeat(tau.mean(axis=1, keepdims=True), n_rows, axis=1),
    }
    targets = {"cate": tau, "ite": ite}

    frames = []
    for task in TASKS.values():
        kind = task.target_kind(targets)
        rows = {**preds, **references}
        for name, p in rows.items():
            frames.append(
                score_method(
                    p,
                    targets[kind],
                    np.full(n_episodes, 0.1),
                    name,
                    task,
                    estimand="ate(x)" if name == "Oracle-ATE" else "cate(x)",
                    is_reference=name in references,
                )
            )
    pd.concat(frames, ignore_index=True).to_csv(
        root / "results/results.csv", index=False
    )

    d = 3
    torch.save(
        {
            "U": torch.randn(n_episodes, 2 * n_rows, d),
            "X": torch.randn(n_episodes, 2 * n_rows, d),
            "Z": torch.randn(n_episodes, 2 * n_rows, d),
            "W": torch.randn(n_episodes, 2 * n_rows, d),
            "A": torch.randint(0, 2, (n_episodes, 2 * n_rows)).float(),
            "Y": torch.randn(n_episodes, 2 * n_rows),
            "d_X": torch.full((n_episodes,), d),
            "d_W": torch.full((n_episodes,), d),
            "d_Z": torch.full((n_episodes,), d),
            "true_ITEs": ite,
            "true_CATEs": tau,
            "cate_s2": np.full(tau.shape, 1e-3),
            "cate_K_eff": torch.full((n_episodes,), 1000),
            "N_TRAIN": n_rows,
            "N_TEST": n_rows,
            "N_BATCH": n_episodes,
        },
        root / "data.pt",
    )
    for name, p in preds.items():
        np.savez(
            root / f"preds/{name}_preds.npz", preds=p, times=np.full(n_episodes, 0.1)
        )
    return root


@pytest.fixture
def run(tmp_path):
    return BenchmarkRun.load(_write_run(tmp_path))


# ─────────────────────────────────────────────────────────────────────────────
# The frame layer
# ─────────────────────────────────────────────────────────────────────────────


def test_load_refuses_a_pre_task_results_file(tmp_path):
    root = tmp_path / "bench"
    (root / "results").mkdir(parents=True)
    pd.DataFrame({"method": ["S-OLS"], "npehe": [1.0]}).to_csv(
        root / "results/results.csv", index=False
    )
    with pytest.raises(SystemExit, match="predates the three-task scoring"):
        BenchmarkRun.load(root)


def test_tasks_and_references(run):
    assert [t.name for t in run.tasks] == ["cate", "ite", "ate"]
    assert run.primary is CATE
    # References are listed apart from competitors, and in a fixed order.
    assert run.references(CATE) == ["Oracle-CATE(X)", "Oracle-ATE"]
    assert not set(run.references(CATE)) & set(run.method_order(CATE))


def test_pivot_keeps_only_complete_episodes(run):
    """The failed episode is dropped for *every* method, not just the one that
    failed — otherwise the surviving methods are compared on different sets."""
    wide, dropped = run.pivot(CATE, "npehe")
    assert dropped == 1
    assert 0 not in wide.index
    assert not wide.isna().any().any()

    loose, dropped_loose = run.pivot(CATE, "npehe", complete=False)
    assert dropped_loose == 0
    assert loose.loc[0].isna().any()


def test_ranks_exclude_references_and_are_paired(run):
    ranks = run.ranks(CATE)
    assert set(ranks.columns) == set(run.method_order(CATE))
    assert "Oracle-CATE(X)" not in ranks.columns
    # Every episode ranks every method exactly once.
    for _, row in ranks.iterrows():
        assert sorted(row) == list(range(1, ranks.shape[1] + 1))


def test_method_order_is_best_first(run):
    order = run.method_order(CATE)
    # ProximalFM is the least noisy estimator in the fixture, so it must lead.
    assert order[0] == "ProximalFM"
    means = run.mean_rank(CATE)
    assert list(means.index) == order
    assert means.is_monotonic_increasing


def test_covariates_are_per_episode_and_finite(run):
    cov = run.covariates()
    assert len(cov) == run.n_episodes(CATE)
    for col in ("d_X", "d_W", "d_Z", "d_U", "overlap", "treated_frac"):
        assert cov[col].notna().all(), col
    assert ((cov["overlap"] >= 0) & (cov["overlap"] <= 0.5)).all()


def test_terciles_refuse_a_degenerate_axis():
    assert terciles(pd.Series([1.0] * 20)).isna().all()
    assert terciles(pd.Series(range(30))).notna().all()


def test_select_methods_pins_the_named_ones():
    order = ["a", "b", "c", "ProximalFM-16k", "d"]
    keep = select_methods(order, 2, always=("ProximalFM",))
    assert keep == ["a", "b", "ProximalFM-16k"]  # order preserved, pin added


# ─────────────────────────────────────────────────────────────────────────────
# The identity two figures are drawn as though it holds
# ─────────────────────────────────────────────────────────────────────────────


def test_decomposition_is_exact(run):
    dec = _decomposition(run, CATE, trim=0.0)
    assert dec is not None
    np.testing.assert_allclose(dec["level"] + dec["shape"], dec["total"], rtol=1e-10)
    # Oracle-ATE predicts the episode's constant true ATE: no level error, no
    # shape recovered, nPEHE exactly 1. It is the plumbing check.
    assert dec.loc["Oracle-ATE", "level"] == pytest.approx(0.0, abs=1e-9)
    assert dec.loc["Oracle-ATE", "total"] == pytest.approx(1.0, abs=1e-9)
    assert dec.loc["Oracle-CATE(X)", "total"] == pytest.approx(0.0, abs=1e-9)


def test_decomposition_survives_the_trim(run):
    """Trimming drops episodes, not components — so the identity still holds."""
    dec = _decomposition(run, CATE, trim=0.2)
    np.testing.assert_allclose(dec["level"] + dec["shape"], dec["total"], rtol=1e-10)


def test_decomposition_is_none_for_a_level_only_task(run):
    assert _decomposition(run, ATE) is None


# ─────────────────────────────────────────────────────────────────────────────
# Every figure runs
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(plots.FIGURES))
def test_every_figure_draws(run, name):
    fig = plots.FIGURES[name]
    opts = plots.Options(top=3, episodes=2)
    if fig.scope == "run":
        fig.fn(run, opts)
    else:
        for task in run.tasks:
            fig.fn(run, task, opts)
    # A figure may legitimately decline (too few episodes for a CD diagram, an
    # oracle at zero error); what it may not do is raise.


def test_degenerate_run_draws_without_raising(tmp_path):
    """One episode, two methods, nothing to rank: still no exception."""
    root = _write_run(tmp_path, n_episodes=1, n_rows=8, fail_method=False)
    run = BenchmarkRun.load(root)
    opts = plots.Options(top=2, episodes=3)
    for fig in plots.FIGURES.values():
        if fig.scope == "run":
            fig.fn(run, opts)
        else:
            for task in run.tasks:
                fig.fn(run, task, opts)
