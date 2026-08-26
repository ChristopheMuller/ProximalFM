"""Tests for benchmarks/metrics.py and benchmarks/estimands.py — the scoring layer.

Three things are pinned here. First the arithmetic of the metrics themselves,
which carries two targets. Second, that each ``Task`` shows the methods it can
rank fairly and no others, and that the reference rows are shown but never
ranked. Third, and most valuable, that this implementation and the trainer's
``CausalEvaluator`` agree number-for-number: after the CATE-target rework both
score the *same* estimand on the *same* sets, so ``synt_point/cate/npehe`` in
wandb and the benchmark's cate-task nPEHE are directly comparable — and will be
compared. Nothing shares code between them (the benchmark deliberately imports
no proximalfm), so only a test keeps them from drifting.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from benchmarks.estimands import (
    canonical_estimand,
    format_estimand,
    parse_estimand,
)
from benchmarks.metrics import (
    ESTIMAND_COLUMN,
    RANK_COLUMN,
    TARGET_VAR_THRESHOLD,
    TASKS,
    TaskReport,
    score_episode,
    score_method,
)

CATE, ITE, ATE = TASKS["cate"], TASKS["ite"], TASKS["ate"]


def _episode(pred, target, task=CATE, **kw):
    return score_episode(np.asarray(pred, float), np.asarray(target, float), task, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode arithmetic
# ─────────────────────────────────────────────────────────────────────────────


def test_perfect_prediction_scores_zero():
    tau = np.array([1.0, -2.0, 0.5, 3.0])
    m = _episode(tau, tau)
    assert m["npehe"] == pytest.approx(0.0)
    assert m["cate_r2"] == pytest.approx(1.0)
    assert m["shape_r2"] == pytest.approx(1.0)
    assert m["ate_bias_norm"] == pytest.approx(0.0)


def test_constant_ate_prediction_is_the_npehe_1_anchor():
    """Predicting the true episode mean sits at exactly nPEHE 1 / Shape R² 0."""
    tau = np.array([1.0, -2.0, 0.5, 3.0])
    m = _episode(np.full_like(tau, tau.mean()), tau)
    assert m["npehe"] == pytest.approx(1.0)
    assert m["cate_r2"] == pytest.approx(0.0)
    assert m["shape_r2"] == pytest.approx(0.0)


def test_npehe_and_r2_are_the_same_number():
    rng = np.random.default_rng(0)
    tau = rng.normal(size=50)
    m = _episode(tau + rng.normal(scale=0.4, size=50), tau)
    assert m["cate_r2"] == pytest.approx(1.0 - m["npehe"] ** 2)


def test_shape_r2_ignores_a_pure_level_shift():
    tau = np.array([1.0, -2.0, 0.5, 3.0])
    m = _episode(tau + 5.0, tau)
    assert m["shape_r2"] == pytest.approx(1.0)  # shape is perfect...
    assert m["cate_r2"] < 0  # ...but the level is not
    assert m["ate_bias"] == pytest.approx(5.0)


def test_homogeneous_episode_yields_no_scale_free_metrics():
    tau = np.full(8, 2.0)
    m = _episode(np.zeros(8), tau)
    assert m["homogeneous"]
    assert np.isnan([m["npehe"], m["shape_r2"], m["ate_bias_norm"]]).all()
    assert m["target_var"] < TARGET_VAR_THRESHOLD


def test_partial_nans_fall_back_to_the_episode_mean():
    tau = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, np.nan, 3.0, 4.0])
    m = _episode(pred, tau)
    assert not m["failed"]
    # The NaN row is scored against the mean of the surviving predictions.
    filled = np.where(np.isnan(pred), np.nanmean(pred), pred)
    assert m["npehe"] == pytest.approx(
        np.sqrt(((filled - tau) ** 2).mean() / tau.var())
    )


def test_all_nan_episode_is_a_failure():
    m = _episode(np.full(4, np.nan), np.array([1.0, 2.0, 3.0, 4.0]))
    assert m["failed"]
    assert np.isnan(m["npehe"])


def test_a_task_computes_only_its_own_metrics():
    """The ATE task is level-only: npehe/shape_r2 are not merely NaN there, they
    are not columns at all."""
    tau = np.array([1.0, -2.0, 0.5, 3.0])
    level = _episode(tau + 1.0, tau, task=ATE)
    assert "npehe" not in level and "shape_r2" not in level
    assert level["ate_bias_norm"] == pytest.approx(1.0 / np.sqrt(tau.var()))


# ─────────────────────────────────────────────────────────────────────────────
# The Monte-Carlo noise share, reported and never subtracted
# ─────────────────────────────────────────────────────────────────────────────


def test_s2_over_var_is_the_noise_share_of_the_target():
    tau = np.array([1.0, -2.0, 0.5, 3.0])
    m = _episode(tau, tau, s2=np.full(4, 0.1))
    assert m["s2_over_var"] == pytest.approx(0.1 / tau.var())


def test_s2_does_not_change_any_score():
    """It is a diagnostic: the metrics score MU1-MU0 exactly as given."""
    rng = np.random.default_rng(1)
    tau, pred = rng.normal(size=30), rng.normal(size=30)
    without = _episode(pred, tau)
    with_s2 = _episode(pred, tau, s2=np.full(30, 0.5))
    assert with_s2["npehe"] == without["npehe"]
    assert with_s2["shape_r2"] == without["shape_r2"]


def test_s2_absent_leaves_the_column_empty():
    tau = np.array([1.0, -2.0, 0.5, 3.0])
    assert np.isnan(_episode(tau, tau)["s2_over_var"])


# ─────────────────────────────────────────────────────────────────────────────
# Estimand tags
# ─────────────────────────────────────────────────────────────────────────────


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
        # An oracle's u is disqualifying on the headline table, same as w/z:
        # cate(x,u) estimates a different quantity than tau*(x).
        ("cate(x,u)", False, True, True),
        ("cate(x,w,u)", False, True, True),
        ("ate(x)", False, False, True),
    ],
)
def test_task_membership_is_kind_plus_a_proxy_check(tag, cate, ite, ate):
    assert CATE.admits(tag) is cate
    assert ITE.admits(tag) is ite
    assert ATE.admits(tag) is ate


def test_legacy_tags_are_stored_in_the_new_spelling():
    """A pre-existing config's `estimand: xw` lands in the frame as cate(x,w)."""
    df = score_method(
        np.zeros((2, 5)), np.ones((2, 5)), np.zeros(2), "m", ITE, estimand="xw"
    )
    assert set(df["estimand"]) == {"cate(x,w)"}


# ─────────────────────────────────────────────────────────────────────────────
# Tasks: which rows a table shows, and who may be ranked
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def results():
    """The long frame run.py writes: one block per (method, task it belongs in)."""
    rng = np.random.default_rng(2)
    tau = rng.normal(size=(4, 25))
    ite = tau + rng.normal(size=(4, 25))
    good = tau + rng.normal(scale=0.3, size=(4, 25))
    const = np.repeat(tau.mean(axis=1, keepdims=True), 25, axis=1)
    times = np.zeros(4)
    targets = {"cate": tau, "ite": ite}

    outputs = [
        ("S-OLS", good, "cate(x)", False),
        ("PMMR-XW", good, "cate(x,w)", False),
        ("P2SLS", const, "ate(x)", False),
        ("Oracle-CATE(X)", tau, "cate(x)", True),
    ]
    blocks = []
    for task in TASKS.values():
        target = targets[task.target_kind(targets)]
        for name, pred, tag, is_ref in outputs:
            if not task.admits(tag):
                continue
            blocks.append(
                score_method(
                    pred,
                    target,
                    times,
                    name,
                    task,
                    estimand=tag,
                    is_reference=is_ref,
                )
            )
    return pd.concat(blocks, ignore_index=True)


def _summary(results, task):
    return TaskReport.from_results(results, task).summary()


def test_report_can_average_ordinary_metrics_over_episodes(results):
    report = TaskReport.from_results(
        results, CATE, episode_aggregation="mean"
    ).summary()
    rows = results[(results["target"] == "cate") & (results["method"] == "S-OLS")]
    assert report.loc["S-OLS", "nPEHE"] == pytest.approx(rows["npehe"].mean())
    assert "nPEHE p25" not in report.columns


def test_cate_task_excludes_proxy_conditioned_and_ate_methods(results):
    assert set(_summary(results, CATE).index) == {"S-OLS", "Oracle-CATE(X)"}


def test_ite_task_admits_the_proxy_conditioned_methods(results):
    assert set(_summary(results, ITE).index) == {
        "S-OLS",
        "PMMR-XW",
        "Oracle-CATE(X)",
    }


def test_ate_task_shows_everyone(results):
    assert set(_summary(results, ATE).index) == {
        "S-OLS",
        "PMMR-XW",
        "P2SLS",
        "Oracle-CATE(X)",
    }


def test_ate_task_reads_the_cate_target_and_falls_back_to_ite():
    """It is a view of the CATE rows, not a third target — unless there is no
    Monte-Carlo target, in which case it reads the ITE."""
    assert ATE.target_kind({"cate": None, "ite": None}) == "cate"
    assert ATE.target_kind({"ite": None}) == "ite"
    assert CATE.target_kind({"ite": None}) is None


def test_reference_rows_are_shown_but_never_ranked(results):
    summary = _summary(results, CATE)
    assert "Oracle-CATE(X)" in summary.index
    assert pd.isna(summary.loc["Oracle-CATE(X)", RANK_COLUMN.label])
    assert summary.loc["Oracle-CATE(X)", "Reference"]
    # The one real method still ranks 1st — the oracle did not displace it.
    assert summary.loc["S-OLS", RANK_COLUMN.label] == pytest.approx(1.0)
    assert not summary.loc["S-OLS", "Reference"]


def test_oracle_is_exact_on_cate_and_noise_bound_on_ite(results):
    """The invariant the whole rework rests on.

    tau*(x) scores 0 against itself — if it does not, the target plumbing is
    wrong — and strictly worse than 0 against the realized ITE, by exactly the
    irreducible E[Var(Y1-Y0 | X)] that no tau(x) method can escape.
    """
    on_cate = _summary(results, CATE).loc["Oracle-CATE(X)", "nPEHE"]
    on_ite = _summary(results, ITE).loc["Oracle-CATE(X)", "nPEHE"]
    assert on_cate == pytest.approx(0.0, abs=1e-12)
    assert on_ite > 0.5


def test_ate_task_ranks_on_the_level_not_the_shape(results):
    """P2SLS predicts the true episode mean, so it must win the ATE table
    outright while being last on nPEHE."""
    summary = _summary(results, ATE)
    assert summary.index[0] == "P2SLS"
    assert summary.loc["P2SLS", "ATE RMSE"] == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation: pooled vs the distribution over episodes
# ─────────────────────────────────────────────────────────────────────────────


def test_median_metrics_carry_their_quartile_companions(results):
    summary = _summary(results, CATE)
    frame = TaskReport.from_results(results, CATE).frame
    npehe = frame[frame["method"] == "S-OLS"]["npehe"]
    assert summary.loc["S-OLS", "nPEHE"] == pytest.approx(npehe.median())
    assert summary.loc["S-OLS", "nPEHE p25"] == pytest.approx(npehe.quantile(0.25))
    assert summary.loc["S-OLS", "nPEHE p75"] == pytest.approx(npehe.quantile(0.75))


def test_ate_rmse_is_pooled_not_a_median(results):
    """The one metric declared `pooled_rms`: it pools both the bias and the
    episode-to-episode spread of the ATE estimate, so it is not a quantile of
    anything and gets no p25/p75."""
    summary = _summary(results, CATE)
    frame = TaskReport.from_results(results, CATE).frame
    bias = frame[frame["method"] == "S-OLS"]["ate_bias_norm"]
    assert summary.loc["S-OLS", "ATE RMSE"] == pytest.approx(
        float(np.sqrt((bias**2).mean()))
    )
    assert "ATE RMSE p25" not in summary.columns


# ─────────────────────────────────────────────────────────────────────────────
# One column spec, both renderers
# ─────────────────────────────────────────────────────────────────────────────


def test_both_renderers_read_the_same_task_columns(results):
    """The point of the Task objects: no second column spec exists, so the
    console table and the LaTeX table cannot list different columns."""
    report = TaskReport.from_results(results, CATE)
    labels = [c.label for c in CATE.columns()]

    headings = [c.heading for c in CATE.columns() if c is not ESTIMAND_COLUMN]
    console_labels = [c.header for c in report.rich_table().columns]
    assert console_labels == ["Method", "Est.", *headings]

    latex = report.latex()
    for label in labels:
        assert label.replace("R²", "$R^2$") in latex


def test_hidden_metrics_stay_in_the_csv_but_out_of_the_tables(results):
    """CATE R² is 1 - nPEHE²: worth keeping in summary_*.csv, not worth a
    column on screen."""
    report = TaskReport.from_results(results, CATE)
    assert "CATE R²" in report.summary().columns
    assert "CATE R²" not in [c.label for c in CATE.columns()]


def test_latex_never_bolds_a_reference_row(results):
    """Oracle-CATE(X) has the best possible nPEHE by construction. It is an
    anchor, not a competitor, so the bold goes to the best real method — and
    picking the winners by row rather than by value is what stops a reference
    row tying its way into one."""
    lines = {
        line.split(" & ")[0]: line
        for line in TaskReport.from_results(results, CATE).latex().splitlines()
        if " & " in line
    }
    assert "\\textbf" not in lines["Oracle-CATE(X)"]
    assert "\\textbf" in lines["S-OLS"]


def test_latex_right_aligns_the_numeric_columns(results):
    """The cells are pre-formatted strings, which pandas would left-align."""
    latex = TaskReport.from_results(results, CATE).latex()
    # index + Estimand are left-aligned; every metric column is right-aligned.
    expected = "l" + "".join(
        "l" if c is ESTIMAND_COLUMN else "r" for c in CATE.columns()
    )
    assert f"\\begin{{tabular}}{{{expected}}}" in latex


def test_latex_escapes_the_percent_sign(results):
    """`%` opens a comment in LaTeX and would swallow the rest of the row."""
    latex = TaskReport.from_results(results, CATE).latex()
    assert "\\%" in latex
    assert not any(
        "%" in line and "\\%" not in line for line in latex.splitlines() if "&" in line
    )


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

    ours = score_method(
        pred_cate.astype(np.float32),
        true_cate.astype(np.float32),
        np.zeros(6),
        "m",
        CATE,
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
