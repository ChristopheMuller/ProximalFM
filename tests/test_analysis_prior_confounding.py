"""Tests for the confounding-strength ladder (B.3.3), on the fast linear model.

The synthetic records below are confounded *by construction*: ``U`` drives both
the treatment logit and the outcome, and nothing else does. So the ladder's two
expected readings are both checkable -- naive and backdoor-``X`` are visibly
biased for the known ATE, while the oracle adjusting on ``(X, U)`` recovers it.
The true ATE is pinned to exactly 1.0 (a constant treatment effect) so the
errors have a closed-form reference rather than a Monte Carlo one.
"""

import numpy as np
import pandas as pd
import pytest

from analysis_prior import generate as gen
from analysis_prior.analyses.confounding import (
    RUNGS,
    ConfoundingResults,
    ConfoundingStrengthAnalysis,
    confounding_frame,
    delivered_shares,
    share_grid,
)
from analysis_prior.config import PriorSpec


def _mechanisms(u_var: float):
    """Fake mechanism diagnostics; ``u_var`` places the record on the delivered
    confounding-share axis."""
    return {
        mechanism: {
            "widths": {"U_sel": 2, "U_aug": 0, "X": 3, "proxy": 1, "eps": 2},
            "first_layer_var": {
                "U_sel": u_var,
                "U_aug": 0.0,
                "X": 3.0,
                "proxy": 1.0,
                "eps": 2.0,
            },
        }
        for mechanism in ("A", "Y")
    }


def _record(index=0, n=600, seed=0, dim_U=1, u_var=2.0, confounding=2.0):
    """A confounded record with a constant treatment effect of exactly 1.0.

    ``confounding`` scales ``U``'s effect on both the treatment logit and the
    outcome; at ``0`` the record is unconfounded and every rung should agree.
    """
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, dim_U)).astype(np.float32)
    X = rng.normal(size=(n, 2)).astype(np.float32)
    u0 = U[:, 0] if dim_U else np.zeros(n)
    # Noisy proxies on purpose: near-perfect ones would let the DAG-blind
    # (X, W, Z) rung all but reach the oracle and collapse the ladder's middle.
    W = (u0[:, None] * 1.5 + rng.normal(scale=0.8, size=(n, 2))).astype(np.float32)
    Z = (u0[:, None] * 1.2 + rng.normal(scale=0.8, size=(n, 2))).astype(np.float32)

    a_logit = confounding * u0 + 0.5 * X[:, 0]
    a_probs = 1.0 / (1.0 + np.exp(-a_logit))
    A = (rng.uniform(size=n) < a_probs).astype(np.float32)

    # W enters the outcome directly, as `W_direct` does in `_dataset.py`. That
    # edge is what makes the (X, U, W) oracle better-specified than (X, U);
    # without it the two oracles would be indistinguishable here.
    y0 = (
        confounding * u0 + 1.0 * X[:, 1] + 0.9 * W[:, 1] + rng.normal(scale=0.1, size=n)
    )
    y1 = y0 + 1.0  # constant effect => true ATE is exactly 1.0
    y_obs = np.where(A == 1, y1, y0)

    knobs = {
        "dim_U": dim_U,
        "confounding_share_of_U_in_A": 0.4,
        "confounding_share_of_U_in_Y": 0.4,
        "covariate_share_of_X_in_A": 0.5,
        "covariate_share_of_X_in_Y": 0.5,
        "direct_edge_share_of_Z_in_A": 0.6,
        "direct_edge_share_of_W_in_Y": 0.6,
        "noise_share_of_input_in_A": 0.3,
        "noise_share_of_input_in_Y": 0.3,
    }
    return gen.PriorDatasetRecord(
        index=index,
        seed=seed,
        U=U,
        X=X,
        W=W,
        Z=Z,
        A=A,
        A_probs=a_probs.astype(np.float32),
        Y_obs=y_obs.astype(np.float32),
        Y0=y0.astype(np.float32),
        Y1=y1.astype(np.float32),
        MU0=y0.astype(np.float32),
        MU1=y1.astype(np.float32),
        sampled_hp=knobs,
        mechanisms=_mechanisms(u_var),
    )


def _frame(records, tmp_path, **kwargs):
    return confounding_frame(
        records,
        cache_key="test",
        model="linear",
        n_splits=3,
        max_rows=None,
        regression_cache_dir=tmp_path,
        progress=False,
        **kwargs,
    )


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def test_frame_shape_and_columns(tmp_path):
    frame = _frame([_record(index=i, seed=i) for i in range(3)], tmp_path)
    assert len(frame) == 3 * len(RUNGS)
    assert set(frame["rung"]) == set(RUNGS)
    assert np.isfinite(frame["abs_error"]).all()
    assert np.isfinite(frame["raw_error"]).all()


def test_raw_error_is_abs_error_times_sd_y(tmp_path):
    """`raw_error` is the unnormalised counterpart of `abs_error`, kept because
    `sd_y` is itself entangled with the delivered U share of f_Y (see the
    module docstring's normalisation caveat) -- so the two must stay
    consistent with each other by construction, not just both present."""
    frame = _frame([_record(index=i, seed=i) for i in range(3)], tmp_path)
    np.testing.assert_allclose(
        frame["raw_error"], frame["abs_error"] * frame["sd_y"], rtol=1e-5
    )


def test_ground_truth_is_the_monte_carlo_ate(tmp_path):
    frame = _frame([_record()], tmp_path)
    # The fixture's effect is constant at 1.0, so MU1 - MU0 averages to it.
    assert frame["ate_true"].unique() == pytest.approx(1.0)


def test_the_ladder_descends(tmp_path):
    """Naive and backdoor-X are biased; the oracle recovers the truth."""
    records = [_record(index=i, seed=i) for i in range(4)]
    frame = _frame(records, tmp_path)
    median = frame.groupby("rung")["abs_error"].median()

    # Adjusting on (X, U) closes the backdoor path. Not exact: the fixture's
    # skewed assignment leaves a little clipped-extrapolation residual (see
    # test_t_learner_recovers_the_effect_the_naive_contrast_misses), well below
    # the confounding being measured.
    assert median["oracle_XU"] < 0.06
    # ...whereas ignoring U leaves a clearly detectable bias. This gap is the
    # premise the paper rests on; if it ever vanishes the proxies are decorative.
    assert median["backdoor_X"] > 10 * median["oracle_XU"]
    assert median["naive"] > 10 * median["oracle_XU"]


def test_both_oracles_are_unbiased_and_below_the_backdoor_rungs(tmp_path):
    """`W` is redundant for identification -- conditioning on `U` already blocks
    the backdoor path -- so adding it must not cost *bias*. Which of the two
    oracles has the smaller error is deliberately **not** asserted: adding `W`
    trades a better-specified outcome model (it is a real parent of `f_Y`)
    against worse cross-arm overlap in predictor space (it is a proxy of `U`,
    which drives assignment), and the winner is an empirical question per SCM.
    Pinning a direction here would bake in whichever way this fixture happens to
    fall."""
    frame = _frame([_record(index=i, seed=i) for i in range(4)], tmp_path)
    median = frame.groupby("rung")["abs_error"].median()
    for oracle in ("oracle_XU", "oracle_XUW"):
        assert median[oracle] < 0.06
        assert median[oracle] < median["backdoor_X"]


def test_z_is_absent_from_both_oracles():
    """`Z` is a parent of `f_A` only and is excluded from `f_Y` by construction,
    so it belongs in the DAG-blind rung and nowhere else."""
    assert "Z" not in RUNGS["oracle_XU"] and "Z" not in RUNGS["oracle_XUW"]
    assert "Z" in RUNGS["backdoor_XWZ"]
    # And W, being a genuine parent of f_Y, is in exactly one oracle.
    assert "W" in RUNGS["oracle_XUW"] and "W" not in RUNGS["oracle_XU"]


def test_proxies_as_plain_covariates_help_but_do_not_close_the_gap(tmp_path):
    """(X, W, Z) beats X alone -- the proxies are noisy substitutes for U --
    without matching the oracle. Both halves matter: the first says the proxies
    carry real information about U, the second says using them as ordinary
    covariates does not identify the effect."""
    frame = _frame([_record(index=i, seed=i) for i in range(4)], tmp_path)
    median = frame.groupby("rung")["abs_error"].median()
    assert median["backdoor_XWZ"] < median["backdoor_X"]
    assert median["backdoor_XWZ"] > median["oracle_XU"]


def test_an_unconfounded_record_needs_no_adjustment(tmp_path):
    """With U wired out of both mechanisms, even the naive rung is unbiased.

    The bound is on *every* rung, so it has to accommodate the regression rungs'
    finite-sample estimation error, not just the naive rung's exact zero bias.
    """
    frame = _frame([_record(confounding=0.0)], tmp_path)
    assert frame["abs_error"].max() < 0.08


def test_unconfounded_datasets_are_kept(tmp_path):
    """Unlike the exclusion audit, d_U = 0 records stay: they are the zero end
    of the axis this analysis measures."""
    frame = _frame([_record(index=0), _record(index=1, dim_U=0)], tmp_path)
    assert set(frame["index"]) == {0, 1}


def test_error_is_normalised_by_the_outcome_scale(tmp_path):
    """Rescaling Y rescales the raw ATE error identically, so the reported
    normalised error is invariant -- which is what makes cross-SCM pooling
    legitimate."""
    record = _record()
    scaled = gen.PriorDatasetRecord(
        **{
            **record.to_dict(),
            **{
                name: getattr(record, name) * 10.0
                for name in ("Y_obs", "Y0", "Y1", "MU0", "MU1")
            },
        }
    )
    base = _frame([record], tmp_path / "a").set_index("rung")["abs_error"]
    rescaled = _frame([scaled], tmp_path / "b").set_index("rung")["abs_error"]
    np.testing.assert_allclose(base.to_numpy(), rescaled.to_numpy(), rtol=1e-3)


def test_max_rows_truncates(tmp_path):
    frame = confounding_frame(
        [_record(n=600)],
        cache_key="test",
        model="linear",
        n_splits=3,
        max_rows=200,
        regression_cache_dir=tmp_path,
        progress=False,
    )
    assert np.isfinite(frame["abs_error"]).all()


def test_regressions_are_cached_and_never_refitted(tmp_path):
    _frame([_record()], tmp_path)
    n_first = len(list(tmp_path.glob("*.npz")))
    _frame([_record()], tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == n_first
    # One entry per regression rung (naive fits nothing).
    assert n_first == len(RUNGS) - 1


# --------------------------------------------------------------------------
# Delivered confounding share
# --------------------------------------------------------------------------


def test_delivered_share_uses_the_first_layer_variance(tmp_path):
    records = [_record(index=0, u_var=2.0), _record(index=1, u_var=8.0)]
    shares = delivered_shares(records).set_index("index")
    # U / (U + X + proxy), eps excluded: 2/(2+3+1) and 8/(8+3+1).
    assert shares.loc[0, "share_A"] == pytest.approx(2 / 6)
    assert shares.loc[1, "share_A"] == pytest.approx(8 / 12)
    # Both mechanisms share the fixture's variance, so share_min agrees.
    assert shares.loc[1, "share_min"] == pytest.approx(shares.loc[1, "share_Y"])


def test_share_min_takes_the_binding_mechanism():
    """Confounding needs U in *both* mechanisms, so the summary statistic is the
    smaller share, not the average."""
    record = _record()
    record.mechanisms["A"]["first_layer_var"]["U_sel"] = 1.0
    record.mechanisms["Y"]["first_layer_var"]["U_sel"] = 9.0
    shares = delivered_shares([record])
    assert shares["share_min"].iloc[0] == pytest.approx(shares["share_A"].iloc[0])


def test_share_grid_bins_and_counts():
    rng = np.random.default_rng(0)
    n = 200
    joined = pd.DataFrame(
        {
            "share_A": rng.uniform(size=n),
            "share_Y": rng.uniform(size=n),
            "abs_error": rng.uniform(size=n),
        }
    )
    median, counts, y_edges, a_edges = share_grid(joined, n_bins=4)
    assert median.shape == counts.shape == (4, 4)
    # Quantile bins on each axis independently: every dataset lands in exactly
    # one cell, and the marginal counts are balanced.
    assert counts.sum() == n
    assert counts.sum(axis=1).min() >= n // 4 - 1
    assert len(y_edges) == len(a_edges) == 5


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_analysis_end_to_end(tmp_path):
    spec = PriorSpec(
        name="tiny-conf",
        n_datasets=6,
        seq_len=300,
        mcmc_cate=1,
        min_dim_U=2,
        max_dim_U=2,
        min_dim_W=2,
        max_dim_W=3,
        min_dim_Z=2,
        max_dim_Z=3,
        min_dim_X=2,
        max_dim_X=3,
        seed=0,
    )
    cache = gen.generate(spec, n_jobs=1, progress=False)

    analysis = ConfoundingStrengthAnalysis()
    analysis.model = "linear"
    analysis.n_splits = 3
    analysis.max_rows = None
    analysis.regression_cache_dir = tmp_path / "reg"

    artifacts = analysis.run(cache, tmp_path / "out")
    for path in artifacts.figures + artifacts.tables:
        assert path.exists() and path.stat().st_size > 0
    names = {p.name for p in artifacts.figures}
    assert {"confounding_ladder.pdf", "confounding_vs_share.pdf"} <= names
    assert "confounding_table.tex" in {p.name for p in artifacts.tables}

    assert artifacts.summary["model"] == "linear"
    for rung in RUNGS:
        assert np.isfinite(artifacts.summary[f"median_error_{rung}"])
        assert 0.0 <= artifacts.summary[f"share_materially_biased_{rung}"] <= 1.0
    for oracle in ("oracle_XU", "oracle_XUW"):
        assert np.isfinite(artifacts.summary[f"backdoor_X_over_{oracle}_median"])
        # The top-right grid cell can be empty on a cache this small, in which
        # case the summary carries None rather than an unserialisable NaN.
        assert f"median_error_{oracle}_top_cell" in artifacts.summary
    assert np.isfinite(artifacts.summary["oracle_XU_over_oracle_XUW_median"])
    assert np.isfinite(artifacts.summary["spearman_share_min_vs_error_backdoor_X"])
    # The normalisation-caveat evidence trail (module docstring): sd_y's
    # entanglement with share_Y, and raw- vs normalised-error correlations for
    # both shares, on both bias-carrying rungs.
    for axis in ("share_A", "share_Y"):
        assert np.isfinite(artifacts.summary[f"spearman_sd_y_vs_{axis}"])
        for rung in ("naive", "backdoor_X"):
            assert np.isfinite(
                artifacts.summary[f"spearman_{axis}_vs_raw_error_{rung}"]
            )
            assert np.isfinite(
                artifacts.summary[f"spearman_{axis}_vs_abs_error_{rung}"]
            )


def test_registered_under_its_name():
    from analysis_prior.analyses import get_analysis

    assert get_analysis("confounding_strength").name == "confounding_strength"


def test_results_accessors(tmp_path):
    records = [_record(index=i, seed=i) for i in range(3)]
    results = ConfoundingResults(
        frame=_frame(records, tmp_path),
        shares=delivered_shares(records),
        model="linear",
    )
    assert results.errors("oracle_XU").shape == (3,)
    joined = results.joined("backdoor_X")
    assert len(joined) == 3
    assert {"share_A", "share_Y", "share_min", "abs_error"} <= set(joined.columns)
