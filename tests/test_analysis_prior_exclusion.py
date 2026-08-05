"""Tests for the exclusion-restriction audit (B.3.2), on the fast linear model.

The synthetic records below wire the *true* exclusion structure -- ``Z`` never
enters ``Y``, ``W`` never enters the treatment logit, and ``U`` is the only path
linking each to the other side -- so the audit's two expected readings are both
checkable: near-zero ``dR2`` when conditioning on ``U``, clearly nonzero once
``U`` is dropped.
"""

import numpy as np
import pandas as pd
import pytest

from analysis_prior import generate as gen
from analysis_prior.analyses.exclusion import (
    CHECKS,
    CONDITIONS,
    ExclusionAnalysis,
    _target,
    exclusion_frame,
)
from analysis_prior.config import PriorSpec


def _record(index=0, n=400, seed=0, dim_U=2):
    """A record satisfying the exclusion restrictions, with U confounding both.

    ``Z`` and ``W`` are proxies of ``U`` only; the treatment logit is driven by
    ``(U, X, Z)`` and never by ``W``, the outcome by ``(U, X, W, A)`` and never
    by ``Z``.
    """
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, dim_U)).astype(np.float32)
    X = rng.normal(size=(n, 2)).astype(np.float32)
    u0 = U[:, 0] if dim_U else np.zeros(n)
    # Proxies: noisy images of U, with no other parent.
    W = (u0[:, None] * 1.5 + rng.normal(scale=0.3, size=(n, 2))).astype(np.float32)
    Z = (u0[:, None] * 1.2 + rng.normal(scale=0.3, size=(n, 2))).astype(np.float32)

    a_logit = 2.0 * u0 + 1.0 * X[:, 0] + 0.8 * Z[:, 0] + rng.normal(scale=0.1, size=n)
    a_probs = 1.0 / (1.0 + np.exp(-a_logit))
    A = (rng.uniform(size=n) < a_probs).astype(np.float32)

    y0 = 1.5 * u0 + 1.0 * X[:, 1] + 0.7 * W[:, 0] + rng.normal(scale=0.1, size=n)
    y1 = y0 + 1.0
    y_obs = np.where(A == 1, y1, y0)
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
        sampled_hp={"dim_U": dim_U},
    )


def _frame(records, tmp_path, **kwargs):
    return exclusion_frame(
        records,
        cache_key="test",
        model="linear",
        n_splits=3,
        max_rows=None,
        regression_cache_dir=tmp_path,
        progress=False,
        **kwargs,
    )


def test_target_a_is_the_logit_and_y_is_the_observed_outcome():
    rec = _record()
    expected = np.log(rec.A_probs / (1 - rec.A_probs))
    np.testing.assert_allclose(_target(rec, "A"), expected, rtol=1e-4)
    np.testing.assert_array_equal(_target(rec, "Y"), rec.Y_obs)


def test_checks_exclude_the_right_block():
    # Z must be absent from f_Y, W from f_A -- and each excluded block must sit
    # in its own check's predictor set (it is what gets dropped).
    assert CHECKS["Y"]["dropped"] == "Z" and "Z" in CHECKS["Y"]["blocks"]
    assert CHECKS["A"]["dropped"] == "W" and "W" in CHECKS["A"]["blocks"]
    # Z stays out of the A-side conditioning set: it is a proxy of U, and would
    # absorb the confounder the positive control needs to remove.
    assert "Z" not in CHECKS["A"]["blocks"]
    # U is in both full sets; the positive control is that set minus U.
    assert all("U" in check["blocks"] for check in CHECKS.values())


def test_frame_shape_and_columns(tmp_path):
    frame = _frame([_record(index=i, seed=i) for i in range(3)], tmp_path)
    assert len(frame) == 3 * len(CHECKS) * len(CONDITIONS)
    assert set(frame["condition"]) == set(CONDITIONS)
    assert set(frame["side"]) == set(CHECKS)


def test_exclusion_holds_and_the_control_has_power(tmp_path):
    records = [_record(index=i, seed=i) for i in range(4)]
    frame = _frame(records, tmp_path)

    for side in CHECKS:
        held = frame[(frame["side"] == side) & (frame["condition"] == "with_U")]
        control = frame[(frame["side"] == side) & (frame["condition"] == "without_U")]
        # Conditioning on U, the excluded block adds essentially nothing...
        assert abs(held["delta_r2"].median()) < 0.02
        # ...but dropping U opens the path, so the same test lights up.
        assert control["delta_r2"].median() > 0.05


def test_datasets_without_u_are_skipped(tmp_path):
    records = [_record(index=0, seed=0), _record(index=1, seed=1, dim_U=0)]
    frame = _frame(records, tmp_path)
    assert set(frame["index"]) == {0}


def test_max_rows_truncates(tmp_path):
    frame = exclusion_frame(
        [_record(n=500)],
        cache_key="test",
        model="linear",
        n_splits=2,
        max_rows=200,
        regression_cache_dir=tmp_path,
        progress=False,
    )
    assert np.isfinite(frame["delta_r2"]).all()


def test_predictor_sets_are_cached_and_never_refitted(tmp_path):
    """Each distinct predictor set is fitted exactly once and then served from
    the regression cache, so re-running writes no new entries."""
    _frame([_record()], tmp_path)
    n_first = len(list(tmp_path.glob("*.npz")))
    _frame([_record()], tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == n_first
    # Per side: with_U full, with_U reduced, control full, control reduced.
    assert n_first == 2 * 4


def test_analysis_end_to_end(tmp_path):
    spec = PriorSpec(
        name="tiny-excl",
        n_datasets=3,
        seq_len=200,
        mc_cate=1,
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

    analysis = ExclusionAnalysis()
    analysis.model = "linear"
    analysis.n_splits = 3
    analysis.max_rows = None
    analysis.regression_cache_dir = tmp_path / "reg"

    artifacts = analysis.run(cache, tmp_path / "out")
    for path in artifacts.figures + artifacts.tables:
        assert path.exists() and path.stat().st_size > 0
    assert "exclusion.pdf" in {p.name for p in artifacts.figures}
    assert "exclusion_table.tex" in {p.name for p in artifacts.tables}
    assert artifacts.summary["model"] == "linear"
    assert artifacts.summary["n_skipped_no_u"] >= 0
    for key in (
        "median_delta_r2_Z_to_Y_with_U",
        "median_delta_r2_W_to_A_with_U",
        # The conditional read: the subset where the control can have power.
        "median_u_dependence_Y",
        "spearman_u_dependence_control_A",
        "median_delta_r2_Z_to_Y_without_U_top_tercile",
        "share_above_threshold_W_to_A_with_U_top_tercile",
    ):
        assert key in artifacts.summary
        assert np.isfinite(artifacts.summary[key])

    # Every (clause, condition) appears both pooled and on the top tercile.
    table = pd.read_csv(next(p for p in artifacts.tables if p.suffix == ".csv"))
    assert len(table) == 8
    assert set(table["datasets"]) == {"all", "top confounded"}


def test_paired_recovers_the_u_dependence(tmp_path):
    """``u_dependence`` is the with-U minus without-U *full*-set R², i.e. what U
    uniquely adds to the target -- the quantity the control's power rides on."""
    from analysis_prior.analyses.exclusion import ExclusionResults

    frame = _frame([_record(index=i, seed=i) for i in range(3)], tmp_path)
    results = ExclusionResults(frame=frame, model="linear", n_skipped_no_u=0)
    paired = results.paired("Y")

    assert list(paired["index"]) == [0, 1, 2]
    wide = frame[frame["side"] == "Y"].set_index(["index", "condition"])["r2_full"]
    expected = [wide[(i, "with_U")] - wide[(i, "without_U")] for i in range(3)]
    np.testing.assert_allclose(paired["u_dependence"], expected, rtol=1e-6)
    # U genuinely drives Y here, so it adds signal even over {A, X, Z} -- only a
    # little, since Z is itself a noisy image of U and absorbs most of it.
    assert paired["u_dependence"].median() > 0.01


def test_strongly_confounded_takes_the_top_third(tmp_path):
    from analysis_prior.analyses.exclusion import ExclusionResults

    frame = _frame([_record(index=i, seed=i) for i in range(6)], tmp_path)
    results = ExclusionResults(frame=frame, model="linear", n_skipped_no_u=0)
    top = results.strongly_confounded("Y")
    assert len(top) == 2  # top third of 6
    assert top["u_dependence"].min() >= results.paired("Y")["u_dependence"].median()


def test_registered_under_its_name():
    from analysis_prior.analyses import get_analysis

    assert get_analysis("exclusion").name == "exclusion"


@pytest.mark.parametrize("side", list(CHECKS))
def test_series_selects_one_cell(tmp_path, side):
    analysis = ExclusionAnalysis()
    analysis.model = "linear"
    analysis.max_rows = None
    analysis.regression_cache_dir = tmp_path
    frame = _frame([_record(index=i, seed=i) for i in range(2)], tmp_path)
    from analysis_prior.analyses.exclusion import ExclusionResults

    results = ExclusionResults(frame=frame, model="linear", n_skipped_no_u=0)
    assert results.series(side, "with_U").shape == (2,)
