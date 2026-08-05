"""Tests for the mechanism output-dependence analysis (fast linear model)."""

import numpy as np
import pytest

from analysis_prior import generate as gen
from analysis_prior.analyses.mechanism_outputs import (
    MECHANISM_BLOCKS,
    MechanismOutputsAnalysis,
    _target,
    output_frame,
)
from analysis_prior.config import PriorSpec


def _record(index=0, n=200, seed=0):
    """A synthetic record with the fields the analysis reads."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, 2)).astype(np.float32)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    W = rng.normal(size=(n, 2)).astype(np.float32)
    Z = rng.normal(size=(n, 2)).astype(np.float32)
    # A depends on U (confounding) and Z; Y0 depends on U and X.
    a_logit = U[:, 0] * 2.0 + Z[:, 0] * 1.0 + rng.normal(scale=0.1, size=n)
    a_probs = 1.0 / (1.0 + np.exp(-a_logit))
    y0 = U[:, 1] * 1.5 + X[:, 0] * 1.0 + rng.normal(scale=0.1, size=n)
    vec = np.zeros(n, dtype=np.float32)
    knobs = {
        "dim_U": 2,
        "confounding_share_of_U_in_A": 0.4,
        "confounding_share_of_U_in_Y": 0.3,
        "covariate_share_of_X_in_A": 0.5,
        "covariate_share_of_X_in_Y": 0.6,
        "direct_edge_share_of_Z_in_A": 0.7,
        "direct_edge_share_of_W_in_Y": 0.2,
        "noise_share_of_input_in_A": 0.3,
        "noise_share_of_input_in_Y": 0.4,
    }
    return gen.PriorDatasetRecord(
        index=index,
        seed=seed,
        U=U,
        X=X,
        W=W,
        Z=Z,
        A=(a_probs > 0.5).astype(np.float32),
        A_probs=a_probs.astype(np.float32),
        Y_obs=vec,
        Y0=y0.astype(np.float32),
        Y1=vec,
        MU0=vec,
        MU1=vec,
        sampled_hp=knobs,
        gamma_hetero=0.5,
    )


def test_target_a_is_logit_of_propensity():
    rec = _record()
    tgt = _target(rec, "A")
    expected = np.log(rec.A_probs / (1 - rec.A_probs))
    np.testing.assert_allclose(tgt, expected, rtol=1e-4)


def test_target_y_is_y0():
    rec = _record()
    np.testing.assert_array_equal(_target(rec, "Y"), rec.Y0)


def test_mechanism_blocks_cover_the_four_channels():
    for mechanism, blocks in MECHANISM_BLOCKS.items():
        assert set(blocks) == {"U", "X", "proxy"}


def test_output_frame_recovers_the_dependence_structure(tmp_path):
    # A depends on U and Z (not X); Y0 depends on U and X (not W).
    records = [_record(index=i, seed=i) for i in range(4)]
    frame = output_frame(
        records,
        cache_key="test",
        model="linear",
        n_splits=3,
        max_rows=None,
        regression_cache_dir=tmp_path,
    )
    assert len(frame) == 8  # 4 datasets x 2 mechanisms

    a = frame[frame["mechanism"] == "A"]
    # U and Z carry A; X does not.
    assert a["delta_r2_U"].median() > 0.1
    assert a["delta_r2_proxy"].median() > 0.1
    assert abs(a["delta_r2_X"].median()) < 0.05

    y = frame[frame["mechanism"] == "Y"]
    # U and X carry Y0; W does not.
    assert y["delta_r2_U"].median() > 0.1
    assert y["delta_r2_X"].median() > 0.1
    assert abs(y["delta_r2_proxy"].median()) < 0.05


def test_output_frame_carries_knobs_and_residual(tmp_path):
    frame = output_frame(
        [_record()],
        cache_key="test",
        model="linear",
        n_splits=3,
        max_rows=None,
        regression_cache_dir=tmp_path,
    )
    row = frame[frame["mechanism"] == "A"].iloc[0]
    assert row["knob_U"] == pytest.approx(0.4)
    assert row["knob_proxy"] == pytest.approx(0.7)
    assert row["knob_eps"] == pytest.approx(0.3)
    assert row["residual"] == pytest.approx(1.0 - row["r2_full"])


def test_output_frame_rejects_records_without_sampled_hp(tmp_path):
    rec = _record()
    stale = gen.PriorDatasetRecord(**{**rec.to_dict(), "sampled_hp": None})
    with pytest.raises(ValueError, match="--force"):
        output_frame(
            [stale], cache_key="t", model="linear", regression_cache_dir=tmp_path
        )


def test_max_rows_subsamples(tmp_path):
    frame = output_frame(
        [_record(n=300)],
        cache_key="test",
        model="linear",
        n_splits=2,
        max_rows=100,
        regression_cache_dir=tmp_path,
    )
    # Just needs to run and produce finite numbers on the truncated data.
    assert np.isfinite(frame["r2_full"]).all()


def test_analysis_end_to_end(tmp_path):
    spec = PriorSpec(
        name="tiny-out",
        n_datasets=4,
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

    # Uses every dataset in the cache (4 here); no per-analysis dataset cap.
    analysis = MechanismOutputsAnalysis()
    analysis.model = "linear"
    analysis.n_splits = 3
    analysis.max_rows = None

    out = tmp_path / "out"
    cache_dir = tmp_path / "reg"
    # Point the regression cache at tmp so the test never writes to the repo.
    import analysis_prior.analyses.mechanism_outputs as mod

    original = mod.DEFAULT_REGRESSION_CACHE_DIR
    mod.DEFAULT_REGRESSION_CACHE_DIR = cache_dir
    try:
        artifacts = analysis.run(cache, out)
    finally:
        mod.DEFAULT_REGRESSION_CACHE_DIR = original

    for path in artifacts.figures + artifacts.tables:
        assert path.exists() and path.stat().st_size > 0
    assert "mechanism_output_dependence.pdf" in {p.name for p in artifacts.figures}
    assert artifacts.summary["n_datasets"] == 4
    assert artifacts.summary["model"] == "linear"


def test_example_script_imports_and_parses_args():
    """Bit-rot guard for the demo script: it imports and its CLI is wired.

    The full pipeline needs the general_ite cache (not present in CI), so we
    only check import + argument parsing here; the analysis itself is covered
    by the tests above.
    """
    from analysis_prior.examples import mechanism_outputs_small as demo

    assert demo.DEFAULT_OUT.name == "mechanism_outputs_small"
    with pytest.raises(SystemExit) as exc:  # --help exits 0
        demo.main(["--help"])
    assert exc.value.code == 0
    # argparse itself validates the choices; a bad model is rejected.
    with pytest.raises(SystemExit):
        demo.main(["--model", "not-a-model"])
