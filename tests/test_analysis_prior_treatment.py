"""Tests for the treatment-assignment walkthrough (appendix B.2.5).

Three layers:

- raw-logit diagnostics collection in ``generate_dataset`` -- the stored
  ``a_logits`` reproduce ``sample_A``'s propensity exactly, and collecting them
  leaves the generated data bit-identical.
- ``analysis_prior.analyses.treatment.reconstruct_pipeline`` -- the pure
  transform matches the stored ``A_probs`` and rejects logit-less records.
- ``TreatmentWalkthroughAnalysis`` -- an end-to-end render on a tiny cache.
"""

import numpy as np
import pytest

from analysis_prior import generate as gen
from analysis_prior.analyses.treatment import (
    TreatmentWalkthroughAnalysis,
    reconstruct_pipeline,
    select_dataset,
)
from analysis_prior.config import PriorSpec


@pytest.fixture(scope="module")
def tiny_spec():
    return PriorSpec(
        name="tiny-treat",
        n_datasets=4,
        seq_len=200,
        mcmc_cate=3,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=2,
        max_dim_W=3,
        min_dim_Z=2,
        max_dim_Z=3,
        min_dim_X=2,
        max_dim_X=3,
        seed=0,
    )


@pytest.fixture(scope="module")
def tiny_cache(tiny_spec):
    return gen.generate(tiny_spec, n_jobs=1, progress=False)


# --------------------------------------------------------------------------
# Diagnostics collection
# --------------------------------------------------------------------------


def test_records_carry_raw_logits(tiny_cache):
    for record in tiny_cache:
        assert record.a_logits is not None
        # Observed replicate only: length T, row-aligned with A / A_probs.
        assert record.a_logits.shape == record.A_probs.shape
        assert record.a_logit_mean is not None
        assert record.a_logit_std is not None
        assert np.isfinite(record.a_logits).all()


def test_reconstruction_matches_stored_propensity(tiny_cache):
    """The invariant that makes the walkthrough faithful: re-applying the
    pipeline to the raw logits reproduces the stored propensity (at the true
    production clamp, not the enlarged display clamp)."""
    for record in tiny_cache:
        pipe = reconstruct_pipeline(record)
        np.testing.assert_allclose(
            pipe["propensity_true"], record.A_probs, rtol=0, atol=1e-5
        )
        assert pipe["recon_residual"] < 1e-5


def test_logits_survive_save_load_round_trip(tiny_spec, tiny_cache, tmp_path):
    gen.save(tiny_cache, tmp_path)
    loaded = gen.load(tiny_spec, tmp_path)
    for a, b in zip(tiny_cache, loaded):
        np.testing.assert_array_equal(a.a_logits, b.a_logits)
        assert a.a_logit_mean == b.a_logit_mean
        assert a.a_logit_std == b.a_logit_std


# --------------------------------------------------------------------------
# Pure transform
# --------------------------------------------------------------------------


def test_standardisation_and_affine_reshape_the_batch(tiny_cache):
    """Standardise pins mean/sd to 0/1; the affine map moves them to b/s."""
    record = tiny_cache[0]
    pipe = reconstruct_pipeline(record)
    # K > 1 here, so the observed slice's own mean/sd need not be exactly 0/1,
    # but the full-population standardisation keeps them close.
    assert pipe["standardised"].mean() == pytest.approx(0.0, abs=0.2)
    assert pipe["standardised"].std(ddof=1) == pytest.approx(1.0, abs=0.2)
    assert pipe["affine"].mean() == pytest.approx(
        pipe["shift"], abs=0.4 * pipe["scale"]
    )
    disp = pipe["display_epsilon"]
    assert (pipe["propensity"] >= disp).all()
    assert (pipe["propensity"] <= 1.0 - disp).all()


def test_reconstruct_rejects_records_without_logits(tiny_cache):
    record = tiny_cache[0]
    stale = gen.PriorDatasetRecord(**{**record.to_dict(), "a_logits": None})
    with pytest.raises(ValueError, match="--force"):
        reconstruct_pipeline(stale)


def test_select_dataset_is_a_valid_index(tiny_cache):
    idx = select_dataset(tiny_cache.records)
    assert 0 <= idx < len(tiny_cache)


# --------------------------------------------------------------------------
# End-to-end render
# --------------------------------------------------------------------------


def test_analysis_end_to_end(tiny_cache, tmp_path):
    artifacts = TreatmentWalkthroughAnalysis().run(tiny_cache, tmp_path)

    for path in artifacts.figures + artifacts.tables:
        assert path.exists() and path.stat().st_size > 0
    names = {p.name for p in artifacts.figures}
    assert "treatment_walkthrough.pdf" in names
    assert (tmp_path / "summary.json").exists()
    assert artifacts.summary["reconstruction_residual"] < 1e-5
    assert artifacts.summary["tracked_A"] in (0, 1)
