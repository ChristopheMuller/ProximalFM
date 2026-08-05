"""Tests for the mechanism input-composition diagnostics and analysis.

Three layers, matching the machinery under test:

- ``proximalfm.prior._diagnostics`` + the fixed-indices ``MLPSCM.forward`` --
  exact unit tests on hand-built mechanisms.
- diagnostics collection in ``generate_dataset`` -- above all, *purity*:
  enabling collection must leave the generated data bit-identical.
- ``analysis_prior.analyses.mechanism_inputs`` -- share arithmetic on
  hand-built records, and an end-to-end render on a tiny cache.
"""

import random

import numpy as np
import pytest
import torch

from analysis_prior import generate as gen
from analysis_prior.analyses.mechanism_inputs import (
    MechanismInputsAnalysis,
    mechanism_frame,
)
from analysis_prior.config import PriorSpec
from proximalfm.prior import PriorDataset
from proximalfm.prior._diagnostics import (
    block_slices,
    first_layer_variances,
)
from proximalfm.prior._mlp_scm import MLPSCM


# --------------------------------------------------------------------------
# Diagnostics primitives
# --------------------------------------------------------------------------


def test_block_slices_partition_the_input():
    slices = block_slices({"a": 2, "b": 0, "c": 3})
    assert slices == {"a": slice(0, 2), "b": slice(2, 2), "c": slice(2, 5)}


def _mlp_with_first_layer(weight_rows):
    """MLPSCM whose first linear layer is set to ``weight_rows`` exactly."""
    weight = torch.tensor(weight_rows, dtype=torch.float32)
    hidden, in_dim = weight.shape
    mlp = MLPSCM(in_dim=in_dim, num_outputs=1, num_layers=3, hidden_dim=hidden)
    with torch.no_grad():
        mlp.layers[0].weight.copy_(weight)
        mlp.layers[0].bias.zero_()
    return mlp


def test_first_layer_variances_are_exact():
    # One hidden unit; block "a" (cols 0-1) enters with weights (1, 0),
    # block "b" (col 2) is wired to zero.
    mlp = _mlp_with_first_layer([[1.0, 0.0, 0.0]])
    x = torch.randn(2000, 3)
    slices = block_slices({"a": 2, "b": 1})
    variances = first_layer_variances(mlp, x, slices)

    expected_a = float(x[:, 0].var(unbiased=False))
    assert variances["a"] == pytest.approx(expected_a, rel=1e-5)
    assert variances["b"] == 0.0


def test_zero_width_block_has_zero_variance():
    torch.manual_seed(0)
    mlp = MLPSCM(in_dim=3, num_outputs=1, num_layers=3, hidden_dim=8)
    x = torch.randn(100, 3)
    slices = block_slices({"a": 3, "empty": 0})
    assert first_layer_variances(mlp, x, slices)["empty"] == 0.0


# --------------------------------------------------------------------------
# Collection in the generator
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_spec():
    return PriorSpec(
        name="tiny-mech",
        n_datasets=2,
        seq_len=60,
        mc_cate=3,
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


@pytest.fixture(scope="module")
def tiny_cache(tiny_spec):
    return gen.generate(tiny_spec, n_jobs=1, progress=False)


def _generate_without_diagnostics(spec, index):
    """Mirror of ``gen._generate_one`` with diagnostics left off."""
    seed = spec.dataset_seed(index)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    fixed_hp, sampled_hp = spec.resolved_hp()
    dataset = PriorDataset(
        batch_size=1,
        batch_size_per_gp=1,
        batch_size_per_subgp=1,
        min_dim_U=spec.min_dim_U,
        max_dim_U=spec.max_dim_U,
        min_dim_W=spec.min_dim_W,
        max_dim_W=spec.max_dim_W,
        min_dim_Z=spec.min_dim_Z,
        max_dim_Z=spec.max_dim_Z,
        min_dim_X=spec.min_dim_X,
        max_dim_X=spec.max_dim_X,
        min_seq_len=None,
        max_seq_len=spec.seq_len,
        log_seq_len=False,
        seq_len_per_gp=False,
        replay_small=False,
        prior_type=spec.prior_type,
        mc_cate=spec.mc_cate,
        scm_fixed_hp=fixed_hp,
        scm_sampled_hp=sampled_hp,
        n_jobs=1,
        device="cpu",
    )
    batch = dataset.get_batch()
    assert dataset.last_diagnostics == [None]
    return batch


def test_collection_leaves_generated_data_bit_identical(tiny_spec, tiny_cache):
    """The purity guarantee: diagnostics ride along without touching the data."""
    for index, record in enumerate(tiny_cache):
        U, X, W, Z, A, A_probs, Y_obs, *_ = _generate_without_diagnostics(
            tiny_spec, index
        )
        d_U = record.U.shape[1]
        np.testing.assert_array_equal(
            record.U, U[0, :, :d_U].numpy().astype(np.float32)
        )
        np.testing.assert_array_equal(record.Y_obs, Y_obs[0].numpy().astype(np.float32))
        np.testing.assert_array_equal(record.A, A[0].numpy().astype(np.float32))


def test_records_carry_diagnostics(tiny_cache):
    for record in tiny_cache:
        assert record.gamma_hetero is not None
        assert 0.0 <= record.gamma_hetero <= 1.0
        assert "confounding_share_of_U_in_A" in record.sampled_hp
        for mechanism in ("A", "Y"):
            mech = record.mechanisms[mechanism]
            assert set(mech["widths"]) == {"U_sel", "X", "proxy", "eps", "U_aug"}
            assert set(mech["first_layer_var"]) == set(mech["widths"])
            assert all(v >= 0 for v in mech["first_layer_var"].values())


def test_diagnostics_survive_save_load_round_trip(tiny_spec, tiny_cache, tmp_path):
    gen.save(tiny_cache, tmp_path)
    loaded = gen.load(tiny_spec, tmp_path)
    for a, b in zip(tiny_cache, loaded):
        assert a.sampled_hp == b.sampled_hp
        assert a.mechanisms == b.mechanisms
        assert a.gamma_hetero == b.gamma_hetero


# --------------------------------------------------------------------------
# Analysis share arithmetic
# --------------------------------------------------------------------------


def _fake_record(index=0):
    matrix = np.zeros((4, 2), dtype=np.float32)
    vector = np.zeros(4, dtype=np.float32)
    knobs = {
        "dim_U": 3,
        "confounding_share_of_U_in_A": 0.4,
        "confounding_share_of_U_in_Y": 0.2,
        "covariate_share_of_X_in_A": 0.5,
        "covariate_share_of_X_in_Y": 0.5,
        "direct_edge_share_of_Z_in_A": 0.6,
        "direct_edge_share_of_W_in_Y": 0.6,
        "noise_share_of_input_in_A": 0.3,
        "noise_share_of_input_in_Y": 0.3,
    }
    mech = {
        "widths": {"U_sel": 2, "X": 3, "proxy": 1, "eps": 2, "U_aug": 1},
        "first_layer_var": {
            "U_sel": 2.0,
            "X": 3.0,
            "proxy": 1.0,
            "eps": 2.0,
            "U_aug": 2.0,
        },
    }
    return gen.PriorDatasetRecord(
        index=index,
        seed=0,
        U=matrix,
        X=matrix,
        W=matrix,
        Z=matrix,
        A=vector,
        A_probs=vector,
        Y_obs=vector,
        Y0=vector,
        Y1=vector,
        MU0=vector,
        MU1=vector,
        sampled_hp=knobs,
        mechanisms={"A": mech, "Y": mech},
        gamma_hetero=0.5,
    )


def test_mechanism_frame_share_arithmetic():
    frame = mechanism_frame([_fake_record()])
    assert len(frame) == 2  # one row per mechanism

    row = frame[frame["mechanism"] == "A"].iloc[0]
    # U channel = U_sel + U_aug = 3 columns; signal = 3 + 3 + 1 = 7.
    assert row["wiring_share_signal"] == pytest.approx(3 / 7)
    # Variances: U channel 4.0, signal 4 + 3 + 1 = 8.
    assert row["first_layer_share_signal"] == pytest.approx(4 / 8)
    # Of-total shares: eps over U+X+proxy+eps.
    assert row["wiring_share_total_eps"] == pytest.approx(2 / 9)
    assert row["first_layer_share_total_eps"] == pytest.approx(2 / 10)
    assert row["target_share"] == pytest.approx(0.4)
    assert row["knob_proxy"] == pytest.approx(0.6)
    assert row["dim_U"] == 3

    row_y = frame[frame["mechanism"] == "Y"].iloc[0]
    assert row_y["target_share"] == pytest.approx(0.2)


def test_mechanism_frame_rejects_records_without_diagnostics():
    record = _fake_record()
    stale = gen.PriorDatasetRecord(
        **{**record.to_dict(), "sampled_hp": None, "mechanisms": None}
    )
    with pytest.raises(ValueError, match="--force"):
        mechanism_frame([stale])


# --------------------------------------------------------------------------
# End-to-end render
# --------------------------------------------------------------------------


def test_analysis_end_to_end(tiny_cache, tmp_path):
    artifacts = MechanismInputsAnalysis().run(tiny_cache, tmp_path)

    for path in artifacts.figures + artifacts.tables:
        assert path.exists() and path.stat().st_size > 0
    names = {p.name for p in artifacts.figures}
    assert "mechanism_input_composition.pdf" in names
    assert (tmp_path / "summary.json").exists()
    assert artifacts.summary["n_datasets"] == len(tiny_cache)
