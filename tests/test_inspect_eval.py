"""Adjustment-ladder targets produced by ``inspect_eval``."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from inspect_eval.build_proximal_datasets import POOL_FIELDS, build, parse_args
from inspect_eval.export_ladder_predictions import (
    LADDER_OUTPUTS,
    export_ladder_predictions,
)
from inspect_eval.realistic_proximal_mechanisms import (
    N_MECHANISMS,
    RealisticGeneratorConfig,
    RealisticProximalGenerator,
)
from inspect_eval.realistic_sidecar import load_realistic_meta
from inspect_eval.realistic_sidecar import RealisticBuildConfig
from prior_eval.build_prior_frozen_set import frozen_content_digest


def make_gen(tmp_path, *, mc_cate=8, seed=123):
    return RealisticProximalGenerator(
        RealisticGeneratorConfig(
            save_dir=tmp_path / "scratch",
            K=1,
            seq_len=12,
            train_size=8,
            min_dim_U=1,
            max_dim_U=2,
            min_dim_X=1,
            max_dim_X=2,
            min_dim_W=1,
            max_dim_W=2,
            min_dim_Z=1,
            max_dim_Z=2,
            mc_cate=mc_cate,
            adjustment_max_iter=5,
            adjustment_min_samples_leaf=2,
            seed=seed,
        )
    )


@pytest.mark.parametrize("mechanism", range(1, N_MECHANISMS + 1))
def test_all_mechanisms_emit_an_aligned_finite_ladder(tmp_path, mechanism):
    gen = make_gen(tmp_path)
    batch = gen.get_batch(mechanism, batch_size=1, seq_len=12, train_size=8)
    p = dict(zip(POOL_FIELDS, batch))

    assert len(batch) == len(POOL_FIELDS) == 24
    for name in ("Q_naive_X", "Q_adj_W", "Q_adj_Z", "Q_adj_WZ", "Q_true"):
        value = p[name]
        assert value.shape == (1, 12, 1)
        assert torch.isnan(value[:, :8]).all()
        assert torch.isfinite(value[:, 8:]).all()
    torch.testing.assert_close(
        p["Q_true"][:, 8:], (p["MU1"] - p["MU0"])[:, 8:], atol=1e-6, rtol=1e-6
    )


def test_adjustment_formula_preserves_arm_selection(tmp_path, monkeypatch):
    """The adjustment target weights Y_a by arm selection before integrating.

    Returning the fine propensity as the observed-set propensity represents a
    proxy block that reveals all treatment-relevant latent information. Its
    adjustment target must then equal the true CATE, while X-only adjustment
    remains the selected treated-minus-control contrast.
    """
    gen = make_gen(tmp_path, mc_cate=4)
    gen._rep = gen._make_rep(T=3, train_n=1)
    r = gen._rep

    # One context row, followed by two x values with four replicates each.
    x = torch.tensor([[9.0], *([[0.0]] * 4), *([[1.0]] * 4)])
    w = torch.arange(r.N, dtype=torch.float32).reshape(-1, 1)
    z = -w
    pi = torch.tensor(
        [0.5, 0.1, 0.2, 0.8, 0.9, 0.2, 0.4, 0.6, 0.8], dtype=torch.float32
    ).reshape(-1, 1)
    y0 = torch.tensor(
        [0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 8.0], dtype=torch.float32
    ).reshape(-1, 1)
    y1 = y0 + torch.tensor(
        [0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float32
    ).reshape(-1, 1)

    monkeypatch.setattr(
        gen,
        "_cross_fitted_observed_propensity",
        lambda _features, fine: fine.clone(),
    )
    q_x, q_w, q_z, q_wz, q_true = gen._adjustment_ladder(x, w, z, pi, y0, y1)

    expected_true = torch.tensor([1.0, 2.0])
    for adjusted in (q_w, q_z, q_wz, q_true):
        torch.testing.assert_close(adjusted[1:, 0], expected_true)

    # Manual Rao-Blackwellized observed arm contrast for the first x group.
    p0 = pi[1:5, 0]
    manual = (p0 * y1[1:5, 0]).sum() / p0.sum() - ((1 - p0) * y0[1:5, 0]).sum() / (
        1 - p0
    ).sum()
    torch.testing.assert_close(q_x[1, 0], manual)
    assert q_x[1, 0] != pytest.approx(float(q_true[1, 0]))


def test_x_only_adjustment_equals_truth_without_treatment_selection(
    tmp_path, monkeypatch
):
    gen = make_gen(tmp_path, mc_cate=4)
    gen._rep = gen._make_rep(T=2, train_n=1)
    x = torch.tensor([[9.0], [0.0], [0.0], [0.0], [0.0]])
    w = torch.arange(5, dtype=torch.float32).reshape(-1, 1)
    z = -w
    pi = torch.full((5, 1), 0.4)
    y0 = torch.tensor([[0.0], [0.0], [1.0], [3.0], [6.0]])
    y1 = torch.tensor([[0.0], [2.0], [4.0], [8.0], [12.0]])
    monkeypatch.setattr(
        gen,
        "_cross_fitted_observed_propensity",
        lambda _features, fine: fine.clone(),
    )

    q_x, *_rest, q_true = gen._adjustment_ladder(x, w, z, pi, y0, y1)
    torch.testing.assert_close(q_x[1], q_true[1])


def test_ladder_is_reproducible_from_the_seed(tmp_path):
    first = dict(
        zip(
            POOL_FIELDS,
            make_gen(tmp_path, seed=991).get_batch(
                3, batch_size=1, seq_len=12, train_size=8
            ),
        )
    )
    second = dict(
        zip(
            POOL_FIELDS,
            make_gen(tmp_path, seed=991).get_batch(
                3, batch_size=1, seq_len=12, train_size=8
            ),
        )
    )
    for name in ("Q_naive_X", "Q_adj_W", "Q_adj_Z", "Q_adj_WZ", "Q_true"):
        torch.testing.assert_close(first[name], second[name], equal_nan=True)


def test_digest_moves_with_propensity_nuisance_settings():
    base = dict(
        n_datasets=12,
        seq_len=128,
        prop_train=0.75,
        mc_cate=64,
        seed=3,
        mechanisms=[1, 2],
    )
    digest = RealisticBuildConfig(**base).content_digest()
    changed = RealisticBuildConfig(**base, adjustment_max_iter=101).content_digest()
    assert changed != digest


def test_build_persists_the_ladder_in_the_sidecar(tmp_path):
    args = parse_args(
        [
            "--name",
            "tiny",
            "--out_root",
            str(tmp_path),
            "--n_datasets",
            "2",
            "--n_max",
            "8",
            "--n_q",
            "4",
            "--mc_cate",
            "8",
            "--mechanisms",
            "1",
            "9",
            "--max_dim_U",
            "2",
            "--max_dim_X",
            "2",
            "--max_dim_W",
            "2",
            "--max_dim_Z",
            "2",
            "--adjustment_max_iter",
            "5",
            "--adjustment_min_samples_leaf",
            "2",
            "--seed",
            "17",
        ]
    )
    build(args)

    root = tmp_path / "tiny"
    meta = load_realistic_meta(root)
    assert meta["global"]["set_kind"] == "inspect"
    assert frozen_content_digest(meta) == meta["global"]["content_digest"]
    assert meta["global"]["adjustment_folds"] == 2
    assert len(meta["episodes"]) == 2
    for record in meta["episodes"]:
        for name in (
            "Q_naive_X",
            "Q_adj_W",
            "Q_adj_Z",
            "Q_adj_WZ",
            "Q_true",
        ):
            assert record[name].shape == (4,)
            assert record[f"{name}_std"].shape == (4,)
            assert torch.isfinite(record[name]).all()
        torch.testing.assert_close(record["Q_true"], record["tau_star"])

    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["adjustment"]["folds"] == 2
    raw = torch.load(root / "val" / "batch_000000.pt", weights_only=True)
    # The reusable val schema stays canonical; inspection-only targets live in
    # the sidecar and do not leak into LoadPriorDataset's positional contract.
    assert "Q_adj_WZ" not in raw

    paths = export_ladder_predictions(root)
    assert set(paths) == set(LADDER_OUTPUTS)
    for method_name, field in LADDER_OUTPUTS.items():
        with np.load(paths[method_name]) as cached:
            assert cached["preds"].shape == (2, 4)
            assert cached["preds"].dtype == np.float32
            assert cached["times"].shape == (2,)
            np.testing.assert_allclose(
                cached["preds"],
                torch.stack([record[field] for record in meta["episodes"]]).numpy(),
            )

    with pytest.raises(FileExistsError, match="pass --force"):
        export_ladder_predictions(root)


def test_build_refuses_a_nonreplicate_set(tmp_path):
    args = parse_args(["--name", "bad", "--out_root", str(tmp_path), "--mc_cate", "1"])
    with pytest.raises(ValueError, match="mc_cate K with K > 1"):
        build(args)
