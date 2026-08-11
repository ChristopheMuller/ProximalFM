"""The ``lit_eval`` skeleton: the plumbing every literature adapter inherits.

The load-bearing test is ``test_mc_target_converges_to_the_closed_form_tau_star``:
the dummy DGP is linear-Gaussian precisely so ``tau*(x)`` has an exact
expression, and that is the only direct evidence that the replicate layout
estimates ``E[Y1 - Y0 | X = x]`` rather than some neighbouring conditional. The
rest pin the contracts a consumer relies on: the ``val/`` round trip, the
sidecar's alignment with what is on disk, the ``mc_cate=1`` gate, and the
refusal to rebuild a named set with different content.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lit_eval.base import POOL_FIELDS, Dims, EpisodeSpec, ProximalDataset, RawEpisode
from lit_eval.build import build, parse_args
from lit_eval.registry import available_datasets, build_dataset
from lit_eval.sidecar import (
    LitBuildConfig,
    LitSetConflictError,
    lit_content_digest,
    load_lit_meta,
)
from prior_eval.build_prior_frozen_set import frozen_content_digest
from proximalfm.prior._genload import LoadPriorDataset

N_MAX = 96
N_Q = 12
K = 512
N_DATASETS = 4


def build_set(root: Path, name: str = "toy", **overrides):
    argv = [
        "--dataset",
        overrides.pop("dataset", "dummy_linear"),
        "--name",
        name,
        "--out_root",
        str(root),
        "--n_datasets",
        str(overrides.pop("n_datasets", N_DATASETS)),
        "--n_max",
        str(overrides.pop("n_max", N_MAX)),
        "--n_q",
        str(overrides.pop("n_q", N_Q)),
        "--mc_cate",
        str(overrides.pop("mc_cate", K)),
        "--seed",
        str(overrides.pop("seed", 7)),
    ]
    for key, value in overrides.pop("options", {}).items():
        argv += ["--option", f"{key}={value}"]
    argv += overrides.pop("extra", [])
    args = parse_args(argv)
    build(args)
    return root / name, args


@pytest.fixture(scope="module")
def toy_set(tmp_path_factory):
    return build_set(tmp_path_factory.mktemp("lit"))


# ---------------------------------------------------------------------------
# The estimand
# ---------------------------------------------------------------------------


def test_mc_target_converges_to_the_closed_form_tau_star(toy_set):
    """``MU1 - MU0`` estimates ``tau*(x) = E[Y1 - Y0 | X = x]``, nothing else.

    ``S2`` is the sampling variance of that estimate, so the standardized
    residual is ~N(0, 1) and a 5-sigma envelope is generous. A systematic
    failure (conditioning on ``W`` instead of marginalizing it, say) shifts the
    mean and blows straight through it; Monte Carlo noise alone does not.
    """
    out, _ = toy_set
    meta = load_lit_meta(out)
    for rec in meta["episodes"]:
        exact = rec["tau_star_exact"]
        assert exact is not None, "dummy_linear must publish its closed form"
        z = (rec["tau_star"] - exact) / rec["s2"].clamp(min=1e-12).sqrt()
        assert z.abs().max() < 5.0, f"episode {rec['index']}: max |z| = {z.abs().max()}"
        # And unbiased in the mean, tightened by averaging over the query rows.
        assert abs(float(z.mean())) < 1.0


def test_s2_is_the_paired_variance_not_the_sum(toy_set):
    """The shared outcome noise cancels in ``Y1 - Y0``; ``S2`` must reflect that.

    Under this DGP the only surviving per-replicate spread is the latent's
    contribution to ``tau``, which is far smaller than ``Var(Y_a)``. Summing the
    two arms' variances instead would inflate ``S2`` by roughly twice the
    outcome variance -- a large, silent over-statement of target noise.
    """
    out, _ = toy_set
    meta = load_lit_meta(out)
    raw = torch.load(out / "val" / "batch_000000.pt", weights_only=True)
    y = raw["Y_obs"].float()
    for rec in meta["episodes"]:
        i = rec["index"]
        var_y = float(y[i].var())
        assert float(rec["s2"].max()) < 0.5 * var_y / K


# ---------------------------------------------------------------------------
# On-disk format
# ---------------------------------------------------------------------------


def test_val_files_carry_every_field_the_loader_expects(toy_set):
    out, _ = toy_set
    raw = torch.load(out / "val" / "batch_000000.pt", weights_only=True)
    for key in ("U", "X", "W", "Z", "A", "Y_obs", "MU0", "MU1", "S2", "K_eff"):
        assert key in raw, key
    meta = json.loads((out / "val" / "metadata.json").read_text())
    assert meta["mc_cate"] == K
    assert meta["mc_targets_on_context"] is False


def test_round_trips_through_the_standard_val_loader(toy_set):
    out, args = toy_set
    loader = LoadPriorDataset(
        data_dir=str(out / "val"), batch_size=args.n_datasets, device="cpu"
    )
    batch = next(iter(loader))
    MU0, MU1, S2, K_eff = batch[9], batch[10], batch[17], batch[18]
    n_max = int(batch[16][0])
    assert n_max == N_MAX
    assert (K_eff == K).all()
    # NaN on the context prefix, finite on the query rows: the contract every
    # consumer reads MU under.
    assert torch.isnan(MU0[:, :n_max]).all()
    assert torch.isfinite(MU1[:, n_max:]).all()
    assert (S2[:, n_max:] >= 0).all()


def test_sidecar_targets_match_the_data_on_disk(toy_set):
    """Record ``i`` must describe episode ``i`` -- the alignment invariant."""
    out, args = toy_set
    meta = load_lit_meta(out)
    assert meta["global"]["set_kind"] == "literature"
    assert meta["global"]["dataset"] == "dummy_linear"
    assert len(meta["episodes"]) == args.n_datasets

    raw = torch.load(out / "val" / "batch_000000.pt", weights_only=True)
    n_max = meta["global"]["n_max"]
    for rec in meta["episodes"]:
        i = rec["index"]
        disk = (raw["MU1"][i] - raw["MU0"][i]).float().reshape(-1)[n_max:]
        assert torch.allclose(disk, rec["tau_star"], atol=1e-6)
        # Head units: tau by y_std, s2 by the *variance*. Getting one of the two
        # wrong is invisible downstream, so it is pinned here.
        y_std = rec["y_context_std"]
        assert torch.allclose(rec["tau_star_std"], rec["tau_star"] / y_std, atol=1e-6)
        assert torch.allclose(rec["s2_std"], rec["s2"] / y_std**2, atol=1e-6)


def test_completeness_and_dims_hold_for_every_episode(toy_set):
    out, _ = toy_set
    meta = load_lit_meta(out)
    for rec in meta["episodes"]:
        assert min(rec["d_W"], rec["d_Z"]) >= rec["d_U"]


# ---------------------------------------------------------------------------
# The mc_cate = 1 gate
# ---------------------------------------------------------------------------


def test_point_only_build_writes_no_sidecar(tmp_path):
    """The sidecar's *absence* is what makes the trainer refuse a K=1 set."""
    out, _ = build_set(tmp_path, name="point", mc_cate=1, n_max=32, n_q=8)
    assert (out / "val" / "metadata.json").exists()
    assert not (out / "frozen_meta.pt").exists()


def test_a_dataset_without_a_forward_model_refuses_replicates():
    """Real data cannot be replicated; asking for it must fail, not degrade."""

    class NoForwardModel(ProximalDataset):
        name = "no_forward_model"
        supports_replicates = False

        @property
        def max_dims(self):
            return Dims(1, 1, 1, 1)

        def sample_episode(self, spec, rep):  # pragma: no cover - never reached
            raise AssertionError("must not be called")

    spec = EpisodeSpec(index=0, seed=0, seq_len=16, train_size=8, mc_cate=4)
    with pytest.raises(ValueError, match="no forward model"):
        NoForwardModel().generate(spec)


# ---------------------------------------------------------------------------
# Identity and reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_rebuilds_the_same_bytes(tmp_path):
    a, _ = build_set(tmp_path / "a", n_max=32, n_q=8, mc_cate=16)
    b, _ = build_set(tmp_path / "b", n_max=32, n_q=8, mc_cate=16)
    ra = torch.load(a / "val" / "batch_000000.pt", weights_only=True)
    rb = torch.load(b / "val" / "batch_000000.pt", weights_only=True)
    for key in ("X", "W", "Z", "A", "Y_obs", "MU0", "MU1", "S2"):
        # equal_nan: MU/S2 are NaN on the context prefix by contract, and
        # torch.equal calls two NaNs different.
        torch.testing.assert_close(
            ra[key], rb[key], rtol=0, atol=0, equal_nan=True, msg=key
        )
    assert lit_content_digest(load_lit_meta(a)) == lit_content_digest(load_lit_meta(b))


def test_a_dataset_option_changes_the_identity(tmp_path):
    a, _ = build_set(tmp_path / "a", n_max=32, n_q=8, mc_cate=16)
    b, _ = build_set(
        tmp_path / "b", n_max=32, n_q=8, mc_cate=16, options={"sigma_w": 0.9}
    )
    assert lit_content_digest(load_lit_meta(a)) != lit_content_digest(load_lit_meta(b))


def test_rebuilding_a_named_set_with_different_content_is_refused(tmp_path):
    """Prevention, not detection: a rewritten sidecar is self-consistent."""
    build_set(tmp_path, name="fixed", n_max=32, n_q=8, mc_cate=16)
    with pytest.raises(LitSetConflictError):
        build_set(tmp_path, name="fixed", n_max=32, n_q=8, mc_cate=16, seed=999)
    # --force is the documented override.
    build_set(
        tmp_path, name="fixed", n_max=32, n_q=8, mc_cate=16, seed=999, extra=["--force"]
    )


def test_digest_is_recomputable_when_the_cached_field_is_missing(toy_set):
    """The stored digest is a cache; identity must survive its absence."""
    out, _ = toy_set
    meta = load_lit_meta(out)
    stored = meta["global"]["content_digest"]
    stripped = {**meta, "global": {k: v for k, v in meta["global"].items()}}
    stripped["global"].pop("content_digest")
    assert lit_content_digest(stripped) == stored
    # And prior_eval's shared entry point dispatches on set_kind to get here.
    assert frozen_content_digest(stripped) == stored


def test_row_split_spellings_agree(tmp_path):
    """``--n_max/--n_q`` and ``--seq_len/--prop_train`` are one parameterisation."""
    a = parse_args(
        ["--dataset", "dummy_linear", "--name", "a", "--n_max", "96", "--n_q", "32"]
    )
    b = parse_args(
        [
            "--dataset",
            "dummy_linear",
            "--name",
            "a",
            "--seq_len",
            "128",
            "--prop_train",
            "0.75",
        ]
    )
    assert (a.seq_len, a.n_max, a.n_q) == (b.seq_len, b.n_max, b.n_q)
    assert a.prop_train == b.prop_train

    def cfg(args):
        return LitBuildConfig(
            dataset="dummy_linear",
            n_datasets=4,
            seq_len=args.seq_len,
            prop_train=args.prop_train,
            mc_cate=8,
            seed=1,
            options={},
        ).content_digest()

    assert cfg(a) == cfg(b)


def test_mixing_the_two_row_spellings_is_refused():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dataset",
                "dummy_linear",
                "--name",
                "a",
                "--n_max",
                "96",
                "--n_q",
                "32",
                "--seq_len",
                "128",
            ]
        )


def test_registry_exposes_the_reference_adapter():
    assert "dummy_linear" in available_datasets()
    ds = build_dataset("dummy_linear", sigma_w=0.25)
    assert ds.options["sigma_w"] == 0.25
    with pytest.raises(TypeError, match="unknown option"):
        build_dataset("dummy_linear", not_a_knob=1)


def test_pool_field_order_matches_the_realistic_builder():
    """Both builders feed one writer; a reordering here corrupts silently."""
    from synt_eval.build_proximal_datasets import POOL_FIELDS as REALISTIC_FIELDS

    assert POOL_FIELDS == REALISTIC_FIELDS
