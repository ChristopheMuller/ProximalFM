"""ProximalFM as an ordinary benchmark method.

The load-bearing claim of the move off the `models:` block is that scoring one
episode at a time through the `CATEEstimator` interface is the *same function*
as the batched pass it replaced — the model is invariant to the episode axis, so
only wall time changed. That is what the equivalence test here pins.
"""

import numpy as np
import pytest
import torch
import yaml

from benchmarks import run
from benchmarks.compute import ComputeConfig
from benchmarks.methods import build_method, method_estimand, normalize_method_cfg
from benchmarks.methods.proximalfm import _MODEL_CACHE, resolve_checkpoints
from proximalfm._model import ProximalFM

MODEL_KWARGS = dict(
    embed_dim=16,
    col_num_blocks=1,
    col_nhead=2,
    col_num_inds=8,
    row_num_blocks=1,
    row_nhead=2,
    row_num_cls=4,
    icl_num_blocks=1,
    icl_nhead=2,
    nbins=32,
)


@pytest.fixture(autouse=True)
def _clear_model_cache():
    _MODEL_CACHE.clear()
    yield
    _MODEL_CACHE.clear()


def write_ckpt(path, model, step=1):
    torch.save(
        {"config": MODEL_KWARGS, "state_dict": model.state_dict(), "curr_step": step},
        path,
    )
    return path


@pytest.fixture
def ckpt(tmp_path):
    torch.manual_seed(0)
    model = ProximalFM(**MODEL_KWARGS).eval()
    return write_ckpt(tmp_path / "step-000010.ckpt", model), model


def make_episode(n_ctx=12, n_q=5, d_X=3, d_W=2, d_Z=2, seed=0):
    rng = np.random.default_rng(seed)
    n = n_ctx + n_q

    def block(d):
        return rng.normal(size=(n, d)).astype(np.float32)

    X, W, Z = block(d_X), block(d_W), block(d_Z)
    A = (rng.random(n_ctx) < 0.5).astype(np.float32)
    Y = rng.normal(size=n_ctx).astype(np.float32)
    return dict(X=X, W=W, Z=Z, A=A, Y=Y, n_ctx=n_ctx)


def fit_predict(cfg, ep, compute=None):
    _, method = build_method(cfg, compute or ComputeConfig())
    k = ep["n_ctx"]
    method.fit(ep["X"][:k], ep["A"], ep["Y"], W=ep["W"][:k], Z=ep["Z"][:k])
    return method.predict_variants(ep["X"][k:], W=ep["W"][k:], Z=ep["Z"][k:])


# ─────────────────────────────────────────────────────────────────────────────
# Equivalence with the batched pass it replaced
# ─────────────────────────────────────────────────────────────────────────────


def test_per_episode_prediction_matches_a_direct_batched_call(ckpt):
    """One episode through fit/predict == that episode inside a batched predict.

    The removed `run_neural_model` fed three episodes at a time into the model's
    own batch dimension. Query rows never attend to each other and every
    standardization statistic is context-only, so the batch axis is pure
    batching — this is the check that the interface change kept it that way.
    """
    path, model = ckpt
    episodes = [make_episode(seed=s) for s in range(3)]

    per_episode = np.stack(
        [
            fit_predict(
                {"name": "PFM", "cls": "ProximalFM", "checkpoint": str(path)}, ep
            )[None]
            for ep in episodes
        ]
    )

    stack = lambda key: torch.tensor(np.stack([ep[key] for ep in episodes]))
    batched = model.predict(
        X=stack("X"),
        W=stack("W"),
        Z=stack("Z"),
        A=stack("A"),
        y_train=stack("Y"),
        d_X=torch.tensor([ep["X"].shape[-1] for ep in episodes]),
        d_W=torch.tensor([ep["W"].shape[-1] for ep in episodes]),
        d_Z=torch.tensor([ep["Z"].shape[-1] for ep in episodes]),
        target="cate",
        output_type="mean",
    ).numpy()

    assert per_episode.shape == batched.shape
    np.testing.assert_allclose(per_episode, batched, atol=1e-5)


def test_the_checkpoint_is_loaded_once_per_process(ckpt):
    """`build_method` runs per episode; the checkpoint load must not."""
    path, _ = ckpt
    cfg = {"name": "PFM", "cls": "ProximalFM", "checkpoint": str(path)}
    for seed in range(3):
        fit_predict(cfg, make_episode(seed=seed))
    assert len(_MODEL_CACHE) == 1


def test_predict_ate_averages_context_x_with_query_proxies_hidden(ckpt):
    path, _ = ckpt
    ep = make_episode(n_ctx=12, n_q=5)
    _, method = build_method(
        {"name": "PFM", "cls": "ProximalFM", "checkpoint": str(path)}
    )
    k = ep["n_ctx"]
    method.fit(ep["X"][:k], ep["A"], ep["Y"], W=ep["W"][:k], Z=ep["Z"][:k])

    # ProximalFM hides query W/Z, so arbitrary proxy values leave the CATEs
    # unchanged. Its scalar ATE is their mean over every fitted context X.
    expected = method.predict(
        ep["X"][:k],
        W=np.full_like(ep["W"][:k], 123.0),
        Z=np.full_like(ep["Z"][:k], -456.0),
    ).mean()
    assert method.predict_ate() == pytest.approx(expected)


def test_swap_diag_exchanges_proxy_values_and_dimensions(ckpt):
    """The diagnostic presents original Z as W and original W as Z."""
    path, model = ckpt
    ep = make_episode(d_W=2, d_Z=3)
    cfg = {
        "name": "PFM-swap",
        "cls": "ProximalFM",
        "checkpoint": str(path),
        "swap_diag": True,
    }

    swapped = fit_predict(cfg, ep)[None]
    expected = (
        model.predict(
            X=torch.tensor(ep["X"]).unsqueeze(0),
            # The role-labelled inputs are exchanged, so these dimensions must be
            # exchanged too when they differ.
            W=torch.tensor(ep["Z"]).unsqueeze(0),
            Z=torch.tensor(ep["W"]).unsqueeze(0),
            A=torch.tensor(ep["A"]).unsqueeze(0),
            y_train=torch.tensor(ep["Y"]).unsqueeze(0),
            d_X=torch.tensor([ep["X"].shape[-1]]),
            d_W=torch.tensor([ep["Z"].shape[-1]]),
            d_Z=torch.tensor([ep["W"].shape[-1]]),
            target="cate",
            output_type="mean",
        )
        .numpy()
        .squeeze(0)
    )

    np.testing.assert_allclose(swapped, expected, atol=1e-5)


def test_device_comes_from_compute_not_from_the_config(ckpt):
    """A stale `device:` in a YAML cannot outrank the resolved ComputeConfig."""
    path, _ = ckpt
    cfg = {
        "name": "PFM",
        "cls": "ProximalFM",
        "checkpoint": str(path),
        "device": "mps",
    }
    _, method = build_method(cfg, ComputeConfig(device="cpu"))
    assert method.device == torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint resolution
# ─────────────────────────────────────────────────────────────────────────────


def test_directory_prefers_best_model_then_highest_step(tmp_path):
    torch.manual_seed(0)
    model = ProximalFM(**MODEL_KWARGS)
    write_ckpt(tmp_path / "step-000010.ckpt", model)
    write_ckpt(tmp_path / "step-000200.ckpt", model)

    cfg = {"name": "PFM", "checkpoint": str(tmp_path)}
    assert resolve_checkpoints(cfg) == [("_200", tmp_path / "step-000200.ckpt")]

    write_ckpt(tmp_path / "best_model.ckpt", model)
    assert resolve_checkpoints(cfg) == [("", tmp_path / "best_model.ckpt")]


def test_all_expands_into_outputs_in_step_order(tmp_path):
    """`all: true` becomes an ordinary multi-output entry before the run starts.

    The output names have to exist before any episode runs: they are what the
    runner pre-allocates its arrays with and what the skip-if-cached check
    globs for.
    """
    torch.manual_seed(0)
    model = ProximalFM(**MODEL_KWARGS)
    for step in (200, 10, 30):
        write_ckpt(tmp_path / f"step-{step:06d}.ckpt", model)

    cfg = normalize_method_cfg(
        {"name": "PFM", "cls": "ProximalFM", "checkpoint": str(tmp_path), "all": True}
    )
    assert [o["name"] for o in cfg["outputs"]] == [
        "PFM_step_10",
        "PFM_step_30",
        "PFM_step_200",
    ]
    assert [method_estimand(cfg, o) for o in cfg["outputs"]] == ["cate(x)"] * 3


def test_every_checkpoint_scores_off_one_stored_context(tmp_path):
    """A multi-checkpoint entry fits once and predicts once per checkpoint."""
    torch.manual_seed(0)
    write_ckpt(tmp_path / "step-000010.ckpt", ProximalFM(**MODEL_KWARGS))
    torch.manual_seed(1)
    write_ckpt(tmp_path / "step-000020.ckpt", ProximalFM(**MODEL_KWARGS))

    cfg = normalize_method_cfg(
        {"name": "PFM", "cls": "ProximalFM", "checkpoint": str(tmp_path), "all": True}
    )
    preds = fit_predict(cfg, make_episode())

    assert sorted(preds) == ["PFM_step_10", "PFM_step_20"]
    assert all(np.isfinite(p).all() and p.shape == (5,) for p in preds.values())
    # Different weights, so this is also a check that each output really used
    # its own checkpoint rather than the first one twice.
    assert not np.allclose(preds["PFM_step_10"], preds["PFM_step_20"])


def test_missing_checkpoint_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_checkpoints({"name": "PFM", "checkpoint": str(tmp_path / "nope")})


def test_empty_checkpoint_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No \\*.ckpt"):
        resolve_checkpoints({"name": "PFM", "checkpoint": str(tmp_path)})


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


def test_config_models_block_is_refused_with_the_translation(tmp_path):
    """`models:` is gone, not aliased — it named a pipeline that no longer exists."""
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "run": {},
                "models": [{"name": "PFM", "checkpoint": "runs/x"}],
            }
        )
    )
    with pytest.raises(ValueError, match="cls: ProximalFM"):
        run.load_config(str(path))


def test_config_data_block_is_refused(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"run": {}, "data": {"val_dir": "x/val"}}))
    with pytest.raises(ValueError, match="--val-dir"):
        run.load_config(str(path))


def test_config_context_sizes_are_cli_only(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"run": {"context_sizes": [128]}}))
    with pytest.raises(ValueError, match="--context-sizes"):
        run.load_config(str(path))
