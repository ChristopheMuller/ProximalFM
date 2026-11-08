"""Integration checks for the proximal ablation prior."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from proximalfm._model import ProximalFM
from proximalfm.prior._ablation_dataset import AblationSCMPrior
from proximalfm.prior._genload import LoadPriorDataset, SavePriorDataset
from proximalfm.train._run import Trainer, compute_cate_noise_aware_nll
from proximalfm.train._train_config import build_parser
from eval_datasets.prior_eval.build_prior_frozen_set import (
    BuildConfig,
    build,
    generate_episode,
    load_frozen_meta,
)


def test_training_bank_round_trip_and_masking(tmp_path):
    args = SimpleNamespace(
        save_dir=str(tmp_path),
        batch_size=2,
        batch_size_per_gp=1,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_X=2,
        max_dim_X=4,
        min_dim_W=2,
        max_dim_W=3,
        min_dim_Z=1,
        max_dim_Z=4,
        min_seq_len=None,
        max_seq_len=64,
        log_seq_len=False,
        seq_len_per_gp=False,
        n_query=16,
        replay_small=False,
        prior_type="ablation_scm",
        mc_cate=16,
        mc_targets_on_context=False,
        num_batches=1,
        resume_from=0,
        n_jobs=1,
        num_threads_per_generate=1,
        device="cpu",
    )
    np.random.seed(13)
    torch.manual_seed(13)
    saver = SavePriorDataset(args)
    original = saver.prior.get_batch()
    with patch.object(saver.prior, "get_batch", return_value=original):
        saver.run()
    loaded = next(iter(LoadPriorDataset(tmp_path, batch_size=2, max_batches=1)))
    for expected, actual in zip(original, loaded):
        # Sparse storage trims trailing padding to the maximum active width.
        if expected.ndim == 3 and expected.shape[-1] != actual.shape[-1]:
            expected = expected[..., : actual.shape[-1]]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert loaded[1].shape == (2, 80, int(loaded[12].max()))
    assert (loaded[12] >= 2).all()
    for i in range(2):
        assert loaded[1][i, :, int(loaded[12][i]) :].count_nonzero() == 0
        assert loaded[2][i, :, int(loaded[13][i]) :].count_nonzero() == 0
        assert loaded[3][i, :, int(loaded[14][i]) :].count_nonzero() == 0
        assert all(loaded[j][i, :64].isnan().all() for j in (9, 10, 17))
        assert all(loaded[j][i, 64:].isfinite().all() for j in (9, 10, 17))
        assert loaded[18][i] == 16

    # Query proxies and treatment must not affect predictions. Query X remains visible.
    U, X, W, Z, A, _, Y = loaded[:7]
    model = ProximalFM(
        embed_dim=16,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=4,
        icl_num_blocks=1,
        icl_nhead=2,
        nbins=20,
        loss_type="cate_single",
    )
    model.eval()
    dims = (loaded[12], loaded[14], loaded[13])
    with torch.no_grad():
        base = model(X, Z, W, A, Y[:, :64], *dims)
        w, z, a = W.clone(), Z.clone(), A.clone()
        w[:, 64:] = 1000
        z[:, 64:] = -1000
        a[:, 64:] = 1 - a[:, 64:]
        changed = model(X, z, w, a, Y[:, :64], *dims)
    torch.testing.assert_close(base, changed, rtol=0, atol=0)

    # Feed the loaded bank through the real training loss and inspect its inputs.
    config = build_parser().parse_args(
        [
            "--max_steps",
            "1",
            "--batch_size",
            "2",
            "--micro_batch_size",
            "2",
            "--embed_dim",
            "16",
            "--row_nhead",
            "4",
            "--col_num_blocks",
            "1",
            "--row_num_blocks",
            "1",
            "--icl_num_blocks",
            "1",
            "--amp",
            "False",
            "--device",
            "cpu",
            "--wandb_mode",
            "disabled",
            "--loss_type",
            "noise_aware",
            "--checkpoint_dir",
            str(tmp_path / "checkpoint"),
        ]
    )
    trainer = Trainer(config)
    with patch(
        "proximalfm.train._run.compute_cate_noise_aware_nll",
        wraps=compute_cate_noise_aware_nll,
    ) as loss:
        metrics = trainer.run_micro_batch(loaded, 0, 1)
    assert metrics and metrics["noise/mean_K"] == 16
    context_std = Y[:, :64].std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-3)
    expected_target = (loaded[10][:, 64:] - loaded[9][:, 64:]) / context_std
    expected_variance = loaded[17][:, 64:] / context_std.square()
    torch.testing.assert_close(loss.call_args.args[1], expected_target)
    torch.testing.assert_close(loss.call_args.kwargs["target_var"], expected_variance)


@pytest.mark.parametrize("dim_x", [1, 4])
def test_multivariate_normal_mechanisms_and_mc_variance(dim_x):
    torch.manual_seed(832)
    prior = AblationSCMPrior(n_jobs=1, device="cpu")
    n_context, n_query, k = 20000, 4096, 16
    params = dict(
        seq_len=n_context + n_query,
        train_size=n_context,
        mc_cate=k,
        mc_targets_on_context=False,
        dim_U=3,
        max_dim_U=4,
        dim_X=dim_x,
        max_dim_X=4,
        dim_W=3,
        max_dim_W=4,
        dim_Z=4,
        max_dim_Z=4,
        collect_mechanism_diagnostics=True,
    )
    (
        u,
        x,
        w,
        z,
        a,
        p,
        y,
        y0,
        y1,
        mu0,
        mu1,
        du,
        dx,
        _,
        _,
        s2,
        _,
        diag,
    ) = prior.generate_dataset(params)
    c, m = diag["params"], diag["mechanisms"]
    assert torch.count_nonzero(u[:, 3:]) == 0
    assert torch.count_nonzero(x[:, dim_x:]) == 0
    u, x, w = u[:, :3], x[:, :dim_x], w[:, :3]
    assert du == 3
    assert dx == dim_x
    torch.testing.assert_close(u.mean(0), torch.zeros(3), rtol=0, atol=0.04)
    torch.testing.assert_close(torch.cov(u.T), torch.eye(3), rtol=0, atol=0.04)
    lw, lz = m["loading_W"], m["loading_Z"]
    torch.testing.assert_close(lw.T @ lw, torch.eye(3), rtol=0, atol=1e-5)
    torch.testing.assert_close(lz.T @ lz, torch.eye(3), rtol=0, atol=1e-5)
    cross_cov = (w - w.mean(0)).T @ (z - z.mean(0)) / len(u)
    torch.testing.assert_close(cross_cov, lw @ lz.T, rtol=0, atol=0.04)
    x_a, x_au, x_y, x_tau = (x @ m["x_directions"]).unbind(1)
    torch.testing.assert_close(m["x_directions"].norm(dim=0), torch.ones(4))
    cate = c["tau0"] + c["beta"] * x_tau
    effect = cate + c["gamma"] * (u @ m["effect_direction"])
    torch.testing.assert_close(y1 - y0, effect, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y, torch.where(a == 1, y1, y0))

    latent_score = u @ m["confounding_direction"]
    logits = (
        c["c0"]
        + c["cx"] * x_a
        + (c["cu"] + c["cxu"] * x_au) * latent_score
        + z @ m["c_Z"]
    )
    torch.testing.assert_close(p, 0.1 + 0.8 * logits.sigmoid())
    proxy_latent = w @ lw
    bridge = (
        c["b0"]
        + c["bx"] * x_y
        + c["bu"] * (proxy_latent @ m["confounding_direction"])
        + a * (cate + c["gamma"] * (proxy_latent @ m["effect_direction"]))
    )
    instruments = torch.column_stack((torch.ones_like(x_a), x, z))
    for arm in (0, 1):
        selected = a == arm
        moments = instruments[selected].T @ (y - bridge)[selected] / selected.sum()
        torch.testing.assert_close(
            moments, torch.zeros_like(moments), rtol=0, atol=0.06
        )

    error = mu1[n_context:] - mu0[n_context:] - cate[n_context:]
    expected_variance = c["gamma"] ** 2 / k
    assert abs(error.mean()) < 0.02
    assert error.var().item() == pytest.approx(expected_variance, rel=0.1)
    assert s2[n_context:].mean().item() == pytest.approx(expected_variance, rel=0.05)


def test_frozen_ablation_set_preserves_high_k_reference(tmp_path):
    cfg = BuildConfig(
        n_datasets=2,
        n_max=32,
        n_q=16,
        mc_cate=20_000,
        seed=431,
        prior_mc_mem_budget=20_000_000,
        prior_type="ablation_scm",
        min_dim_U=2,
        max_dim_U=2,
        min_dim_X=4,
        max_dim_X=4,
        min_dim_W=2,
        max_dim_W=3,
        min_dim_Z=3,
        max_dim_Z=4,
    )
    val = build(cfg, tmp_path, datasets_per_file=2, progress=False)
    records = load_frozen_meta(tmp_path)["episodes"]
    batch = next(iter(LoadPriorDataset(val, batch_size=2, max_batches=1)))
    for i, record in enumerate(records):
        q = slice(cfg.n_max, None)
        c = record["sampled_hp"]
        saved = batch[10][i, q] - batch[9][i, q]
        direction = record["mechanisms"]["x_directions"][:, -1]
        expected = c["tau0"] + c["beta"] * (batch[1][i, q] @ direction)
        assert (saved - expected).square().mean().sqrt() < 2 * cfg.mc_cate**-0.5
        assert batch[18][i] == record["k_eff"] == cfg.mc_cate
        assert (batch[17][i, q] > 0).all()
        torch.testing.assert_close(record["s2"], batch[17][i, q], rtol=0, atol=0)
        torch.testing.assert_close(record["tau_star"], saved, rtol=0, atol=0)


def test_scalar_ablation_varies_only_cate_slope_and_preserves_mc_noise():
    cfg = BuildConfig(
        n_datasets=4,
        n_max=64,
        n_q=512,
        mc_cate=16,
        seed=431,
        prior_mc_mem_budget=20_000_000,
        max_dim_U=1,
        max_dim_X=1,
        max_dim_W=1,
        max_dim_Z=1,
        prior_type="ablation_scm_slope",
    )
    episodes = [generate_episode(cfg, i) for i in range(cfg.n_datasets)]
    records = [episode["record"] for episode in episodes]
    coefficients = [record["sampled_hp"] for record in records]
    slopes = [c["beta"] for c in coefficients]
    assert len(set(slopes)) == cfg.n_datasets
    assert all(0.75 <= abs(slope) <= 1.25 for slope in slopes)
    for record in records[1:]:
        for key, value in records[0]["mechanisms"].items():
            torch.testing.assert_close(record["mechanisms"][key], value)
    assert all(
        c[key] == value
        for c in coefficients[1:]
        for key, value in coefficients[0].items()
        if key != "beta"
    )

    low_k = episodes[0]
    high_k = generate_episode(replace(cfg, mc_cate=250), 0)
    for key in ("U", "X", "W", "Z", "A", "Y_obs"):
        torch.testing.assert_close(low_k["tensors"][key], high_k["tensors"][key])
    expected = slopes[0] * low_k["tensors"]["X"][cfg.n_max :, 0]
    for episode, k in ((low_k, 16), (high_k, 250)):
        record = episode["record"]
        error = record["tau_star"] - expected
        assert record["k_eff"] == k
        assert error.var().item() == pytest.approx(1 / k, rel=0.2)
        assert record["s2"].mean().item() == pytest.approx(1 / k, rel=0.1)
