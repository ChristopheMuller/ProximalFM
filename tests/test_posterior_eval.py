"""The distributional evaluator (EVAL_PLAN step 5).

Two kinds of check, and both are needed:

* the accumulator reproduces the step-4 closed forms *through* ``update`` /
  ``compute`` -- a streaming sum is exactly the place a weight gets dropped;
* the driver puts predictions and targets in the same space. The frozen set
  exists so that ``tau_star`` is low-noise, and the scale plumbing between
  ``predict``'s moving context statistics and the sidecar's frozen one is the
  cheapest thing in the stack to get silently wrong. The pooled RMSE is
  therefore asserted equal to ``posterior_eval/report.py``'s, which computes the
  same quantity by an independent path.
"""

import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from posterior_eval.build_prior_frozen_set import BuildConfig, build, load_frozen_meta
from posterior_eval.report import _cate_rmse_at_prefix
from proximalfm.train._dist_eval import (
    PosteriorEvaluator,
    UnsupportedLossTypeError,
    episode_predictive,
    iter_frozen_episodes,
    score_frozen_set,
)
from proximalfm.train._dist_metrics import predictive_quantile
from proximalfm.train._losses import gaussian_bin_mass


def make_grid(vmin=-8.0, vmax=8.0, nbins=800):
    return torch.linspace(vmin, vmax, nbins + 1)


def discretized_gaussian(bin_edges, center, sigma):
    p = gaussian_bin_mass(
        bin_edges, torch.as_tensor(center, dtype=torch.float32), sigma
    )
    return p / p.sum(dim=-1, keepdim=True)


# --------------------------------------------------------------------------
# The accumulator
# --------------------------------------------------------------------------


def _gaussian_episodes(evaluator, edges, sigma_pred, sigma_true, n_ep=8, n=256, seed=0):
    """Feed ``n_ep`` episodes of N(mu, sigma_pred^2) forecasts for y ~ N(mu, sigma_true^2)."""
    gen = torch.Generator().manual_seed(seed)
    for _ in range(n_ep):
        mu = torch.randn(n, generator=gen) * 0.7
        y = mu + sigma_true * torch.randn(n, generator=gen)
        probs = discretized_gaussian(edges, mu, sigma_pred)
        evaluator.update(probs=probs, bin_edges=edges, y=y)


def test_calibrated_forecast_recovers_the_closed_forms():
    edges = make_grid()
    ev = PosteriorEvaluator()
    _gaussian_episodes(ev, edges, sigma_pred=1.0, sigma_true=1.0, n_ep=16, n=512)
    m = ev.compute()

    # A correct forecast: unit width ratio, nominal coverage, uniform PIT.
    assert m["prior/dist/sd_over_rmse"] == pytest.approx(1.0, abs=0.05)
    assert m["prior/dist/cov90"] == pytest.approx(0.90, abs=0.02)
    assert m["prior/dist/cov50"] == pytest.approx(0.50, abs=0.03)
    # Exact width of the central 90% interval of a standard normal.
    assert m["prior/dist/width90"] == pytest.approx(2 * 1.6448536, abs=0.01)
    assert m["prior/dist/pit_ks"] < m["prior/dist/pit_ks_band"]
    # CRPS of N(0,1) against its own draws: sigma / sqrt(pi).
    assert m["prior/dist/crps"] == pytest.approx(1.0 / math.sqrt(math.pi), rel=0.03)
    assert m["prior/dist/crps_over_oracle"] > 1.0
    assert m["prior/dist/out_of_grid_frac"] == 0.0
    assert m["prior/dist/eval/episodes"] == 16
    assert m["prior/dist/eval/rows"] == 16 * 512


def test_too_narrow_forecast_fails_both_diagnostics():
    edges = make_grid()
    ev = PosteriorEvaluator()
    _gaussian_episodes(ev, edges, sigma_pred=0.5, sigma_true=1.0, n_ep=16, n=512)
    m = ev.compute()

    # Both, because either alone is ambiguous: a U-shaped PIT is also produced
    # by a biased forecast, and sd/rmse alone says nothing about shape.
    assert m["prior/dist/sd_over_rmse"] == pytest.approx(0.5, abs=0.05)
    assert m["prior/dist/cov90"] < 0.75
    assert m["prior/dist/pit_ks"] > m["prior/dist/pit_ks_band"]


def test_s2_subtraction_restores_the_ratio_on_a_noisy_target():
    """A *correct* posterior scored against tau* + e reads over-confident unless
    s^2 comes out of the denominator. Same argument as step 7's, at the
    evaluator level."""
    edges = make_grid()
    gen = torch.Generator().manual_seed(3)
    s2 = 0.25

    with_s2 = PosteriorEvaluator()
    without_s2 = PosteriorEvaluator()
    for _ in range(16):
        mu = torch.randn(512, generator=gen) * 0.7
        tau_star = mu + torch.randn(512, generator=gen)  # posterior is correct
        y = tau_star + math.sqrt(s2) * torch.randn(512, generator=gen)
        probs = discretized_gaussian(edges, mu, 1.0)
        with_s2.update(probs=probs, bin_edges=edges, y=y, s2=torch.full((512,), s2))
        without_s2.update(probs=probs, bin_edges=edges, y=y)

    assert with_s2.compute()["prior/dist/sd_over_rmse"] == pytest.approx(1.0, abs=0.05)
    # Uncorrected, the same posterior reads as 1/sqrt(1.25) = 0.894.
    assert without_s2.compute()["prior/dist/sd_over_rmse"] == pytest.approx(
        1.0 / math.sqrt(1.25), abs=0.05
    )
    assert with_s2.compute()["prior/dist/s_over_sd"] == pytest.approx(0.5, abs=0.05)


def _episode_with_pit(edges, u_target, sigma=1.0):
    """An episode whose PIT values are exactly ``u_target``.

    Built by inverting the forecast: ``y = F^{-1}(u)`` gives ``F(y) = u``, so the
    calibration shape is prescribed rather than sampled.
    """
    n = u_target.numel()
    probs = discretized_gaussian(edges, torch.zeros(n), sigma)
    q = predictive_quantile(probs, edges, u_target)  # (n, n): all rows identical
    y = q.diagonal()
    return probs, y


def _tilted_pit(v, c):
    """PIT sample whose empirical CDF is exactly ``G(t) = t + c t(1 - t)``.

    Inverting that quadratic gives the sample; the two signs of ``c`` are
    mirrored deviations, so a set made of equal halves has ``(G_+ + G_-)/2 = t``
    -- pooled uniformity by construction, with every episode off by ``|c|/4``.
    """
    if c == 0:
        return v
    disc = (1 + c) ** 2 - 4 * c * v
    return ((1 + c) - disc.sqrt()) / (2 * c)


def test_per_episode_ks_catches_what_pooling_cancels():
    """Half the episodes over-confident, half under-confident, in mirrored
    amounts: the pooled PIT is uniform while every single episode fails."""
    edges = make_grid()
    n = 200
    v = (torch.arange(n, dtype=torch.float64) + 0.5) / n

    ev = PosteriorEvaluator()
    for c in (0.8, -0.8) * 10:  # 20 episodes, mirrored deviations
        probs, y = _episode_with_pit(edges, _tilted_pit(v, c).float())
        ev.update(probs=probs, bin_edges=edges, y=y)
    m = ev.compute()

    # Pooled: the two tilts cancel and the set looks calibrated.
    assert m["prior/dist/pit_ks"] < m["prior/dist/pit_ks_band"]
    # Per-episode: every episode is outside its own band.
    assert m["prior/dist/pit_ks_episode_median"] > m["prior/dist/pit_ks_episode_band"]
    assert m["prior/dist/pit_ks_episode_p90"] > m["prior/dist/pit_ks_episode_band"]


def test_empty_evaluator_computes_to_nothing():
    ev = PosteriorEvaluator()
    assert ev.empty
    assert ev.compute() == {}


def test_out_of_grid_targets_are_counted_not_clipped():
    edges = make_grid(-1.0, 1.0, 40)
    ev = PosteriorEvaluator()
    probs = discretized_gaussian(edges, torch.zeros(4), 0.3)
    y = torch.tensor([0.0, 0.1, 5.0, -5.0])
    ev.update(probs=probs, bin_edges=edges, y=y)
    m = ev.compute()
    assert m["prior/dist/out_of_grid_frac"] == pytest.approx(0.5)
    # The two far targets are charged their distance to the edge (~4 each).
    assert m["prior/dist/crps"] > 1.5


# --------------------------------------------------------------------------
# The driver, against a real frozen set
# --------------------------------------------------------------------------


class _FakeCateModel:
    """A checkpoint-shaped stand-in whose predictive is a known Gaussian.

    Mirrors ``ProximalFM.predict``'s ``cate_single`` branch exactly, including
    the de-standardization (``vals = cate_bin_centers * y_context_std``, no mean
    offset), so the scale plumbing under test is the real one.
    """

    loss_type = "cate_single"

    def __init__(self, nbins=64, vmin=-4.0, vmax=4.0, sigma=0.4):
        self.cate_bin_edges = torch.linspace(vmin, vmax, nbins + 1)
        width = self.cate_bin_edges[1] - self.cate_bin_edges[0]
        self.cate_bin_centers = self.cate_bin_edges[:-1] + 0.5 * width
        self.sigma = sigma

    def _probs(self, n_q):
        centers = 0.1 * torch.arange(n_q, dtype=torch.float32) - 0.05
        return discretized_gaussian(self.cate_bin_edges, centers, self.sigma)

    def predict(self, *, X, y_train, output_type="mean", **kwargs):
        n_q = X.shape[1] - y_train.shape[1]
        y_std = y_train.reshape(-1).float().std(unbiased=False).clamp(min=1e-5)
        probs = self._probs(n_q)
        if output_type == "distribution":
            vals = self.cate_bin_centers.view(1, 1, -1) * y_std
            return {"values": vals, "weights": probs.unsqueeze(0)}
        if output_type == "mean":
            return ((probs * self.cate_bin_centers).sum(-1) * y_std).reshape(1, -1)
        raise ValueError(output_type)


class _FakeTwoHeadModel(_FakeCateModel):
    loss_type = "bins_po"


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    out = tmp_path_factory.mktemp("frozen_score")
    cfg = BuildConfig(
        n_datasets=2,
        n_max=32,
        n_q=8,
        mc_cate=8,
        seed=20260805,
        prior_mc_mem_budget=2_000_000_000,
    )
    build(cfg, out, datasets_per_file=2, progress=False)
    return cfg, out


def test_score_frozen_set_emits_both_surfaces(frozen):
    cfg, out = frozen
    metrics = score_frozen_set(_FakeCateModel(), out)

    assert metrics["prior/point/eval/episodes"] == cfg.n_datasets
    assert metrics["prior/dist/eval/rows"] == cfg.n_datasets * cfg.n_q
    assert metrics["prior/n_ctx"] == cfg.n_max
    for key in (
        "prior/point/cate/rmse",
        "prior/point/cate/rmse_pooled",
        "prior/point/ate/rmse",
        "prior/dist/crps",
        "prior/dist/cov90",
        "prior/dist/width90",
        "prior/dist/sd_over_rmse",
        "prior/dist/mean_sd",
        "prior/dist/crps_over_oracle",
        "prior/dist/out_of_grid_frac",
        "prior/dist/pit_ks",
        "prior/dist/pit_ks_episode_p90",
        "prior/dist/s_over_sd",
    ):
        assert key in metrics, key
    # No per-arm predictions exist for a single-CATE head (EVAL_PLAN step 2).
    assert not any(k.startswith("prior/point/po/") for k in metrics)


def test_pooled_rmse_matches_the_report_path(frozen):
    """The one cross-check that would catch a frozen-scale bug: the same
    quantity computed by ``posterior_eval/report.py``'s independent path."""
    cfg, out = frozen
    model = _FakeCateModel()
    meta = load_frozen_meta(out)

    sq_sum, rows = 0.0, 0
    for i, batch in enumerate(iter_frozen_episodes(out / "val")):
        rmse = _cate_rmse_at_prefix(
            model, batch, meta["episodes"][i], cfg.n_max, cfg.n_max, "cpu"
        )
        sq_sum += rmse**2 * cfg.n_q
        rows += cfg.n_q

    metrics = score_frozen_set(model, out)
    assert metrics["prior/point/cate/rmse_pooled"] == pytest.approx(
        math.sqrt(sq_sum / rows), rel=1e-5
    )


def test_grid_is_rescaled_into_the_frozen_space(frozen):
    """``predict``'s grid is in raw-y units; the evaluator's must be in the
    sidecar's frozen units, i.e. scaled by y_std_predict / y_std_frozen."""
    cfg, out = frozen
    model = _FakeCateModel()
    meta = load_frozen_meta(out)
    batch = next(iter(iter_frozen_episodes(out / "val")))
    record = meta["episodes"][0]

    probs, edges = episode_predictive(
        model, batch, record, n_ctx=cfg.n_max, n_max=cfg.n_max
    )
    assert probs.shape == (cfg.n_q, model.cate_bin_edges.numel() - 1)

    # At n_ctx = n_max the two scales are the same rows, so the grid is the
    # head's own grid (the trainer's 1e-3 floor vs predict's 1e-5 aside).
    torch.testing.assert_close(edges, model.cate_bin_edges, rtol=1e-5, atol=1e-6)

    # At a shorter prefix the context std moves, and the grid must move with it
    # -- by one common factor, since the target scale stays frozen at n_max.
    _, edges_s = episode_predictive(model, batch, record, n_ctx=16, n_max=cfg.n_max)
    ratio = float(edges_s[-1] / model.cate_bin_edges[-1])
    assert ratio != pytest.approx(1.0, abs=1e-3)
    torch.testing.assert_close(edges_s, model.cate_bin_edges * ratio)


def test_grid_scale_is_reported_and_is_the_factor_actually_applied(frozen):
    """``prior/grid_scale`` converts head units into the frozen space, and the
    trainer multiplies its sigma schedule by it (EVAL_PLAN step 6e). A caller
    that hardcoded 1.0 would be right at n_max and wrong on every sweep prefix,
    so pin both ends."""
    cfg, out = frozen
    model = _FakeCateModel()

    at_n_max = score_frozen_set(model, out)
    assert at_n_max["prior/grid_scale"] == pytest.approx(1.0, abs=1e-5)

    short = score_frozen_set(model, out, n_ctx=16)
    assert short["prior/grid_scale"] != pytest.approx(1.0, abs=1e-3)

    # It must be the factor on the grid that was actually scored, not a
    # separately recomputed one -- averaged over episodes, which all carry the
    # same row count here.
    meta = load_frozen_meta(out)
    ratios = []
    for i, batch in enumerate(iter_frozen_episodes(out / "val")):
        _, edges = episode_predictive(
            model, batch, meta["episodes"][i], n_ctx=16, n_max=cfg.n_max
        )
        ratios.append(float(edges[-1] / model.cate_bin_edges[-1]))
    assert short["prior/grid_scale"] == pytest.approx(
        sum(ratios) / len(ratios), rel=1e-6
    )


def test_real_model_predict_path(frozen):
    """The fake model above pins the *scale*; this pins the *interface* -- that
    ``predict(output_type="distribution")`` really returns ``weights`` shaped
    ``(1, n_q, nbins)`` on a single shared grid."""
    from proximalfm._model import ProximalFM

    cfg, out = frozen
    torch.manual_seed(0)
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
        nbins=32,
        loss_type="cate_single",
    ).eval()

    metrics = score_frozen_set(model, out)
    assert metrics["prior/dist/eval/rows"] == cfg.n_datasets * cfg.n_q
    assert 0.0 <= metrics["prior/dist/cov90"] <= 1.0
    assert metrics["prior/dist/crps"] > 0.0
    assert metrics["prior/dist/mean_sd"] > 0.0


def test_two_head_checkpoints_are_refused(frozen):
    _, out = frozen
    with pytest.raises(UnsupportedLossTypeError, match="two per-arm histograms"):
        score_frozen_set(_FakeTwoHeadModel(), out)


# --------------------------------------------------------------------------
# DDP: sharded episodes, synced metrics (EVAL_PLAN step 6d)
# --------------------------------------------------------------------------


def test_sync_ddp_false_is_the_single_process_path(frozen):
    """The regression guard: with no process group, the step-6d parameters are
    inert and the numbers are step 6c's."""
    _, out = frozen
    model = _FakeCateModel()
    baseline = score_frozen_set(model, out)
    assert score_frozen_set(model, out, rank=0, world_size=1, sync_ddp=True) == baseline


@pytest.fixture(scope="module")
def frozen4(tmp_path_factory):
    """Four episodes, so a world of 2 gets a genuinely split shard."""
    out = tmp_path_factory.mktemp("frozen_score_ddp")
    cfg = BuildConfig(
        n_datasets=4,
        n_max=32,
        n_q=8,
        mc_cate=8,
        seed=20260805,
        prior_mc_mem_budget=2_000_000_000,
    )
    build(cfg, out, datasets_per_file=2, progress=False)
    return cfg, out


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_sharded(out_dir, tmp_path, world_size, max_episodes=None, timeout=300):
    """Launch ``world_size`` real gloo ranks; return each rank's metric dict.

    Every rank's JSON is returned, not just rank 0's: after an all-reduce all
    ranks must agree, and a rank that quietly skipped the collectives would
    still write a file -- just a different one.
    """
    script = Path(__file__).resolve().parent / "ddp_dist_eval_worker.py"
    env = dict(os.environ)
    env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(_free_port()))
    env.pop("RANK", None)

    procs, paths = [], []
    for rank in range(world_size):
        path = Path(tmp_path) / f"rank{rank}.json"
        paths.append(path)
        cmd = [sys.executable, str(script), "--dir", str(out_dir), "--out", str(path)]
        if max_episodes is not None:
            cmd += ["--max-episodes", str(max_episodes)]
        procs.append(
            subprocess.Popen(
                cmd,
                env={**env, "RANK": str(rank), "WORLD_SIZE": str(world_size)},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )

    outputs = []
    try:
        for proc in procs:
            outputs.append(proc.communicate(timeout=timeout)[0])
    except subprocess.TimeoutExpired:
        for proc in procs:
            proc.kill()
        pytest.fail(f"a rank did not finish within {timeout}s -- deadlock")

    for rank, (proc, text) in enumerate(zip(procs, outputs)):
        assert proc.returncode == 0, f"rank {rank} failed:\n{text}"
    return [json.loads(p.read_text()) for p in paths]


def test_sharded_metrics_match_the_single_process_ones(frozen4, tmp_path):
    cfg, out = frozen4
    single = score_frozen_set(_FakeCateModel(), out)
    per_rank = _run_sharded(out, tmp_path, world_size=2)

    assert per_rank[0] == per_rank[1], "ranks disagree; a reduction was skipped"
    sharded = per_rank[0]

    # Counts first: the shard split is the thing most likely to lose episodes.
    assert sharded["prior/dist/eval/rows"] == cfg.n_datasets * cfg.n_q
    assert sharded["prior/point/eval/episodes"] == cfg.n_datasets

    # Weighted sums reduce exactly, so these are equalities, not tolerances --
    # including the driver-accumulated pooled RMSE, which lives outside either
    # evaluator and has its own reduce.
    for key in (
        "prior/dist/crps",
        "prior/dist/crps_oracle",
        "prior/dist/cov50",
        "prior/dist/cov90",
        "prior/dist/width50",
        "prior/dist/width90",
        "prior/dist/mean_sd",
        "prior/dist/sd_over_rmse",
        "prior/dist/out_of_grid_frac",
        "prior/point/cate/rmse_pooled",
        "prior/grid_scale",
        "prior/point/ate/rmse",
    ):
        assert sharded[key] == pytest.approx(single[key], rel=1e-9), key

    # Per-episode statistics are gathered, so the *set* is the same and the
    # quantiles over it are exact too.
    for key in ("prior/dist/pit_ks_episode_median", "prior/dist/pit_ks_episode_p90"):
        assert sharded[key] == pytest.approx(single[key], rel=1e-9), key

    # The pooled PIT is subsampled per rank before the gather, so the pooled KS
    # is the same statistic on a possibly different sample. Compare as a
    # distribution, not for equality.
    assert sharded["prior/dist/pit_ks"] == pytest.approx(
        single["prior/dist/pit_ks"], abs=0.05
    )


def test_empty_shard_still_completes(frozen4, tmp_path):
    """One episode, two ranks: rank 1 accumulates nothing and must still enter
    every collective. This is the deadlock the commit exists to get right."""
    cfg, out = frozen4
    single = score_frozen_set(_FakeCateModel(), out, max_episodes=1)
    per_rank = _run_sharded(out, tmp_path, world_size=2, max_episodes=1)

    assert per_rank[0] == per_rank[1]
    assert per_rank[0]["prior/dist/eval/rows"] == cfg.n_q
    assert per_rank[0]["prior/point/eval/episodes"] == 1
    assert per_rank[0]["prior/point/cate/rmse_pooled"] == pytest.approx(
        single["prior/point/cate/rmse_pooled"], rel=1e-9
    )
    assert per_rank[0]["prior/dist/crps"] == pytest.approx(
        single["prior/dist/crps"], rel=1e-9
    )
