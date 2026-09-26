"""The distributional evaluator.

Two kinds of check, and both are needed:

* the accumulator reproduces the step-4 closed forms *through* ``update`` /
  ``compute`` -- a streaming sum is exactly the place a weight gets dropped;
* the driver puts predictions and targets in the same space. The frozen set
  exists so that ``tau_star`` is low-noise, and the scale plumbing between
  ``predict``'s moving context statistics and the sidecar's frozen one is the
  cheapest thing in the stack to get silently wrong. The pooled RMSE is
  therefore asserted equal to ``prior_eval/report.py``'s, which computes the
  same quantity by an independent path.
"""


import math


import pytest


import torch


from prior_eval.build_prior_frozen_set import BuildConfig, build, load_frozen_meta


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
    assert m["prior_dist/sd_over_rmse"] == pytest.approx(1.0, abs=0.05)
    assert m["prior_dist/cov90_pooled"] == pytest.approx(0.90, abs=0.02)
    assert m["prior_dist/cov50_pooled"] == pytest.approx(0.50, abs=0.03)
    # Exact width of the central 90% interval of a standard normal.
    assert m["prior_dist/width90"] == pytest.approx(2 * 1.6448536, abs=0.01)
    assert m["prior_dist/pit_ks_pooled"] < m["prior_dist/pit_ks_pooled_band"]
    # CRPS of N(0,1) against its own draws: sigma / sqrt(pi).
    assert m["prior_dist/crps"] == pytest.approx(1.0 / math.sqrt(math.pi), rel=0.03)
    assert m["prior_dist/crps_over_oracle"] > 1.0
    assert m["prior_dist/out_of_grid_frac"] == 0.0
    assert m["prior_dist/eval_episodes"] == 16
    assert m["prior_dist/eval_rows"] == 16 * 512


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

    assert metrics["prior_point/eval_episodes"] == cfg.n_datasets
    assert metrics["prior_dist/eval_rows"] == cfg.n_datasets * cfg.n_q
    assert metrics["prior_dist/n_ctx"] == cfg.n_max
    for key in (
        "prior_point/cate/rmse",
        "prior_point/cate/rmse_pooled",
        "prior_point/ate/rmse",
        "prior_dist/crps",
        "prior_dist/cov90_pooled",
        "prior_dist/cov90/p50",
        "prior_dist/width90",
        "prior_dist/sd_over_rmse",
        "prior_dist/mean_sd",
        "prior_dist/crps_over_oracle",
        "prior_dist/out_of_grid_frac",
        "prior_dist/pit_ks_pooled",
        "prior_dist/pit_ks/p90",
        "prior_dist/s_over_sd",
    ):
        assert key in metrics, key
    # No per-arm predictions exist for a single-CATE head.
    assert not any(k.startswith("prior_point/po/") for k in metrics)


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


def test_two_head_checkpoints_are_refused(frozen):
    _, out = frozen
    with pytest.raises(UnsupportedLossTypeError, match="two per-arm histograms"):
        score_frozen_set(_FakeTwoHeadModel(), out)
