"""The n-sweep (ROADMAP A6).

Three things are pinned here, in decreasing order of how expensive the bug
would be:

1. **The sweep at ``n_ctx = n_max`` reproduces ``score_frozen_set`` exactly.**
   ``sweep.py`` accumulates its own pass (it needs the raw ``probs`` for the
   mixture CDF, which ``score_episode`` reduces away), so the two paths could
   drift. At the frozen comparison point they must not.
2. **``fit_power_law`` recovers a known exponent**, including on a curve with a
   floor bolted onto its upper end -- the attenuation that ``fit_window``
   exists to keep out of the fit.
3. **The pooled mixture is a mixture**: its CDF and its sd are what averaging
   the per-row predictives gives, not what averaging their summaries gives.
"""

import math

import numpy as np
import pytest
import torch

from posterior_eval.build_prior_frozen_set import BuildConfig, build, load_frozen_meta
from posterior_eval.sweep import (
    MIN_N,
    _NAccumulator,
    contraction_summary,
    default_n_list,
    fit_power_law,
    fit_window,
    parse_n_list,
    sweep_n,
)
from proximalfm.train._dist_eval import score_frozen_set
from proximalfm.train._dist_metrics import predictive_cdf, predictive_mean_var
from test_posterior_eval import _FakeCateModel, discretized_gaussian


class _ContractingModel(_FakeCateModel):
    """A predictive whose width falls as ``n_ctx ** exponent``.

    The one property the sweep is supposed to measure, made exact so the fitted
    ``beta`` has a right answer.
    """

    def __init__(self, exponent=-0.5, sigma0=4.0, **kwargs):
        super().__init__(**kwargs)
        self.exponent = exponent
        self.sigma0 = sigma0
        self._n_ctx = 1

    def _probs(self, n_q):
        sigma = self.sigma0 * self._n_ctx**self.exponent
        centers = torch.zeros(n_q)
        return discretized_gaussian(self.cate_bin_edges, centers, sigma)

    def predict(self, *, X, y_train, **kwargs):
        self._n_ctx = int(y_train.shape[1])
        return super().predict(X=X, y_train=y_train, **kwargs)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    out = tmp_path_factory.mktemp("frozen_sweep")
    cfg = BuildConfig(
        n_datasets=4,
        n_max=64,
        n_q=8,
        mc_cate=8,
        seed=20260807,
        prior_mc_mem_budget=2_000_000_000,
    )
    build(cfg, out, datasets_per_file=2, progress=False)
    return cfg, out


# --------------------------------------------------------------------------
# 1. The two passes agree where they must
# --------------------------------------------------------------------------


def test_sweep_at_n_max_matches_score_frozen_set(frozen):
    cfg, out = frozen
    model = _FakeCateModel()

    reference = score_frozen_set(model, out)
    result = sweep_n(model, out, n_list=[cfg.n_max], progress=False)
    row = result["by_n"][cfg.n_max]

    for key in ("crps", "mean_sd", "sd_over_rmse", "cov90_pooled", "pit_ks_pooled"):
        assert row[key] == pytest.approx(reference[f"prior_dist/{key}"], rel=1e-9)

    # The noise-corrected RMSE is the denominator of sd_over_rmse; deriving it
    # from the evaluator's own sums is the check that it is the same one.
    assert row["rmse_pooled"] == pytest.approx(
        math.sqrt(reference["prior_dist/mse_obs"] - reference["prior_dist/mean_s2"]),
        rel=1e-9,
    )
    # grid_scale is exactly 1 at the frozen comparison point, by construction.
    assert row["grid_scale/p50"] == pytest.approx(1.0, rel=1e-5)


def test_default_grid_extends_below_the_sidecar_grid(frozen):
    cfg, out = frozen
    ns = default_n_list(load_frozen_meta(out))
    assert ns[0] == MIN_N
    assert ns[-1] == cfg.n_max
    assert ns == sorted(set(ns))


def test_out_of_range_prefixes_are_refused(frozen):
    _, out = frozen
    meta = load_frozen_meta(out)
    with pytest.raises(ValueError):
        parse_n_list("8,100000", meta)
    with pytest.raises(ValueError):
        parse_n_list("0,8", meta)


# --------------------------------------------------------------------------
# 2. The fit
# --------------------------------------------------------------------------


def test_fit_recovers_a_known_exponent():
    ns = [8, 16, 32, 64, 128, 256]
    fit = fit_power_law(ns, [3.0 * n**-0.5 for n in ns])
    assert fit["beta"] == pytest.approx(-0.5, abs=1e-9)
    assert math.exp(fit["alpha"]) == pytest.approx(3.0, rel=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)


def test_degenerate_widths_are_dropped_not_logged():
    """A zero width is a degenerate episode, not a point at ``-inf``."""
    ns = [8, 16, 32]
    fit = fit_power_law(ns, [1.0, 0.0, 0.25])
    assert fit["n_points"] == 2
    assert fit["beta"] == pytest.approx(-1.0, abs=1e-9)


def test_floor_bound_prefixes_are_excluded_from_the_window():
    """The attenuation ``fit_window`` exists to prevent.

    A curve that is a clean ``n^-1/2`` until it lands on a floor reads as much
    flatter if the floor is fitted through -- which is the C2 misreading.
    """
    ns = [8, 16, 32, 64, 128, 256, 512, 1024]
    floor = 0.05

    def sd(n):
        # The real shape: the schedule/grid floor adds in quadrature, so the
        # curve bends onto it rather than crossing it.
        return math.hypot(n**-0.5, floor)

    result = {
        "n_list": ns,
        "by_n": {
            n: {
                "mean_sd_geo": sd(n),
                "mean_sd": sd(n),
                "mean_sd_head_geo": sd(n),
                "rmse_geo": n**-0.5,
                "rmse_pooled": n**-0.5,
                "sd_floor": floor,
            }
            for n in ns
        },
        "per_episode": {
            "mean_sd": {n: [sd(n)] for n in ns},
            "mean_sd_head": {n: [sd(n)] for n in ns},
            "rmse": {n: [n**-0.5] for n in ns},
        },
    }
    window = fit_window(result)
    assert window == [8, 16, 32]

    naive = fit_power_law(ns, [sd(n) for n in ns])
    windowed = contraction_summary(result)
    assert windowed["sd_geo"]["beta"] == pytest.approx(-0.48, abs=0.02)
    assert windowed["per_episode_sd"]["p50"] == pytest.approx(
        windowed["sd_geo"]["beta"], abs=1e-9
    )
    # Fitting through the floor reads as materially less contraction.
    assert naive["beta"] > windowed["sd_geo"]["beta"] + 0.05
    assert windowed["monotone_sd"]


def test_sweep_measures_a_contracting_posterior(frozen):
    """``beta`` in head units is the model's own contraction, recovered exactly.

    Head units are where the fake model's exponent is a fact; the frozen-unit
    curve additionally carries ``grid_scale``, which on a 4-episode toy set with
    heavy-tailed ``y`` moves more than the contraction does. That gap is the
    point of reporting both, so it is pinned here rather than tolerated.
    """
    cfg, out = frozen
    ns = [8, 16, 32, cfg.n_max]
    result = sweep_n(_ContractingModel(exponent=-0.5), out, n_list=ns, progress=False)

    fit = result["fit"]
    # Not exact to 1e-6: the widest predictive (n=8) is truncated by the head's
    # own +/-4 grid, so the measured sd is slightly below the nominal sigma at
    # the small-n end. That attenuation is the model's, not the sweep's.
    assert fit["per_episode_sd_head"]["p50"] == pytest.approx(-0.5, abs=0.02)
    assert fit["sd_head_geo"]["beta"] == pytest.approx(-0.5, abs=0.02)
    # The frozen-unit curve is the head curve times grid_scale, exactly.
    for n in ns:
        row = result["by_n"][n]
        assert row["mean_sd_geo"] == pytest.approx(
            row["mean_sd_head_geo"] * _geomean_scale(result, n), rel=1e-6
        )


def test_a_flat_posterior_has_no_contraction(frozen):
    """A point estimate wearing a histogram: beta ~ 0 in head units, which is
    the failure the roadmap's done-when is phrased to catch."""
    cfg, out = frozen
    ns = [8, 16, 32, cfg.n_max]
    result = sweep_n(_FakeCateModel(sigma=0.4), out, n_list=ns, progress=False)
    assert result["fit"]["per_episode_sd_head"]["p50"] == pytest.approx(0.0, abs=1e-6)


def _geomean_scale(result, n):
    sd = result["per_episode"]["mean_sd"][n]
    head = result["per_episode"]["mean_sd_head"][n]
    return math.exp(sum(math.log(a / b) for a, b in zip(sd, head)) / len(sd))


def test_grid_scale_is_the_only_difference_between_the_two_width_curves(frozen):
    """At ``n_ctx = n_max`` the two curves coincide: ``grid_scale`` is 1 there.

    This is what makes the head-unit curve a decomposition rather than a second
    opinion -- they are pinned together at the frozen comparison point.
    """
    cfg, out = frozen
    result = sweep_n(_ContractingModel(), out, n_list=[cfg.n_max], progress=False)
    row = result["by_n"][cfg.n_max]
    assert row["mean_sd_geo"] == pytest.approx(row["mean_sd_head_geo"], rel=1e-5)


# --------------------------------------------------------------------------
# 3. The mixture
# --------------------------------------------------------------------------


def _fake_rows(rng, n_rows, n_bins=32):
    logits = torch.from_numpy(rng.normal(size=(n_rows, n_bins))).float()
    probs = torch.softmax(logits, dim=-1)
    edges = torch.linspace(-3.0, 3.0, n_bins + 1)
    y = torch.from_numpy(rng.normal(scale=0.5, size=n_rows)).float()
    s2 = torch.zeros(n_rows)
    return probs, edges, y, s2


def test_pooled_cdf_is_the_average_of_the_row_cdfs():
    rng = np.random.default_rng(0)
    grid = torch.linspace(-3.0, 3.0, 17, dtype=torch.float64)
    acc = _NAccumulator(8, grid)

    every_row = []
    for _ in range(3):
        probs, edges, y, s2 = _fake_rows(rng, n_rows=5)
        acc.update(probs, edges, y, s2, scale=1.0)
        every_row.append(
            predictive_cdf(
                probs.unsqueeze(1), edges, grid.float().expand(5, grid.numel())
            ).double()
        )

    expected = torch.cat(every_row).mean(dim=0)
    torch.testing.assert_close(acc.cdf_sum / acc.rows, expected, rtol=1e-6, atol=1e-8)


def test_mixture_sd_is_the_law_of_total_variance():
    """Not the mean of the per-row sds: a mixture is wider than its components
    by the spread of their means, and that term is exactly the prior-limit
    signal at small ``n``."""
    rng = np.random.default_rng(1)
    grid = torch.linspace(-3.0, 3.0, 9, dtype=torch.float64)
    acc = _NAccumulator(8, grid)

    means, variances = [], []
    for _ in range(4):
        probs, edges, y, s2 = _fake_rows(rng, n_rows=6)
        acc.update(probs, edges, y, s2, scale=1.0)
        m, v = predictive_mean_var(probs, edges)
        means.append(m.double())
        variances.append(v.double())

    m = torch.cat(means)
    v = torch.cat(variances)
    expected = math.sqrt(float(v.mean() + m.var(unbiased=False)))

    mu = acc.mu_sum / acc.rows
    mix_var = acc.var_sum / acc.rows + (acc.mu2_sum / acc.rows - mu**2)
    assert math.sqrt(mix_var) == pytest.approx(expected, rel=1e-6)


def test_within_frac_is_the_variance_split_not_a_width(frozen):
    """``within_frac`` is what actually discriminates the prior limit.

    ``prior_ks`` does not: averaging a calibrated posterior over the data
    returns the prior marginal at *every* ``n``, so a small KS is a marginal
    calibration statement. The split into "not yet known" and "already resolved"
    is the one that has to move with ``n``.
    """
    cfg, out = frozen
    result = sweep_n(_ContractingModel(), out, n_list=[8, cfg.n_max], progress=False)
    for n in (8, cfg.n_max):
        row = result["by_n"][n]
        assert 0.0 <= row["within_frac"] <= 1.0
        # Var(mix) = E[Var] + Var(means), exactly.
        assert row["mixture_sd"] ** 2 == pytest.approx(
            row["within_frac"] * row["mixture_sd"] ** 2 + row["between_sd"] ** 2,
            rel=1e-6,
        )


def test_sigma_floor_is_absent_until_it_is_supplied(frozen):
    """The C2 trap: without ``--hl_sigma`` the reported floor is the grid term
    alone, which is usually orders of magnitude below the binding one."""
    cfg, out = frozen
    ns = [8, cfg.n_max]
    bare = sweep_n(_ContractingModel(), out, n_list=ns, progress=False)
    with_sigma = sweep_n(
        _ContractingModel(), out, n_list=ns, hl_sigma=0.5, progress=False
    )

    assert bare["by_n"][ns[0]]["sd_floor_is_grid_only"] == 1.0
    assert with_sigma["by_n"][ns[0]]["sd_floor_is_grid_only"] == 0.0
    for n in ns:
        assert with_sigma["by_n"][n]["sd_floor"] > 10 * bare["by_n"][n]["sd_floor"]
    # The floors add in quadrature, in the frozen space.
    n = ns[-1]  # grid_scale is exactly 1 here, so head and frozen units coincide
    assert with_sigma["by_n"][n]["sd_floor"] == pytest.approx(
        math.hypot(bare["by_n"][n]["sd_floor"], 0.5), rel=1e-6
    )


def test_prior_limit_distances_are_reported_per_prefix(frozen):
    cfg, out = frozen
    ns = [MIN_N, 8, cfg.n_max]
    result = sweep_n(_ContractingModel(), out, n_list=ns, progress=False)

    for n in ns:
        row = result["by_n"][n]
        assert 0.0 <= row["prior_ks"] <= 1.0
        assert row["prior_w1"] >= 0.0
        assert row["mixture_sd"] > 0.0
    assert result["prior"]["tau_sd"] > 0.0
    assert len(result["prior"]["cdf"]) == len(result["prior"]["grid"])
