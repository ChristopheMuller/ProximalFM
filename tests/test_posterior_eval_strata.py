"""Stratified evaluation of a frozen set (ROADMAP A2).

The whole design rests on one property: ``PosteriorEvaluator`` accumulates
row-weighted sums, so a stratum's metrics are *exactly* what a separate pass
over only that stratum's rows would produce. That is what lets every cut be
scored in one pass over the data instead of one pass per cut. The first test
here pins it; if it ever fails, ``score_strata`` is silently reporting numbers
that no subset of the data actually has.

The complementary fact is pinned too: ``pit_ks_pooled`` is a *sample statistic*, not a
sum, so it is deliberately NOT additive. Losing that distinction would be the
easy way to "fix" a future additivity failure by averaging something that
cannot be averaged.
"""

import warnings

import numpy as np
import pytest
import torch

from posterior_eval.build_prior_frozen_set import BuildConfig, build, load_frozen_meta
from posterior_eval.strata import (
    MIN_EPISODES,
    ManifestMismatchError,
    build_manifest,
    load_manifest,
    save_manifest,
)
from proximalfm.train._dist_eval import PosteriorEvaluator

# Metrics that are a row-weighted mean of a per-row quantity, hence exactly
# recoverable from disjoint subsets. Kept explicit rather than derived from the
# evaluator, so a new non-additive key cannot join the list by accident.
ADDITIVE = (
    "crps",
    "crps_oracle",
    "mean_var",
    "mean_sd",
    "mse_obs",
    "cov90_pooled",
    "cov50_pooled",
)


def _cfg(**overrides):
    base = dict(
        n_datasets=6,
        n_max=32,
        n_q=8,
        mc_cate=16,
        seed=20260806,
        prior_mc_mem_budget=2_000_000_000,
    )
    base.update(overrides)
    return BuildConfig(**base)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    out = tmp_path_factory.mktemp("frozen_strata")
    build(_cfg(), out, datasets_per_file=2, progress=False)
    return out


def _fake_episode(rng, n_rows, n_bins=24):
    """A predictive histogram plus targets, with no model involved."""
    logits = torch.from_numpy(rng.normal(size=(n_rows, n_bins))).float()
    probs = torch.softmax(logits, dim=-1)
    edges = torch.linspace(-3.0, 3.0, n_bins + 1)
    y = torch.from_numpy(rng.normal(scale=0.8, size=n_rows)).float()
    s2 = torch.from_numpy(rng.uniform(1e-4, 1e-3, size=n_rows)).float()
    return probs, edges, y, s2


def test_pooled_metrics_are_the_row_weighted_sum_of_disjoint_subsets():
    """The property every stratified number depends on.

    Split each episode's rows into two disjoint halves, accumulate them into
    separate evaluators, and recombine by row weight. Every additive metric
    must land on the pooled value.
    """
    rng = np.random.default_rng(0)
    pooled = PosteriorEvaluator()
    left, right = PosteriorEvaluator(), PosteriorEvaluator()

    for _ in range(5):
        probs, edges, y, s2 = _fake_episode(rng, n_rows=40)
        mask = torch.from_numpy(rng.random(40) < 0.5)
        pooled.update(probs=probs, bin_edges=edges, y=y, s2=s2)
        left.update(probs=probs[mask], bin_edges=edges, y=y[mask], s2=s2[mask])
        right.update(probs=probs[~mask], bin_edges=edges, y=y[~mask], s2=s2[~mask])

    p = pooled.compute(prefix="")
    a = left.compute(prefix="")
    b = right.compute(prefix="")
    n_a, n_b, n_p = a["eval_rows"], b["eval_rows"], p["eval_rows"]
    assert n_a + n_b == n_p

    for key in ADDITIVE:
        recombined = (a[key] * n_a + b[key] * n_b) / n_p
        assert recombined == pytest.approx(p[key], rel=1e-9, abs=1e-9), key


def test_pit_ks_is_not_additive_and_must_not_be_recombined():
    """``pit_ks_pooled`` is a sample statistic; recombining it by row weight is wrong.

    Pinned deliberately: if a future change makes the additivity test fail, the
    tempting fix is to average everything, and this is the key that would
    silently become meaningless.
    """
    rng = np.random.default_rng(1)
    pooled = PosteriorEvaluator()
    left, right = PosteriorEvaluator(), PosteriorEvaluator()

    for _ in range(5):
        probs, edges, y, s2 = _fake_episode(rng, n_rows=60)
        mask = torch.zeros(60, dtype=torch.bool)
        # A deliberately non-random split: the halves have different PIT
        # distributions, which is exactly the case a row-weighted average of
        # KS statistics gets wrong.
        mask[y > y.median()] = True
        pooled.update(probs=probs, bin_edges=edges, y=y, s2=s2)
        left.update(probs=probs[mask], bin_edges=edges, y=y[mask], s2=s2[mask])
        right.update(probs=probs[~mask], bin_edges=edges, y=y[~mask], s2=s2[~mask])

    p = pooled.compute(prefix="")
    a = left.compute(prefix="")
    b = right.compute(prefix="")
    n_a, n_b, n_p = a["eval_rows"], b["eval_rows"], p["eval_rows"]

    recombined = (a["pit_ks_pooled"] * n_a + b["pit_ks_pooled"] * n_b) / n_p
    assert recombined != pytest.approx(p["pit_ks_pooled"], rel=1e-3)


def test_per_episode_coverage_quantiles_are_not_additive():
    """``cov90/p*`` are quantiles over episodes, not row-weighted means.

    The pooled twin ``cov90_pooled`` *is* additive and is covered by
    :data:`ADDITIVE`. Pinned separately so the two never get conflated: a
    quantile of a subset's per-episode coverages says nothing about the whole,
    and row-weighting them is exactly the mistake the pooled key exists to
    absorb.
    """
    rng = np.random.default_rng(2)
    edges = torch.linspace(-3.0, 3.0, 25)
    centers = (edges[:-1] + edges[1:]) / 2

    # `_fake_episode`'s softmax-of-noise forecast is so wide that cov90 is 1.0
    # on every row, which makes any recombination trivially exact. Two
    # properties are needed instead: forecasts sharp enough to miss, and
    # *skew across episodes* -- with homogeneous episodes the median is the
    # common value and coincides with the row-weighted mean.
    pooled = PosteriorEvaluator()
    left, right = PosteriorEvaluator(), PosteriorEvaluator()

    for sigma in (0.15, 0.15, 0.2, 0.2, 0.25, 0.3, 1.5, 2.0):
        n = 60
        probs = torch.softmax(-(centers[None, :] ** 2) / (2 * sigma**2), dim=-1)
        probs = probs.repeat(n, 1)
        y = torch.from_numpy(rng.normal(scale=0.8, size=n)).float()
        s2 = torch.full((n,), 1e-4)
        # Split near-centre rows from tail rows, so the two sides have very
        # different coverage -- the cancellation the per-episode keys exist to
        # expose.
        mask = y.abs() < 0.4
        pooled.update(probs=probs, bin_edges=edges, y=y, s2=s2)
        left.update(probs=probs[mask], bin_edges=edges, y=y[mask], s2=s2[mask])
        right.update(probs=probs[~mask], bin_edges=edges, y=y[~mask], s2=s2[~mask])

    p = pooled.compute(prefix="")
    a = left.compute(prefix="")
    b = right.compute(prefix="")
    n_a, n_b, n_p = a["eval_rows"], b["eval_rows"], p["eval_rows"]

    recombined = (a["cov90/p50"] * n_a + b["cov90/p50"] * n_b) / n_p
    assert recombined != pytest.approx(p["cov90/p50"], rel=1e-2)

    # ...while the pooled key over the same split recombines exactly.
    pooled_recombined = (a["cov90_pooled"] * n_a + b["cov90_pooled"] * n_b) / n_p
    assert pooled_recombined == pytest.approx(p["cov90_pooled"], rel=1e-9)

    # And the reason both keys are logged: on this deliberately split set the
    # pooled coverage sits well above the typical episode's.
    assert p["cov90_pooled"] > p["cov90/p50"] + 0.1


def test_manifest_covers_every_episode_and_records_its_thresholds(frozen):
    manifest = build_manifest(frozen)
    meta = load_frozen_meta(frozen)
    n = int(meta["global"]["n_datasets"])
    n_q = int(meta["global"]["n_q"])

    assert manifest["cuts"], "no cuts were built"
    for name, cut in manifest["cuts"].items():
        if cut["kind"] == "episode":
            assert cut["mask"].shape == (n,), name
        else:
            assert cut["mask"].shape == (n, n_q), name
        assert cut["direction"] in ("below", "above"), name
        assert cut["description"] and cut["source"], name


def test_manifest_is_deterministic(frozen):
    a = build_manifest(frozen)
    b = build_manifest(frozen)
    assert set(a["cuts"]) == set(b["cuts"])
    for name in a["cuts"]:
        assert torch.equal(a["cuts"][name]["mask"], b["cuts"][name]["mask"]), name


def test_load_manifest_refuses_a_manifest_from_a_different_set(frozen, tmp_path):
    """Membership is positional, so the wrong pairing is wrong but not anomalous."""
    other = tmp_path / "other"
    build(_cfg(seed=999, n_datasets=6), other, datasets_per_file=2, progress=False)

    save_manifest(other, build_manifest(frozen))
    with pytest.raises(ManifestMismatchError, match="does not describe the set"):
        load_manifest(other)


def test_load_manifest_refuses_a_stale_format_version(frozen):
    manifest = build_manifest(frozen)
    manifest["format_version"] = -1
    save_manifest(frozen, manifest)
    with pytest.raises(ManifestMismatchError, match="format_version"):
        load_manifest(frozen)
    save_manifest(frozen, build_manifest(frozen))


def test_missing_manifest_points_at_the_command_that_builds_it(tmp_path):
    out = tmp_path / "empty"
    out.mkdir()
    with pytest.raises(FileNotFoundError, match="posterior_eval.strata"):
        load_manifest(out)


def test_row_level_cut_partitions_the_query_rows(frozen):
    """``in`` and ``out`` of a row cut must cover every row exactly once."""
    manifest = build_manifest(frozen)
    cut = manifest["cuts"]["covariate_extrapolation"]
    mask = cut["mask"]
    assert cut["kind"] == "row"
    assert int(mask.sum()) + int((~mask).sum()) == mask.numel()


def test_a_small_stratum_warns_because_its_per_episode_stats_are_noise(frozen):
    """The guard that stops a 12-episode cut being read like a 333-episode one."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_manifest(frozen)
    messages = [str(w.message) for w in caught]
    assert any(
        f"< {MIN_EPISODES}" in m for m in messages
    ), f"expected a small-stratum warning, got {messages}"


def test_score_strata_buckets_partition_the_pooled_rows(frozen, monkeypatch):
    """Each cut's in/out row counts must add up to the pooled count.

    Uses a stub model so the test pins the bucketing logic rather than a
    checkpoint's behaviour.
    """
    from posterior_eval import score_strata as mod

    meta = load_frozen_meta(frozen)
    manifest = build_manifest(frozen)
    save_manifest(frozen, manifest)
    n_bins = 16

    def fake_predictive(model, batch, record, *, n_ctx, n_max, device="cpu"):
        n_rows = record["tau_star_std"].numel()
        probs = torch.full((n_rows, n_bins), 1.0 / n_bins)
        return probs, torch.linspace(-3.0, 3.0, n_bins + 1)

    monkeypatch.setattr(mod, "score_episode", _stub_score_episode(fake_predictive))
    results = mod.score_strata(object(), frozen, meta=meta, manifest=manifest)

    pooled_rows = results["pooled"]["eval_rows"]
    for name in manifest["cuts"]:
        inside = results.get(f"{name}/in", {}).get("eval_rows", 0)
        outside = results.get(f"{name}/out", {}).get("eval_rows", 0)
        assert inside + outside == pooled_rows, name


def _verdict_text(weak_in, weak_out, capsys):
    from posterior_eval import score_strata as mod

    results = {
        "pooled": {"sd_over_rmse": (weak_in + weak_out) / 2, "episodes": 1000},
        "weak_proxy/in": {"sd_over_rmse": weak_in, "episodes": 334},
        "weak_proxy/out": {"sd_over_rmse": weak_out, "episodes": 666},
    }
    mod._print_verdict(results, {"cuts": {"weak_proxy": {}}})
    return capsys.readouterr().out


def test_the_a2_alarm_needs_absolute_over_confidence_not_just_a_gap(capsys):
    """A globally over-wide model must not be reported as over-confident.

    ``sd_over_rmse`` far above 1 everywhere means the posterior is
    over-*dispersed*. One stratum being relatively less wide inside that is not
    the A2 failure, and firing on it would cry wolf on every early checkpoint --
    which is exactly what the first full run on frozen_prod_v1 did (weak_proxy
    2.86 vs 3.14, both ~3x too wide).
    """
    text = _verdict_text(2.86, 3.14, capsys)
    assert "over-DISPERSED" in text
    # The alarm itself must not fire. Matched on the phrase the alarm branch
    # owns, not on wrapped prose -- the explanatory branch also mentions A2.
    assert "ROADMAP A2 failure" not in text


def test_the_a2_alarm_fires_when_the_stratum_is_actually_over_confident(capsys):
    text = _verdict_text(0.72, 0.98, capsys)
    assert "ROADMAP A2 failure" in text
    assert "over-confident in absolute terms" in text


def test_no_alarm_when_the_weak_stratum_is_not_narrower(capsys):
    text = _verdict_text(1.05, 0.99, capsys)
    assert "not narrower than its complement" in text


def _stub_score_episode(fake_predictive):
    """A ``score_episode`` that skips the model but keeps the fan-out logic."""

    def _score(model, batch, record, *, n_ctx, n_max, device="cpu", **kwargs):
        probs, edges = fake_predictive(
            model, batch, record, n_ctx=n_ctx, n_max=n_max, device=device
        )
        y = record["tau_star_std"].float().reshape(-1)
        s2 = record["s2_std"].float().reshape(-1)
        for evaluator, mask in kwargs.get("dist_evaluators", ()):
            if mask is None:
                evaluator.update(probs=probs, bin_edges=edges, y=y, s2=s2)
                continue
            rows = mask.reshape(-1)
            if not bool(rows.any()):
                continue
            evaluator.update(probs=probs[rows], bin_edges=edges, y=y[rows], s2=s2[rows])
        return {"mse": 0.0, "rows": int(y.numel()), "grid_scale": 1.0}

    return _score
