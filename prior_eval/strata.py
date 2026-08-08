"""ROADMAP A2 -- stratified cuts of a frozen evaluation set.

    uv run python -m prior_eval.strata --dir data/prior_eval/frozen_prod_v1

A single pooled ``prior_dist/*`` number hides the failure that matters most.
Under **weak proxies** proximal identification is fragile, and a posterior that
stays narrow there is not merely inaccurate -- it is confidently wrong exactly
where the method's assumptions are weakest. The same hiding applies to strong
confounding, high outcome noise and poor overlap. This module defines the cuts;
:mod:`prior_eval.score_strata` scores them.

**No rebuild of the set is needed, and none is wanted.** The cuts are a
*manifest* -- a ``strata.pt`` sidecar of per-episode membership -- for three
reasons: the cuts overlap (an episode can be both weak-proxy and
poorly-overlapping, so they do not partition), the 365 MB of shards would
otherwise be duplicated per cut, and
:class:`~proximalfm.train._dist_eval.PosteriorEvaluator` accumulates row-weighted
sums, so every stratum is scored in **one** pass over the data rather than one
pass each.

Two kinds of cut, because not every question is about whole episodes:

``kind="episode"``
    A boolean per episode, shape ``(n_datasets,)``. All the DGP-property cuts.

``kind="row"``
    A boolean per query row, shape ``(n_datasets, n_q)``. Covariate
    extrapolation is inherently row-level: query rows are drawn i.i.d. with the
    context, so no *episode* extrapolates -- individual rows do.

Thresholds are absolute where the repo already owns a defensible constant
(``proxy_relevance.WEAK_THRESHOLD``, ``treatment.OVERLAP_BAND``) and terciles
otherwise. Terciles are not a statistical preference; they guarantee ~n/3
episodes per stratum, and the per-episode PIT statistics stop meaning anything
below roughly :data:`MIN_EPISODES`. Realised thresholds and counts are recorded
in the manifest so a cut is reproducible from it alone.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from analysis_prior.analyses.confounding import delivered_shares
from analysis_prior.analyses.mechanism_inputs import mechanism_frame
from analysis_prior.analyses.proxy_relevance import WEAK_THRESHOLD, _knobs
from analysis_prior.analyses.treatment import OVERLAP_BAND
from prior_eval._frozen_records import iter_frozen_arrays, sidecar_records
from prior_eval.build_prior_frozen_set import load_frozen_meta

#: Bump when a change makes an existing strata.pt *wrong* rather than merely
#: differently labelled -- e.g. a cut keeping its name but changing definition.
STRATA_FORMAT_VERSION = 1

#: Below this many episodes a stratum's ``pit_ks/p90`` is noise: it is
#: the 90th percentile of a per-episode statistic, so it needs episodes, not
#: rows. Cuts under it are still scored, with a warning attached.
MIN_EPISODES = 100

#: Ridge added to the context covariance before inversion in the Mahalanobis
#: cut. X blocks can be rank-deficient (categorised columns, duplicated draws),
#: and a pseudo-inverse would silently drop those directions instead of saying
#: the distance there is undefined.
_MAHALANOBIS_RIDGE = 1e-6

MANIFEST_NAME = "strata.pt"


@dataclass
class Cut:
    """One stratum: its membership, the score behind it, and how it was cut."""

    name: str
    kind: str  # "episode" | "row"
    description: str
    source: str
    mask: np.ndarray
    score: np.ndarray
    threshold: float
    direction: str  # "below" | "above"

    @property
    def n_selected(self) -> int:
        return int(self.mask.sum())

    @property
    def n_total(self) -> int:
        return int(self.mask.size)

    def episodes_selected(self) -> int:
        """Episodes touched -- for a row cut, those with at least one row in."""
        return (
            self.n_selected if self.kind == "episode" else int(self.mask.any(1).sum())
        )


def _threshold_mask(score: np.ndarray, direction: str, q: float) -> tuple:
    """Quantile cut on the finite scores; NaN never joins a stratum.

    A NaN score means the diagnostic could not be computed for that episode
    (an absent hyperparameter, a degenerate block). Excluding it is the honest
    default: an episode whose weak-proxy score is unknown is not evidence that
    it is weak.
    """
    finite = np.isfinite(score)
    if not finite.any():
        return np.zeros_like(score, dtype=bool), float("nan")
    threshold = float(np.quantile(score[finite], q))
    if direction == "below":
        mask = finite & (score <= threshold)
    elif direction == "above":
        mask = finite & (score >= threshold)
    else:
        raise ValueError(f"direction must be below/above, got {direction!r}")
    return mask, threshold


def _absolute_mask(score: np.ndarray, direction: str, threshold: float) -> np.ndarray:
    finite = np.isfinite(score)
    if direction == "below":
        return finite & (score < threshold)
    return finite & (score > threshold)


# --------------------------------------------------------------------------
# The cuts. Each returns (score, description, source); thresholding is common.
# --------------------------------------------------------------------------


def score_weak_proxy(records: Sequence[Any]) -> np.ndarray:
    """``u_share_expected``, minimised over the two proxies.

    ``analysis_prior.analyses.proxy_relevance._knobs`` derives, from the two
    knobs that dilute ``U`` out of ``f_P``, the expected share of that
    mechanism's input width which is ``U``-derived. **The minimum over W and Z
    is the binding direction**: proximal identification needs *both* proxies, so
    the weaker one governs, exactly as ``confounding.delivered_shares`` takes
    ``share_min`` for the same reason on the confounding side.

    It is an expectation, not a measurement -- the covariate subsample is a
    Binomial draw whose realisation is not recorded. ``weak_proxy_measured`` is
    the measured version.
    """
    return np.array(
        [
            min(
                _knobs(r, "W")["u_share_expected"],
                _knobs(r, "Z")["u_share_expected"],
            )
            for r in records
        ],
        dtype=np.float64,
    )


def score_strong_confounding(records: Sequence[Any]) -> np.ndarray:
    """Delivered ``U`` share of ``f_A`` and ``f_Y``, minimised.

    ``delivered_shares`` -> ``share_min``. Delivered (first-layer injected
    variance), not the knob: block-wise dropout can zero a wired block, so the
    target share and the realised one come apart. Bias needs a ``U`` path into
    *both* mechanisms, hence the min.
    """
    frame = delivered_shares(records).set_index("index")
    return frame.loc[[r.index for r in records], "share_min"].to_numpy(np.float64)


def score_outcome_noise(records: Sequence[Any]) -> np.ndarray:
    """Delivered exogenous-noise share of ``f_Y``'s first layer.

    ``mechanism_frame`` -> ``first_layer_share_total_eps`` for mechanism ``Y``.
    Again delivered rather than the ``noise_share_of_input_in_Y`` knob, for the
    same dropout reason.
    """
    frame = mechanism_frame(records)
    y = frame[frame["mechanism"] == "Y"].set_index("index")
    return y.loc[[r.index for r in records], "first_layer_share_total_eps"].to_numpy(
        np.float64
    )


def score_homogeneous_effect(records: Sequence[Any]) -> np.ndarray:
    """``gamma_hetero``, the outcome remix dial.

    ``gamma -> 0`` collapses the SCM's heterogeneity into a constant treatment
    effect. Not in ROADMAP's list, but it earns a cut: ``prior_eval.verify``
    showed these episodes drive the out-of-grid tail (a flat ``tau`` sitting past
    the grid edge), and a near-constant target is where a posterior can look
    well-calibrated for reasons that have nothing to do with the proxies.
    """
    return np.array(
        [np.nan if r.gamma_hetero is None else float(r.gamma_hetero) for r in records],
        dtype=np.float64,
    )


def score_poor_overlap(out: Path, meta: Dict[str, Any], n: int) -> np.ndarray:
    """Share of context rows whose true propensity is outside ``OVERLAP_BAND``.

    Reads ``A_probs`` -- the ground-truth ``P(A=1 | X, U)`` -- straight off the
    shards, exactly as ``analysis_prior.analyses.treatment.OverlapAnalysis``
    does; fitting a propensity model would only re-estimate a known quantity
    with error. Context rows, because overlap is a property of the sample a
    method would actually be handed.
    """
    lo, hi = OVERLAP_BAND
    score = np.full(n, np.nan)
    for arrays in iter_frozen_arrays(out, meta=meta):
        pi = arrays["A_probs_context"]
        score[arrays["index"]] = float(np.mean((pi < lo) | (pi > hi)))
    return score


def score_covariate_extrapolation(
    out: Path, meta: Dict[str, Any], n: int, n_q: int, q: float = 0.99
) -> tuple:
    """Row-level: query rows far from the context covariate cloud.

    Mahalanobis distance of each query row's ``X`` from the *context* rows'
    mean and covariance, thresholded at the context's own ``q``-quantile of the
    same distance. Self-referential on purpose: it asks "is this query row
    further out than the context rows themselves typically are", which needs no
    cross-episode scale and no tuning constant beyond ``q``.

    This is the one cut in A2 with no prior art in the repo, so the definition
    is a judgement call rather than an inherited constant. Returns
    ``(mask, distance)`` both shaped ``(n_datasets, n_q)``.
    """
    mask = np.zeros((n, n_q), dtype=bool)
    dist = np.full((n, n_q), np.nan)
    for arrays in iter_frozen_arrays(out, meta=meta):
        i = arrays["index"]
        ctx, qry = arrays["X_context"], arrays["X_query"]
        mu = ctx.mean(0)
        cov = np.cov(ctx, rowvar=False)
        cov = np.atleast_2d(cov)
        cov.flat[:: cov.shape[0] + 1] += _MAHALANOBIS_RIDGE
        try:
            prec = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue
        d_ctx = np.einsum("ij,jk,ik->i", ctx - mu, prec, ctx - mu)
        d_qry = np.einsum("ij,jk,ik->i", qry - mu, prec, qry - mu)
        cutoff = float(np.quantile(d_ctx, q))
        dist[i] = d_qry
        mask[i] = d_qry > cutoff
    return mask, dist


def score_weak_proxy_measured(
    out: Path, meta: Dict[str, Any], n: int, model: str = "rf", seed: int = 0
) -> np.ndarray:
    """Cross-fitted ``delta_r2 = R2(u_j | X, P) - R2(u_j | X)``, worst direction.

    The measured counterpart of :func:`score_weak_proxy`, via
    ``analysis_prior.analyses.proxy_relevance.relevance_frame``. Expensive --
    cross-fitted regressions over every ``(episode, U column, proxy)`` -- but
    disk-cached under ``cache/regression/`` and paid **once**, at manifest-build
    time, never at score time. That is the whole reason membership lives in a
    manifest rather than being recomputed per scoring run.

    Aggregated as the minimum over ``U`` columns *and* over proxies: a proxy
    that is uninformative about even one confounder direction breaks
    identification for that direction.
    """
    from analysis_prior.analyses.proxy_relevance import relevance_frame
    from prior_eval._frozen_records import frozen_to_records

    records = frozen_to_records(out, rows="context", meta=meta)
    cache_key = f"frozen-{meta['global'].get('git_rev', 'nogit')}-{n}"
    frame = relevance_frame(records, cache_key=cache_key, model=model, seed=seed)
    worst = frame.groupby("index")["delta_r2"].min()
    score = np.full(n, np.nan)
    score[worst.index.to_numpy()] = worst.to_numpy(np.float64)
    return score


def build_manifest(
    out: Path,
    *,
    with_regression: bool = False,
    tercile: float = 1.0 / 3.0,
    extrapolation_q: float = 0.99,
    regression_model: str = "rf",
    seed: int = 0,
    cli_command: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute every cut over the frozen set at ``out``.

    ``with_regression`` adds ``weak_proxy_measured``, the only cut that costs
    more than a few seconds.
    """
    out = Path(out)
    meta = load_frozen_meta(out)
    glob = meta["global"]
    n = len(meta["episodes"])
    n_q = int(glob["n_q"])
    records = sidecar_records(out, meta=meta)

    cuts: List[Cut] = []

    def add_quantile(name, score, direction, q, description, source):
        mask, thr = _threshold_mask(score, direction, q)
        cuts.append(
            Cut(
                name=name,
                kind="episode",
                description=description,
                source=source,
                mask=mask,
                score=score,
                threshold=thr,
                direction=direction,
            )
        )

    add_quantile(
        "weak_proxy",
        score_weak_proxy(records),
        "below",
        tercile,
        "Bottom tercile of expected U-share of the weaker proxy mechanism.",
        "proxy_relevance._knobs -> u_share_expected, min over W/Z",
    )
    add_quantile(
        "strong_confounding",
        score_strong_confounding(records),
        "above",
        1.0 - tercile,
        "Top tercile of delivered U share, minimised over f_A and f_Y.",
        "confounding.delivered_shares -> share_min",
    )
    add_quantile(
        "high_outcome_noise",
        score_outcome_noise(records),
        "above",
        1.0 - tercile,
        "Top tercile of delivered exogenous-noise share of f_Y.",
        "mechanism_inputs.mechanism_frame -> Y first_layer_share_total_eps",
    )
    add_quantile(
        "homogeneous_effect",
        score_homogeneous_effect(records),
        "below",
        tercile,
        "Bottom tercile of gamma_hetero: near-constant treatment effect.",
        "prior diagnostics -> gamma_hetero",
    )
    add_quantile(
        "poor_overlap",
        score_poor_overlap(out, meta, n),
        "above",
        1.0 - tercile,
        f"Top tercile of the share of context rows outside {OVERLAP_BAND}.",
        "A_probs on context rows vs treatment.OVERLAP_BAND",
    )

    extrap_mask, extrap_dist = score_covariate_extrapolation(
        out, meta, n, n_q, q=extrapolation_q
    )
    cuts.append(
        Cut(
            name="covariate_extrapolation",
            kind="row",
            description=(
                "Query rows whose X Mahalanobis distance from the context cloud "
                f"exceeds the context's own p{extrapolation_q:.0%}."
            ),
            source="Mahalanobis(X_query; context mean/cov), self-referential cutoff",
            mask=extrap_mask,
            score=extrap_dist,
            threshold=float(extrapolation_q),
            direction="above",
        )
    )

    if with_regression:
        score = score_weak_proxy_measured(
            out, meta, n, model=regression_model, seed=seed
        )
        mask = _absolute_mask(score, "below", WEAK_THRESHOLD)
        cuts.append(
            Cut(
                name="weak_proxy_measured",
                kind="episode",
                description=(
                    "Cross-fitted delta_r2 below WEAK_THRESHOLD "
                    f"({WEAK_THRESHOLD}) in the worst U-direction."
                ),
                source=f"proxy_relevance.relevance_frame(model={regression_model!r})",
                mask=mask,
                score=score,
                threshold=float(WEAK_THRESHOLD),
                direction="below",
            )
        )

    for cut in cuts:
        selected = cut.episodes_selected()
        if selected < MIN_EPISODES:
            warnings.warn(
                f"cut {cut.name!r} selects {selected} episodes (< {MIN_EPISODES}); "
                "its pit_ks/p90 will be dominated by noise and its "
                "pit_ks_pooled band will be wide. Read it as indicative only.",
                RuntimeWarning,
            )

    return {
        "format_version": STRATA_FORMAT_VERSION,
        # Provenance: which set these cuts describe. load_manifest refuses a
        # mismatch, which is ROADMAP A4 enforced at the one place it can be.
        "source": {
            "n_datasets": int(glob["n_datasets"]),
            "n_q": n_q,
            "seed": int(glob["seed"]),
            "git_rev": glob.get("git_rev"),
            "meta_format_version": int(meta.get("format_version", 0)),
        },
        "build": {
            "cli_command": cli_command,
            "tercile": tercile,
            "extrapolation_q": extrapolation_q,
            "with_regression": with_regression,
            "regression_model": regression_model if with_regression else None,
            "seed": seed,
        },
        "cuts": {
            c.name: {
                "kind": c.kind,
                "description": c.description,
                "source": c.source,
                "mask": torch.from_numpy(c.mask),
                "score": torch.from_numpy(c.score),
                "threshold": c.threshold,
                "direction": c.direction,
                "n_selected": c.n_selected,
                "n_total": c.n_total,
                "episodes_selected": c.episodes_selected(),
            }
            for c in cuts
        },
    }


def save_manifest(out: Path, manifest: Dict[str, Any]) -> Path:
    """Write ``strata.pt`` through a ``.tmp`` sibling, as the builder does."""
    out = Path(out)
    tmp = out / f"{MANIFEST_NAME}.tmp"
    torch.save(manifest, tmp)
    target = out / MANIFEST_NAME
    tmp.replace(target)
    return target


class ManifestMismatchError(RuntimeError):
    """Raised when a manifest does not describe the set it is loaded against."""


def load_manifest(out: Path, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read ``strata.pt`` and refuse it if it describes a different set.

    Membership is positional -- cut ``k``'s element ``i`` is episode ``i`` of
    the ``val/`` directory -- so a manifest paired with the wrong set produces
    numbers that are wrong without being anomalous. Checking provenance is the
    only way that failure ever surfaces.
    """
    out = Path(out)
    path = out / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"no {MANIFEST_NAME} in {out}. Build it first:\n"
            f"  uv run python -m prior_eval.strata --dir {out}"
        )
    manifest = torch.load(path, weights_only=False)
    meta = load_frozen_meta(out) if meta is None else meta
    glob = meta["global"]
    src = manifest.get("source", {})
    mismatches = [
        f"{k}: manifest={src.get(k)!r} set={v!r}"
        for k, v in (
            ("n_datasets", int(glob["n_datasets"])),
            ("n_q", int(glob["n_q"])),
            ("seed", int(glob["seed"])),
            ("git_rev", glob.get("git_rev")),
        )
        if src.get(k) != v
    ]
    if mismatches:
        raise ManifestMismatchError(
            f"{path} does not describe the set at {out}:\n  "
            + "\n  ".join(mismatches)
            + "\nMembership is positional, so scoring with it would silently "
            "attribute each episode's metrics to the wrong stratum. Rebuild:\n"
            f"  uv run python -m prior_eval.strata --dir {out}"
        )
    if manifest.get("format_version") != STRATA_FORMAT_VERSION:
        raise ManifestMismatchError(
            f"{path} has format_version={manifest.get('format_version')}, this "
            f"build expects {STRATA_FORMAT_VERSION}. Rebuild it."
        )
    return manifest


def _print_summary(out: Path, manifest: Dict[str, Any]) -> None:
    cuts = manifest["cuts"]
    n_total = manifest["source"]["n_datasets"]
    print(f"\nStrata manifest for {out}  ({n_total} episodes)")
    print(f"  git_rev={manifest['source']['git_rev']}")
    print()
    print(f"  {'cut':<26} {'kind':<8} {'selected':>18}  threshold")
    print("  " + "-" * 68)
    for name, c in cuts.items():
        if c["kind"] == "episode":
            sel = f"{c['n_selected']:>5} ep ({c['n_selected'] / n_total:>5.1%})"
        else:
            sel = f"{c['n_selected']:>6} rows, {c['episodes_selected']:>4} ep"
        print(f"  {name:<26} {c['kind']:<8} {sel:>18}  {c['threshold']:.4g}")
    print()
    for name, c in cuts.items():
        print(f"  {name}: {c['description']}")
        print(f"      source: {c['source']}")

    episode_cuts = {k: v for k, v in cuts.items() if v["kind"] == "episode"}
    if len(episode_cuts) > 1:
        print("\n  Pairwise overlap (share of the row cut also in the column cut).")
        print("  The cuts do NOT partition; a stratum result that looks specific")
        print("  to weak proxies may just be strong confounding if these are high.")
        names = list(episode_cuts)
        width = max(len(n) for n in names) + 1
        print("  " + " " * width + " ".join(f"{n[:7]:>8}" for n in names))
        for a in names:
            ma = episode_cuts[a]["mask"].numpy()
            row = []
            for b in names:
                mb = episode_cuts[b]["mask"].numpy()
                row.append(
                    f"{(ma & mb).sum() / max(1, ma.sum()):>8.2f}"
                    if ma.sum()
                    else "     nan"
                )
            print(f"  {a:<{width}}" + " ".join(row))

    small = [n for n, c in cuts.items() if c["episodes_selected"] < MIN_EPISODES]
    if small:
        print(
            f"\n  ! {', '.join(small)} select fewer than {MIN_EPISODES} episodes; "
            "their per-episode PIT statistics are noise."
        )
    print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m prior_eval.strata",
        description="Build the stratified-cut manifest for a frozen eval set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="frozen set ROOT (holding val/ and frozen_meta.pt)",
    )
    p.add_argument(
        "--with_regression",
        action="store_true",
        help="also compute weak_proxy_measured (cross-fitted delta_r2; slow, "
        "disk-cached, paid once)",
    )
    p.add_argument("--regression_model", type=str, default="rf")
    p.add_argument(
        "--tercile",
        type=float,
        default=1.0 / 3.0,
        help="quantile for the quantile-based cuts",
    )
    p.add_argument(
        "--extrapolation_q",
        type=float,
        default=0.99,
        help="context-distance quantile defining covariate_extrapolation",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--summary_only",
        action="store_true",
        help="print an existing manifest without rebuilding it",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.summary_only:
        _print_summary(args.dir, load_manifest(args.dir))
        return 0

    cli_command = shlex.join(
        [sys.executable, "-m", "prior_eval.strata", *(argv or sys.argv[1:])]
    )
    manifest = build_manifest(
        args.dir,
        with_regression=args.with_regression,
        tercile=args.tercile,
        extrapolation_q=args.extrapolation_q,
        regression_model=args.regression_model,
        seed=args.seed,
        cli_command=cli_command,
    )
    path = save_manifest(args.dir, manifest)
    _print_summary(args.dir, manifest)
    print(f"Written to {path}")
    print(
        "Next: uv run python -m prior_eval.score_strata "
        f"--dir {args.dir} --ckpt <ckpt>"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
