"""CLI entry point for the prior audits.

Examples
--------
Run every analysis on the smoke preset (fast; a few tiny datasets)::

    PYTHONPATH=. uv run python -m analysis_prior.main --spec smoke

Run one analysis on the default (training-scale) prior, forcing regeneration::

    PYTHONPATH=. uv run python -m analysis_prior.main \\
        --spec default --analysis dimensions --force --n-jobs -1

Override individual spec fields on top of a preset::

    PYTHONPATH=. uv run python -m analysis_prior.main \\
        --spec default --n-datasets 128 --seq-len 1000 --mc-cate 50

Discovery::

    PYTHONPATH=. uv run python -m analysis_prior.main --list-analyses
    PYTHONPATH=. uv run python -m analysis_prior.main --list-caches
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import List, Optional, Sequence

from .analyses import ComparisonAnalysis, all_analyses, get_analysis
from .config import FAMILIES, PRESETS, PriorSpec, SpecFamily
from .generate import DEFAULT_CACHE_DIR, list_caches, load_or_generate
from .generate import PriorCache

#: Default location for rendered figures and tables.
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"

#: Spec fields overridable from the CLI, mapped to their argparse dest names.
_SPEC_OVERRIDE_FIELDS = (
    "n_datasets",
    "seq_len",
    "mc_cate",
    "seed",
    "prior_type",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analysis_prior",
        description="Reproducible audits of the ProximalFM synthetic prior.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--spec",
        default="default",
        help=f"Preset spec to start from. One of: {sorted(PRESETS)}.",
    )
    parser.add_argument(
        "--family",
        default=None,
        help=(
            f"Run a comparison family instead of a single spec. One of: "
            f"{sorted(FAMILIES)}. Overrides --spec; spec overrides like "
            "--n-datasets apply to the family's base."
        ),
    )
    parser.add_argument(
        "--analysis",
        nargs="+",
        default=["all"],
        help="Analyses to run, space-separated, or 'all'.",
    )

    # Spec overrides. Default None means "keep the preset's value".
    parser.add_argument(
        "--name",
        default=None,
        help="Rename the spec (affects the cache key/output dir).",
    )
    parser.add_argument("--n-datasets", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--mc-cate", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--prior-type", default=None, choices=["mlp_scm", "tree_scm", "mix_scm"]
    )

    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--force", action="store_true", help="Regenerate the cache even if present."
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for generation (-1 = all cores).",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show a live progress bar while generating datasets.",
    )

    parser.add_argument(
        "--list-analyses", action="store_true", help="List analyses and exit."
    )
    parser.add_argument(
        "--list-caches", action="store_true", help="List cached specs and exit."
    )
    parser.add_argument(
        "--list-families",
        action="store_true",
        help="List comparison families and exit.",
    )

    return parser


def _cli_overrides(args: argparse.Namespace) -> dict:
    """Collect the spec-field overrides given on the command line."""
    overrides = {}
    for field in _SPEC_OVERRIDE_FIELDS:
        value = getattr(args, field)
        if value is not None:
            overrides[field] = value
    if args.name is not None:
        overrides["name"] = args.name
    return overrides


def build_spec(args: argparse.Namespace) -> PriorSpec:
    """Build a :class:`PriorSpec` from a preset plus CLI overrides."""
    if args.spec not in PRESETS:
        raise SystemExit(
            f"Unknown --spec {args.spec!r}. Available presets: {sorted(PRESETS)}"
        )

    # dataclasses.replace re-runs __post_init__, so an invalid override (e.g.
    # seq_len=1) is rejected here with the same validation as direct construction.
    return dataclasses.replace(PRESETS[args.spec], **_cli_overrides(args))


def build_family(args: argparse.Namespace) -> SpecFamily:
    """Build a :class:`SpecFamily` from a registered family plus CLI overrides.

    Overrides apply to the family's *base*, so ``--family dim_ablation
    --n-datasets 32`` runs the whole comparison at a smaller size.
    """
    if args.family not in FAMILIES:
        raise SystemExit(
            f"Unknown --family {args.family!r}. Available: {sorted(FAMILIES)}"
        )

    family = FAMILIES[args.family]
    overrides = _cli_overrides(args)
    if overrides:
        family = dataclasses.replace(
            family, base=dataclasses.replace(family.base, **overrides)
        )
    return family


def resolve_analyses(names: Sequence[str]) -> List[str]:
    """Expand the ``--analysis`` argument into concrete analysis names."""
    if list(names) == ["all"]:
        return sorted(all_analyses())
    if "all" in names:
        raise SystemExit("--analysis 'all' cannot be combined with other names")
    # Validate every name up front so a typo fails before any generation runs.
    for name in names:
        get_analysis(name)
    return list(names)


def _print_analyses() -> None:
    print("Available analyses:")
    for name, analysis in sorted(all_analyses().items()):
        print(f"  {name:16s} {analysis.description}")


def _print_caches(cache_dir: Path) -> None:
    entries = list_caches(cache_dir)
    if not entries:
        print(f"No caches in {cache_dir}")
        return
    # Caches are content-addressed, so they have no name -- identify them by the
    # generative fields that distinguish them.
    print(f"Caches in {cache_dir}:")
    for entry in entries:
        spec = entry["generative_spec"]
        print(
            f"  n={entry.get('n_stored'):<5} T={spec.get('seq_len'):<6} "
            f"K={spec.get('mc_cate'):<5} {spec.get('prior_type'):<9} "
            f"{Path(entry['path']).name}"
        )


def _print_families() -> None:
    print("Available families:")
    for name, family in sorted(FAMILIES.items()):
        labels = ", ".join(family.interventions)
        print(f"  {name:16s} base={family.base.name}  variants: {labels}")


def _print_summary(name: str, description: str, summary: dict) -> None:
    print(f"\n[{name}] {description}")
    for key, value in summary.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else value
        print(f"    {key}: {formatted}")


def _run_single(args: argparse.Namespace, analysis_names: List[str]) -> int:
    """Run analyses against one spec (the pre-family behaviour)."""
    # Comparison analyses contrast several variants; a single spec cannot feed
    # them. Skip them under the implicit 'all', but treat an explicit request
    # as a user error rather than silently doing nothing.
    explicit = list(args.analysis) != ["all"]
    comparisons = [
        n for n in analysis_names if isinstance(get_analysis(n), ComparisonAnalysis)
    ]
    if comparisons:
        if explicit:
            raise SystemExit(
                f"Analyses {comparisons} compare prior variants; run them with "
                "--family instead of --spec."
            )
        analysis_names = [n for n in analysis_names if n not in comparisons]
        print(f"Skipping comparison analyses (need --family): {', '.join(comparisons)}")

    spec = build_spec(args)
    # Parameter tables are pure functions of the specification.  In particular
    # the production table describes a variable-length, query-only-MC training
    # stream that is intentionally not materialised by the dense audit cache.
    if analysis_names == ["hyperparameters"]:
        cache = PriorCache(spec=spec, records=[])
    else:
        cache = load_or_generate(
            spec,
            cache_dir=args.cache_dir,
            force=args.force,
            n_jobs=args.n_jobs,
            progress_bar=args.progress_bar,
        )

    # Results are grouped by the spec's *name*: caches are content-addressed
    # (an opaque hash), but output directories should stay human-browsable.
    # spec.json inside each directory records the exact configuration.
    spec_out = Path(args.out_dir) / spec.name
    print(f"\nWriting results to {spec_out}")

    for name in analysis_names:
        analysis = get_analysis(name)
        artifacts = analysis.run(cache, spec_out / name)
        _print_summary(name, analysis.description, artifacts.summary)

    return 0


def _run_family(args: argparse.Namespace, analysis_names: List[str]) -> int:
    """Run analyses across every variant of a comparison family.

    Plain analyses run once per variant, into ``<family>/<label>/<analysis>/``;
    comparison analyses run once with all caches, into ``<family>/<analysis>/``.
    """
    family = build_family(args)
    variants = family.resolve()

    caches = {
        label: load_or_generate(
            spec,
            cache_dir=args.cache_dir,
            force=args.force,
            n_jobs=args.n_jobs,
            progress_bar=args.progress_bar,
        )
        for label, spec in variants.items()
    }

    family_out = Path(args.out_dir) / family.name
    family_out.mkdir(parents=True, exist_ok=True)
    with open(family_out / "specs.json", "w") as f:
        json.dump(family.to_dict(), f, indent=2)
    print(f"\nWriting results to {family_out}")

    for name in analysis_names:
        analysis = get_analysis(name)
        if isinstance(analysis, ComparisonAnalysis):
            artifacts = analysis.run(caches, family_out / name)
            _print_summary(name, analysis.description, artifacts.summary)
        else:
            for label, cache in caches.items():
                artifacts = analysis.run(cache, family_out / label / name)
                _print_summary(
                    f"{name} · {label}", analysis.description, artifacts.summary
                )

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_analyses:
        _print_analyses()
        return 0
    if args.list_caches:
        _print_caches(args.cache_dir)
        return 0
    if args.list_families:
        _print_families()
        return 0

    analysis_names = resolve_analyses(args.analysis)

    if args.family is not None:
        return _run_family(args, analysis_names)
    return _run_single(args, analysis_names)


if __name__ == "__main__":
    raise SystemExit(main())
