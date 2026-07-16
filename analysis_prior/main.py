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
        --spec default --n-datasets 128 --seq-len 1000 --mcmc-cate 50

Discovery::

    PYTHONPATH=. uv run python -m analysis_prior.main --list-analyses
    PYTHONPATH=. uv run python -m analysis_prior.main --list-caches
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import List, Optional, Sequence

from .analyses import all_analyses, get_analysis
from .config import PRESETS, PriorSpec
from .generate import DEFAULT_CACHE_DIR, list_caches, load_or_generate

#: Default location for rendered figures and tables.
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"

#: Spec fields overridable from the CLI, mapped to their argparse dest names.
_SPEC_OVERRIDE_FIELDS = (
    "n_datasets",
    "seq_len",
    "mcmc_cate",
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
    parser.add_argument("--mcmc-cate", type=int, default=None)
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
        "--list-analyses", action="store_true", help="List analyses and exit."
    )
    parser.add_argument(
        "--list-caches", action="store_true", help="List cached specs and exit."
    )

    return parser


def build_spec(args: argparse.Namespace) -> PriorSpec:
    """Build a :class:`PriorSpec` from a preset plus CLI overrides."""
    if args.spec not in PRESETS:
        raise SystemExit(
            f"Unknown --spec {args.spec!r}. Available presets: {sorted(PRESETS)}"
        )

    overrides = {}
    for field in _SPEC_OVERRIDE_FIELDS:
        value = getattr(args, field)
        if value is not None:
            overrides[field] = value
    if args.name is not None:
        overrides["name"] = args.name

    # dataclasses.replace re-runs __post_init__, so an invalid override (e.g.
    # seq_len=1) is rejected here with the same validation as direct construction.
    return dataclasses.replace(PRESETS[args.spec], **overrides)


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
    print(f"Caches in {cache_dir}:")
    for entry in entries:
        spec = entry["spec"]
        print(
            f"  {spec.get('name'):20s} "
            f"n={spec.get('n_datasets'):<5} T={spec.get('seq_len'):<6} "
            f"K={spec.get('mcmc_cate'):<5} {Path(entry['path']).name}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_analyses:
        _print_analyses()
        return 0
    if args.list_caches:
        _print_caches(args.cache_dir)
        return 0

    spec = build_spec(args)
    analysis_names = resolve_analyses(args.analysis)

    cache = load_or_generate(
        spec, cache_dir=args.cache_dir, force=args.force, n_jobs=args.n_jobs
    )

    # Group outputs by spec so multiple specs never clobber each other.
    spec_out = Path(args.out_dir) / spec.cache_key
    print(f"\nWriting results to {spec_out}")

    for name in analysis_names:
        analysis = get_analysis(name)
        artifacts = analysis.run(cache, spec_out / name)
        print(f"\n[{name}] {analysis.description}")
        for key, value in artifacts.summary.items():
            formatted = f"{value:.4f}" if isinstance(value, float) else value
            print(f"    {key}: {formatted}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
