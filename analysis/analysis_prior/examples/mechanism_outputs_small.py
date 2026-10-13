"""Fast, small example of the mechanism output-dependence figure (B.2.3).

The production figure (``--analysis mechanism_outputs``) runs a cross-fitted
**TabICL** regression over 256 datasets and takes a couple of hours on the
first pass (predictions are then cached, so re-renders are instant). This
script produces the *same* figure on a handful of datasets with a fast model
(random forest by default), so the whole pipeline -- generate, cross-fit,
leave-one-out ΔR², render -- can be exercised end to end in well under a
minute. It is a demo/smoke of the analysis, not the paper artifact: RF is a
weaker predictor than TabICL, so absolute R² is lower, but the qualitative
"each knob steers its channel" trend is already visible.

Examples
--------
Default (32 datasets, random forest)::

    PYTHONPATH=. uv run python -m analysis.analysis_prior.examples.mechanism_outputs_small

A quick TabICL taste on fewer datasets::

    PYTHONPATH=. uv run python -m analysis.analysis_prior.examples.mechanism_outputs_small \\
        --n 16 --model tabicl

The output lands in ``results/general_ite/mechanism_outputs_small/`` by default,
next to (but not clobbering) the production ``mechanism_outputs/`` figure. The
regression cache is shared and keyed by model, so RF and TabICL runs never
collide.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from ..analyses.mechanism_outputs import MechanismOutputsAnalysis
from ..config import PRESETS
from ..generate import load_or_generate

DEFAULT_OUT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "general_ite"
    / "mechanism_outputs_small"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Small, fast mechanism output-dependence figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=32, help="Datasets to use.")
    parser.add_argument(
        "--model",
        default="rf",
        choices=["linear", "rf", "tabicl"],
        help="Predictor. 'rf' is fast; 'tabicl' is the paper model (slow).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1024,
        help="Cap rows per dataset for a faster fit.",
    )
    parser.add_argument("--n-splits", type=int, default=3, help="CV folds.")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Output directory."
    )
    args = parser.parse_args(argv)

    # The analysis uses every dataset in the cache it is given, so the count is
    # controlled through the spec: a general_ite spec with n_datasets=--n is
    # served as a prefix of the 1024-dataset cache -- no regeneration (the cache
    # is content-addressed on the generative fields, which n_datasets is not).
    spec = dataclasses.replace(PRESETS["general_ite"], n_datasets=args.n)
    cache = load_or_generate(spec, n_jobs=1, progress=True)

    analysis = MechanismOutputsAnalysis()
    analysis.model = args.model
    analysis.n_splits = args.n_splits
    analysis.max_rows = args.max_rows

    print(
        f"\nmechanism_outputs example: {args.n} datasets, model={args.model}, "
        f"max_rows={args.max_rows}, {args.n_splits}-fold"
    )
    artifacts = analysis.run(cache, args.out)

    print(f"\nWrote to {args.out}:")
    for path in artifacts.figures + artifacts.tables:
        print(f"  {path.name}")
    print("\nSummary:")
    for key, value in artifacts.summary.items():
        print(f"  {key}: {round(value, 4) if isinstance(value, float) else value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
