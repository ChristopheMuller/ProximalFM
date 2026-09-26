"""Generate the seven-context semisynthetic selection-method nPEHE table.

Run after scripts/run_semisynt_selection_benchmarks.sh::

    uv run python -m analysis_eval.selection_tables
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .exhaustive_tables import (
    DEFAULT_ALPHA,
    DEFAULT_EXTENDED_CASE,
    DEFAULT_EXTENDED_CONTEXTS,
    DEFAULT_MATRIX_ROOT,
    DEFAULT_REPLICATES,
    aggregate_replicates,
    load_results,
    render_extended_table,
    summarize_extended_case,
    write_summary_csv,
)

DEFAULT_OUT = (
    Path(__file__).resolve().parent / "results" / "semisynt_selection_npehe.tex"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--case", default=DEFAULT_EXTENDED_CASE)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=DEFAULT_EXTENDED_CONTEXTS
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print", dest="to_stdout", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out.suffix.lower() != ".tex":
        raise ValueError("--out must be a .tex path")
    frame = load_results(args.matrix_root, dimension=args.dim, selected_case=args.case)
    source_means = aggregate_replicates(frame, expected_replicates=args.replicates)
    summary = summarize_extended_case(
        source_means,
        case=args.case,
        expected_contexts=args.contexts,
        alpha=args.alpha,
    )
    latex = render_extended_table(summary, alpha=args.alpha, replicates=args.replicates)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(latex, encoding="utf-8")
    csv_path = args.out.with_suffix(".csv")
    write_summary_csv(summary, csv_path)
    print(f"wrote {args.out}")
    print(f"wrote {csv_path}")
    if args.to_stdout:
        print(latex, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
