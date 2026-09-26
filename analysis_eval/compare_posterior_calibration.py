"""Combine cached prior and semi-synthetic posterior-calibration figures.

This command is cache-only: it reads ``points.parquet`` and ``coverage.csv``
from two completed posterior-calibration runs and writes one two-row figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from analysis_eval.posterior_calibration import make_comparison_figure
from analysis_eval.plotting import save_figure


def _load_cached_run(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(path).resolve()
    coverage_path = root / "coverage.csv"
    points_path = root / "points.parquet"
    missing = [path.name for path in (coverage_path, points_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{root} is not a posterior-calibration cache; missing {', '.join(missing)}"
        )
    return pd.read_csv(coverage_path), pd.read_parquet(points_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis_eval.compare_posterior_calibration",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--semisynt-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default="posterior_calibration_comparison")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    prior_coverage, prior_points = _load_cached_run(args.prior_dir)
    semisynt_coverage, semisynt_points = _load_cached_run(args.semisynt_dir)
    figure = make_comparison_figure(
        prior_coverage, prior_points, semisynt_coverage, semisynt_points
    )
    paths = save_figure(figure, args.out_dir.resolve(), args.name)
    print("Wrote:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
