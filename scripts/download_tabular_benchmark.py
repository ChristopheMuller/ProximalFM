#!/usr/bin/env python3
"""Download the selected INRA/SODA tabular-benchmark CSV files.

The data live in one Hugging Face dataset repository.  Each task variant is a
separate CSV, so a dataset that appears in more than one task (for example
``pol``) is downloaded once per requested variant.

Examples
--------
python scripts/download_tabular_benchmark.py
python scripts/download_tabular_benchmark.py --output /scratch/tabular-benchmark
python scripts/download_tabular_benchmark.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

from huggingface_hub import HfApi, hf_hub_download


REPOSITORY = "inria-soda/tabular-benchmark"

SELECTED_FILES = (
    "clf_num/pol.csv",
    "clf_num/MiniBooNE.csv",
    "clf_num/default-of-credit-card-clients.csv",
    "clf_num/Higgs.csv",
    "clf_num/jannis.csv",
    "clf_num/heloc.csv",
    "clf_cat/covertype.csv",
    "clf_cat/albert.csv",
    "clf_cat/road-safety.csv",
    "reg_num/Ailerons.csv",
    "reg_num/superconduct.csv",
    "reg_cat/Allstate_Claims_Severity.csv",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("semisynt_eval/datasets/"),
        help="Directory where the selected CSV files are written.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face branch, tag, or commit to resolve (default: main).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download files even when they are already present locally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected repository paths without accessing the network.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        print(f"Would download {len(SELECTED_FILES)} files from {REPOSITORY}:")
        print("\n".join(f"  {path}" for path in SELECTED_FILES))
        return

    # Resolve a moving branch such as ``main`` once.  Passing the resulting
    # commit to every download prevents a repository update mid-download from
    # producing a mixed-version local data set.
    revision = HfApi().dataset_info(REPOSITORY, revision=args.revision).sha
    print(f"Downloading {len(SELECTED_FILES)} files from {REPOSITORY}@{revision}")
    args.output.mkdir(parents=True, exist_ok=True)
    for filename in SELECTED_FILES:
        cached_path = hf_hub_download(
            repo_id=REPOSITORY,
            repo_type="dataset",
            filename=filename,
            revision=revision,
            force_download=args.force_download,
        )
        destination = args.output / Path(filename).name
        shutil.copy2(cached_path, destination)
        print(destination)


if __name__ == "__main__":
    main()
