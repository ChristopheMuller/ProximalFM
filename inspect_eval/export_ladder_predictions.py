"""Export the inspection ladder as benchmark prediction caches.

The benchmark scores predictions in context-standardized outcome units.  The
inspection sidecar already stores each ladder target in exactly those units, so
this script only validates the episode alignment and stacks the vectors into
the benchmark's ``preds/<name>_preds.npz`` format.

Usage::

    python -m inspect_eval.export_ladder_predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from inspect_eval.realistic_sidecar import load_realistic_meta


DEFAULT_ROOT = Path("data/inspect_eval/adjustment_small")

# Display name in the benchmark -> standardized sidecar field.
LADDER_OUTPUTS = {
    "Q-X-naive": "Q_naive_X_std",
    "Q-W-adj": "Q_adj_W_std",
    "Q-Z-adj": "Q_adj_Z_std",
    "Q-WZ-adj": "Q_adj_WZ_std",
    "Q-true": "Q_true_std",
}


def _stack(records: list[dict], field: str) -> np.ndarray:
    values = [record[field].detach().cpu().reshape(-1).float() for record in records]
    widths = {int(value.numel()) for value in values}
    if len(widths) != 1:
        raise ValueError(f"{field} has inconsistent query lengths: {sorted(widths)}")
    predictions = torch.stack(values).numpy().astype(np.float32, copy=False)
    if not np.isfinite(predictions).all():
        raise ValueError(f"{field} contains non-finite values")
    return predictions


def export_ladder_predictions(
    root: Path,
    *,
    output_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Write one benchmark prediction cache for every inspection target."""
    root = Path(root)
    output_dir = (
        Path(output_dir) if output_dir is not None else root / "bench" / "preds"
    )
    meta = load_realistic_meta(root)
    if meta["global"].get("set_kind") != "inspect":
        raise ValueError(f"{root} is not an inspect_eval set")

    records = meta["episodes"]
    if not records:
        raise ValueError(f"{root / 'frozen_meta.pt'} contains no episodes")

    expected_shape = (len(records), int(meta["global"]["n_q"]))
    benchmark_data = root / "bench" / "data.pt"
    if benchmark_data.exists():
        data = torch.load(benchmark_data, map_location="cpu", weights_only=False)
        benchmark_shape = (int(data["N_BATCH"]), int(data["N_TEST"]))
        if benchmark_shape != expected_shape:
            raise ValueError(
                f"sidecar shape {expected_shape} disagrees with benchmark data "
                f"shape {benchmark_shape}; rebuild the benchmark data cache"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        method_name: output_dir / f"{method_name}_preds.npz"
        for method_name in LADDER_OUTPUTS
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"{names} already exist; pass --force to replace them")

    paths: dict[str, Path] = {}
    for method_name, field in LADDER_OUTPUTS.items():
        path = destinations[method_name]
        predictions = _stack(records, field)
        if predictions.shape != expected_shape:
            raise ValueError(
                f"{field} has shape {predictions.shape}, expected {expected_shape}"
            )
        # These are population-limit reference calculations rather than fitted
        # estimators. Zero records that no benchmark fit/predict time was spent.
        times = np.zeros(expected_shape[0], dtype=np.float32)
        np.savez(path, preds=predictions, times=times)
        paths[method_name] = path
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export inspect_eval adjustment targets as benchmark caches."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to <root>/bench/preds.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = export_ladder_predictions(
        args.root, output_dir=args.output_dir, force=args.force
    )
    for method_name, path in paths.items():
        print(f"{method_name:<12} -> {path}")


if __name__ == "__main__":
    main()
