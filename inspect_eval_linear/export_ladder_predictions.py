"""Export the linear adjustment ladder as benchmark prediction caches."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .sidecar import load_linear_meta

DEFAULT_ROOT = Path("data/inspect_eval_linear/proxy_sweep")
LADDER_OUTPUTS = {
    "Q-X-naive": "Q_naive_X_std",
    "Q-W-adj": "Q_adj_W_std",
    "Q-Z-adj": "Q_adj_Z_std",
    "Q-WZ-adj": "Q_adj_WZ_std",
    "Q-true": "Q_true_std",
}


def export_ladder_predictions(
    root: Path, *, output_dir: Path | None = None, force: bool = False
) -> dict[str, Path]:
    root = Path(root)
    meta = load_linear_meta(root)
    if meta["global"].get("set_kind") != "inspect_linear":
        raise ValueError(f"{root} is not an inspect_eval_linear set")
    records = meta["episodes"]
    if not records:
        raise ValueError("sidecar contains no episodes")
    expected = (len(records), int(meta["global"]["n_q"]))
    output_dir = Path(output_dir) if output_dir else root / "bench" / "preds"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for method, field in LADDER_OUTPUTS.items():
        path = output_dir / f"{method}_preds.npz"
        if path.exists() and not force:
            raise FileExistsError(f"{path} exists; pass --force to replace it")
        predictions = torch.stack(
            [record[field].reshape(-1).float() for record in records]
        ).numpy()
        if predictions.shape != expected or not np.isfinite(predictions).all():
            raise ValueError(f"{field} has invalid shape or non-finite values")
        np.savez(
            path,
            preds=predictions.astype(np.float32, copy=False),
            times=np.zeros(expected[0], dtype=np.float32),
        )
        paths[method] = path
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    for name, path in export_ladder_predictions(
        args.root, output_dir=args.output_dir, force=args.force
    ).items():
        print(f"{name:<12} -> {path}")


if __name__ == "__main__":
    main()
