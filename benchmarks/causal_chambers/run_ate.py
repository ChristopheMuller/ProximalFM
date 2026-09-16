"""Run the growing-context Causal Chambers ATE benchmark.

Each CSV is a physical operating condition.  Estimators fit on every factual
row in a nested context prefix, predict a contrast at every same-prefix
covariate value, and the runner averages those contrasts.  The paired
Y1 - Y0 measurements are strictly evaluation data and are never supplied to
``fit``.  A shared seeded row permutation makes the prefixes identical across
methods; several permutations prevent one arbitrary acquisition order from
determining the learning curve.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.compute import (
    apply_compute,
    parse_global_compute,
    resolve_method_compute,
)
from benchmarks.methods import build_method

REPO_ROOT = Path(__file__).resolve().parents[2]


def _columns(frame: pd.DataFrame, configured, role: str) -> np.ndarray:
    names = [configured] if isinstance(configured, str) else list(configured)
    missing = [name for name in names if name not in frame]
    if missing:
        raise ValueError(f"{role}: missing column(s) {missing}")
    return frame[names].to_numpy(dtype=np.float64)


def _binary_treatment(values: np.ndarray, name: str) -> np.ndarray:
    levels = np.unique(values)
    if len(levels) != 2:
        raise ValueError(
            f"{name}: expected exactly two treatment values, found {levels.tolist()}"
        )
    return (values == levels[1]).astype(np.float64)


def _read_episode(path: Path, columns: dict) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    if len(frame.columns) and str(frame.columns[0]).startswith("Unnamed:"):
        frame = frame.iloc[:, 1:]

    required = ("X", "W", "Z", "A", "Y", "Y0", "Y1")
    missing = [key for key in required if key not in columns]
    if missing:
        raise ValueError(f"config data.columns missing {missing}")

    a = _binary_treatment(frame[columns["A"]].to_numpy(), columns["A"])
    return {
        "X": _columns(frame, columns["X"], "X"),
        "W": _columns(frame, columns["W"], "W"),
        "Z": _columns(frame, columns["Z"], "Z"),
        "A": a,
        "Y": frame[columns["Y"]].to_numpy(dtype=np.float64),
        # These remain out of the fitted-data dictionary on purpose.
        "tau_ref": frame[columns["Y1"]].to_numpy(dtype=np.float64)
        - frame[columns["Y0"]].to_numpy(dtype=np.float64),
        "n": len(frame),
    }


def _resolve_path(path: str | Path) -> Path:
    """Resolve config paths relative to the repository, not the caller's cwd."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _context_sizes(configured: list[int], n_rows: int) -> list[int]:
    sizes = [int(size) for size in configured]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError(f"context sizes must be positive and nonempty, got {sizes}")
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"context sizes must be unique, got {sizes}")
    too_large = [size for size in sizes if size > n_rows]
    if too_large:
        raise ValueError(
            f"context sizes {too_large} exceed the {n_rows} available rows"
        )
    return sorted(sizes)


def _permutation(seed: int, episode_index: int, repeat: int, n_rows: int) -> np.ndarray:
    """One reproducible row order, independent of methods and potential outcomes."""
    return np.random.default_rng(
        np.random.SeedSequence([seed, episode_index, repeat])
    ).permutation(n_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config_ate.yaml"),
        help="ATE benchmark YAML configuration.",
    )
    parser.add_argument("--output-dir", type=Path, help="Override config output_dir.")
    parser.add_argument(
        "--context-sizes",
        type=int,
        nargs="+",
        metavar="N_CTX",
        help="Override ate.context_sizes from the config.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        help="Override ate.n_permutations from the config.",
    )
    args = parser.parse_args()

    with args.config.open() as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict) or "run" not in cfg or "data" not in cfg:
        raise ValueError("config must contain run: and data: mappings")

    data_cfg = cfg["data"]
    columns = data_cfg.get("columns")
    if not isinstance(columns, dict):
        raise ValueError("data.columns must be a mapping")
    paths = sorted(
        _resolve_path(data_cfg["csv_dir"]).glob(data_cfg.get("csv_pattern", "*.csv"))
    )
    if not paths:
        raise FileNotFoundError("no Causal Chambers CSVs matched data.csv_dir/pattern")

    output_dir = args.output_dir or _resolve_path(cfg.get("output_dir", "ate_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    compute = parse_global_compute(cfg["run"])
    ate_cfg = cfg.get("ate") or {}
    n_permutations = int(
        args.n_permutations
        if args.n_permutations is not None
        else ate_cfg.get("n_permutations", 1)
    )
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    configured_sizes = args.context_sizes or ate_cfg.get("context_sizes")
    if configured_sizes is None:
        raise ValueError("set ate.context_sizes or pass --context-sizes")
    episodes = [_read_episode(path, columns) for path in paths]
    context_sizes = _context_sizes(configured_sizes, episodes[0]["n"])
    unequal = [
        path
        for path, episode in zip(paths, episodes)
        if episode["n"] != episodes[0]["n"]
    ]
    if unequal:
        raise ValueError(f"all CSVs need the same row count; differs at {unequal}")

    print(
        f"Running {len(paths)} conditions × {n_permutations} permutations × "
        f"context sizes {context_sizes}"
    )
    rows: list[dict[str, object]] = []

    methods = cfg.get("methods", [])
    total_fits = len(methods) * len(paths) * n_permutations * len(context_sizes)
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("·"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Fitting ATE estimators", total=total_fits)
        for method_cfg in methods:
            method_compute = resolve_method_compute(compute, method_cfg)
            apply_compute(method_compute)
            for episode_index, (path, episode) in enumerate(zip(paths, episodes)):
                for repeat in range(n_permutations):
                    order = _permutation(
                        int(cfg["run"].get("seed", 0)),
                        episode_index,
                        repeat,
                        episode["n"],
                    )
                    for n_ctx in context_sizes:
                        progress.update(
                            task,
                            description=(
                                f"{method_cfg['name']} · {path.stem} · "
                                f"permutation {repeat + 1}/{n_permutations} · n={n_ctx}"
                            ),
                        )
                        context = order[:n_ctx]
                        # The method gets precisely the factual prefix; the paired
                        # intervention contrasts below remain evaluation-only.
                        _, method = build_method(method_cfg, method_compute)
                        method.fit(
                            episode["X"][context],
                            episode["A"][context],
                            episode["Y"][context],
                            W=episode["W"][context],
                            Z=episode["Z"][context],
                        )
                        predict_ate = getattr(method, "predict_ate", None)
                        if not callable(predict_ate):
                            raise TypeError(
                                f"{method_cfg['name']!r} must implement predict_ate() "
                                "for the Causal Chambers ATE protocol."
                            )
                        ate_hat = float(predict_ate())
                        ate_ref = float(episode["tau_ref"][context].mean())
                        rows.append(
                            {
                                "condition": path.stem,
                                "permutation": repeat,
                                "n_ctx": n_ctx,
                                "method": method_cfg["name"],
                                "ate_hat": ate_hat,
                                "ate_ref": ate_ref,
                                "error": ate_hat - ate_ref,
                                "abs_error": abs(ate_hat - ate_ref),
                            }
                        )
                        progress.advance(task)

    results = pd.DataFrame(rows)
    destination = output_dir / "results.csv"
    results.to_csv(destination, index=False)
    summary = (
        results.groupby(["method", "n_ctx"], as_index=False)
        .agg(
            mean_error=("error", "mean"),
            mae=("abs_error", "mean"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        )
        .sort_values(["method", "n_ctx"])
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {destination}")


if __name__ == "__main__":
    main()
