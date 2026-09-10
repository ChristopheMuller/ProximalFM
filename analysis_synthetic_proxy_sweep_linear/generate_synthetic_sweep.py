"""Generate a paired synthetic proximal study with exactly controlled population R².

Open this file in PyCharm and press Run. Defaults create 300 datasets beside
this script, each with 50,000 rows. NumPy is the only required package.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np


# Change these defaults here, or supply the corresponding command-line options.
N_ROWS = 5_000
N_LEVELS = 30
DIMENSIONS = (1, 3, 5)
SEED = 20260908
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "data"
VERSION = "1.0.0"


@dataclass(frozen=True)
class StudyConfig:
    n_rows: int = N_ROWS
    n_levels: int = N_LEVELS
    dimensions: tuple[int, ...] = DIMENSIONS
    seed: int = SEED
    r2_min: float = 0.01
    r2_max: float = 0.99
    train_fraction: float = 0.8
    observed_treatment_coefficient: float = 0.5
    latent_treatment_coefficient: float = 1.25
    observed_baseline_coefficient: float = 0.5
    latent_baseline_coefficient: float = 1.5
    cate_intercept: float = 0.5
    cate_coefficient: float = 1.0
    latent_effect_coefficient: float = 0.5
    outcome_noise_sd: float = 1.0
    propensity_floor: float = 0.02

    def validate(self):
        if self.n_rows < 200 or self.n_levels < 2:
            raise ValueError("Use at least 200 rows and two R² levels.")
        if not self.dimensions or any(d < 1 for d in self.dimensions):
            raise ValueError("Dimensions must be positive integers.")
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("Dimensions must be distinct.")
        if self.seed < 0 or not 0 < self.r2_min < self.r2_max < 1:
            raise ValueError("Use a nonnegative seed and 0 < r2_min < r2_max < 1.")
        if not 0 < self.train_fraction < 1 or not 0 < self.propensity_floor < 0.5:
            raise ValueError("Invalid train fraction or propensity floor.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(f"{value.shape}:{value.dtype.str}:".encode())
    h.update(value.tobytes())
    return h.hexdigest()


def matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Evaluate small dense products consistently across CPU libraries."""
    return np.einsum("ni,ij->nj", left, right, optimize=False)


def write_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz(path: Path, arrays: dict[str, np.ndarray]):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def normal_columns(seed: int, tag: int, n: int, d: int) -> np.ndarray:
    # Per-column streams pair shared coordinates across dimensions and keep
    # selected-source runs consistent with the complete sweep.
    return np.column_stack(
        [
            np.random.default_rng(
                np.random.SeedSequence([seed, tag, j])
            ).standard_normal(n)
            for j in range(d)
        ]
    )


def orthogonal_matrix(seed: int, tag: int, d: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, tag, d]))
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.where(np.diag(r) >= 0, 1.0, -1.0)


def make_base(config: StudyConfig, dimension: int) -> dict:
    """Draw one cohort and fixed mechanisms, shared across the whole R² grid."""
    n, d = config.n_rows, dimension
    x = normal_columns(config.seed, 10, n, d)
    u = normal_columns(config.seed, 20, n, d)
    eps_w = normal_columns(config.seed, 30, n, d)
    eps_z = normal_columns(config.seed, 40, n, d)
    eps_y = normal_columns(config.seed, 50, n, 1)[:, 0]
    uniform_a = np.random.default_rng(np.random.SeedSequence([config.seed, 60])).random(
        n
    )
    sx, latent = x.sum(axis=1) / np.sqrt(d), u.sum(axis=1) / np.sqrt(d)
    logit = (
        config.observed_treatment_coefficient * sx
        + config.latent_treatment_coefficient * latent
    )
    propensity = np.clip(
        1 / (1 + np.exp(-np.clip(logit, -40, 40))),
        config.propensity_floor,
        1 - config.propensity_floor,
    )
    treatment = (uniform_a < propensity).astype(np.uint8)
    y0 = (
        config.observed_baseline_coefficient * sx
        + config.latent_baseline_coefficient * latent
        + config.outcome_noise_sd * eps_y
    )
    cate = config.cate_intercept + config.cate_coefficient * sx
    ite = cate + config.latent_effect_coefficient * latent
    y1 = y0 + ite
    outcome = np.where(treatment == 1, y1, y0)
    n_train = int(config.train_fraction * n)
    split = np.zeros(n, dtype=np.uint8)
    split[n_train:] = 1
    observed = {
        "row_id": np.arange(n, dtype=np.int32),
        "split": split,
        "X": x,
        "A": treatment,
        "Y": outcome,
    }
    truth = {
        "row_id": observed["row_id"],
        "U": u,
        "Y0": y0,
        "Y1": y1,
        "CATE": cate,
        "ITE": ite,
        "propensity": propensity,
    }
    hashes = {f"observed.{k}": array_hash(v) for k, v in observed.items()}
    hashes.update({f"truth.{k}": array_hash(v) for k, v in truth.items()})
    return {
        "observed": observed,
        "truth": truth,
        "eps_w": eps_w,
        "eps_z": eps_z,
        "Q_W": orthogonal_matrix(config.seed, 70, d),
        "Q_Z": orthogonal_matrix(config.seed, 80, d),
        "latent_score": latent,
        "observed_score": sx,
        "n_train": n_train,
        "paired_array_hashes": hashes,
    }


def add_proxies(base: dict, joint_r2: float) -> tuple[dict, dict, float]:
    """Make each channel's R² q=s/(2-s), giving joint R² exactly s."""
    if not 0 < joint_r2 < 1:
        raise ValueError("Joint R² must be strictly between zero and one.")
    q = joint_r2 / (2 - joint_r2)
    u = base["truth"]["U"]
    w = np.sqrt(q) * matmul(u, base["Q_W"].T) + np.sqrt(1 - q) * base["eps_w"]
    z = np.sqrt(q) * matmul(u, base["Q_Z"].T) + np.sqrt(1 - q) * base["eps_z"]
    observed = {**base["observed"], "W": w, "Z": z}
    return observed, base["truth"], q


def regression_r2(
    features: np.ndarray, target: np.ndarray, n_train: int
) -> list[float]:
    """Fit OLS on train rows and report one held-out R² per latent coordinate."""
    x_mean = features[:n_train].mean(axis=0)
    y_mean = target[:n_train].mean(axis=0)
    centered = features[:n_train] - x_mean
    beta = np.linalg.lstsq(centered, target[:n_train] - y_mean, rcond=None)[0]
    prediction = matmul(features[n_train:] - x_mean, beta) + y_mean
    heldout = target[n_train:]
    denominator = ((heldout - heldout.mean(axis=0)) ** 2).sum(axis=0)
    return (1 - ((heldout - prediction) ** 2).sum(axis=0) / denominator).tolist()


def check_arrays(
    config: StudyConfig, base: dict, observed: dict, truth: dict, q: float
) -> dict:
    x, w, z, u = (observed["X"], observed["W"], observed["Z"], truth["U"])
    n, d = x.shape
    for group in (observed, truth):
        for key, value in group.items():
            if value.shape[0] != n or not np.isfinite(value).all():
                raise AssertionError(f"Invalid array: {key}")
    np.testing.assert_allclose(truth["Y1"] - truth["Y0"], truth["ITE"], atol=1e-12)
    np.testing.assert_allclose(
        observed["Y"], np.where(observed["A"], truth["Y1"], truth["Y0"]), atol=1e-12
    )
    np.testing.assert_allclose(
        truth["CATE"],
        config.cate_intercept + config.cate_coefficient * x.sum(1) / np.sqrt(d),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        truth["ITE"] - truth["CATE"],
        config.latent_effect_coefficient * u.sum(1) / np.sqrt(d),
        atol=1e-12,
    )
    unbiased_latent = matmul(w, base["Q_W"]).sum(1) / np.sqrt(q * d)
    a = observed["A"].astype(float)
    bridge = (
        config.observed_baseline_coefficient * base["observed_score"]
        + config.latent_baseline_coefficient * unbiased_latent
        + a * (truth["CATE"] + config.latent_effect_coefficient * unbiased_latent)
    )
    eps_y = (
        truth["Y0"]
        - config.observed_baseline_coefficient * base["observed_score"]
        - config.latent_baseline_coefficient * base["latent_score"]
    )
    expected_residual = eps_y - (
        config.latent_baseline_coefficient + a * config.latent_effect_coefficient
    ) * (unbiased_latent - base["latent_score"])
    error = float(np.max(np.abs(observed["Y"] - bridge - expected_residual)))
    if error > 1e-10:
        raise AssertionError("Outcome bridge residual identity failed.")
    n_train = base["n_train"]
    r_w = regression_r2(w, u, n_train)
    r_z = regression_r2(z, u, n_train)
    r_joint = regression_r2(np.column_stack([w, z]), u, n_train)
    # The known conditional mean independently characterizes joint information.
    conditional_mean = (
        np.sqrt(q) / (1 + q) * (matmul(w, base["Q_W"]) + matmul(z, base["Q_Z"]))
    )
    test_u = u[n_train:]
    denominator = ((test_u - test_u.mean(0)) ** 2).sum(0)
    oracle_r2 = 1 - ((test_u - conditional_mean[n_train:]) ** 2).sum(0) / denominator
    return {
        "r2_w_ols_per_coordinate": r_w,
        "r2_z_ols_per_coordinate": r_z,
        "r2_joint_ols_per_coordinate": r_joint,
        "r2_w_ols_mean": float(np.mean(r_w)),
        "r2_z_ols_mean": float(np.mean(r_z)),
        "r2_joint_ols_mean": float(np.mean(r_joint)),
        "r2_joint_known_predictor_mean": float(np.mean(oracle_r2)),
        "treatment_fraction": float(a.mean()),
        "overlap_05_95": float(
            np.mean((truth["propensity"] > 0.05) & (truth["propensity"] < 0.95))
        ),
        "cate_mean": float(truth["CATE"].mean()),
        "cate_sd": float(truth["CATE"].std()),
        "ite_minus_cate_sd": float((truth["ITE"] - truth["CATE"]).std()),
        "bridge_residual_identity_max_error": error,
    }


def payload_size(config: StudyConfig) -> int:
    # Observed: 3*d float64 + Y + int32 row ID + uint8 split/A.
    # Truth: d float64 + five scalar float64 arrays + int32 row ID.
    return config.n_rows * config.n_levels * sum(32 * d + 58 for d in config.dimensions)


def generate_sweep(output: Path, config: StudyConfig) -> dict:
    config.validate()
    output = Path(output).expanduser().resolve()
    recipe = {
        "version": VERSION,
        **asdict(config),
        "storage_dtype": "float64",
        "generator_sha256": sha256(Path(__file__)),
    }
    recipe = json.loads(json.dumps(recipe))
    marker = output / "run_config.json"
    if output.exists() and not output.is_dir():
        raise ValueError("The output path must be a directory.")
    if marker.exists():
        if json.loads(marker.read_text()) != recipe:
            raise ValueError(
                "Choose a fresh output directory for a different recipe or seed."
            )
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory for the initial generation.")
    output.mkdir(parents=True, exist_ok=True)
    expected_bytes = payload_size(config)
    if shutil.disk_usage(output).free < expected_bytes * 1.2 + 100_000_000:
        raise OSError("Available disk space is below the estimated output requirement.")
    write_json(marker, recipe)
    write_json(output / "manifest.json", {"status": "building", "recipe": recipe})
    grid = np.linspace(config.r2_min, config.r2_max, config.n_levels)
    total = len(config.dimensions) * config.n_levels
    print(f"Creating {total} datasets x {config.n_rows:,} rows.", flush=True)
    print(
        f"Uncompressed numeric payload: {expected_bytes / 1e9:.3f} GB; compressed files will be measured.",
        flush=True,
    )
    entries = []
    began = time.monotonic()
    for dimension in config.dimensions:
        base = make_base(config, dimension)
        for level_id, target in enumerate(grid):
            observed, truth, q = add_proxies(base, float(target))
            diagnostics = check_arrays(config, base, observed, truth, q)
            dataset_id = f"d{dimension:02d}_level{level_id:03d}"
            relative = (
                Path(f"dim_{dimension:02d}") / f"level_{level_id:03d}_r2_{target:.6f}"
            )
            directory = output / relative
            directory.mkdir(parents=True, exist_ok=True)
            obs_path, truth_path = directory / "observed.npz", directory / "truth.npz"
            write_npz(obs_path, observed)
            write_npz(truth_path, truth)
            entry = {
                "dataset_id": dataset_id,
                "dimension": dimension,
                "level_id": level_id,
                "joint_r2_target": float(target),
                "single_proxy_r2": q,
                "n_rows": config.n_rows,
                "n_train": base["n_train"],
                "n_test": config.n_rows - base["n_train"],
                "seed": config.seed,
                "observed_path": (relative / "observed.npz").as_posix(),
                "truth_path": (relative / "truth.npz").as_posix(),
                "metadata_path": (relative / "metadata.json").as_posix(),
                "observed_sha256": sha256(obs_path),
                "truth_sha256": sha256(truth_path),
                "observed_bytes": obs_path.stat().st_size,
                "truth_bytes": truth_path.stat().st_size,
                **{
                    key: value
                    for key, value in diagnostics.items()
                    if not isinstance(value, list)
                },
            }
            metadata = {
                **entry,
                "recipe": recipe,
                "Q_W": base["Q_W"].tolist(),
                "Q_Z": base["Q_Z"].tolist(),
                "paired_array_hashes": base["paired_array_hashes"],
                "diagnostics": diagnostics,
                "split_codes": {"train": 0, "test": 1},
                "population_r2_definition": "Per-coordinate R2 of U predicted from W and Z jointly; uniform average over U coordinates.",
                "population_posterior_variance": float(1 - target),
                "d_X": dimension,
                "d_U": dimension,
                "d_W": dimension,
                "d_Z": dimension,
                "observed_fields": list(observed),
                "truth_fields": list(truth),
                "array_payload_bytes": sum(v.nbytes for v in observed.values())
                + sum(v.nbytes for v in truth.values()),
            }
            write_json(directory / "metadata.json", metadata)
            entries.append(entry)
            if (
                (level_id + 1) % 10 == 0
                or level_id == 0
                or level_id + 1 == config.n_levels
            ):
                print(
                    f"[{len(entries):3d}/{total}] d={dimension}, joint R²={target:.4f}, test R²={diagnostics['r2_joint_ols_mean']:.4f}",
                    flush=True,
                )
    temporary = output / "index.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    temporary.replace(output / "index.csv")
    compressed = sum(e["observed_bytes"] + e["truth_bytes"] for e in entries)
    manifest = {
        "status": "complete",
        "recipe": recipe,
        "n_datasets": len(entries),
        "n_rows_per_dataset": config.n_rows,
        "total_rows": config.n_rows * len(entries),
        "dimensions": list(config.dimensions),
        "r2_grid": grid.tolist(),
        "index_sha256": sha256(output / "index.csv"),
        "uncompressed_array_bytes": expected_bytes,
        "compressed_array_bytes": compressed,
        "generation_seconds": time.monotonic() - began,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "independent_grid_replicates": 1,
        "pairing": "Common Gaussian roots and noises across proxy levels; common per-coordinate streams across dimensions.",
        "effect_target": (
            f"CATE(X)={config.cate_intercept}+{config.cate_coefficient}*sum(X)/sqrt(d); "
            f"ITE=CATE+{config.latent_effect_coefficient}*sum(U)/sqrt(d)."
        ),
    }
    write_json(output / "manifest.json", manifest)
    lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS.txt", "verification.json"}:
            lines.append(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n")
    (output / "SHA256SUMS.txt").write_text("".join(lines), encoding="utf-8")
    print(
        f"Completed: {len(entries)} datasets; arrays occupy {compressed / 1e9:.3f} GB.",
        flush=True,
    )
    print(f"Saved to {output}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--rows", type=int, default=N_ROWS)
    parser.add_argument("--levels", type=int, default=N_LEVELS)
    parser.add_argument("--dimensions", nargs="+", type=int, default=list(DIMENSIONS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    config = StudyConfig(
        n_rows=args.rows,
        n_levels=args.levels,
        dimensions=tuple(args.dimensions),
        seed=args.seed,
    )
    config.validate()
    if args.estimate_only:
        print(
            json.dumps(
                {
                    "datasets": config.n_levels * len(config.dimensions),
                    "total_rows": config.n_levels
                    * len(config.dimensions)
                    * config.n_rows,
                    "uncompressed_array_bytes": payload_size(config),
                    "uncompressed_GB": payload_size(config) / 1e9,
                },
                indent=2,
            )
        )
    else:
        generate_sweep(args.output, config)


if __name__ == "__main__":
    main()
