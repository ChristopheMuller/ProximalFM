#!/usr/bin/env python3
"""Generate a self-contained nonlinear proximal proxy-strength sweep.

The generator draws its own Gaussian roots, treatment, outcomes, truth, and
proxy grid. Only the variables exposed to estimators (X, W, and Z) are mapped
through an invertible standardized cubic, so proxy information is controlled
exactly while the observed-scale regression problem is nonlinear.
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
import time

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "data"
N_ROWS = 12_000
N_LEVELS = 20
DIMENSIONS = (1, 3, 5)
SEED = 20260908
VERSION = "2.0.0"

TRANSFORM_ALPHA = 0.25
TRANSFORM_VARIANCE = 1.0 + 6.0 * TRANSFORM_ALPHA + 15.0 * TRANSFORM_ALPHA**2
TRANSFORM_SCALE = float(np.sqrt(TRANSFORM_VARIANCE))
TRANSFORM_FORMULA = "g(t)=(t+0.25*t^3)/sqrt(3.4375)"


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

    def validate(self) -> None:
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


def nonlinear_transform(value: np.ndarray) -> np.ndarray:
    """Apply the monotone population-standardized cubic transform."""
    value = np.asarray(value, dtype=np.float64)
    return (value + TRANSFORM_ALPHA * value**3) / TRANSFORM_SCALE


def inverse_nonlinear_transform(value: np.ndarray) -> np.ndarray:
    """Return the analytic real inverse of :func:`nonlinear_transform`."""
    value = np.asarray(value, dtype=np.float64)
    scaled = TRANSFORM_SCALE * value
    discriminant = 4.0 * scaled**2 + 64.0 / 27.0
    root = np.sqrt(discriminant)
    return np.cbrt(2.0 * scaled + root) + np.cbrt(2.0 * scaled - root)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{value.shape}:{value.dtype.str}:".encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Evaluate small dense products consistently across CPU libraries."""
    return np.einsum("ni,ij->nj", left, right, optimize=False)


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def normal_columns(seed: int, tag: int, n: int, d: int) -> np.ndarray:
    # Per-column streams pair shared coordinates across dimensions and keep
    # filtered dimension runs consistent with a complete sweep.
    return np.column_stack(
        [
            np.random.default_rng(
                np.random.SeedSequence([seed, tag, coordinate])
            ).standard_normal(n)
            for coordinate in range(d)
        ]
    )


def orthogonal_matrix(seed: int, tag: int, d: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, tag, d]))
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.where(np.diag(r) >= 0, 1.0, -1.0)


def make_base(config: StudyConfig, dimension: int) -> dict:
    """Draw one cohort and fixed mechanisms shared by every strength level."""
    n, d = config.n_rows, dimension
    x_raw = normal_columns(config.seed, 10, n, d)
    u = normal_columns(config.seed, 20, n, d)
    eps_w = normal_columns(config.seed, 30, n, d)
    eps_z = normal_columns(config.seed, 40, n, d)
    eps_y = normal_columns(config.seed, 50, n, 1)[:, 0]
    uniform_a = np.random.default_rng(np.random.SeedSequence([config.seed, 60])).random(
        n
    )

    observed_score = x_raw.sum(axis=1) / np.sqrt(d)
    latent_score = u.sum(axis=1) / np.sqrt(d)
    logit = (
        config.observed_treatment_coefficient * observed_score
        + config.latent_treatment_coefficient * latent_score
    )
    propensity = np.clip(
        1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))),
        config.propensity_floor,
        1.0 - config.propensity_floor,
    )
    treatment = (uniform_a < propensity).astype(np.uint8)
    y0 = (
        config.observed_baseline_coefficient * observed_score
        + config.latent_baseline_coefficient * latent_score
        + config.outcome_noise_sd * eps_y
    )
    cate = config.cate_intercept + config.cate_coefficient * observed_score
    ite = cate + config.latent_effect_coefficient * latent_score
    y1 = y0 + ite
    outcome = np.where(treatment == 1, y1, y0)

    n_train = int(config.train_fraction * n)
    split = np.zeros(n, dtype=np.uint8)
    split[n_train:] = 1
    observed = {
        "row_id": np.arange(n, dtype=np.int32),
        "split": split,
        "X": nonlinear_transform(x_raw),
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
    hashes = {f"observed.{key}": array_hash(value) for key, value in observed.items()}
    hashes.update({f"truth.{key}": array_hash(value) for key, value in truth.items()})
    return {
        "observed": observed,
        "truth": truth,
        "x_raw": x_raw,
        "eps_w": eps_w,
        "eps_z": eps_z,
        "Q_W": orthogonal_matrix(config.seed, 70, d),
        "Q_Z": orthogonal_matrix(config.seed, 80, d),
        "latent_score": latent_score,
        "observed_score": observed_score,
        "n_train": n_train,
        "paired_array_hashes": hashes,
    }


def add_proxies(base: dict, joint_r2: float) -> tuple[dict, dict, float, dict]:
    """Add transformed proxies whose joint nonlinear-prediction R² is exact."""
    if not 0 < joint_r2 < 1:
        raise ValueError("Joint R² must be strictly between zero and one.")
    q = joint_r2 / (2.0 - joint_r2)
    u = base["truth"]["U"]
    w_raw = np.sqrt(q) * matmul(u, base["Q_W"].T) + np.sqrt(1 - q) * base["eps_w"]
    z_raw = np.sqrt(q) * matmul(u, base["Q_Z"].T) + np.sqrt(1 - q) * base["eps_z"]
    observed = {
        **base["observed"],
        "W": nonlinear_transform(w_raw),
        "Z": nonlinear_transform(z_raw),
    }
    return observed, base["truth"], q, {"W": w_raw, "Z": z_raw}


def regression_r2(
    features: np.ndarray, target: np.ndarray, n_train: int
) -> list[float]:
    """Fit OLS on context rows and report held-out R² by latent coordinate."""
    x_mean = features[:n_train].mean(axis=0)
    y_mean = target[:n_train].mean(axis=0)
    beta = np.linalg.lstsq(
        features[:n_train] - x_mean, target[:n_train] - y_mean, rcond=None
    )[0]
    prediction = matmul(features[n_train:] - x_mean, beta) + y_mean
    heldout = target[n_train:]
    denominator = np.sum((heldout - heldout.mean(axis=0)) ** 2, axis=0)
    return (1.0 - np.sum((heldout - prediction) ** 2, axis=0) / denominator).tolist()


def check_arrays(
    config: StudyConfig,
    base: dict,
    observed: dict,
    truth: dict,
    q: float,
    raw_proxies: dict,
) -> dict:
    x = observed["X"]
    w = observed["W"]
    z = observed["Z"]
    u = truth["U"]
    n, d = x.shape
    for group in (observed, truth):
        for key, value in group.items():
            if value.shape[0] != n or not np.isfinite(value).all():
                raise AssertionError(f"Invalid array: {key}")
    np.testing.assert_allclose(truth["Y1"] - truth["Y0"], truth["ITE"], atol=1e-12)
    np.testing.assert_allclose(
        observed["Y"], np.where(observed["A"], truth["Y1"], truth["Y0"]), atol=1e-12
    )

    reconstructed = {
        "X": inverse_nonlinear_transform(x),
        "W": inverse_nonlinear_transform(w),
        "Z": inverse_nonlinear_transform(z),
    }
    inverse_error = max(
        float(np.max(np.abs(reconstructed["X"] - base["x_raw"]))),
        float(np.max(np.abs(reconstructed["W"] - raw_proxies["W"]))),
        float(np.max(np.abs(reconstructed["Z"] - raw_proxies["Z"]))),
    )
    if inverse_error > 1e-10:
        raise AssertionError("Nonlinear transform inverse roundtrip failed.")

    np.testing.assert_allclose(
        truth["CATE"],
        config.cate_intercept
        + config.cate_coefficient * reconstructed["X"].sum(axis=1) / np.sqrt(d),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        truth["ITE"] - truth["CATE"],
        config.latent_effect_coefficient * u.sum(axis=1) / np.sqrt(d),
        atol=1e-12,
    )

    w_raw, z_raw = reconstructed["W"], reconstructed["Z"]
    unbiased_latent = matmul(w_raw, base["Q_W"]).sum(axis=1) / np.sqrt(q * d)
    treatment = observed["A"].astype(np.float64)
    bridge = (
        config.observed_baseline_coefficient * base["observed_score"]
        + config.latent_baseline_coefficient * unbiased_latent
        + treatment
        * (truth["CATE"] + config.latent_effect_coefficient * unbiased_latent)
    )
    eps_y = (
        truth["Y0"]
        - config.observed_baseline_coefficient * base["observed_score"]
        - config.latent_baseline_coefficient * base["latent_score"]
    )
    expected_residual = eps_y - (
        config.latent_baseline_coefficient
        + treatment * config.latent_effect_coefficient
    ) * (unbiased_latent - base["latent_score"])
    bridge_error = float(np.max(np.abs(observed["Y"] - bridge - expected_residual)))
    if bridge_error > 1e-10:
        raise AssertionError("Nonlinear outcome bridge residual identity failed.")

    n_train = base["n_train"]
    conditional_mean = (
        np.sqrt(q)
        / (1.0 + q)
        * (matmul(w_raw, base["Q_W"]) + matmul(z_raw, base["Q_Z"]))
    )
    heldout = u[n_train:]
    denominator = np.sum((heldout - heldout.mean(axis=0)) ** 2, axis=0)
    known_r2 = (
        1.0 - np.sum((heldout - conditional_mean[n_train:]) ** 2, axis=0) / denominator
    )
    observed_joint_ols = regression_r2(np.column_stack([w, z]), u, n_train)
    observed_w_ols = regression_r2(w, u, n_train)
    observed_z_ols = regression_r2(z, u, n_train)
    return {
        "inverse_roundtrip_max_error": inverse_error,
        "nonlinear_bridge_residual_identity_max_error": bridge_error,
        "r2_joint_known_nonlinear_predictor_mean": float(np.mean(known_r2)),
        "r2_joint_known_nonlinear_predictor_per_coordinate": known_r2.tolist(),
        "r2_joint_observed_scale_ols_mean": float(np.mean(observed_joint_ols)),
        "r2_joint_observed_scale_ols_per_coordinate": observed_joint_ols,
        "r2_w_observed_scale_ols_mean": float(np.mean(observed_w_ols)),
        "r2_w_observed_scale_ols_per_coordinate": observed_w_ols,
        "r2_z_observed_scale_ols_mean": float(np.mean(observed_z_ols)),
        "r2_z_observed_scale_ols_per_coordinate": observed_z_ols,
        "treatment_fraction": float(treatment.mean()),
        "overlap_05_95": float(
            np.mean((truth["propensity"] > 0.05) & (truth["propensity"] < 0.95))
        ),
        "cate_mean": float(truth["CATE"].mean()),
        "cate_sd": float(truth["CATE"].std()),
        "ite_minus_cate_sd": float((truth["ITE"] - truth["CATE"]).std()),
    }


def payload_size(config: StudyConfig) -> int:
    # Observed: 3*d float64 + Y + int32 row ID + uint8 split/A.
    # Truth: d float64 + five scalar float64 arrays + int32 row ID.
    return (
        config.n_rows
        * config.n_levels
        * sum(32 * dimension + 58 for dimension in config.dimensions)
    )


def generate_sweep(output: Path, config: StudyConfig) -> dict:
    """Generate all dimensions and strength levels into ``output``."""
    config.validate()
    output = Path(output).expanduser().resolve()
    recipe = {
        "version": VERSION,
        **asdict(config),
        "storage_dtype": "float64",
        "transform": {
            "formula": TRANSFORM_FORMULA,
            "alpha": TRANSFORM_ALPHA,
            "gaussian_population_mean": 0.0,
            "gaussian_population_variance": TRANSFORM_VARIANCE / TRANSFORM_SCALE**2,
            "strictly_increasing": True,
            "invertible": True,
        },
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
        "Uncompressed numeric payload: "
        f"{expected_bytes / 1e9:.3f} GB; compressed files will be measured.",
        flush=True,
    )
    entries = []
    began = time.monotonic()
    for dimension in config.dimensions:
        base = make_base(config, dimension)
        for level_id, target in enumerate(grid):
            observed, truth, q, raw_proxies = add_proxies(base, float(target))
            diagnostics = check_arrays(config, base, observed, truth, q, raw_proxies)
            dataset_id = f"d{dimension:02d}_level{level_id:03d}"
            relative = (
                Path(f"dim_{dimension:02d}") / f"level_{level_id:03d}_r2_{target:.6f}"
            )
            directory = output / relative
            directory.mkdir(parents=True, exist_ok=True)
            observed_path = directory / "observed.npz"
            truth_path = directory / "truth.npz"
            write_npz(observed_path, observed)
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
                "observed_sha256": sha256(observed_path),
                "truth_sha256": sha256(truth_path),
                "observed_bytes": observed_path.stat().st_size,
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
                "population_r2_definition": (
                    "Per-coordinate correlation-ratio R2 of U predicted from the "
                    "invertibly transformed W and Z jointly; uniform average over "
                    "U coordinates."
                ),
                "population_posterior_variance": float(1.0 - target),
                "observed_scale_ols_r2_is_controlled": False,
                "d_X": dimension,
                "d_U": dimension,
                "d_W": dimension,
                "d_Z": dimension,
                "observed_fields": list(observed),
                "truth_fields": list(truth),
                "array_payload_bytes": sum(value.nbytes for value in observed.values())
                + sum(value.nbytes for value in truth.values()),
            }
            write_json(directory / "metadata.json", metadata)
            entries.append(entry)
            if (
                (level_id + 1) % 10 == 0
                or level_id == 0
                or level_id + 1 == config.n_levels
            ):
                print(
                    f"[{len(entries):3d}/{total}] d={dimension}, "
                    f"joint R²={target:.4f}, nonlinear-predictor test "
                    f"R²={diagnostics['r2_joint_known_nonlinear_predictor_mean']:.4f}",
                    flush=True,
                )

    temporary = output / "index.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    temporary.replace(output / "index.csv")

    compressed = sum(
        entry["observed_bytes"] + entry["truth_bytes"] for entry in entries
    )
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
        "pairing": (
            "Common Gaussian roots and noises across proxy levels; common "
            "per-coordinate streams across dimensions."
        ),
        "effect_target": (
            f"CATE(X)={config.cate_intercept}+{config.cate_coefficient}*"
            "sum(g^-1(X))/sqrt(d); "
            f"ITE=CATE+{config.latent_effect_coefficient}*sum(U)/sqrt(d)."
        ),
    }
    write_json(output / "manifest.json", manifest)
    checksum_lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS.txt", "verification.json"}:
            checksum_lines.append(
                f"{sha256(path)}  {path.relative_to(output).as_posix()}\n"
            )
    (output / "SHA256SUMS.txt").write_text("".join(checksum_lines), encoding="utf-8")
    print(
        f"Completed: {len(entries)} datasets; arrays occupy {compressed / 1e9:.3f} GB.",
        flush=True,
    )
    print(f"Saved to {output}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=N_ROWS)
    parser.add_argument("--levels", type=int, default=N_LEVELS)
    parser.add_argument("--dimensions", nargs="+", type=int, default=list(DIMENSIONS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--r2-min", type=float, default=0.01)
    parser.add_argument("--r2-max", type=float, default=0.99)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    config = StudyConfig(
        n_rows=args.rows,
        n_levels=args.levels,
        dimensions=tuple(args.dimensions),
        seed=args.seed,
        r2_min=args.r2_min,
        r2_max=args.r2_max,
    )
    config.validate()
    if args.estimate_only:
        size = payload_size(config)
        print(
            json.dumps(
                {
                    "datasets": config.n_levels * len(config.dimensions),
                    "total_rows": config.n_levels
                    * len(config.dimensions)
                    * config.n_rows,
                    "uncompressed_array_bytes": size,
                    "uncompressed_GB": size / 1e9,
                },
                indent=2,
            )
        )
    else:
        generate_sweep(args.output, config)


if __name__ == "__main__":
    main()
