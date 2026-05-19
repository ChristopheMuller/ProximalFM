"""
generate_xu_kanagawa.py

Generates synthetic data from the Xu and Kanagawa DGP and formats it 
into a data.pt file compatible with bench_run.py.
"""

# uv run benchmarks/demand_synthetic_data.py --out-dir synthetic/demand --batches 10 --rows 1000

# in config:
# data:
#   source: load
#   output_dir: synthetic/demand

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# ── 1. DGP Definition ─────────────────────────────────────────────────────────


def g_(d):
    return 2 * (((d - 5) ** 4) / 600 + np.exp(-4 * (d - 5) ** 2) + (d / 10) - 2)


def generate_proximal_data(n_samples=1000):
    """
    Generates synthetic data for Proximal Causal Learning.
    Seed is managed globally in the parent loop.
    """
    D = np.random.uniform(0, 10, size=n_samples)

    C1 = 2 * np.sin(2 * D * np.pi / 10) + np.random.normal(0, 1, size=n_samples)
    C2 = 2 * np.cos(2 * D * np.pi / 10) + np.random.normal(0, 1, size=n_samples)

    V = 7 * g_(D) + 45 + np.random.normal(0, 1, size=n_samples)

    P = 35 + (C1 + 3) * g_(D) + C2 + np.random.normal(0, 1, size=n_samples)
    A = np.where(P >= 30, 1, 0)
    P0, P1 = 25, 35

    Y0 = P0 * np.minimum(np.exp((V - P0) / 10), 5) - 5 * g_(D)
    Y1 = P1 * np.minimum(np.exp((V - P1) / 10), 5) - 5 * g_(D)

    noise = np.random.normal(0, 1, size=n_samples)
    Y0 += noise
    Y1 += noise
    Y_obs = A * Y1 + (1 - A) * Y0

    return pd.DataFrame(
        {
            "D": D,
            "C1": C1,
            "C2": C2,
            "V": V,
            "A": A,
            "Y_obs": Y_obs,
            "Y0": Y0,
            "Y1": Y1,
            "ITE": Y1 - Y0,
        }
    )


# ── 2. Benchmark Formatter ───────────────────────────────────────────────────


def safe_standardize(x: torch.Tensor) -> torch.Tensor:
    """Matches the standardization logic from bench_run.py"""
    mu = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    std = torch.where(std < 1e-5, torch.ones_like(std), std)
    return (x - mu) / std


def build_benchmark_data(
    output_dir: Path, n_batch=10, n_rows=1000, prop_train=0.75, seed=42
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    N_TRAIN = int(n_rows * prop_train)
    N_TEST = n_rows - N_TRAIN

    U_list, Z_list, W_list, A_list, Y_obs_list, Y0_list, Y1_list = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    print(f"Generating {n_batch} batches of size {n_rows}...")
    for _ in range(n_batch):
        df = generate_proximal_data(n_samples=n_rows)

        # Extract and format features to (N, Dim)
        U_list.append(df[["D"]].values)
        Z_list.append(df[["C1", "C2"]].values)
        W_list.append(df[["V"]].values)
        A_list.append(df["A"].values)
        Y_obs_list.append(df["Y_obs"].values)
        Y0_list.append(df["Y0"].values)
        Y1_list.append(df["Y1"].values)

    # Stack into batched tensors (B, N, Dim) or (B, N)
    U = torch.tensor(np.stack(U_list), dtype=torch.float32)
    Z = torch.tensor(np.stack(Z_list), dtype=torch.float32)
    W = torch.tensor(np.stack(W_list), dtype=torch.float32)
    A = torch.tensor(np.stack(A_list), dtype=torch.float32)
    Y_obs = torch.tensor(np.stack(Y_obs_list), dtype=torch.float32)
    Y0 = torch.tensor(np.stack(Y0_list), dtype=torch.float32)
    Y1 = torch.tensor(np.stack(Y1_list), dtype=torch.float32)

    # This DGP lacks direct observed covariates X, so we provide an empty tensor
    # of dimension 0 to satisfy the pipeline's shape expectations without adding dummy data.
    X_std = torch.zeros((n_batch, n_rows, 0), dtype=torch.float32)

    # Standardize Covariates
    U_std = safe_standardize(U)
    Z_std = safe_standardize(Z)
    W_std = safe_standardize(W)

    # Standardize Targets based ONLY on the observed training context
    y_train_obs = Y_obs[:, :N_TRAIN]
    y_context_mean = y_train_obs.mean(dim=1, keepdim=True)
    y_context_std = y_train_obs.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-5)

    Y_obs_std = (Y_obs - y_context_mean) / y_context_std

    # Compute true standardized ITE for the test segment
    y_test_0 = Y0[:, N_TRAIN:]
    y_test_1 = Y1[:, N_TRAIN:]
    true_ITEs_std = ((y_test_1 - y_test_0) / y_context_std).numpy()

    # Package exactly as bench_run.py expects
    data = {
        "U": U_std,
        "X": X_std,
        "Z": Z_std,
        "W": W_std,
        "A": A,
        "Y": Y_obs_std,
        "d_X": torch.tensor([0] * n_batch),
        "d_Z": torch.tensor([2] * n_batch),
        "d_W": torch.tensor([1] * n_batch),
        "true_ITEs": true_ITEs_std,
        "N_TRAIN": N_TRAIN,
        "N_TEST": N_TEST,
        "N_BATCH": n_batch,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "data.pt"
    torch.save(data, out_path)
    print(f"Success! Data saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=str, default="./outputs", help="Directory to save data.pt"
    )
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--prop-train", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_benchmark_data(
        Path(args.out_dir),
        n_batch=args.batches,
        n_rows=args.rows,
        prop_train=args.prop_train,
        seed=args.seed,
    )
