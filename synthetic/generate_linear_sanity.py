#%%

import os
import torch
import numpy as np
from pathlib import Path


def generate_perfect_linear_sanity(
    save_dir="linear",
    n_batch=30,  # Match your standard benchmark batch size
    seq_len=2048,
    prop_train=0.75,
    true_tau=0.75,  # Constant homogeneous treatment effect
    seed=123,
):
    """
    Generates a standalone, purely linear 1D proximal setup with heavy unmeasured confounding.
    Saves directly to save_dir/data.pt in the format expected by bench_run.py.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_size = int(round(prop_train * seq_len))
    n_test = seq_len - train_size

    # Initialize lists to hold data across all batches
    U_list, X_list, Z_list, W_list, A_list, Y_obs_list, Y0_list, Y1_list = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    print(f"Generating {n_batch} linear proximal sanity batches...")

    for b in range(n_batch):
        # Helper to ensure coefficients don't vanish
        def rand_coeff(low=0.5, high=1.5):
            val = low + (high - low) * torch.rand(1).item()
            sign = -1.0 if torch.rand(1).item() < 0.5 else 1.0
            return sign * val

        U_b = torch.randn(seq_len, 1)
        X_b = rand_coeff() * U_b + torch.randn(seq_len, 1)

        # Z and W get bounded coefficients to ensure they are always strong proxies
        Z_b = (
            rand_coeff() * U_b
            + rand_coeff() * X_b
            + torch.randn(1) * torch.randn(seq_len, 1)
        )
        W_b = (
            rand_coeff() * U_b
            - rand_coeff() * X_b
            + torch.randn(1) * torch.randn(seq_len, 1)
        )

        raw_logit = (
            rand_coeff() * X_b
            + rand_coeff() * Z_b
            + rand_coeff() * U_b
            + torch.randn(seq_len, 1)
        )
        logit_std = (raw_logit - raw_logit.mean()) / raw_logit.std().clamp(min=1e-5)

        # Note: -10 to 10 allows probabilities of 0.00004 and 0.99995.
        # You might want -6 to 6 for a more realistic overlap!
        logit_clipped = logit_std.clamp(-6.0, 6.0)

        p_b = torch.sigmoid(logit_clipped)
        A_b = (torch.rand(seq_len, 1) < p_b).float()

        f0 = (
            rand_coeff() * X_b
            + rand_coeff() * U_b
            + rand_coeff() * W_b
            + torch.randn(seq_len, 1)
        )
        tau_b = torch.full((seq_len, 1), true_tau)

        eps_y = 0.5 * torch.randn(seq_len, 1)
        Y0_b = f0 + eps_y
        Y1_b = f0 + tau_b + eps_y
        Y_obs_b = A_b * Y1_b + (1.0 - A_b) * Y0_b + 0.03 * torch.randn(seq_len, 1)
        # Append arrays to compile dataset
        X_list.append(X_b)
        U_list.append(U_b)
        Z_list.append(Z_b)
        W_list.append(W_b)
        A_list.append(A_b)
        Y_obs_list.append(Y_obs_b)
        Y0_list.append(Y0_b)
        Y1_list.append(Y1_b)

    # Stack along the batch dimension (dim=0) -> Shape: [B, T, 1]
    X = torch.stack(X_list)
    U = torch.stack(U_list)
    Z = torch.stack(Z_list)
    W = torch.stack(W_list)
    A = torch.stack(A_list)
    Y_obs = torch.stack(Y_obs_list)
    Y0 = torch.stack(Y0_list)
    Y1 = torch.stack(Y1_list)

    # --- 5. Pipeline Standardization (Matches bench_run.py exactly) ---
    def safe_standardize(tensor: torch.Tensor) -> torch.Tensor:
        mu = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True)
        std = torch.where(std < 1e-5, torch.ones_like(std), std)
        return (tensor - mu) / std

    X_std = safe_standardize(X)
    W_std = safe_standardize(W)
    Z_std = safe_standardize(Z)
    U_std = safe_standardize(U)

    # Target Standardization (Based entirely on training split context)
    y_train_obs = Y_obs[:, :train_size]
    y_context_mean = y_train_obs.mean(dim=1, keepdim=True)
    y_context_std = y_train_obs.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-5)

    Y_obs_std = (Y_obs - y_context_mean) / y_context_std

    # Compute explicit standardized ITEs for the held-out test split
    y_test_0 = Y0[:, train_size:]
    y_test_1 = Y1[:, train_size:]
    true_ITEs_std = ((y_test_1 - y_test_0) / y_context_std).cpu().numpy()

    # --- 6. Pack into unified data schema ---
    final_data = {
        "U": U_std.cpu(),
        "X": X_std.cpu(),
        "Z": Z_std.cpu(),
        "W": W_std.cpu(),
        "A": A.cpu(),  # Left raw (binary)
        "Y": Y_obs_std.cpu(),
        "d_X": torch.ones(n_batch, dtype=torch.long),
        "d_Z": torch.ones(n_batch, dtype=torch.long),
        "d_W": torch.ones(n_batch, dtype=torch.long),
        "true_ITEs": true_ITEs_std,
        "N_TRAIN": train_size,
        "N_TEST": n_test,
        "N_BATCH": n_batch,
    }

    out_file = save_dir / "data.pt"
    torch.save(final_data, out_file)
    print(f"[✓] Standalone linear sanity dataset built successfully at: {out_file}")


if __name__ == "__main__":
    generate_perfect_linear_sanity()
