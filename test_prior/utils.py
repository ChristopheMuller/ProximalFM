import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from proximalfm.prior import PriorDataset


def safe_standardize(x):
    """Standardize features, preventing division by zero for padded features."""
    mu = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    std = torch.where(std < 1e-5, torch.ones_like(std), std)
    return (x - mu) / std


def generate_eval_datasets(n_batches=2, batch_size=2, prior_kwargs=None):
    """
    Generates batches from PriorDataset and formats them into a list
    of pandas DataFrames for easy downstream evaluation.
    """
    if prior_kwargs is None:
        prior_kwargs = {
            "min_dim_U": 1,
            "max_dim_U": 1,
            "min_dim_W": 1,
            "max_dim_W": 1,
            "min_dim_Z": 1,
            "max_dim_Z": 1,
            "min_dim_X": 1,
            "max_dim_X": 1,
            "max_seq_len": 2048,
            "min_train_size": 0.5,
            "max_train_size": 0.95,
            "prior_type": "mlp_scm",
            "device": "cpu",
        }

    dataset = PriorDataset(batch_size=batch_size, batch_size_per_gp=1, **prior_kwargs)
    iterator = iter(dataset)

    all_datasets = []

    for _ in tqdm(range(n_batches), desc="Sampling Batches"):
        batch = next(iterator)
        U, X, W, Z, A, A_probs, y_obs, y0, y1 = batch[:9]
        seq_lens, train_sizes = batch[-2:]

        # Standardize features
        U = safe_standardize(U)
        X = safe_standardize(X)
        W = safe_standardize(W)
        Z = safe_standardize(Z)

        # Standardize targets based on context (train) set
        y_obs_ctx = y_obs[:, : train_sizes[0]]
        y_obs_ctx_mean = y_obs_ctx.mean(dim=1, keepdim=True)
        y_obs_ctx_std = y_obs_ctx.std(dim=1, unbiased=False, keepdim=True).clamp(
            min=1e-5
        )

        y_obs_std = (y_obs - y_obs_ctx_mean) / y_obs_ctx_std
        y0_std = (y0 - y_obs_ctx_mean) / y_obs_ctx_std
        y1_std = (y1 - y_obs_ctx_mean) / y_obs_ctx_std

        for i in range(dataset.batch_size):
            seq_len = int(seq_lens[i].item())
            train_size = int(train_sizes[i].item())
            train_size = max(1, min(train_size, seq_len))

            sample_data = {
                "is_train": np.arange(seq_len) < train_size,
                "A": A[i, :seq_len].cpu().numpy().flatten(),
                "A_probs": A_probs[i, :seq_len].cpu().numpy().flatten(),
                "Y_obs": y_obs_std[i, :seq_len].cpu().numpy().flatten(),
                "Y0": y0_std[i, :seq_len].cpu().numpy().flatten(),
                "Y1": y1_std[i, :seq_len].cpu().numpy().flatten(),
            }

            # Flatten multi-dimensional variables
            for var_name, data in [("U", U), ("X", X), ("W", W), ("Z", Z)]:
                raw_data = data[i, :seq_len].cpu().numpy()
                if raw_data.ndim == 1:
                    sample_data[var_name] = raw_data
                else:
                    for col in range(raw_data.shape[1]):
                        sample_data[f"{var_name}_{col}"] = raw_data[:, col]

            df = pd.DataFrame(sample_data)
            df.attrs["train_size"] = train_size
            df.attrs["seq_len"] = seq_len
            all_datasets.append(df)

    return all_datasets
