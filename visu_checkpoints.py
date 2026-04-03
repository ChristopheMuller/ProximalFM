#%%

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from proximalfm.Model import ProximalNanoTabICLv2
from proximalfm.Structure import ProximalPriorConfig
from proximalfm.PriorDataLoader import create_causal_data_loaders

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_epoch(filepath: str) -> int:
    """Extract epoch number from checkpoint filename for sorting."""
    match = re.search(r"model_epoch_(\d+)", os.path.basename(filepath))
    return int(match.group(1)) if match else -1

def main():
    config = load_config()
    train_cfg = config["training"]
    prior_cfg = config["prior"]
    model_cfg = config["model"]

    # ---------------------------------------------------------
    # 1. Setup Device & Model
    # ---------------------------------------------------------
    device_str = train_cfg.get("device", "cpu")
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{train_cfg.get('gpu', 0)}")
    elif device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    dtype = getattr(torch, model_cfg.get("dtype", "float32"))

    model = ProximalNanoTabICLv2(
        out_dim=2, # 2 dimensions: 1 for CATE mean, 1 for log_var
        embed_dim=model_cfg.get("embed_dim", 64),
        col_num_blocks=model_cfg.get("col_num_blocks", 3), 
        row_num_blocks=model_cfg.get("row_num_blocks", 3),
        icl_num_blocks=model_cfg.get("icl_num_blocks", 12),
        col_nhead=model_cfg.get("col_nhead", 8),
        row_nhead=model_cfg.get("row_nhead", 8),
        icl_nhead=model_cfg.get("icl_nhead", 8),
        feature_group_size_x=prior_cfg["d_x"],
        feature_group_size_w=prior_cfg["d_w"],
        feature_group_size_z=prior_cfg["d_z"],
    ).to(device, dtype=dtype)
    model.eval()

    # ---------------------------------------------------------
    # 2. Generate exactly K test datasets (1 Batch)
    # ---------------------------------------------------------
    k_datasets = 150 # Number of test datasets
    samples_per_episode = config["data"]["samples_per_episode"]
    
    # We want 1 query point per episode, the rest is context
    single_eval_pos = samples_per_episode - 1 

    prior_config = ProximalPriorConfig(
        type=prior_cfg["type"],
        d_x=prior_cfg["d_x"],
        d_z=prior_cfg["d_z"],
        d_w=prior_cfg["d_w"],
        d_u=prior_cfg["d_u"],
        dtype=dtype,
    )

    loaders = create_causal_data_loaders(
        prior_config=prior_config,
        batch_size=k_datasets,       # Set to k_datasets to get them all in one batch
        samples_per_episode=samples_per_episode,
        train_episodes=1,            # Don't need train
        val_episodes=k_datasets,     # Exactly one batch worth of episodes
        num_workers=1,
        seed=1234, # Fixed seed for reproducible evaluation 
    )
    
    # Grab the single batch
    batch = next(iter(loaders["val"]))
    batch = {k: v.to(device) for k, v in batch.items()}

    x = batch["X"].transpose(0, 1)  # (B, N_total, d_x)
    z = batch["Z"].transpose(0, 1)  # (B, N_total, d_z)
    w = batch["W"].transpose(0, 1)  # (B, N_total, d_w)
    
    # A and Y transposed to (B, N_total, 1)
    a_full = batch["A"].transpose(0, 1)  
    y_full = batch["Y"].transpose(0, 1)  

    # True ITE at the query position: shape (B,)
    true_ite = batch["ITE"][single_eval_pos, :, 0].cpu().numpy()

    # ---------------------------------------------------------
    # 3. Find and sort checkpoints
    # ---------------------------------------------------------
    path = os.path.join(train_cfg["save_dir"], train_cfg.get("exp_name", "cate_run"))
    checkpoints = glob.glob(os.path.join(path, "model_epoch_*.pth"))
    
    # Filter out "best" to avoid duplicate epoch plots, keeping only numbered epochs
    checkpoints = [c for c in checkpoints if "best" not in c]
    checkpoints.sort(key=extract_epoch)

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {path} matching pattern 'model_epoch_*.pth'")

    # Select all checkpoints to plot
    num_plots = len(checkpoints)
    selected_checkpoints = checkpoints

    # ---------------------------------------------------------
    # 4. Evaluate and Plot
    # ---------------------------------------------------------
    import math
    cols = 4
    rows = math.ceil(num_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(24, 5 * rows))

    # If there's only 1 row and 1 col, axes might not be an array, but here cols=4 so it will be an array
    # Flattening makes it easier to iterate over
    axes = np.array(axes).flatten()

    for idx, ckpt_path in enumerate(tqdm(selected_checkpoints, desc="Evaluating Checkpoints")):
        epoch = extract_epoch(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        with torch.no_grad():
            # Slice A and Y to only include the context rows (n_train), and drop the last dim
            a_train = a_full[:, :single_eval_pos, 0] # (B, n_train)
            y_train = y_full[:, :single_eval_pos, 0] # (B, n_train)
            
            # Forward pass
            out = model(x=x, a=a_train, w=w, z=z, y=y_train)
            
            # Extract query predictions. `out` has shape (B, N_query, 2)
            # Since N_query is 1, shape is (B, 1, 2)
            pred_mean = out[:, 0, 0].cpu().numpy()      # Mean is the 0th index
            pred_log_var = out[:, 0, 1].cpu().numpy()   # Log Var is the 1st index
            
            # Standard deviation for error bars
            pred_std = np.exp(0.5 * pred_log_var)

        # Plotting
        ax = axes[idx]
        
        # Perfect prediction line
        min_val = min(true_ite.min(), pred_mean.min()) - 1
        max_val = max(true_ite.max(), pred_mean.max()) + 1
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label="Perfect Prediction")

        # Scatter with error bars (2 std devs for ~95% confidence interval)
        ax.errorbar(
            true_ite, 
            pred_mean, 
            yerr=2 * pred_std, 
            fmt='o', 
            alpha=0.6, 
            capsize=3,
            ecolor='gray',
            markerfacecolor='royalblue',
            markeredgecolor='darkblue',
            label=r"Pred Mean $\pm 2\sigma$"
        )

        ax.set_title(f"Epoch {epoch}", fontsize=14)
        ax.set_xlabel("True ITE")
        ax.set_ylabel("Predicted CATE")
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

    # Hide any unused subplots
    for j in range(len(selected_checkpoints), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
# %%
