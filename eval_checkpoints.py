#%%

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from proximalfm.Model import PerFeatureTransformerCATE
from proximalfm.Structure import ProximalPriorConfig
from proximalfm.PriorDataLoader import create_causal_data_loaders

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_epoch(filepath: str) -> int:
    """Extract epoch number from checkpoint filename for sorting."""
    match = re.search(r"model_epoch_(\d+)", os.path.basename(filepath))
    return int(match.group(1)) if match else -1

def _to_seq_first(t: torch.Tensor) -> torch.Tensor:
    """Convert DataLoader output (B, N, D) to Model input (N, B, D)."""
    return t.permute(1, 0, 2).contiguous()

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

    model = PerFeatureTransformerCATE(
        ninp=model_cfg["ninp"],
        nhead=model_cfg["nhead"],
        nhid=model_cfg["nhid"],
        nlayers=model_cfg["nlayers"],
        layer_norm_with_elementwise_affine=model_cfg.get("layer_norm_with_elementwise_affine", True),
    ).to(device, dtype=dtype)
    model.eval()

    # ---------------------------------------------------------
    # 2. Generate exactly K test datasets (1 Batch)
    # ---------------------------------------------------------
    k_datasets = 50 # Number of test datasets (batch size)
    samples_per_episode = config["data"]["samples_per_episode"]
    
    # We want 1 query point per episode, the rest is context
    single_eval_pos = samples_per_episode - 1 

    prior_config = ProximalPriorConfig(
        d_x=prior_cfg["d_x"],
        d_z=prior_cfg["d_z"],
        d_w=prior_cfg["d_w"],
        d_u=prior_cfg["d_u"],
        dtype=dtype,
    )

    loaders = create_causal_data_loaders(
        prior_config=prior_config,
        batch_size=k_datasets,
        samples_per_episode=samples_per_episode,
        train_episodes=1,            # Don't need train
        val_episodes=k_datasets,     # Exactly one batch
        num_workers=1,
        seed=1234, # Fixed seed for reproducible evaluation 
    )
    
    # Grab the single batch
    batch = next(iter(loaders["val"]))
    batch = {k: v.to(device) for k, v in batch.items()}

    x = {"main": _to_seq_first(batch["X"])}
    z = {"main": _to_seq_first(batch["Z"])}
    w = {"main": _to_seq_first(batch["W"])}
    a = {"main": _to_seq_first(batch["A"])}
    y = {"main": _to_seq_first(batch["Y"])}
    
    # True ITE at the query position: shape (B,)
    true_ite = batch["ITE"][:, single_eval_pos, 0].cpu().numpy()

    # ---------------------------------------------------------
    # 3. Find and sort checkpoints
    # ---------------------------------------------------------
    save_dir = train_cfg.get("save_dir", "checkpoints")
    checkpoints = glob.glob(os.path.join(save_dir, "model_epoch_*.pth"))
    
    # Filter out "best" to avoid duplicate epoch plots, keeping only numbered epochs
    checkpoints = [c for c in checkpoints if "best" not in c]
    checkpoints.sort(key=extract_epoch)

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {save_dir} matching pattern 'model_epoch_*.pth'")

    # Select up to 6 checkpoints evenly spaced to plot
    num_plots = min(6, len(checkpoints))
    indices = np.linspace(0, len(checkpoints) - 1, num_plots, dtype=int)
    selected_checkpoints = [checkpoints[i] for i in indices]

    # ---------------------------------------------------------
    # 4. Evaluate and Plot
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, ckpt_path in enumerate(tqdm(selected_checkpoints, desc="Evaluating Checkpoints")):
        epoch = extract_epoch(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        with torch.no_grad():
            out = model(x, z, w, a, y, single_eval_pos=single_eval_pos)
            
            # Extract query predictions: shape (B, N_query, 1) -> (B,)
            # Our query is at index single_eval_pos: inside the sequence
            pred_mean = out["cate_mean"][:, single_eval_pos:, :].squeeze().cpu().numpy()
            pred_log_var = out["cate_log_var"][:, single_eval_pos:, :].squeeze().cpu().numpy()
            
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
        ax.set_xlabel("True CATE")
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
