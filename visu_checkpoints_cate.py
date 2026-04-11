#%%

import glob
import os
import re
import math

# import matplotlib.subplots
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from proximalfm.Model import ProximalNanoTabICLv2
from proximalfm.Structure import ProximalPriorConfig, ModelConfig
from proximalfm.PriorDataLoader import create_causal_data_loaders

MODEL = "cate_big"
FILTER_REMOVE_EPOCHS = []
FILTER_KEEP_EPOCHS = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 440]

def load_config() -> dict:
    full_config = {}
    config_training_path = os.path.join("checkpoints", MODEL, "train_config.yaml")
    with open(config_training_path, "r") as f:
        full_config["training"] = yaml.safe_load(f)

    config_prior_path = os.path.join("checkpoints", MODEL, "prior_config.yaml")
    with open(config_prior_path, "r") as f:
        full_config["prior"] = yaml.safe_load(f)

    config_model_path = os.path.join("checkpoints", MODEL, "model_config.yaml")
    with open(config_model_path, "r") as f:
        full_config["model"] = yaml.safe_load(f)

    config_data_path = os.path.join("checkpoints", MODEL, "data_config.yaml")
    with open(config_data_path, "r") as f:
        full_config["data"] = yaml.safe_load(f)

    return full_config

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

    config_model = ModelConfig(
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
    )
    model = ProximalNanoTabICLv2(
        config=config_model
    ).to(device, dtype=dtype)
    model.eval()

    # ---------------------------------------------------------
    # 2. Generate exactly 1 episode with K test points
    # ---------------------------------------------------------
    n_train = int(config["data"].get("samples_per_episode", 256) * 0.85)
    k_test_points = 150
    samples_per_episode = n_train + k_test_points
    batch_size = 1

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
        batch_size=batch_size,       
        samples_per_episode=samples_per_episode,
        train_episodes=1,            
        val_episodes=1,     
        num_workers=1,
        seed=np.random.randint(0, 10000),
    )
    
    # Grab the single batch
    batch = next(iter(loaders["val"]))
    batch = {k: v.to(device) for k, v in batch.items()}

    x = batch["X"].transpose(0, 1)  # (1, N_total, d_x)
    z = batch["Z"].transpose(0, 1)  # (1, N_total, d_z)
    w = batch["W"].transpose(0, 1)  # (1, N_total, d_w)
    
    # A and Y transposed to (1, N_total, 1)
    a_full = batch["A"].transpose(0, 1)  
    y_full = batch["Y"].transpose(0, 1)  

    # True ITE for the k test points: shape (k_test_points,)
    # Original shape before transpose is (seq_len, batch_size, 1)
    true_ite = batch["ITE"][n_train:, 0, 0].cpu().numpy()

    # ---------------------------------------------------------
    # 3. Find and sort checkpoints
    # ---------------------------------------------------------
    path = os.path.join(train_cfg["save_dir"], train_cfg.get("exp_name", "cate_run"))
    checkpoints = glob.glob(os.path.join(path, "model_epoch_*.pth"))
    
    checkpoints = [c for c in checkpoints if "best" not in c]
    checkpoints.sort(key=extract_epoch)

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {path} matching pattern 'model_epoch_*.pth'")

    if FILTER_KEEP_EPOCHS:
        checkpoints = [c for c in checkpoints if extract_epoch(c) in FILTER_KEEP_EPOCHS]
    selected_checkpoints = [c for c in checkpoints if extract_epoch(c) not in FILTER_REMOVE_EPOCHS]
    num_plots = len(selected_checkpoints)

    # ---------------------------------------------------------
    # 4. Evaluate and Plot
    # ---------------------------------------------------------
    cols = 3
    rows = math.ceil(num_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(24, 5 * rows))

    axes = np.array(axes).flatten()

    for idx, ckpt_path in enumerate(tqdm(selected_checkpoints, desc="Evaluating Checkpoints")):
        epoch = extract_epoch(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        with torch.no_grad():
            # Slice A and Y to only include the context rows
            a_train = a_full[:, :n_train, 0] # (1, n_train)
            y_train = y_full[:, :n_train, 0] # (1, n_train)
            
            # Forward pass
            out = model(x=x, a=a_train, w=w, z=z, y=y_train)
            
            # Extract query predictions. `out` has shape (1, k_test_points, 2)
            pred_mean = out[0, :, 0].cpu().numpy()      
            pred_log_var = out[0, :, 1].cpu().numpy()   
            
            # Standard deviation for error bars
            pred_std = np.exp(0.5 * pred_log_var)

        # Plotting
        ax = axes[idx]
        
        # Perfect prediction line
        min_val = min(true_ite.min(), pred_mean.min()) - 0.2
        max_val = max(true_ite.max(), pred_mean.max()) + 0.2
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
        ax.set_xlim([true_ite.min(), true_ite.max()])
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
