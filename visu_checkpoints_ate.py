#%%

import glob
import os
import re
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from proximalfm.models import build_model
from proximalfm.Structure import ProximalPriorConfig, ModelConfig
from proximalfm.PriorDataLoader import create_causal_data_loaders

MODEL = "bnn_complete__longRunMSE"
device_str = "mps"
FILTER_REMOVE_EPOCHS = []
FILTER_KEEP_EPOCHS = [1, 10, 30, 50, 70, 100, 120, 150, 200, 250, 300, 330]

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
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{train_cfg.get('gpu', 0)}")
    elif device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    dtype = getattr(torch, model_cfg.get("dtype", "float32"))

    config_model = ModelConfig(**model_cfg)
    model = build_model(model_config=config_model).to(device, dtype=dtype)
    model.eval()

    # ---------------------------------------------------------
    # 2. Generate exactly K test datasets
    # ---------------------------------------------------------
    k_datasets = 100 # Number of test datasets (episodes)
    
    # Define how many points are used as context vs test
    n_train = config["data"].get("min_context_size", 200)
    n_test = 500
    samples_per_episode = n_train + n_test

    prior_config = ProximalPriorConfig(**prior_cfg)

    loaders = create_causal_data_loaders(
        prior_config=prior_config,
        batch_size=k_datasets,       
        min_episode_size=samples_per_episode,
        max_episode_size=samples_per_episode,
        train_episodes=1,            
        val_episodes=k_datasets,     
        num_workers=1,
        seed=9999999, 
    )
    
    # Grab the single batch
    batch = next(iter(loaders["val"]))
    batch = {k: v.to(device) for k, v in batch.items()}

    x = batch["X"]
    z = batch["Z"]
    w = batch["W"]
    
    # A and Y transposed to (B, N_total, 1)
    a_full = batch["A"]
    y_full = batch["Y"]

    # True ITE shape from loader is (N_total, B, 1). 
    # We slice out the test portion, transpose to (B, n_test), and calculate the ATE per episode.
    true_ite_test = batch["ITE"][:, n_train:, 0].cpu().numpy().T
    true_ate = true_ite_test.mean(axis=0) # shape (B,)

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
    num_plots = len(checkpoints)

    # ---------------------------------------------------------
    # 4. Evaluate and Plot
    # ---------------------------------------------------------
    cols = 3
    rows = math.ceil(num_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(24, 5 * rows))

    axes = np.array(axes).flatten()

    for idx, ckpt_path in enumerate(tqdm(selected_checkpoints, desc="Evaluating Checkpoints")):
        epoch = extract_epoch(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        chunk = 10
        pred_ates = np.zeros((x.size(0),))  # (B,)
        with torch.no_grad():

            for start in range(0, k_datasets, chunk):
                end = min(start + chunk, k_datasets)
                x_chunk = x[start:end, :, :]
                z_chunk = z[start:end, :, :]
                w_chunk = w[start:end, :, :]
                a_chunk = a_full[start:end, :n_train, 0]
                y_chunk = y_full[start:end, :n_train, 0]
                out = model.estimate_CATE(x_chunk, a_chunk, w_chunk, z_chunk, y_chunk)
                pred_cate = out[:, :, 0].cpu().numpy()
                pred_ate_chunk = pred_cate.mean(axis=1)
                pred_ates[start:end] = pred_ate_chunk

        # Plotting
        ax = axes[idx]
        
        # Perfect prediction line
        min_val = min(true_ate.min(), pred_ates.min()) - 1
        max_val = max(true_ate.max(), pred_ates.max()) + 1
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label="Perfect Prediction")

        # Scatter the ATE predictions
        ax.scatter(
            true_ate, 
            pred_ates, 
            alpha=0.6, 
            s=40,
            edgecolors='w',
            color='royalblue',
            label=f"Predicted ATE (n={n_test})"
        )

        ax.set_title(f"Epoch {epoch}", fontsize=14)
        ax.set_xlabel("True ATE")
        ax.set_ylabel("Predicted ATE")
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
