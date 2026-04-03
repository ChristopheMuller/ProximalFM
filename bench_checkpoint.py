
#%%

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm

from other_methods import *

import yaml
import seaborn as sns

from proximalfm.Model import ProximalNanoTabICLv2
from proximalfm.Structure import ProximalPriorConfig
from proximalfm.PriorDataLoader import create_causal_data_loaders

#%%
# -- 0. Parameters of Benchmarking --

N_TRAIN = 100
N_TEST = 10 # 1 test per episode
N_BATCH = 150 # Number of episodes (datasets) to evaluate on

methods_to_compare = [
    S_OLS,
    # S_RF,
    # S_TabICL,
    T_OLS,
    # T_RF,
    # T_TabICL,
    P2SLS,
]

#%%
# ── 1. Load Configuration ──────────────────────────────────────────────────
def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
train_cfg = config["training"]
prior_cfg = config["prior"]
model_cfg = config["model"]

#%%
# ── 2. Device & Precision Setup ────────────────────────────────────────────
# device_str = train_cfg.get("device", "cpu")
device_str = "cpu"
if device_str == "cuda" and torch.cuda.is_available():
    device = torch.device(f"cuda:{train_cfg.get('gpu', 0)}")
elif device_str == "mps" and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
dtype = getattr(torch, model_cfg.get("dtype", "float32"))

#%%
# ── 3. Model Initialization ────────────────────────────────────────────────
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

#%%
# ── 4. Data Loader & Evaluation Setup ──────────────────────────────────────
samples_per_episode = N_TRAIN + N_TEST
k_datasets = N_BATCH
single_eval_pos = samples_per_episode - N_TEST

prior_config = ProximalPriorConfig(
    type=prior_cfg.get("type", "linear"),
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
    train_episodes=1,            
    val_episodes=k_datasets,     
    num_workers=0,
    seed=1234, 
)

print(f"✓ Initialization complete.")
print(f"  Device : {device}")
print(f"  Batch  : {k_datasets} episodes")
print(f"  Context: {single_eval_pos} samples")

#%%
# ── 5. Load the Best Model ─────────────────────────────────────────────────
best_ckpt_path = os.path.join(
    config["training"]["save_dir"], 
    config["training"].get("exp_name", "cate_run"), 
    "best_model.pth"
)

if not os.path.exists(best_ckpt_path):
    raise FileNotFoundError(f"Could not find best model at {best_ckpt_path}")

checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"✓ Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")

#%%
# ── 6. Get Data Batch & True ITE ───────────────────────────────────────────
batch = next(iter(loaders["val"]))
batch = {k: v.to(device) for k, v in batch.items()}

x_full = batch["X"].transpose(0, 1) # (batch, N_total, d_x)
z_full = batch["Z"].transpose(0, 1)
w_full = batch["W"].transpose(0, 1)
a_full = batch["A"].transpose(0, 1)
y_full = batch["Y"].transpose(0, 1)

# True ITE for the query points: shape (B,)
true_ITEs = batch["ITE"][single_eval_pos:, :, 0].transpose(0,1).cpu().numpy()
true_ATEs = true_ITEs.mean(axis=1)

#%%
# ── 7. Neural Model Inference ──────────────────────────────────────────────
with torch.no_grad():
    a_train = a_full[:, :N_TRAIN, 0]
    y_train = y_full[:, :N_TRAIN, 0]
    
    out = model(x=x_full, a=a_train, w=w_full, z=z_full, y=y_train)
    cate_pred_nn_mean = out[:, :, 0].cpu().numpy()
    cate_pred_nn_var = out[:, :, 1].cpu().numpy()

#%%
# ── 8. Baseline Inference & Neural Model Post-Processing ───────────────────

# Convert full tensors to numpy for sklearn
X_np = x_full.cpu().numpy()
Z_np = z_full.cpu().numpy()
W_np = w_full.cpu().numpy()
A_np = a_full.cpu().numpy()
Y_np = y_full.cpu().numpy()

# Initialize a dictionary to hold predictions. Shape: (N_BATCH, N_TEST)
preds_dict = {method.__name__: np.zeros((N_BATCH, N_TEST)) for method in methods_to_compare}
preds_dict["ProximalNanoTabICLv2"] = cate_pred_nn_mean

# Loop through episodes to fit/predict baselines
for b in tqdm(range(N_BATCH)):
    # Context Data (Fit)
    X_ctx = X_np[b, :N_TRAIN]
    W_ctx = W_np[b, :N_TRAIN]
    Z_ctx = Z_np[b, :N_TRAIN]
    A_ctx = A_np[b, :N_TRAIN, 0]
    Y_ctx = Y_np[b, :N_TRAIN, 0]
    
    # Query Data (Predict)
    X_q = X_np[b, N_TRAIN:]
    W_q = W_np[b, N_TRAIN:]
    Z_q = Z_np[b, N_TRAIN:]
    
    # Run each baseline method
    for method_class in methods_to_compare:
        name = method_class.__name__
        model_inst = method_class()
        
        try:
            model_inst.fit(X_ctx, A_ctx, Y_ctx, W=W_ctx, Z=Z_ctx)
            preds_dict[name][b] = model_inst.predict(X_q, W=W_q, Z=Z_q)
        except Exception as e:
            print(f"Error in {name} on episode {b}: {e}")
            preds_dict[name][b] = np.nan

#%%
# ── 9. Compiling Metrics and Evaluation ────────────────────────────────────

results = []

# true_ITEs shape is (N_BATCH, N_TEST)
flattened_true_ITEs = true_ITEs.flatten()
true_ate_global = flattened_true_ITEs.mean()

for name, preds in preds_dict.items():
    flattened_preds = preds.flatten()
    
    # Calculate Bias, RMSE, and R2 globally across all test points
    bias = flattened_preds.mean() - true_ate_global
    
    # Avoid errors if a model failed and populated NaNs
    valid_mask = ~np.isnan(flattened_preds)
    if valid_mask.sum() == 0:
        rmse, r2 = np.nan, np.nan
    else:
        rmse = np.sqrt(np.mean((flattened_preds[valid_mask] - flattened_true_ITEs[valid_mask]) ** 2))
        r2 = r2_score(flattened_true_ITEs[valid_mask], flattened_preds[valid_mask])
    
    results.append({
        "Estimator": name,
        "ATE Bias": bias,
        "CATE RMSE": rmse,
        "CATE R²": r2
    })

df_results = pd.DataFrame(results).set_index("Estimator")
print("\n" + "="*70)
print(f"  BENCHMARK SUMMARY ({N_BATCH} episodes, {N_TEST} queries/ep)")
print("="*70)
print(df_results.round(4).to_string())
print("="*70)

plt.figure(figsize=(10, 5))
sns.barplot(x=df_results.index, y=df_results["CATE RMSE"], palette="viridis", hue=df_results.index)
plt.title("CATE RMSE Comparison")
plt.ylabel("RMSE (Lower is Better)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()