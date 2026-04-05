
#%%

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

N_TRAIN = 200
N_TEST = 500
N_BATCH = 250 # Number of episodes (datasets) to evaluate on

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

prior_kwargs = prior_cfg.copy()
prior_kwargs["dtype"] = dtype
prior_config = ProximalPriorConfig(**prior_kwargs)


loaders = create_causal_data_loaders(
    prior_config=prior_config,
    batch_size=k_datasets,
    samples_per_episode=samples_per_episode,
    train_episodes=1,            
    val_episodes=k_datasets,     
    num_workers=0,
    seed=np.random.randint(0, 10000)
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
# ── 9. Compare ATE & CATE ────────────────────────────────────────────────

results = []

for name, preds in preds_dict.items():
    for b in range(N_BATCH):
        pred_b   = preds[b]
        true_b   = true_ITEs[b]
        pred_ate = np.nanmean(pred_b)
        true_ate = true_b.mean()

        # R² per episode — degenerate (=nan) with constant ITE, meaningful once you add heterogeneity
        ite_var = true_b.var()
        if ite_var < 1e-8 or np.all(np.isnan(pred_b)):
            r2 = np.nan
        else:
            r2 = r2_score(true_b, pred_b)

        results.append({
            "Estimator": name,
            "Episode":   b,
            "Pred ATE":  pred_ate,
            "True ATE":  true_ate,
            "Error":     pred_ate - true_ate,
            "R²":        r2,
        })

df_results = pd.DataFrame(results)

# Updated summary table to include MedAE (Median Absolute Error) for outlier robustness
df_summary = df_results.groupby("Estimator")["Error"].agg(
    Bias  = "mean",
    MedAE = lambda e: e.abs().median(), 
    MAE   = lambda e: e.abs().mean(),
    RMSE  = lambda e: np.sqrt((e**2).mean()),
    Std   = "std",
).round(4)

print(df_summary)

#%%

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_results, x="Estimator", y="Error", palette="viridis")
plt.axhline(0, color='red', linestyle='--')
plt.title("Distribution of ATE Estimation Error across Episodes")
plt.ylabel("Error (Pred ATE - True ATE)")
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()