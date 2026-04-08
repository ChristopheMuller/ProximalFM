#%% 3_eval_and_plot.py
import os
import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from proximalfm.Structure import ModelConfig
from proximalfm.Model import ProximalNanoTabICLv2
from other_methods import S_OLS, T_OLS, P2SLS, S_RF, S_TabICL

#%%
# -- Configuration --
# Make sure to update this to the timestamped folder you want to evaluate
DATA_FOLDER = "eval_data/conf_2000"
MODEL_CHECK = "run4_256"

methods_to_compare = [
    S_OLS,
    S_RF,
    S_TabICL,
    T_OLS,
    P2SLS,
]

#%%
print("Loading configs and establishing device...")

model_config_path = os.path.join("checkpoints", MODEL_CHECK, "model_config.yaml")

# Load model and training configs (for neural model initialization)
with open(model_config_path, "r") as f:
    model_cfg = yaml.safe_load(f)

# Load the specific prior config used to generate this dataset
prior_config_path = os.path.join(DATA_FOLDER, "prior_config.yaml")
with open(prior_config_path, "r") as f:
    prior_cfg = yaml.safe_load(f)

device_str = "mps"
device = torch.device(device_str)
dtype = getattr(torch, model_cfg.get("dtype", "float32"))

#%%
# ── 1. Load Data ───────────────────────────────────────────────────────────
data_path = os.path.join(DATA_FOLDER, "evaluation_data.pt")
print(f"Loading data from {data_path}...")
data = torch.load(data_path, weights_only=False)

N_TRAIN = data["N_TRAIN"]
N_TEST = data["N_TEST"]
N_BATCH = data["N_BATCH"]
true_ITEs = data["true_ITEs"]

#%%
# ── 2. Setup Model & Load Checkpoint ───────────────────────────────────────

model_config = ModelConfig(
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
    config=model_config
).to(device, dtype=dtype)

best_ckpt_path = os.path.join("checkpoints", MODEL_CHECK, "best_model.pth")

if not os.path.exists(best_ckpt_path):
    raise FileNotFoundError(f"Could not find best model at {best_ckpt_path}")

checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"✓ Loaded best neural model from epoch {checkpoint.get('epoch', 'unknown')}")

#%%
# ── 3. Run Neural Inference ────────────────────────────────────────────────
print("Running Neural Inference...")
x_full = data["X"].to(device, dtype=dtype)
z_full = data["Z"].to(device, dtype=dtype)
w_full = data["W"].to(device, dtype=dtype)
a_full = data["A"].to(device, dtype=dtype)
y_full = data["Y"].to(device, dtype=dtype)

max_batch_size = 10
cate_preds_nn = np.zeros((N_BATCH, N_TEST))

n_bat = x_full.shape[0] // max_batch_size

with torch.no_grad():
    for i in range(n_bat):
        start_idx = i * max_batch_size
        end_idx = min((i + 1) * max_batch_size, x_full.shape[0])
        a_train = a_full[start_idx:end_idx, :N_TRAIN, 0]
        y_train = y_full[start_idx:end_idx, :N_TRAIN, 0]
        w_full_b = w_full[start_idx:end_idx]
        z_full_b = z_full[start_idx:end_idx]
        x_full_b = x_full[start_idx:end_idx]
        out = model(x=x_full_b, a=a_train, w=w_full_b, z=z_full_b, y=y_train)
        cate_pred_nn_mean = out[:, :, 0].cpu().numpy()
        cate_preds_nn[i * max_batch_size:(i + 1) * max_batch_size] = cate_pred_nn_mean

#%% Save the prediction as "exp_name_nn_preds.npz" in the same folder as baselines
nn_pred_file = os.path.join(DATA_FOLDER, f"{MODEL_CHECK}_preds.npz")
np.savez(nn_pred_file, preds=cate_preds_nn)
methods_to_compare.append(MODEL_CHECK)
print(f"✓ Saved neural model predictions to {nn_pred_file}")

#%%
# ── 4. Aggregate Predictions ───────────────────────────────────────────────
print(f"Aggregating baseline predictions from {DATA_FOLDER}...")
preds_dict = {}

for meth in methods_to_compare:
    if hasattr(meth, "__name__"):
        name = meth.__name__
    else:
        name = str(meth)

    pred_file = os.path.join(DATA_FOLDER, f"{name}_preds.npz")
    
    if os.path.exists(pred_file):
        baseline_data = np.load(pred_file)
        preds_dict[name] = baseline_data["preds"]
    else:
        print(f"Warning: Baseline predictions for {name} not found at {pred_file}")


#%%
# ── 5. Evaluate and Plot ───────────────────────────────────────────────────
results = []
for name, preds in preds_dict.items():
    for b in range(N_BATCH):
        pred_b   = preds[b]
        true_b   = true_ITEs[b]
        
        # Calculate ATE for the episode
        pred_ate = np.nanmean(pred_b)
        true_ate = true_b.mean()

        # R² per episode
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

df_summary = df_results.groupby("Estimator")["Error"].agg(
    Bias  = "mean",
    MedAE = lambda e: e.abs().median(), 
    MAE   = lambda e: e.abs().mean(),
    RMSE  = lambda e: np.sqrt((e**2).mean()),
    RMedSE = lambda e: np.sqrt(np.median(e**2)),
    Std   = "std",
).round(4)

print("\n--- Evaluation Summary ---")
print(df_summary)
