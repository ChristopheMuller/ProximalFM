#%%

import torch
import numpy as np
import matplotlib.pyplot as plt

#%%

from proximalfm.prior import PriorDataset

#%%

prior = PriorDataset(
    batch_size=1,
    batch_size_per_gp=1,
    min_dim_U=1,
    max_dim_U=1,
    min_dim_W=1,
    max_dim_W=2,
    min_dim_Z=1,
    max_dim_Z=2,
    min_dim_X=1,
    max_dim_X=3,
    min_seq_len=None,
    max_seq_len=256,
    log_seq_len=False,
    seq_len_per_gp=False,
    min_train_size=0.8,
    max_train_size=0.95,
    replay_small=False,
    prior_type="mlp_scm",
    device="cpu",
)

batch = prior.get_batch(batch_size=2)

U, X, W, Z, A, Y_obs, Y0, Y1, d_U, d_X, d_W, d_Z, seq_len, train_sizes = batch

if len(Y0.shape) == 1:
    Y0 = Y0.unsqueeze(0)
if len(Y1.shape) == 1:
    Y1 = Y1.unsqueeze(0)
if len(A.shape) == 1:
    A = A.unsqueeze(0)
if len(Y_obs.shape) == 1:
    Y_obs = Y_obs.unsqueeze(0)
y_obs_train = Y_obs[:, : train_sizes[0]]
y_obs_test = Y_obs[:, train_sizes[0] :]

#%%
import os

checkpoint_path = "check_1/"

all_checkpoints = [
    f
    for f in os.listdir(checkpoint_path)
    if f.startswith("step-") and f.endswith(".ckpt")
]

all_checkpoints.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))

for checkp in all_checkpoints:
    print(checkpoint_path + checkp)

first_checkpoint = torch.load(checkpoint_path + all_checkpoints[0], map_location="cpu")

#%%
from proximalfm._model import TabICL

model = TabICL(**first_checkpoint["config"])

# figures: 4 columns, and enough rows to fit all checkpoints
num_checkpoints = len(all_checkpoints)
num_cols = 4
num_rows = (num_checkpoints + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
axes = axes.flatten()

max_y_target = y_obs_test[0].max().item()
min_y_target = y_obs_test[0].min().item()

for i, checkp in enumerate(all_checkpoints):
    checkpoint = torch.load(checkpoint_path + checkp, map_location="cpu")
    print(f"Checkpoint: {checkp}, Step: {checkpoint['curr_step']}")

    model.load_state_dict(checkpoint["state_dict"])

    out = model.predict(
        X,
        W,
        Z,
        A.unsqueeze(-1).float(),
        y_train=y_obs_train,
        output_type="mean",
        target="cate",
    )[0]

    axes[i].scatter(y_obs_test[0].cpu(), out.cpu())
    min_x = min(min_y_target, out.min().item())
    max_x = max(max_y_target, out.max().item())
    axes[i].plot([min_x, max_x], [min_x, max_x], color="red", linestyle="--")
    axes[i].set_title(f"{checkp}")
    axes[i].set_xlabel("True CATE")
    axes[i].set_ylabel("Predicted CATE")


#%%


for i, checkp in enumerate(all_checkpoints):
    checkpoint = torch.load(checkpoint_path + checkp, map_location="cpu")
    print(f"Checkpoint: {checkp}, Step: {checkpoint['curr_step']}")

    model.load_state_dict(checkpoint["state_dict"])

    out = model.predict(
        X,
        W,
        Z,
        A.unsqueeze(-1).float(),
        y_train=y_obs_train,
        output_type="distribution",
        target="cate",
    )

    out_weights = out["y0"]["weights"][0, 0]

    # plot the weight shape:
    plt.figure()
    plt.bar(height=out_weights.cpu().numpy(), x=np.arange(len(out_weights)))
    plt.title(f"Weights for {checkp}")
    plt.xlabel("Bin Index")
    plt.ylabel("Weight")


#%%
