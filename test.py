

#%%

from proximalfm.Prior import ProximalPrior
import numpy as np
import matplotlib.pyplot as plt
from proximalfm.Structure import ProximalPriorConfig

#%%

seed = np.random.randint(0, 10000)
config = ProximalPriorConfig(
    temperature=1.5,
)
prior = ProximalPrior(seed=seed, config=config)

dataset = prior.generate_dataset(
    num_episodes=1,
    batch_size=1000,
    num_workers=1,
)

dataset[0].X.shape, dataset[0].Z.shape, dataset[0].W.shape, dataset[0].U.shape, dataset[0].A.shape, dataset[0].Y.shape, dataset[0].Y0.shape, dataset[0].Y1.shape, dataset[0].propensity.shape

#%%

episode = dataset[0]

x_dim = episode.X.shape[1]
z_dim = episode.Z.shape[1]
w_dim = episode.W.shape[1]
u_dim = episode.U.shape[1]

total_plots = x_dim + z_dim + w_dim + u_dim
cols = 4
rows = (total_plots + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
axes = axes.flatten()

idx = 0

for i in range(x_dim):
    axes[idx].scatter(episode.X[:, i].numpy(), episode.Y.numpy(), alpha=0.5)
    axes[idx].set_title(f'X[{i}] vs Y')
    idx += 1

for i in range(z_dim):
    axes[idx].scatter(episode.Z[:, i].numpy(), episode.Y.numpy(), alpha=0.5)
    axes[idx].set_title(f'Z[{i}] vs Y')
    idx += 1

for i in range(w_dim):
    axes[idx].scatter(episode.W[:, i].numpy(), episode.Y.numpy(), alpha=0.5)
    axes[idx].set_title(f'W[{i}] vs Y')
    idx += 1

for i in range(u_dim):
    axes[idx].scatter(episode.U[:, i].numpy(), episode.Y.numpy(), alpha=0.5)
    axes[idx].set_title(f'U[{i}] vs Y')
    idx += 1

for i in range(idx, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

#%%

cate = episode.Y1 - episode.Y0
fig, axes = plt.subplots(2, x_dim//2, figsize=(4 * x_dim, 8), squeeze=False)
axes = axes.flatten()

for i in range(x_dim):
    axes[i].scatter(episode.X[:, i].numpy(), cate.numpy(), alpha=0.5)
    axes[i].set_title(f'CATE vs X[{i}]')

plt.tight_layout()
plt.show()


#%%

plt.figure(figsize=(8, 6))
plt.hist(cate.numpy(), bins=50, density=True, alpha=0.6, color='g')
plt.title('Density of CATE')
plt.xlabel('CATE')
plt.ylabel('Density')
plt.grid()
plt.show()

#%%

plt.figure(figsize=(8, 6))
plt.hist(episode.propensity.numpy(), bins=50, density=True, alpha=0.6, color='b')
plt.title('Density of Propensity Scores')
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.grid()
plt.show()


#%%
from proximalfm.Model import PerFeatureTransformerCATE

model = PerFeatureTransformerCATE(
    ninp=128,
    nhead=4,
    nhid=128 * 4,
    nlayers=10,
    layer_norm_with_elementwise_affine=True,
)

model
# %%

import time
import torch
from proximalfm.Structure import ProximalPriorConfig
from proximalfm.PriorDataLoader import create_causal_data_loaders

print("1. Initializing Prior and DataLoaders...")
prior_config = ProximalPriorConfig(
    d_x=5, d_z=2, d_w=2, d_u=1,
    dtype=torch.float32
)

# create_causal_data_loaders now returns a dict {"train": ..., "val": ...}
loaders = create_causal_data_loaders(
    prior_config=prior_config,
    batch_size=4,
    samples_per_episode=128,
    train_episodes=100,
    val_episodes=10,
    num_workers=0,
)
train_loader = loaders["train"]
val_loader   = loaders["val"]

print("\n2. Fetching the first batch (testing generation speed)...")
start_time = time.time()
batch = next(iter(train_loader))
fetch_time = time.time() - start_time
print(f"Time to generate and fetch 1 batch: {fetch_time:.4f} seconds")

print("\n3. Inspecting Dictionary Keys and Tensor Shapes...")
# Keys are now uppercase and X/Z/W are separate tensors
expected_keys = ["X", "Z", "W", "A", "Y", "Y0", "Y1", "ITE"]
for key in expected_keys:
    if key in batch:
        print(f"  - {key:3s}  shape: {batch[key].shape}  |  dtype: {batch[key].dtype}")
    else:
        print(f"  - MISSING KEY: '{key}'")

print("\n4. Statistical Sanity Checks (on the first episode in the batch)...")
X   = batch["X"][0]    # (N, d_x)
Z   = batch["Z"][0]    # (N, d_z)
W   = batch["W"][0]    # (N, d_w)
A   = batch["A"][0]    # (N, 1)
Y   = batch["Y"][0]    # (N, 1)
Y0  = batch["Y0"][0]   # (N, 1)
Y1  = batch["Y1"][0]   # (N, 1)
ITE = batch["ITE"][0]  # (N, 1)

print(f"  - X  shape: {X.shape}  |  Z shape: {Z.shape}  |  W shape: {W.shape}")

print(f"  - Treatment A unique values: {torch.unique(A).tolist()}  (expected: [0.0, 1.0])")
print(f"  - Treatment A mean (propensity proxy): {A.mean().item():.3f}  (expected: ~0.1 – 0.9)")

print(f"  - Are Y0 and Y1 identical? {torch.allclose(Y0, Y1)}  (expected: False)")
print(f"  - ITE mean: {ITE.mean().item():.4f}  |  ITE std: {ITE.std().item():.4f}")

# Factual consistency: Y must equal Y1 where A=1 and Y0 where A=0
mask_treated = (A == 1.0).squeeze(-1)
mask_control = (A == 0.0).squeeze(-1)
y_treated_match = torch.allclose(Y[mask_treated], Y1[mask_treated])
y_control_match = torch.allclose(Y[mask_control], Y0[mask_control])
print(f"  - Factual Y == Y1 where A=1? {y_treated_match}  (expected: True)")
print(f"  - Factual Y == Y0 where A=0? {y_control_match}  (expected: True)")

# Proxy exclusion sanity: Z should be more correlated with A than with Y|A,
# and W should be more correlated with Y than with A|Y.
# We check this loosely via marginal absolute correlations.
def marginal_corr(u: torch.Tensor, v: torch.Tensor) -> float:
    """Mean absolute Pearson correlation of each column of u with v."""
    v_flat = v.squeeze(-1).float()
    corrs = []
    for col in u.T:
        c = torch.corrcoef(torch.stack([col.float(), v_flat]))[0, 1].abs().item()
        corrs.append(c)
    return sum(corrs) / len(corrs)

corr_Z_A = marginal_corr(Z, A)
corr_Z_Y = marginal_corr(Z, Y)
corr_W_A = marginal_corr(W, A)
corr_W_Y = marginal_corr(W, Y)
print(f"\n  Proxy correlation heuristics (marginal absolute Pearson):")
print(f"  - |corr(Z, A)|: {corr_Z_A:.3f}  |  |corr(Z, Y)|: {corr_Z_Y:.3f}  "
      f"(Z should correlate more with A)")
print(f"  - |corr(W, A)|: {corr_W_A:.3f}  |  |corr(W, Y)|: {corr_W_Y:.3f}  "
      f"(W should correlate more with Y)")

print("\n5. NaN / Inf Checks...")
has_nans = any(torch.isnan(batch[k]).any().item() for k in expected_keys)
has_infs = any(torch.isinf(batch[k]).any().item() for k in expected_keys)
print(f"  - Contains NaNs? {has_nans}  (expected: False)")
print(f"  - Contains Infs? {has_infs}  (expected: False)")

print("\n6. Validation Loader Smoke Test...")
val_batch = next(iter(val_loader))
print(f"  - Val batch keys:  {list(val_batch.keys())}")
print(f"  - Val X shape:     {val_batch['X'].shape}  (expected: [4, 128, 5])")

print("\nAll checks complete.")

#%%
