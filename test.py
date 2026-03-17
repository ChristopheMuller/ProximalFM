

#%%

from proximalfm.prior import ProximalPrior
import torch
import numpy as np
import pandas as pd
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