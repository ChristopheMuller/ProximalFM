#%%

from prior._dataset import PriorDataset

#%%

print("Initializing Proximal PriorDataset...")

dataset = PriorDataset(
    batch_size=8,  # Small batch size for quick testing
    batch_size_per_gp=4,
    min_dim_U=3,
    max_dim_U=5,  # Unmeasured confounder
    min_dim_W=4,
    max_dim_W=6,  # Outcome-inducing proxy
    min_dim_Z=4,
    max_dim_Z=6,  # Treatment-inducing proxy
    min_dim_X=10,
    max_dim_X=15,  # Baseline covariates (Legacy generic features)
    min_seq_len=500,
    max_seq_len=1000,
    prior_type="mlp_scm",
    device="cpu",
)

print("\nDataset configured:")
print(dataset)

print("\nGenerating a batch of data...")
# Fetch a batch
X, W, Z, A, Y_obs, Y0, Y1, d_X, d_W, d_Z, seq_lens, train_sizes = dataset.get_batch()

print("\nSuccess! Batch generated.")
print("--- Output Tensor Shapes ---")
print(f"X shape:           {X.shape}  -> (Batch, Sequence Length, Max Features)")
print(f"W shape:           {W.shape}  -> (Batch, Sequence Length, Max Features)")
print(f"Z shape:           {Z.shape}  -> (Batch, Sequence Length, Max Features)")
print(f"A shape:           {A.shape}  -> (Batch, Sequence Length, Max Features)")
print(f"Y_obs shape:      {Y_obs.shape}  -> (Batch, Sequence Length, Max Features)")
print(f"Y0 shape:        {Y0.shape}  -> (Batch, Sequence Length, Max Features)")
print(f"Y1 shape:        {Y1.shape}  -> (Batch, Sequence Length, Max Features)")
print(f"d_X shape:       {d_X.shape}  -> (Batch,) Active features per dataset")
print(f"d_W shape:       {d_W.shape}  -> (Batch,) Active features per dataset")
print(f"d_Z shape:       {d_Z.shape}  -> (Batch,) Active features per dataset")
print(f"seq_lens shape:    {seq_lens.shape}  -> (Batch,) Sequence lengths")
print(f"train_sizes shape: {train_sizes.shape}  -> (Batch,) Train split indices")

#%%

print("\nX:")
print(X[0, :, : d_X[0]])

print("\nW:")
print(W[0, :, : d_W[0]])

print("\nZ:")
print(Z[0, :, : d_Z[0]])

print("\nA:")
print(A[0, :20])

print("\nY_obs:")
print(Y_obs[0, :20])

print("\nY0:")
print(Y0[0, :20])

print("\nY1:")
print(Y1[0, :20])

#%%
