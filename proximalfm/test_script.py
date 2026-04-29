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
X, y, d, seq_lens, train_sizes = dataset.get_batch()

print("\nSuccess! Batch generated.")
print("--- Output Tensor Shapes ---")
print(f"X shape:           {X.shape}  -> (Batch, Sequence Length, Max Features)")
print(f"y shape:           {y.shape}  -> (Batch, Sequence Length)")
print(f"d shape:           {d.shape}  -> (Batch,) Active features per dataset")
print(f"seq_lens shape:    {seq_lens.shape}  -> (Batch,) Sequence lengths")
print(f"train_sizes shape: {train_sizes.shape}  -> (Batch,) Train split indices")
