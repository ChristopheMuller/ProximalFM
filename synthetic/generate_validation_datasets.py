#%%
from validation_proximal_mechanisms import (
    ValidationGeneratorConfig,
    ProximalValidationGenerator,
)
import torch

cfg = ValidationGeneratorConfig(
    save_dir="test",
    K=5,
    seq_len=512,
    train_size=0.75,
    min_dim_U=1,
    max_dim_U=10,
    min_dim_X=1,
    max_dim_X=10,
    min_dim_W=1,
    max_dim_W=10,
    min_dim_Z=1,
    max_dim_Z=10,
    seed=123,
)

gen = ProximalValidationGenerator(cfg)

mechanisms_to_compile = range(1, 19)
all_batches = []

print("Generating and collecting mechanism episodes...")
for mech_id in mechanisms_to_compile:
    batch = gen.get_batch(mechanism_id=mech_id)
    all_batches.append(batch)

unpacked = list(zip(*all_batches))
merged_tensors = [torch.cat(tensors, dim=0) for tensors in unpacked]
gen.save_batch_sparse(*merged_tensors)
# %%
