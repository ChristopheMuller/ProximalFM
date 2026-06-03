#%%

from validation_proximal_mechanisms import (
    ValidationGeneratorConfig,
    ProximalValidationGenerator,
)

cfg = ValidationGeneratorConfig(
    save_dir="validation_batches",
    K=16,
    seq_len=2048,
    train_size=0.75,
    min_dim_U=1,
    max_dim_U=1,
    min_dim_X=1,
    max_dim_X=1,
    min_dim_W=1,
    max_dim_W=1,
    min_dim_Z=1,
    max_dim_Z=1,
    seed=123,
)

gen = ProximalValidationGenerator(cfg)
paths = gen.save_all_mechanisms(separate_dirs=False)
# %%
