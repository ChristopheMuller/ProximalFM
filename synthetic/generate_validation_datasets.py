#%%

from validation_proximal_mechanisms import (
    ValidationGeneratorConfig,
    ProximalValidationGenerator,
)

cfg = ValidationGeneratorConfig(
    save_dir="new_synt_10dim",
    K=3,
    seq_len=2048,
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
paths = gen.save_all_mechanisms(separate_dirs=False)
# %%
