#%% 1_generate_data.py
import yaml
import torch
import os
import datetime
import numpy as np
from proximalfm.Structure import ProximalPriorConfig
from proximalfm.PriorDataLoader import create_causal_data_loaders

#%%
# -- Parameters --
N_TRAIN = 2000
N_TEST = 100
N_BATCH = 1000
# name_save = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
name_save = "conf_2000"
OUTPUT_FOLDER = f"eval_data/{name_save}/"

def load_config(config_path: str = "config.yaml") -> dict:
    # with open(config_path, "r") as f:
    #     return yaml.safe_load(f)["prior"]
    return {
        'type': 'test',
        'd_x': 1,
        'd_u': 2,
        'd_z': 2,
        'd_w': 2,
        'noise_scale_w': 1,
        'noise_scale_z': 1,
        'noise_scale_x': 1,
        'temperature': 1,
        'confounding_strength': 3.0
    }

#%%

print("Loading configuration...")
prior_cfg = load_config()

# save prior config in output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
with open(os.path.join(OUTPUT_FOLDER, "prior_config.yaml"), "w") as f:
    yaml.dump(prior_cfg, f)

dtype = getattr(torch, prior_cfg.get("dtype", "float32"))

samples_per_episode = N_TRAIN + N_TEST

prior_kwargs = prior_cfg.copy()
prior_kwargs["dtype"] = dtype
prior_config = ProximalPriorConfig(**prior_kwargs)

#%%
print(f"Generating {N_BATCH} datasets...")
loaders = create_causal_data_loaders(
    prior_config=prior_config,
    batch_size=N_BATCH,
    samples_per_episode=samples_per_episode,
    train_episodes=1,            
    val_episodes=N_BATCH,     
    num_workers=0,
    seed=np.random.randint(0, 10000)
)

batch = next(iter(loaders["val"]))

x_full = batch["X"].transpose(0, 1).cpu()
z_full = batch["Z"].transpose(0, 1).cpu()
w_full = batch["W"].transpose(0, 1).cpu()
a_full = batch["A"].transpose(0, 1).cpu()
y_full = batch["Y"].transpose(0, 1).cpu()

true_ITEs = batch["ITE"][N_TRAIN:, :, 0].transpose(0, 1).cpu().numpy()

data_to_save = {
    "X": x_full,
    "Z": z_full,
    "W": w_full,
    "A": a_full,
    "Y": y_full,
    "true_ITEs": true_ITEs,
    "N_TRAIN": N_TRAIN,
    "N_TEST": N_TEST,
    "N_BATCH": N_BATCH
}

torch.save(data_to_save, os.path.join(OUTPUT_FOLDER, "evaluation_data.pt"))
print(f"✓ Saved generated datasets to {OUTPUT_FOLDER}")