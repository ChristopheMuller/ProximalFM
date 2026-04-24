import time
import torch
import warnings
from proximalfm.Train import CATETrainer
from proximalfm.Structure import ProximalPriorConfig, ModelConfig, ProximalEpisode
from proximalfm.PriorDataLoader import create_causal_data_loaders
import dataclasses
import yaml

warnings.filterwarnings('ignore')

# Mock yaml.dump to prevent issues
def dummy_dump(*args, **kwargs):
    pass
yaml.dump = dummy_dump

def run_bench():
    print("Setting up simple training bench for 5 epochs...")
    # Setup some minimal prior config
    prior_cfg = ProximalPriorConfig()

    # Setup a small model config
    model_cfg = ModelConfig(
        embed_dim=64,
        col_num_blocks=2,
        row_num_blocks=2,
        icl_num_blocks=2,
        col_nhead=2,
        row_nhead=2,
        icl_nhead=2,
        type="gmm"
    )

    # Train config for 5 epochs
    train_cfg = {
        "epochs": 5,
        "batch_size": 16,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "warmup_epochs": 1,
        "device": "cpu",
        "output_dir": "bench_outputs"
    }

    config = {
        "training": train_cfg,
        "prior": dataclasses.asdict(prior_cfg),
        "model": dataclasses.asdict(model_cfg),
        "data": {
            "context_size": 100,
            "test_size": 10,
            "train_episodes": 10 * 16, # Small batches
            "val_episodes": 2 * 16,
        }
    }

    loaders = create_causal_data_loaders(
        prior_config=prior_cfg,
        batch_size=train_cfg["batch_size"],
        train_episodes=config["data"]["train_episodes"],
        val_episodes=config["data"]["val_episodes"],
        min_episode_size=110,
        max_episode_size=110,
        num_workers=0
    )

    trainer = CATETrainer(config=config)
    trainer.prior_cfg = prior_cfg # Patch config object instead of dict
    trainer.train_loader = loaders['train']
    trainer.val_loader = loaders['val']

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    print(f"Total time for 5 epochs: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    run_bench()
