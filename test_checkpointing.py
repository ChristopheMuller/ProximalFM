import os
import yaml
import torch
from proximalfm.Train import CATETrainer

config = {
    "training": {
        "epochs": 2,
        "save_dir": "test_save",
        "exp_name": "test_exp",
        "seed": 42,
        "early_stop": 10
    },
    "data": {
        "train_episodes": 2,
        "val_episodes": 2,
        "samples_per_episode": 10,
        "batch_size": 2
    },
    "prior": {},
    "model": {
        "embed_dim": 8,
        "col_num_blocks": 1,
        "row_num_blocks": 1,
        "icl_num_blocks": 1,
        "col_nhead": 1,
        "row_nhead": 1,
        "icl_nhead": 1,
    }
}

os.makedirs("test_save/test_exp", exist_ok=True)
with open("test_save/test_exp/test_config.yaml", "w") as f:
    yaml.dump(config, f)

trainer = CATETrainer(config)
trainer.train()

print(os.listdir("test_save/test_exp/"))

# Now resume
config["training"]["epochs"] = 4
config["training"]["start_checkpoint"] = "test_save/test_exp/model_epoch_3_final.pth"

with open("test_save/test_exp/test_config.yaml", "w") as f:
    yaml.dump(config, f)

trainer_resume = CATETrainer(config)
print("Trainer start epoch:", trainer_resume.start_epoch)
trainer_resume.train()
