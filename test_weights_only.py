import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 2)
optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "epoch": 5,
}

torch.save(checkpoint, "test_checkpoint.pth")

try:
    loaded = torch.load("test_checkpoint.pth", weights_only=True)
    print("Loaded successfully with weights_only=True")
except Exception as e:
    print(f"Failed: {e}")
