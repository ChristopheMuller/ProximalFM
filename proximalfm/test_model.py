# %%

import torch

# Adjust these imports to match your exact directory structure
from _model.learning import ICLearning
from _model.tabicl import TabICL

print("Initializing synthetic dataloader outputs...")

# 1. Define dimensions
B = 2  # Batch size
T = 100  # Total sequence length (context + query)
train_size = 50  # Number of in-context training examples
test_size = T - train_size
d_model = 128  # Embedding dimension

# 2. Mock Dataloader Outputs
# (Assuming feature dimensions: X=5, W=3, Z=3, A=1)
X_out = torch.randn(B, T, 5)
W_out = torch.randn(B, T, 3)
Z_out = torch.randn(B, T, 3)
A_out = torch.randint(0, 2, (B, T, 1)).float()

# The model only needs the observed Y for the in-context training rows!
Y_obs_out = torch.randn(B, train_size)

print("-" * 50)
print(" PHASE 1: Testing Isolated ICLearning (Step A)")
print("-" * 50)
try:
    icl_model = ICLearning(
        d_model=d_model,
        num_blocks=2,  # Kept small for fast initialization
        nhead=4,
        dim_feedforward=256,
    )
    icl_model.eval()

    # Mock the output of RowInteraction
    R_dummy = torch.randn(B, T, d_model)

    out = icl_model(R_dummy, Y_obs_out)
    print("✅ ICLearning Forward Pass Successful!")
    print(f"   Expected Shape: ({B}, {test_size}, 4)")
    print(f"   Actual Shape:   {tuple(out.shape)}")
    assert out.shape == (B, test_size, 4), "Shape mismatch in ICLearning!"

except Exception as e:
    print(f"❌ ICLearning Test Failed:\n{e}")


print("\n" + "-" * 50)
print(" PHASE 2: Testing Full TabICL (For Step B & C)")
print("-" * 50)
try:
    # Initialize full model.
    # CRITICAL: max_classes=0 triggers the continuous regression pathways
    tabicl_model = TabICL(
        max_classes=0,
        embed_dim=d_model,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
    )
    tabicl_model.eval()

    # BEFORE STEP B IS COMPLETE:
    # TabICL still expects a single X tensor of shape (B, T, H).
    # We concatenate our features to mock this temporarily.
    X_concat = torch.cat([X_out, W_out, Z_out, A_out], dim=-1)

    # Run forward pass
    out = tabicl_model(X=X_concat, y_train=Y_obs_out)

    print("✅ TabICL Forward Pass Successful!")
    print(f"   Expected Shape: ({B}, {test_size}, 4)")
    print(f"   Actual Shape:   {tuple(out.shape)}")

    # NOTE: After you complete Step B & C, you will change this call to:
    # out = tabicl_model(X_out, W_out, Z_out, A_out, Y_obs_out)

except Exception as e:
    print(f"❌ Full TabICL Test Failed:\n{e}")

# %%
