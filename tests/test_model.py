import torch
import pytest

# Adjust this import path based on your actual project structure
from proximalfm._model import TabICL


@pytest.fixture
def setup_model_and_data():
    """
    Fixture to initialize a minimal TabICL model and synthetic data.
    """
    # Dimensions
    B = 2  # Batch size (number of tables)
    T = 15  # Total sequence length (rows per table)
    train_size = 10  # Number of training samples

    H_X = 5  # Baseline covariates
    H_Z = 3  # Treatment proxies
    H_W = 4  # Outcome proxies
    H_A = 1  # Treatment variable

    # Initialize a miniature version of TabICL for fast testing
    model = TabICL(
        embed_dim=16,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=4,
        icl_num_blocks=1,
        icl_nhead=2,
    )

    # Generate synthetic input tensors
    X = torch.randn(B, T, H_X)
    Z = torch.randn(B, T, H_Z)
    W = torch.randn(B, T, H_W)
    A = torch.randn(B, T, H_A)

    # y_train only covers the first `train_size` rows
    y_train = torch.randn(B, train_size)

    # Simulate dynamic feature counts (assuming full capacity for simplicity here)
    d_X = torch.full((B,), H_X, dtype=torch.long)
    d_Z = torch.full((B,), H_Z, dtype=torch.long)
    d_W = torch.full((B,), H_W, dtype=torch.long)

    return model, (X, Z, W, A, y_train, d_X, d_Z, d_W), (B, T, train_size)


def test_tabicl_training_mode(setup_model_and_data):
    """
    Validates the standard forward pass while model is in .train() mode.
    """
    model, inputs, dims = setup_model_and_data
    B, T, train_size = dims
    test_size = T - train_size

    model.train()

    # Execute forward pass
    out = model(*inputs)

    # Check shape: (Batch, Test_Size, Out_Dim)
    # Note: Out_Dim is currently hardcoded to 4 in your ICLearning module
    expected_shape = (B, test_size, 4)
    assert (
        out.shape == expected_shape
    ), f"Training output shape mismatch. Expected {expected_shape}, got {out.shape}"
    assert not torch.isnan(out).any(), "NaNs detected in training output."


def test_tabicl_padding_logic(setup_model_and_data):
    """
    Validates that the model can handle a batch where datasets have fewer
    features than the maximum allocated tensor size.
    """
    model, inputs, dims = setup_model_and_data
    X, Z, W, A, y_train, _, _, _ = inputs
    B, T, train_size = dims
    test_size = T - train_size

    # artificially lower the number of valid features for the batch
    d_X_reduced = torch.tensor([X.shape[2] - 1, X.shape[2] - 2], dtype=torch.long)
    d_Z_reduced = torch.tensor([Z.shape[2] - 1, Z.shape[2] - 1], dtype=torch.long)
    d_W_reduced = torch.tensor([W.shape[2] - 2, W.shape[2] - 1], dtype=torch.long)

    model.train()
    out = model(X, Z, W, A, y_train, d_X=d_X_reduced, d_Z=d_Z_reduced, d_W=d_W_reduced)

    expected_shape = (B, test_size, 4)
    assert (
        out.shape == expected_shape
    ), "Model failed to handle reduced feature counts properly."
