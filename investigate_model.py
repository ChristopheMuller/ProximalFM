#%%

import torch
from proximalfm.Model import PerFeatureTransformerCATE
from proximalfm.Structure import ProximalPriorConfig
from proximalfm.Prior import ProximalPrior

#%%
def main():
    # 1. Setup device and types
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32

    # 2. Instantiate the model
    model_kwargs = {
        "ninp": 16,
        "nhead": 1,
        "nhid": 32,
        "nlayers": 4,
        "layer_norm_with_elementwise_affine": True,
    }
    
    print("Initializing model...")
    model = PerFeatureTransformerCATE(**model_kwargs).to(device=device, dtype=dtype)

    # 3. Load the checkpoint
    checkpoint_path = "checkpoints/best_model.pth" # <-- Update this path
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Handle cases where the checkpoint is just a state dict vs a full dict saving epoch/optimizer
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Running with untrained weights for demonstration.")
    
    model.eval()

    # 4. Generate simple synthetic data for inference
    # We use the prior to guarantee the data aligns with the expected causal structures
    prior_config = ProximalPriorConfig(
        d_x=8, d_z=6, d_w=6, d_u=4, 
        device="cpu", dtype=dtype
    )
    prior = ProximalPrior(config=prior_config, seed=42)
    
    # Generate 100 samples
    total_samples = 100
    episode = prior.generate_episode(batch_size=total_samples)

    # 5. Split into "Context" (Train) and "Query" (Test)
    # The transformer uses the context rows to infer the causal structure, 
    # and predicts the CATE for the query rows.
    n_context = 80
    
    # Context data (Fully observed)
    X_context = episode.X[:n_context].to(device)
    Z_context = episode.Z[:n_context].to(device)
    W_context = episode.W[:n_context].to(device)
    A_context = episode.A[:n_context].unsqueeze(-1).to(device) # Shape [N, 1]
    Y_context = episode.Y[:n_context].unsqueeze(-1).to(device) # Shape [N, 1]

    # Query data (Proxies and covariates observed; A and Y unknown/to-be-inferred)
    X_query = episode.X[n_context:].to(device)
    Z_query = episode.Z[n_context:].to(device)
    W_query = episode.W[n_context:].to(device)

    # True ITE for evaluating the prediction
    true_ite_query = (episode.Y1 - episode.Y0)[n_context:].to(device)

    # 6. Run Inference
    print(f"\nRunning inference on {total_samples - n_context} query data points using {n_context} context data points...")
    with torch.no_grad():
        predictions = model.estimate_cate(
            X_train=X_context, Z_train=Z_context, W_train=W_context, A_train=A_context, Y_train=Y_context,
            X_test=X_query, Z_test=Z_query, W_test=W_query,
            device=device, dtype=dtype
        )

    # 7. Display Results
    pred_mean = predictions["cate_mean"]
    pred_std = predictions["cate_std"]

    print("\n--- Inference Results ---")
    for i in range(5):  # Print first 5 results
        print(f"Query {i+1}:")
        print(f"  Predicted CATE: {pred_mean[i].item():.4f} (± {pred_std[i].item():.4f})")
        print(f"  True ITE:       {true_ite_query[i].item():.4f}")
        print(f"  Absolute Error: {abs(pred_mean[i].item() - true_ite_query[i].item()):.4f}\n")

if __name__ == "__main__":
    main()