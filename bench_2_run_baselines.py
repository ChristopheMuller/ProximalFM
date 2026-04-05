#%% 2_run_baselines.py
import os
import torch
import numpy as np
from tqdm import tqdm
from other_methods import S_OLS, T_OLS, P2SLS

# -- Configuration --
DATA_FOLDER = "eval_data/test"
OUTPUT_PREDS = "_preds.npz"

methods_to_compare = [
    S_OLS,
    T_OLS,
    P2SLS,
]

#%%

print(f"Loading data from {DATA_FOLDER}...")
data = torch.load(os.path.join(DATA_FOLDER, "evaluation_data.pt"), weights_only=False)

N_TRAIN = data["N_TRAIN"]
N_TEST = data["N_TEST"]
N_BATCH = data["N_BATCH"]

X_np = data["X"].numpy()
Z_np = data["Z"].numpy()
W_np = data["W"].numpy()
A_np = data["A"].numpy()
Y_np = data["Y"].numpy()

preds_dict = {method.__name__: np.zeros((N_BATCH, N_TEST)) for method in methods_to_compare}

print("Running Baseline Models...")
for b in tqdm(range(N_BATCH)):

    X_ctx = X_np[b, :N_TRAIN]
    W_ctx = W_np[b, :N_TRAIN]
    Z_ctx = Z_np[b, :N_TRAIN]
    A_ctx = A_np[b, :N_TRAIN, 0]
    Y_ctx = Y_np[b, :N_TRAIN, 0]
    
    X_q = X_np[b, N_TRAIN:]
    W_q = W_np[b, N_TRAIN:]
    Z_q = Z_np[b, N_TRAIN:]
    
    for method_class in methods_to_compare:
        name = method_class.__name__
        model_inst = method_class()
        
        try:
            model_inst.fit(X_ctx, A_ctx, Y_ctx, W=W_ctx, Z=Z_ctx)
            preds_dict[name][b] = model_inst.predict(X_q, W=W_q, Z=Z_q)
        except Exception as e:
            print(f"Error in {name} on episode {b}: {e}")
            preds_dict[name][b] = np.nan
        

for meth in methods_to_compare:
    print(f"✓ Completed predictions for {meth.__name__}")
    output_file = os.path.join(DATA_FOLDER, f"{meth.__name__}_preds.npz")
    np.savez(output_file, preds=preds_dict[meth.__name__])

