#%% 2_run_baselines.py

import multiprocessing
multiprocessing.set_start_method("fork")  # macOS defaults to spawn; force fork

import os
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from other_methods import S_OLS, T_OLS, P2SLS

# -- Configuration --
DATA_FOLDER = "eval_data/conf_2000"
N_WORKERS = os.cpu_count() - 2  # or set manually, e.g. 8

METHODS = [
    S_OLS, 
    # S_RF, 
    # S_TabICL, 
    T_OLS, 
    P2SLS
]

#%% ── Data loading ────────────────────────────────────────────────────────────

print(f"Loading data from {DATA_FOLDER}...")
data = torch.load(os.path.join(DATA_FOLDER, "evaluation_data.pt"), weights_only=False)

N_TRAIN = data["N_TRAIN"]
N_TEST  = data["N_TEST"]
N_BATCH = data["N_BATCH"]

# Convert once; pre-slice context vs query to avoid repeated indexing in workers
X_np = data["X"].numpy()
Z_np = data["Z"].numpy()
W_np = data["W"].numpy()
A_np = data["A"].numpy()
Y_np = data["Y"].numpy()

# Build a list of per-batch dicts so each worker receives only what it needs
batches = [
    {
        "b":     b,
        "X_ctx": X_np[b, :N_TRAIN],
        "W_ctx": W_np[b, :N_TRAIN],
        "Z_ctx": Z_np[b, :N_TRAIN],
        "A_ctx": A_np[b, :N_TRAIN, 0],
        "Y_ctx": Y_np[b, :N_TRAIN, 0],
        "X_q":   X_np[b, N_TRAIN:],
        "W_q":   W_np[b, N_TRAIN:],
        "Z_q":   Z_np[b, N_TRAIN:],
    }
    for b in range(N_BATCH)
]

#%% ── Worker function ──────────────────────────────────────────────────────────

def run_batch(batch: dict) -> dict:
    """Fit + predict all methods for a single batch. Runs in a worker process."""
    b = batch["b"]
    results = {}
    for method_class in METHODS:
        name = method_class.__name__
        try:
            model = method_class()
            model.fit(
                batch["X_ctx"], batch["A_ctx"], batch["Y_ctx"],
                W=batch["W_ctx"], Z=batch["Z_ctx"],
            )
            results[name] = model.predict(batch["X_q"], W=batch["W_q"], Z=batch["Z_q"])
        except Exception as e:
            print(f"  [batch {b}] {name} failed: {e}")
            results[name] = np.full(N_TEST, np.nan)
    return b, results

# ── Parallel execution ───────────────────────────────────────────────────────

preds_dict = {m.__name__: np.zeros((N_BATCH, N_TEST)) for m in METHODS}

print(f"Running {len(METHODS)} baselines over {N_BATCH} batches "
      f"({N_WORKERS} workers)...")

with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
    futures = {pool.submit(run_batch, batch): batch["b"] for batch in batches}
    for future in tqdm(as_completed(futures), total=N_BATCH):
        b, results = future.result()
        for name, preds in results.items():
            preds_dict[name][b] = preds

# ── Save outputs ─────────────────────────────────────────────────────────────

for method in METHODS:
    name = method.__name__
    out = os.path.join(DATA_FOLDER, f"{name}_preds.npz")
    np.savez(out, preds=preds_dict[name])
    print(f"✓ Saved {name} → {out}")