#%%

"""
evaluate.py  –  Load a ProximalFM checkpoint and run a few predictions.

Usage
-----
    python evaluate.py                              # uses best_model.pth + config.yaml
    python evaluate.py --checkpoint checkpoints/best_model.pth --config config.yaml
    python evaluate.py --episodes 10 --seed 9999   # custom number of episodes / seed
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from proximalfm.Model import PerFeatureTransformerCATE
from proximalfm.Prior import ProximalPrior
from proximalfm.Structure import ProximalPriorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> PerFeatureTransformerCATE:
    mc = cfg["model"]
    dtype = getattr(torch, mc.get("dtype", "float32"))

    model = PerFeatureTransformerCATE(
        ninp=mc["ninp"],
        nhead=mc["nhead"],
        nhid=mc["nhid"],
        nlayers=mc["nlayers"],
        layer_norm_with_elementwise_affine=mc.get("layer_norm_with_elementwise_affine", True),
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device=device, dtype=dtype)
    model.eval()

    saved_epoch = ckpt.get("epoch", "?")
    saved_val   = ckpt.get("val_loss", float("nan"))
    print(f"  Loaded checkpoint: epoch={saved_epoch}, val_loss={saved_val:.4f}")
    return model, dtype


def make_prior(cfg: dict, dtype: torch.dtype) -> ProximalPrior:
    pc = cfg["prior"]
    prior_config = ProximalPriorConfig(
        d_x=pc["d_x"],
        d_z=pc["d_z"],
        d_w=pc["d_w"],
        d_u=pc["d_u"],
        dtype=dtype,
        device="cpu",
    )
    return ProximalPrior(config=prior_config)


# ---------------------------------------------------------------------------
# Per-episode evaluation
# ---------------------------------------------------------------------------

def evaluate_episode(
    model: PerFeatureTransformerCATE,
    prior: ProximalPrior,
    episode_seed: int,
    context_fraction: float,
    samples_per_episode: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """
    Generate one episode, split into context / query, run estimate_cate,
    return a dict of scalar metrics.
    """
    ep = prior.generate_episode(batch_size=samples_per_episode, episode_seed=episode_seed)

    n_ctx = int(samples_per_episode * context_fraction)

    def _cpu(t):
        return t.to(device="cpu", dtype=dtype)

    # Context (training) rows
    X_tr = _cpu(ep.X[:n_ctx])
    Z_tr = _cpu(ep.Z[:n_ctx])
    W_tr = _cpu(ep.W[:n_ctx])
    A_tr = _cpu(ep.A[:n_ctx].unsqueeze(-1))
    Y_tr = _cpu(ep.Y[:n_ctx].unsqueeze(-1))

    # Query rows — model sees X, Z, W but NOT A or Y
    X_te = _cpu(ep.X[n_ctx:])
    Z_te = _cpu(ep.Z[n_ctx:])
    W_te = _cpu(ep.W[n_ctx:])

    # Ground truth
    true_ite = _cpu(ep.Y1[n_ctx:] - ep.Y0[n_ctx:])   # (N_q,)
    true_ate = true_ite.mean().item()

    out = model.estimate_cate(
        X_tr, Z_tr, W_tr, A_tr, Y_tr,
        X_te, Z_te, W_te,
        device=device,
        dtype=dtype,
    )

    pred_mean = out["cate_mean"].cpu()   # (N_q,)
    pred_std  = out["cate_std"].cpu()    # (N_q,)
    pred_ate  = pred_mean.mean().item()

    rmse      = math.sqrt(((pred_mean - true_ite) ** 2).mean().item())
    ate_err   = abs(pred_ate - true_ate)
    mean_std  = pred_std.mean().item()

    # 90% prediction interval coverage
    z90 = 1.645
    lo  = pred_mean - z90 * pred_std
    hi  = pred_mean + z90 * pred_std
    coverage = ((true_ite >= lo) & (true_ite <= hi)).float().mean().item()

    return {
        "seed":     episode_seed,
        "true_ate": true_ate,
        "pred_ate": pred_ate,
        "ate_err":  ate_err,
        "rmse":     rmse,
        "mean_std": mean_std,
        "cov90":    coverage,
        "pred_mean": pred_mean,
        "true_ite":  true_ite,
        "pred_std":  pred_std,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_episode_summary(r: dict, show_samples: int = 5) -> None:
    print(f"\n{'─'*60}")
    print(f"  Episode seed = {r['seed']}")
    print(f"  True ATE          : {r['true_ate']:+.4f}")
    print(f"  Predicted ATE     : {r['pred_ate']:+.4f}   (|err| = {r['ate_err']:.4f})")
    print(f"  RMSE (unit ITE)   : {r['rmse']:.4f}")
    print(f"  Mean pred. std    : {r['mean_std']:.4f}")
    print(f"  90% PI coverage   : {r['cov90']*100:.1f}%  (ideal ≈ 90%)")

    # Show a handful of individual predictions
    print(f"\n  {'Unit':>5}  {'True ITE':>9}  {'Pred mean':>10}  {'Pred std':>9}  {'in 90% PI':>10}")
    z90 = 1.645
    for i in range(min(show_samples, len(r["true_ite"]))):
        t  = r["true_ite"][i].item()
        pm = r["pred_mean"][i].item()
        ps = r["pred_std"][i].item()
        inside = "✓" if abs(t - pm) <= z90 * ps else "✗"
        print(f"  {i:>5}  {t:>+9.4f}  {pm:>+10.4f}  {ps:>9.4f}  {inside:>10}")


def print_aggregate(results: list[dict]) -> None:
    rmses     = [r["rmse"]     for r in results]
    ate_errs  = [r["ate_err"]  for r in results]
    covs      = [r["cov90"]    for r in results]
    stds      = [r["mean_std"] for r in results]

    print(f"\n{'═'*60}")
    print(f"  Aggregate over {len(results)} episodes")
    print(f"{'═'*60}")
    print(f"  RMSE (unit ITE)  mean={np.mean(rmses):.4f}  std={np.std(rmses):.4f}")
    print(f"  ATE |error|      mean={np.mean(ate_errs):.4f}  std={np.std(ate_errs):.4f}")
    print(f"  Mean pred. std   mean={np.mean(stds):.4f}  std={np.std(stds):.4f}")
    print(f"  90% PI coverage  mean={np.mean(covs)*100:.1f}%")

    # Null baseline: predict zero everywhere → RMSE = std(true ITE)
    # We can't compute this exactly without the raw ITE tensors, but we note:
    print(f"\n  💡 Null-model RMSE ≈ std(true ITE) per episode.")
    print(f"     If RMSE above is close to mean_std of pred, model is near-null.")
    print(f"{'═'*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ProximalFM quick evaluation")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--episodes",   type=int, default=5)
    parser.add_argument("--seed",       type=int, default=99999,
                        help="Base seed for eval episodes (should differ from training seeds)")
    parser.add_argument("--samples",    type=int, default=None,
                        help="Override samples_per_episode from config")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    requested = cfg["training"].get("device", "cpu")
    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif requested == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDevice : {device}")
    print(f"Checkpoint : {args.checkpoint}")

    model, dtype = load_model(args.checkpoint, cfg, device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}")

    prior = make_prior(cfg, dtype)

    samples = args.samples or cfg["data"]["samples_per_episode"]
    ctx_frac = cfg["data"].get("context_fraction", 0.75)

    # ── Run episodes ──────────────────────────────────────────────────────
    results = []
    for i in range(args.episodes):
        seed = args.seed + i
        r = evaluate_episode(
            model, prior, seed,
            context_fraction=ctx_frac,
            samples_per_episode=samples,
            device=device,
            dtype=dtype,
        )
        print_episode_summary(r, show_samples=5)
        results.append(r)

    print_aggregate(results)


if __name__ == "__main__":
    main()
# %%
