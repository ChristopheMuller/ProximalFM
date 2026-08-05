"""The ``s/sigma_post`` read-off for a frozen set (``POSTERIOR_EVAL_NOTES.md`` 2.1).

The residual Monte Carlo noise on the target is ``t = tau* + e``,
``e ~ N(0, s^2)``. Whether that noise is worth correcting for -- i.e. whether M2
and M3 need to convolve the predictive before scoring -- depends entirely on
``s / sigma_post``, and the notes give the thresholds:

===============  =============  ==========================================
``s/sigma_post``  sd inflation   verdict
===============  =============  ==========================================
0.1               0.5 %          negligible, carry nothing
0.2               2 %            negligible
0.3               4.4 %          correct (a), (b), (c)
0.5               12 %           correct, and raise ``K``
===============  =============  ==========================================

``sigma_post`` is stood in for by today's CATE RMSE, valid because ``sd/RMSE``
tends to 1 at convergence. Read the ratio at the **largest** ``n``: the smallest
``sigma_post`` is the binding case, not the average one. And note it drifts
upward as the fine-tune sharpens the posterior, so a ``K`` that is comfortably
negligible at step 0 can become the constraint later.

Everything is reported in the frozen comparison space of the sidecar, so the
numbers are stable across ``n`` and across checkpoints.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from proximalfm.prior._genload import LoadPriorDataset

# CRPS of the best possible predictive against a target carrying noise s:
# min_P E_e[CRPS(P, tau*+e)] is attained at N(tau*, s^2) and equals
# s*(sqrt(2)-1)/sqrt(pi). Loose, but it settles whether target noise is
# anywhere near binding.
CRPS_FLOOR_COEF = (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)

# The s/sigma_post the notes size K against.
TARGET_RATIO = 0.2


def load_model(ckpt_path: Path, device: str = "cpu"):
    """Load a ProximalFM checkpoint the way the benchmark runner does."""
    from proximalfm._model.proximalfm import ProximalFM

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ProximalFM(**ckpt.get("config", {})).to(device)
    state_dict = ckpt.get("state_dict", ckpt)
    clean = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(clean, strict=False)
    model.eval()
    return model, ckpt.get("curr_step")


def iter_episodes(val_dir: Path, max_episodes: Optional[int] = None):
    """Yield frozen episodes one at a time, in the order they were built."""
    val_dir = Path(val_dir)
    n_files = len([p for p in val_dir.iterdir() if p.name.startswith("batch_")])
    loader = LoadPriorDataset(
        data_dir=str(val_dir), batch_size=1, max_batches=n_files, device="cpu"
    )
    for i, batch in enumerate(loader):
        if max_episodes is not None and i >= max_episodes:
            return
        yield batch


def _cate_rmse_at_prefix(
    model, batch, record, n: int, n_max: int, device: str
) -> float:
    """CATE RMSE in the frozen space, using the first ``n`` context rows."""
    (
        U,
        X,
        W,
        Z,
        A,
        A_probs,
        Y_obs,
        Y0,
        Y1,
        MU0,
        MU1,
        d_U,
        d_X,
        d_W,
        d_Z,
        *_rest,
    ) = batch

    def rows(t):
        t = t.to(device)
        return torch.cat([t[:, :n], t[:, n_max:]], dim=1)

    y_train = Y_obs.to(device)[:, :n].reshape(1, n)
    pred_raw = model.predict(
        X=rows(X),
        Z=rows(Z),
        W=rows(W),
        A=rows(A).reshape(1, -1),
        y_train=y_train,
        target="cate",
        output_type="mean",
        d_X=d_X.to(device),
        d_Z=d_Z.to(device),
        d_W=d_W.to(device),
    ).reshape(-1)

    # predict() returns raw-y units; divide by the *frozen* scale so every n and
    # every checkpoint land in the same comparison space (section 2, nuisance 3).
    y_std = record["y_context_std"].to(pred_raw.device)
    pred_std = pred_raw / y_std
    tau_star_std = record["tau_star_std"].to(pred_raw.device)
    return float(torch.sqrt(torch.mean((pred_std - tau_star_std) ** 2)))


def compute_report(
    out: Path,
    ckpt: Optional[Path] = None,
    device: str = "cpu",
    max_episodes: Optional[int] = 64,
) -> Dict[str, Any]:
    """Gather the section-2.1 quantities for a frozen set."""
    out = Path(out)
    meta = torch.load(out / "frozen_meta.pt", weights_only=False)
    glob = meta["global"]
    records = meta["episodes"]
    n_grid: List[int] = glob["n_grid"]
    n_max: int = glob["n_max"]
    K: int = glob["mcmc_cate"]

    # s in the frozen comparison space, pooled over all query rows.
    s_all = torch.cat([torch.sqrt(r["s2_std"].clamp_min(0)) for r in records])
    s_mean = float(s_all.mean())
    s_median = float(s_all.median())

    result: Dict[str, Any] = {
        "n_datasets": len(records),
        "mcmc_cate": K,
        "n_grid": n_grid,
        "s_mean": s_mean,
        "s_median": s_median,
        "s_p95": float(s_all.quantile(0.95)),
        "crps_floor": CRPS_FLOOR_COEF * s_mean,
        "k_eff_uniform": len({int(r["k_eff"]) for r in records}) == 1,
        "by_n": {},
    }

    if ckpt is None:
        return result

    model, step = load_model(Path(ckpt), device=device)
    result["ckpt"] = str(ckpt)
    result["ckpt_step"] = step

    # Per episode: mean squared error, and mean s^2. Variances, not sds --
    # the inflation the ratio predicts is sqrt(1 + (s/sigma_post)^2), so the
    # two quantities have to be paired on the same (squared) scale.
    per_n_mse: Dict[int, List[float]] = {n: [] for n in n_grid}
    per_n_s2: Dict[int, List[float]] = {n: [] for n in n_grid}
    with torch.inference_mode():
        for i, batch in enumerate(iter_episodes(out / "val", max_episodes)):
            record = records[i]
            s2_ep = float(record["s2_std"].clamp_min(0).mean())
            for n in n_grid:
                per_n_mse[n].append(
                    _cate_rmse_at_prefix(model, batch, record, n, n_max, device) ** 2
                )
                per_n_s2[n].append(s2_ep)

    for n in n_grid:
        if not per_n_mse[n]:
            continue
        mse = per_n_mse[n]
        s2 = per_n_s2[n]
        rmse = math.sqrt(sum(mse) / len(mse))
        s_n = math.sqrt(sum(s2) / len(s2))
        ratio = s_n / rmse if rmse > 0 else float("inf")
        # s^2 spans orders of magnitude across episodes, so the pooled ratio is
        # driven by the noisiest few. The median of the per-episode ratios says
        # what a typical episode looks like; report both and believe neither
        # alone if they disagree by much.
        per_ep = sorted(
            math.sqrt(v) / math.sqrt(e) if e > 0 else float("inf")
            for v, e in zip(s2, mse)
        )
        result["by_n"][n] = {
            "sigma_post": rmse,
            "s": s_n,
            "ratio": ratio,
            "ratio_median": per_ep[len(per_ep) // 2],
            "implied_k_star": int(math.ceil(K * (ratio / TARGET_RATIO) ** 2)),
        }
    result["episodes_scored"] = len(per_n_mse[n_grid[0]])
    return result


def _verdict(ratio: float) -> str:
    if ratio < 0.2:
        return "negligible -- carry no correction"
    if ratio < 0.3:
        return "borderline"
    if ratio < 0.5:
        return "correct 2.1 (a), (b), (c)"
    return "correct 2.1, and raise K"


def report(
    out: Path,
    ckpt: Optional[Path] = None,
    device: str = "cpu",
    max_episodes: Optional[int] = 64,
) -> Dict[str, Any]:
    """Compute and print the report; returns the same dict."""
    r = compute_report(out, ckpt=ckpt, device=device, max_episodes=max_episodes)

    print()
    print(f"Frozen set: {out}")
    print(f"  episodes         {r['n_datasets']}   K = {r['mcmc_cate']}")
    print(f"  uniform K_eff    {r['k_eff_uniform']}")
    print("  target noise s (frozen units)")
    print(
        f"    mean {r['s_mean']:.5f}   median {r['s_median']:.5f}   p95 {r['s_p95']:.5f}"
    )
    print(f"  CRPS floor = 0.2337 * mean(s) = {r['crps_floor']:.5f}")

    if not r["by_n"]:
        print("  (no --ckpt: sigma_post not estimated)")
        return r

    print(
        f"  checkpoint {r['ckpt']} (step {r['ckpt_step']}), "
        f"{r['episodes_scored']} episodes scored"
    )
    print()
    print(
        f"  {'n':>6}  {'sigma_post':>11}  {'s':>9}  {'s/sigma':>9}"
        f"  {'median':>9}  {'K* for 0.2':>11}"
    )
    for n, row in r["by_n"].items():
        print(
            f"  {n:>6}  {row['sigma_post']:>11.5f}  {row['s']:>9.5f}  "
            f"{row['ratio']:>9.4f}  {row['ratio_median']:>9.4f}  "
            f"{row['implied_k_star']:>11,}"
        )
    binding = r["by_n"][max(r["by_n"])]
    print()
    print(
        f"  binding case (largest n): s/sigma_post = {binding['ratio']:.4f}"
        f"  ->  {_verdict(binding['ratio'])}"
    )
    print(
        f"  K for s/sigma_post = {TARGET_RATIO}: {binding['implied_k_star']:,}"
        f"  (built at {r['mcmc_cate']:,})"
    )
    print()
    return r


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m posterior_eval.report",
        description="Report s/sigma_post for a frozen posterior-evaluation set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", type=Path, required=True, help="frozen set directory")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max_episodes", type=int, default=64)
    args = p.parse_args(argv)
    report(args.out, ckpt=args.ckpt, device=args.device, max_episodes=args.max_episodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
