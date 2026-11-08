"""Compare CATE recovery across dimension choices and nested context sizes.

Run from the repo root with python -m analysis.ablation_scm_verification.context_diagnostic.
"""

from itertools import product

import numpy as np
import torch

from analysis.ablation_scm_verification.continuous_diagnostics import estimate
from proximalfm.prior._ablation_dataset import AblationSCMPrior


CONTEXT_SIZES = (10, 32, 64, 96, 128, 192, 256, 384, 512)
METHODS = ("X only", "OLS with proxies", "Bridge 2SLS", "Oracle U")
GEOMETRIES = [
    (u, x, w, z)
    for u in (1, 2, 3)
    for x, w, z in product(range(1, 5), range(u, 5), range(u, 5))
]
PILOTS = ((1, 1, 1, 1), (1, 2, 2, 2), (2, 2, 3, 3), (3, 4, 4, 4))


def check(prior, geometry, seeds):
    du, dx, dw, dz = geometry
    queries, k = 64, 2048
    n_max = CONTEXT_SIZES[-1]
    # Columns are nPEHE and centered shape R². A rank-deficient fit is a failure.
    scores = np.empty((len(seeds), len(CONTEXT_SIZES), len(METHODS), 2))
    for i, seed in enumerate(seeds):
        torch.manual_seed(seed)
        episode = prior.generate_dataset(
            dict(
                dim_U=du,
                max_dim_U=du,
                dim_X=dx,
                max_dim_X=dx,
                dim_W=dw,
                max_dim_W=dw,
                dim_Z=dz,
                max_dim_Z=dz,
                seq_len=n_max + queries,
                train_size=n_max,
                mc_cate=k,
                mc_targets_on_context=False,
            )
        )
        u, x, w, z, a, _, y, _, _, mu0, mu1 = (t.numpy() for t in episode[:11])
        target = (mu1 - mu0)[n_max:]
        target_var = np.var(target, dtype=np.float64)
        for j, n in enumerate(CONTEXT_SIZES):
            observed = (v[:n] for v in (x, w, z, u, a, y))
            args = (*observed, x[n_max:])
            for m, method in enumerate(METHODS):
                error = estimate(*args, method=method) - target
                if np.isfinite(error).all():
                    scores[i, j, m] = (
                        np.sqrt(np.mean(error**2) / target_var),
                        1 - np.var(error) / target_var,
                    )
                else:
                    scores[i, j, m] = (np.inf, -np.inf)
    return scores


def summary(label, scores):
    print(f"\n{label} ({len(scores)} episodes)")
    print(
        "Rows  P2SLS nPEHE   P2SLS nPEHE ≤ .5  P2SLS shape R² > 0  Oracle nPEHE ≤ .5  OLS+W,Z nPEHE"
    )
    for n, values in zip(CONTEXT_SIZES, scores.transpose(1, 0, 2, 3)):
        proxy_ols, bridge, oracle = values[:, 1], values[:, 2], values[:, 3]
        print(
            f"{n:4d}  {np.median(bridge[:, 0]):11.2f}   "
            f"{np.mean(bridge[:, 0] <= 0.5):16.0%}  {np.mean(bridge[:, 1] > 0):18.0%}  "
            f"{np.mean(oracle[:, 0] <= 0.5):18.0%}  {np.median(proxy_ols[:, 0]):13.2f}"
        )


def main():
    torch.set_num_threads(1)
    prior = AblationSCMPrior(device="cpu", n_jobs=1)
    repeats = 24
    sweep = {}
    for i, geometry in enumerate(GEOMETRIES):
        start = 702_000 + repeats * i
        sweep[geometry] = check(prior, geometry, range(start, start + repeats))

    summary("All valid geometries", np.concatenate(list(sweep.values())))
    for du in (1, 2, 3):
        summary(
            f"d_U={du}", np.concatenate([s for g, s in sweep.items() if g[0] == du])
        )
    for dx in (1, 2, 3, 4):
        summary(
            f"d_X={dx}", np.concatenate([s for g, s in sweep.items() if g[1] == dx])
        )

    for i, geometry in enumerate(PILOTS):
        start = 780_000 + 128 * i
        summary(
            f"(d_U,d_X,d_W,d_Z)={geometry}",
            check(prior, geometry, range(start, start + 128)),
        )


if __name__ == "__main__":
    main()
