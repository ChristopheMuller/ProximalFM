"""Run a fixed-geometry check of regression bias and bridge recovery.

From the repo root, run python -m analysis.ablation_scm_verification.continuous_diagnostics.
"""

import numpy as np
import torch

from proximalfm.prior._ablation_dataset import AblationSCMPrior


def estimate(x, w, z, u, a, y, x_query, method):
    # Center adjustment variables on the whole context. The resulting contrasts
    # average over the pre-treatment context distribution of the proxies or U.
    a = a[:, None]
    base = np.column_stack((np.ones_like(x), x, a, a * x))
    if method == "X only":
        regressors = base
    elif method == "OLS with proxies":
        proxies = np.column_stack((w - w.mean(0), z - z.mean(0)))
        regressors = np.column_stack((base, proxies, a * proxies))
    elif method == "Oracle U":
        u = u - u.mean(0)
        regressors = np.column_stack((base, u, a * u))
    else:
        # Estimate the signal subspaces from context only. Use the known latent
        # width for this diagnostic. The fitted coefficients never use U.
        w, z = w - w.mean(0), z - z.mean(0)
        left, _, right = np.linalg.svd(w.T @ z, full_matrices=False)
        rank = u.shape[1]
        ws, zs = w @ left[:, :rank], z @ right[:rank].T
        regressors = np.column_stack((base, ws, a * ws))
        instruments = np.column_stack((base, zs, a * zs))
        regressors = (
            instruments @ np.linalg.lstsq(instruments, regressors, rcond=None)[0]
        )
    coefficients = np.linalg.lstsq(regressors, y, rcond=None)[0]
    return coefficients[2] + coefficients[3] * x_query[:, 0]


def main():
    torch.set_num_threads(1)
    context, large_context, queries, k = 256, 30_000, 64, 20_000
    dims = dict(dim_U=2, max_dim_U=2, dim_W=3, max_dim_W=3, dim_Z=4, max_dim_Z=4)
    prior = AblationSCMPrior(device="cpu", n_jobs=1)
    methods = ("X only", "OLS with proxies", "Bridge 2SLS", "Oracle U")
    results = {(n, method): [] for n in (context, large_context) for method in methods}
    small_arms, mc_se, outside_support = [], [], 0
    for seed in range(601_000, 601_048):
        torch.manual_seed(seed)
        episode = prior.generate_dataset(
            dict(
                **dims,
                seq_len=large_context + queries,
                train_size=large_context,
                mc_cate=k,
                mc_targets_on_context=False,
            )
        )
        u, x, w, z, a, _, y, _, _, mu0, mu1 = [t.numpy() for t in episode[:11]]
        target = (mu1 - mu0)[-queries:]
        mc_se.extend(episode[15][-queries:].sqrt().tolist())
        small_arms.append(min(a[:context].sum(), context - a[:context].sum()))
        outside_support += np.count_nonzero(np.abs(target / y[:context].std()) > 5)
        for n in (context, large_context):
            observed = (v[:n] for v in (x, w, z, u, a, y))
            args = (*observed, x[-queries:])
            for method in methods:
                error = estimate(*args, method=method) - target
                rmse = np.sqrt(np.mean(error**2))
                shape_r2 = 1 - np.var(error) / np.var(target)
                results[n, method].append((rmse, shape_r2))

    print("48 SCMs with (d_X,d_U,d_W,d_Z) = (1,2,3,4); K = 20,000")
    print(
        f"Reference MC SE median {np.median(mc_se):.4f}; "
        f"smallest arm at {context} rows: {min(small_arms)}; "
        f"labels outside normalized [-5, 5]: {outside_support}/{len(mc_se)}"
    )
    print("Rows  Method            Median RMSE  Median shape R²")
    for (n, method), errors in results.items():
        rmse, shape = np.median(errors, axis=0)
        print(f"{n:5} {method:17} {rmse:11.3f} {shape:16.3f}")


if __name__ == "__main__":
    main()
