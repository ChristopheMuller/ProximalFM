# %%
# IMPORTS
####

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from filelock import FileLock

from analysis.analysis_prior import generate as prior_generation
from analysis.analysis_prior.config import CACHE_FORMAT_VERSION, PriorSpec

# %%
# CONFIG
####

dim_X = 0
dimensions = (1, 3)
bin_counts = (2, 4, 8, 16, 32, 64)
n_samples = 50_000
n_scms = 20
seed = 0
checkpoint_dir = (
    Path(prior_generation.__file__).resolve().parents[2]
    / "rebuttal/results/completeness_scm"
)
CHECKPOINT_VERSION = 1

# %%
# FUNCTIONS
####


def median_split(block, rows):
    """Choose the most balanced axis split that keeps tied values together."""
    split = None
    best_balance = 0
    for column in block.T:
        ordered_rows = rows[np.argsort(column[rows])]
        values = column[ordered_rows]
        cuts = np.flatnonzero(values[1:] != values[:-1]) + 1
        if not cuts.size:
            continue
        cut = cuts[np.abs(2 * cuts - len(rows)).argmin()]
        balance = min(cut, len(rows) - cut)
        if balance > best_balance:
            split = ordered_rows[:cut], ordered_rows[cut:]
            best_balance = balance
    return split


def discretize(block, bin_counts):
    """Split the largest remaining bin and reuse that partition at finer k."""
    leaves = [np.arange(len(block))]
    unsplittable = set()
    partitions = {}
    for k in sorted(bin_counts):
        while len(leaves) < k:
            candidates = (i for i in range(len(leaves)) if i not in unsplittable)
            index = max(candidates, key=lambda i: len(leaves[i]), default=None)
            if index is None:
                break
            split = median_split(block, leaves[index])
            if split is None:
                unsplittable.add(index)
            else:
                leaves[index], right = split
                leaves.append(right)

        codes = np.empty(len(block), dtype=np.intp)
        for index, rows in enumerate(leaves):
            codes[rows] = index
        partitions[k] = codes, len(leaves)
    return partitions


def probability_matrix(rows, cols, n_rows, n_cols):
    """Estimate P(column category | row category), or return None for an empty row."""
    counts = np.bincount(rows * n_cols + cols, minlength=n_rows * n_cols)
    counts = counts.reshape(n_rows, n_cols)
    totals = counts.sum(axis=1, keepdims=True)
    if not totals.all():
        return None
    return counts / totals


def latent_spectrum(matrix, n_u):
    """Pad the SVD with zeros when the proxy has fewer categories than U."""
    if matrix is None:
        return None
    values = np.linalg.svd(matrix, compute_uv=False)
    return np.pad(values, (0, n_u - len(values)))


def analyze_scm(spec, index):
    sample = prior_generation.generate(spec, indices=[index], progress=False)[0]
    dim = spec.min_dim_U
    if (sample.U.shape[1], sample.W.shape[1], sample.Z.shape[1]) != (dim,) * 3:
        print(f"d={dim}, SCM {index}: skipped after constant-column filtering")
        return []

    u_bins, w_bins, z_bins = (
        discretize(block, bin_counts) for block in (sample.U, sample.W, sample.Z)
    )
    rows = []
    for arm in (0, 1):
        mask = sample.A == arm
        for k in bin_counts:
            u, n_u = u_bins[k]
            w, n_w = w_bins[k]
            z, n_z = z_bins[k]
            p_u_given_z = probability_matrix(z[mask], u[mask], n_z, n_u)
            p_w_given_u = probability_matrix(u[mask], w[mask], n_u, n_w)
            rows.append(
                {
                    "dim": dim,
                    "k": k,
                    "scm": index,
                    "arm": arm,
                    "n_rows": int(mask.sum()),
                    "p_u_given_z": p_u_given_z,
                    "p_w_given_u": p_w_given_u,
                    "sv_z": latent_spectrum(p_u_given_z, n_u),
                    "sv_w": latent_spectrum(p_w_given_u, n_u),
                }
            )
    return rows


def save_checkpoint(path, checkpoint):
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(checkpoint, file, default=np.ndarray.tolist, separators=(",", ":"))
    tmp.replace(path)


# %%
# RUN
####


def run():
    """Resume this configuration's sweep or load its saved results."""
    if dim_X != 0:
        raise ValueError("This experiment requires dim_X = 0")
    if not bin_counts or min(bin_counts) < 2:
        raise ValueError("k must be at least 2")

    specs = [
        PriorSpec(
            name="rebuttal_completeness_scm",
            n_datasets=n_scms,
            seq_len=n_samples,
            mc_cate=1,
            seed=seed,
            min_dim_X=dim_X,
            max_dim_X=dim_X,
            min_dim_U=dim,
            max_dim_U=dim,
            min_dim_W=dim,
            max_dim_W=dim,
            min_dim_Z=dim,
            max_dim_Z=dim,
        )
        for dim in dimensions
    ]
    config = {
        "version": CHECKPOINT_VERSION,
        "prior_version": CACHE_FORMAT_VERSION,
        "prior_specs": [spec.to_dict() for spec in specs],
        "bin_counts": list(bin_counts),
    }
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{config_id}.json"

    with FileLock(path.with_suffix(".lock")):
        if path.exists():
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
            if checkpoint["config"] != config:
                raise ValueError(f"Checkpoint configuration mismatch in {path}")
        else:
            checkpoint = {"config": config, "completed": [], "results": []}

        completed = {tuple(key) for key in checkpoint["completed"]}
        print(f"Checkpoint {path} ({len(completed)}/{len(specs) * n_scms} SCMs done)")
        for spec in specs:
            for index in range(spec.n_datasets):
                key = spec.min_dim_U, index
                if key in completed:
                    continue
                checkpoint["results"].extend(analyze_scm(spec, index))
                checkpoint["completed"].append(key)
                save_checkpoint(path, checkpoint)

    for row in checkpoint["results"]:
        for field in ("p_u_given_z", "p_w_given_u", "sv_z", "sv_w"):
            if row[field] is not None:
                row[field] = np.asarray(row[field])
    return checkpoint["results"]


# %%
# RUN THE EXPERIMENT
#####

results = run()

# %%
# PLOT THE RESULTS
#####


def plot_results(results, n_scms, dimensions, bin_counts):
    by_arm = {(r["dim"], r["k"], r["scm"], r["arm"]): r for r in results}
    fig, axes = plt.subplots(
        2,
        len(bin_counts),
        figsize=(3.3 * len(bin_counts), 6),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for proxy_axes, field in zip(axes, ("sv_z", "sv_w"), strict=True):
        for ax, k in zip(proxy_axes, bin_counts, strict=True):
            for color_index, dim in enumerate(dimensions):
                smallest = []
                for scm in range(n_scms):
                    arm_0 = by_arm.get((dim, k, scm, 0), {}).get(field)
                    arm_1 = by_arm.get((dim, k, scm, 1), {}).get(field)
                    if arm_0 is not None and arm_1 is not None:
                        smallest.append(min(arm_0[-1], arm_1[-1]))

                label = f"d={dim} ({len(smallest)}/{n_scms})"
                color = f"C{color_index}"
                if smallest:
                    ax.ecdf(smallest, compress=True, color=color, label=label)
                else:
                    ax.plot([], [], color=color, label=label)
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left", fontsize="small", frameon=False)

    for ax, k in zip(axes[0], bin_counts, strict=True):
        ax.set_title(f"k = {k}")
    axes[0, 0].set_ylabel("Z proxy\nCumulative fraction")
    axes[1, 0].set_ylabel("W proxy\nCumulative fraction")
    axes[0, 0].set_xscale("symlog", linthresh=1e-8)
    axes[0, 0].set_xlim(left=0)
    axes[0, 0].set_ylim(0, 1.02)
    fig.supxlabel("Smallest singular value across treatment arms")
    return fig


fig = plot_results(results, n_scms, dimensions, bin_counts)
plt.show()
