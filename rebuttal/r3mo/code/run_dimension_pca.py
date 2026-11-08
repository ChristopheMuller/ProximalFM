"""Released ProximalFM: original semi-synthetic widths and context-only PCA.

Allstate and superconduct are the original sources eligible for30 columns per
X/U role. Raw and PCA variants share each task's original CATE target. The
primary compression rule is m=min(k,10); other retained widths are exploratory.
No fitting step uses query rows, treatment, outcomes, latent U, or true CATE.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

SOURCES = ["Allstate_Claims_Severity.csv", "superconduct.csv"]
WIDTHS = [5, 10, 20, 30]
CASES = ["linear_conf_high_proxy_high", "nonlinear_conf_high_proxy_high"]
N_CONTEXT, N_QUERY, N_STORED = 2048, 256, 8192


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    import numpy as np
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.shape).encode() + a.dtype.str.encode() + a.tobytes()).hexdigest()


def gpu_snapshot():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def fit_pca(context):
    """No query/target argument: only the evaluated observed context block."""
    import numpy as np
    x = np.asarray(context, dtype=np.float64)
    assert x.ndim == 2 and x.shape[0] == N_CONTEXT
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    standardized = (x - mean) / scale
    _, singular, components = np.linalg.svd(standardized, full_matrices=False)
    pivots = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(len(components)), pivots])
    signs[signs == 0] = 1
    components *= signs[:, None]
    assert np.all(components[np.arange(len(components)), pivots] >= 0)
    reconstruction = (standardized @ components.T) @ components
    residual = float(np.max(np.abs(reconstruction - standardized)))
    assert residual < 1e-10 * max(1.0, float(np.max(np.abs(standardized))))
    explained = singular ** 2 / (len(x) - 1)
    ratio = explained / explained.sum() if explained.sum() > 0 else np.zeros_like(explained)
    return dict(mean=mean, scale=scale, components=components,
                explained_variance=explained, explained_ratio=ratio,
                full_rank_max_abs_reconstruction_error=residual)


def apply_pca(values, state, retained):
    import numpy as np
    return (((np.asarray(values, dtype=np.float64) - state["mean"]) / state["scale"])
            @ state["components"][:retained].T).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--support-code", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--timing-repeats", type=int, default=3)
    args = p.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    sys.path.insert(0, str(args.support_code.resolve()))
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from benchmarks.data import build_benchmark_dict
    from eval_datasets.semisynt_eval._config import load_case_configs
    from eval_datasets.semisynt_eval.build import generate_paired_episodes
    from eval_datasets.semisynt_eval.source import load_csv, dimension_view
    from released_adapter import ReleasedProximalFM
    from metrics import point_metrics, histogram_moments

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.csv").exists():
        raise FileExistsError("Existing results are not overwritten; use a new output directory")
    (args.output / "predictions").mkdir(exist_ok=True)
    (args.output / "pca_fits").mkdir(exist_ok=True)
    torch.set_num_threads(2)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.12, torch.device(args.device))
    recipe_path = args.repo / "eval_datasets/semisynt_eval/config.yaml"
    recipe = yaml.safe_load(recipe_path.read_text())
    all_cases = list(load_case_configs(recipe["common"], recipe["cases"]).values())
    selected = [next(c for c in all_cases if c.name == name) for name in CASES]
    selected_rows = np.r_[np.arange(N_CONTEXT), np.arange(N_STORED, N_STORED + N_QUERY)]
    adapter = ReleasedProximalFM(args.checkpoint, device=args.device, threads=2,
                                timing_repeats=args.timing_repeats, vendor_path=args.repo)
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "complete": False,
        "label": "New matched two-source dimension/PCA cohort; context-only normalization",
        "sources": SOURCES, "selection": "All original benchmark sources with at least60 usable columns after original calibration preprocessing; fixed before inference.",
        "widths": WIDTHS, "width_definition": "Original d_X=d_U=d_Z=d_W=k; PCA retained m applied separately to X/Z/W, latent mechanism unchanged within each task.",
        "primary_pca_rule": "m=min(k,10); k5 uses full-rank PCA5; all other m choices exploratory",
        "cases": CASES, "n_context": N_CONTEXT, "n_query": N_QUERY,
        "generated_context_pool": N_STORED, "n_calibration": 1000,
        "seed": 1234, "split_seed": 1234, "replicates": [1, 2, 3], "expected_predictions": 168,
        "normalization": "Generate original8192+256 row pool, select first2048 context plus original256 queries, then standardize X/Z/W/Y using only these2048 context rows. This differs from the older frozen3-source study's8192-row normalization.",
        "pca": "Separate observed-block context mean/SD, full NumPy SVD, loading sign canonicalized by largest absolute coordinate positive; no whitening. No treatment, outcome, U, query or truth used to fit/select components.",
        "target": "Original tau(X) held bitwise fixed within task; compressed-X causal estimand is not substituted. PCA truncation need not preserve proximal identification or effect modifiers.",
        "aggregation": "Average three replicates within each source, then weight two source means equally.",
        "inference_timing": f"Warm median of{args.timing_repeats} device-synchronized model forwards; cache cleared each pass; excludes PCA and input transfer.",
        "pca_timing": "Context mean/SD/SVD fitting for three blocks plus transform time for variant; full fit shared computationally, charged in full to every PCA variant. Excludes common source preprocessing and base benchmark normalization.",
        "gpu_before": gpu_snapshot(), "model": adapter.provenance,
        "recipe_sha256": digest(recipe_path), "runner_sha256": digest(__file__),
        "sources_metadata": [], "code_sha256": {n: digest(args.support_code / n) for n in ["released_adapter.py", "metrics.py"]},
    }
    rows, explained_rows, row_hashes = [], [], {}
    for name in SOURCES:
        print(f"Preparing {name}", flush=True)
        source = load_csv(name, data_dir=args.data_root, n_features_x=30,
                          n_features_u=30, n_calibration=1000, split_seed=1234)
        metadata["sources_metadata"].append({"name": name, "sha256": source.digest,
            "rows": len(source.calibration_row_indices) + len(source.evaluation_row_indices),
            "x_columns_30": list(source.x_columns), "u_columns_30": list(source.u_columns)})
        for k in WIDTHS:
            generated = generate_paired_episodes([dimension_view(source, k)], all_cases, selected,
                row_sampling="without_replacement", replicates=[1, 2, 3],
                n_rows=N_STORED + N_QUERY, seed=1234)
            for case in CASES:
                _, episodes, records = generated[case]
                for episode, record in zip(episodes, records):
                    rep = int(record["replicate"])
                    task = f"{Path(name).stem}_{record['family']}_k{k}_rep{rep}"
                    row_hash = record["source_row_indices_sha256"]
                    pair = (name, rep)
                    if pair in row_hashes:
                        assert row_hash == row_hashes[pair]
                    row_hashes[pair] = row_hash
                    data = build_benchmark_dict(
                        **{q: getattr(episode, q)[selected_rows].unsqueeze(0) for q in
                           ["U", "X", "W", "Z", "A", "Y_obs", "Y0", "Y1", "CATE"]},
                        **{q: torch.tensor([k]) for q in ["d_X", "d_W", "d_Z"]}, train_size=N_CONTEXT)
                    base = {q: np.asarray(data[q][0]).copy() for q in ["X", "Z", "W"]}
                    a, y = np.asarray(data["A"][0, :N_CONTEXT]).ravel().copy(), np.asarray(data["Y"][0, :N_CONTEXT]).ravel().copy()
                    truth = np.asarray(data["true_CATEs"][0]).ravel().copy()
                    hashes = {"A_sha256": array_digest(a), "Y_sha256": array_digest(y), "truth_sha256": array_digest(truth)}
                    start = time.perf_counter()
                    states = {q: fit_pca(base[q][:N_CONTEXT].copy()) for q in ["X", "Z", "W"]}
                    fit_seconds = time.perf_counter() - start
                    fit_arrays = {f"{q}_{field}": value for q, state in states.items() for field, value in state.items()}
                    # Context arrays permit direct verification of the fitted PCs.
                    fit_arrays.update({f"{q}_context": base[q][:N_CONTEXT] for q in base})
                    np.savez_compressed(args.output / "pca_fits" / f"{task}.npz", **fit_arrays)
                    for block, state in states.items():
                        for j, (variance, ratio) in enumerate(zip(state["explained_variance"], state["explained_ratio"])):
                            explained_rows.append({"task": task, "source": name, "replicate": rep,
                                "family": record["family"], "k": k, "block": block, "component": j + 1,
                                "explained_variance": variance, "explained_variance_ratio": ratio})
                    variants = [("raw", k)] + [("pca", m) for m in WIDTHS if m <= k]
                    for method, m in variants:
                        projection_seconds = 0.0
                        if method == "raw":
                            observed = {q: value.copy() for q, value in base.items()}
                            pca_seconds = 0.0
                        else:
                            start = time.perf_counter()
                            observed = {q: apply_pca(value, states[q], m) for q, value in base.items()}
                            projection_seconds = time.perf_counter() - start
                            pca_seconds = fit_seconds + projection_seconds
                        # Query proxies are not observed by the estimator.
                        observed["Z"][N_CONTEXT:] = 0
                        observed["W"][N_CONTEXT:] = 0
                        evaluation = SimpleNamespace(**observed, A=a, Y=y, tau=truth, n_context=N_CONTEXT)
                        assert {"A_sha256": array_digest(evaluation.A), "Y_sha256": array_digest(evaluation.Y),
                                "truth_sha256": array_digest(evaluation.tau)} == hashes
                        pred = adapter.predict(evaluation)
                        mean, _ = histogram_moments(pred["weights"], pred["edges"])
                        scores = point_metrics(mean, truth)
                        assert np.isclose(scores["npehe"] ** 2, scores["shape_error"] + scores["level_error"], rtol=1e-10)
                        relative = f"predictions/{task}_{method}{m}.npz"
                        np.savez_compressed(args.output / relative, prediction=mean, truth=truth, A=a, Y=y,
                            X_query=observed["X"][N_CONTEXT:], source_row_indices=episode.source_row_indices[selected_rows].numpy())
                        retained = {f"pca_{q}_explained_ratio": float(states[q]["explained_ratio"][:m].sum()) if method == "pca" else 1.0 for q in states}
                        rows.append({"task": task, "source": name, "replicate": rep, "family": record["family"],
                            "case": case, "k": k, "method": method, "retained_m": m, "n_context": N_CONTEXT,
                            "primary_pca": method == "pca" and m == min(k, 10),
                            "full_rank_pca": method == "pca" and m == k,
                            "source_row_indices_sha256": row_hash, "prediction_file": relative,
                            **hashes, **scores, **pred["performance"], **retained,
                            "pca_fit_seconds": fit_seconds if method == "pca" else 0.0,
                            "pca_projection_seconds": projection_seconds, "pca_preprocessing_seconds": pca_seconds,
                            "pca_full_rank_max_abs_reconstruction_error": max(s["full_rank_max_abs_reconstruction_error"] for s in states.values()),
                        })
                        pd.DataFrame(rows).to_csv(args.output / "results.csv", index=False)
                        print(f"[{len(rows)}/168] {task} {method}{m}: nPEHE={scores['npehe']:.4f} shape={scores['shape_r2']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    assert len(df) == 168 and not df.duplicated(["task", "method", "retained_m"]).any()
    metrics = ["npehe", "shape_r2", "level_error", "warm_forward_seconds", "pca_preprocessing_seconds",
               "pca_fit_seconds", "pca_projection_seconds", "pca_X_explained_ratio", "pca_Z_explained_ratio", "pca_W_explained_ratio"]
    if "device_peak_allocated_bytes" in df:
        metrics.append("device_peak_allocated_bytes")
    assert np.isfinite(df[metrics].to_numpy()).all()
    keys = ["family", "case", "k", "method", "retained_m", "primary_pca", "full_rank_pca"]
    source_means = df.groupby(keys + ["source"], as_index=False)[metrics].mean()
    source_means.to_csv(args.output / "source_summary.csv", index=False)
    summary = source_means.groupby(keys, as_index=False)[metrics].mean()
    summary["n_sources"], summary["n_replicates"], summary["n_tasks"] = 2, 3, 6
    summary["warm_forward_ms"] = summary["warm_forward_seconds"] * 1000
    summary["pca_preprocessing_ms"] = summary["pca_preprocessing_seconds"] * 1000
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(explained_rows).to_csv(args.output / "pca_explained_variance.csv", index=False)
    metadata.update(complete=True, n_predictions=len(df), n_base_tasks=df.task.nunique(),
        all_within_task_truth_A_Y_hashes_match=all(df.groupby("task")[c].nunique().eq(1).all() for c in hashes),
        maximum_full_rank_reconstruction_error=float(df.pca_full_rank_max_abs_reconstruction_error.max()),
        finished_utc=datetime.now(timezone.utc).isoformat(), gpu_after=gpu_snapshot())
    assert metadata["all_within_task_truth_A_Y_hashes_match"]
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
