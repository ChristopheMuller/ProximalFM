"""Frozen ProximalFM: input-dimension sweep and main-grid shape R-squared.

Uses the repository's CSV loader, calibrated mechanisms, and model directly.
All outputs are fresh predictions. From the repository root, run:

python rebuttal/r3mo/forward_passes/run_forward.py \
  --checkpoint /path/to/proximalfm.ckpt \
  --data-dir eval_datasets/semisynt_eval/datasets \
  --mode dimensions --output rebuttal/r3mo/forward_passes/raw/results/dimensions

For the main-grid shape scores, use --mode shape and an output ending in
raw/results/shape. Run summarize.py afterward to regenerate tables and figures.
The checkpoint and source-data revisions are recorded in provenance.json.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from eval_datasets.semisynt_eval._config import load_case_configs
from eval_datasets.semisynt_eval.build import generate_paired_episodes
from eval_datasets.semisynt_eval.source import load_csv
from proximalfm._model.proximalfm import ProximalFM


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def view(source, dx, du):
    """Keep nested, disjoint source-column prefixes in their original roles."""
    xn, un = source.x_columns[:dx], source.u_columns[:du]
    names = (*un, *xn)
    positions = [source.columns.index(c) for c in names]
    return replace(source, calibration_values=source.calibration_values[:, positions],
                   evaluation_values=source.evaluation_values[:, positions], columns=names,
                   u_indices=tuple(range(du)), x_indices=tuple(range(du, du + dx)),
                   x_columns=xn, u_columns=un,
                   preprocessing={c: source.preprocessing[c] for c in names})


def metrics(prediction, truth):
    p, t = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    variance = t.var()
    if not np.isfinite(p).all() or variance <= 0:
        raise ValueError("Predictions must be finite and the CATE must vary.")
    error = p - t
    shape = 1 - error.var() / variance
    level = error.mean() ** 2 / variance
    npehe2 = np.mean(error ** 2) / variance
    assert np.isclose(npehe2, level + 1 - shape, rtol=1e-10, atol=1e-10)
    return dict(npehe=float(np.sqrt(npehe2)), shape_r2=float(shape),
                level_error=float(level), heterogeneity_error=float(1 - shape),
                npehe_squared=float(npehe2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["dimensions", "shape"], required=True)
    p.add_argument("--sources", nargs="+")
    p.add_argument("--totals", nargs="+", type=int, default=list(range(5, 51, 5)))
    p.add_argument("--contexts", nargs="+", type=int)
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--queries", type=int, default=256)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predictions").mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    contexts = args.contexts or ([512] if args.mode == "dimensions" else [128, 512, 1024, 2048])
    totals = args.totals if args.mode == "dimensions" else [15]
    names = args.sources or (["Allstate_Claims_Severity", "superconduct", "MiniBooNE"]
                            if args.mode == "dimensions" else
                            sorted(f.stem for f in args.data_dir.glob("*.csv")))
    recipe_path = REPO / "eval_datasets/semisynt_eval/config.yaml"
    recipe = yaml.safe_load(recipe_path.read_text())
    cases = list(load_case_configs(recipe["common"], recipe["cases"]).values())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ProximalFM(**checkpoint["config"])
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v
             for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    initial_version = [param._version for param in model.parameters()]
    meta = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                checkpoint_sha256=sha256(args.checkpoint), model_config=checkpoint["config"],
                model_parameters=sum(t.numel() for t in model.parameters()),
                script_sha256=sha256(__file__), recipe_sha256=sha256(recipe_path),
                source_commit=((REPO / "SOURCE_COMMIT.txt").read_text().strip()
                               if (REPO / "SOURCE_COMMIT.txt").exists() else
                               subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()),
                host=platform.node(), torch=torch.__version__, numpy=np.__version__,
                device=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                contexts=contexts, totals=totals, sources=[],
                total_dimension_definition="d_X + d_W + d_Z, excluding treatment/outcome and latent U",
                query_rows=f"Fixed final {args.queries} rows; context is a nested prefix of the remaining pool.",
                timing="Synchronized full model.predict, inputs already on GPU, one warm-up per shape.")
    rows = []
    warmed = set()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for name in names:
        max_du = max(t // 3 for t in totals)
        max_dx = max(t - 2 * (t // 3) for t in totals)
        source = load_csv(name, data_dir=args.data_dir, n_features_x=max_dx,
                          n_features_u=max_du, n_calibration=1000, split_seed=args.seed)
        meta["sources"].append(dict(name=name, sha256=source.digest,
                                    csv_columns=len(pd.read_csv(source.path, nrows=0).columns),
                                    x_columns=list(source.x_columns), u_columns=list(source.u_columns),
                                    calibration_rows=list(source.calibration_row_indices)))
        for total in totals:
            du, dx = total // 3, total - 2 * (total // 3)
            selected = view(source, dx, du)
            pool = generate_paired_episodes(
                [selected], cases, cases, row_sampling="without_replacement",
                replicates=list(range(1, args.replicates + 1)),
                n_rows=max(contexts) + args.queries, seed=args.seed)
            for case in cases:
                _, episodes, records = pool[case.name]
                for episode, record in zip(episodes, records):
                    query = slice(max(contexts), max(contexts) + args.queries)
                    truth = episode.CATE[query].numpy().reshape(-1)
                    for n in contexts:
                        inputs = {}
                        for block in ("X", "W", "Z"):
                            x = getattr(episode, block)
                            q = x[query] if block == "X" else torch.zeros_like(x[query])
                            inputs[block] = torch.cat([x[:n], q]).unsqueeze(0).to(device)
                            inputs["d_" + block] = torch.tensor([x.shape[1]], device=device)
                        inputs["A"] = episode.A[:n].reshape(1, -1).to(device)
                        inputs["y_train"] = episode.Y_obs[:n].reshape(1, -1).to(device)
                        with torch.inference_mode():
                            if (total, n) not in warmed:
                                model.predict(**inputs, target="cate", output_type="mean")
                                sync()
                                warmed.add((total, n))
                            sync()
                            start = time.perf_counter()
                            prediction = model.predict(**inputs, target="cate", output_type="mean")
                            sync()
                            seconds = time.perf_counter() - start
                            prediction = prediction[0].cpu().numpy()
                        row = dict(source=name, case=case.name, family=case.family,
                                   confounding=case.confounding, proxy=case.proxy,
                                   replicate=record["replicate"], total_dimension=total,
                                   d_X=dx, d_W=du, d_Z=du, d_U=du, context=n,
                                   query_count=args.queries, seconds=seconds,
                                   **metrics(prediction, truth))
                        filename = f"{name}__{case.name}__r{record['replicate']}__d{total}__n{n}.npz"
                        np.savez_compressed(args.output / "predictions" / filename,
                                            prediction=prediction, truth=truth,
                                            context_rows=episode.source_row_indices[:n].numpy(),
                                            query_rows=episode.source_row_indices[query].numpy())
                        row["prediction_file"] = filename
                        rows.append(row)
            pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
            (args.output / "metadata.json").write_text(json.dumps(meta, indent=2))
            print(f"{name} total={total}: {len(rows)} forward passes saved", flush=True)
    assert initial_version == [param._version for param in model.parameters()]
    meta.update(completed=True, forward_passes=len(rows), parameters_unchanged=True,
                predictions_finite=bool(np.isfinite(pd.DataFrame(rows).npehe).all()))
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ["completed", "forward_passes", "model_parameters",
                                          "checkpoint_sha256", "parameters_unchanged"]}), flush=True)


if __name__ == "__main__":
    main()
