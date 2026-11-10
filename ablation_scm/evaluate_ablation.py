"""Score the variable-width SCM ablations on a common frozen validation set.

From the repository root, build the set with scripts/ablation/build_ablation_data.sh
if needed, then run `OMP_NUM_THREADS=1 python -m ablation_scm.evaluate_ablation`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from analysis.analysis_eval.posterior_calibration import (
    normalize_levels,
    resolve_dataset,
    run_calibration,
)
from eval_datasets.prior_eval.report import load_model

HERE = Path(__file__).resolve().parent
GEOMETRY = "u1_x2-3_w1-2_z1-2"
CONTEXTS = (64, 128, 256)
DEFAULT_OUT = HERE / "results" / GEOMETRY / "steps_15000" / "seed_1234"

# Training K and correction come from the launchers and run directory names.
# Checkpoints store the model config but omit these two training settings.
VARIANTS = (
    ("default", "Default (K=250, correction on)", 250, True, "loss"),
    ("k250_no_correction", "K=250, correction off", 250, False, "loss"),
    ("k16_correction", "K=16, correction on", 16, True, "loss"),
    ("k16_no_correction", "K=16, correction off", 16, False, "loss"),
    ("k1_no_correction", "K=1, correction off", 1, False, "loss"),
    ("no_role_tags", "No role tags", 250, True, "architecture"),
)
ADDITIONAL_VARIANTS = (
    ("additive_encoding", "Additive encoding", 250, True, "architecture"),
    ("joint_column_only", "Joint encoding, column only", 250, True, "architecture"),
    ("joint_icl_only", "Joint encoding, ICL only", 250, True, "architecture"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=HERE / GEOMETRY / "steps_15000" / "seed_1234"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=HERE.parent
        / "data"
        / "prior_eval"
        / f"prior_val_ablation_{GEOMETRY}_v1",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--step", type=int, default=15000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoints = {
        arm: args.run_dir / arm / f"step-{args.step:06d}.ckpt"
        for arm, *_ in (*VARIANTS, *ADDITIONAL_VARIANTS)
    }
    missing = [
        str(checkpoints[arm]) for arm, *_ in VARIANTS if not checkpoints[arm].is_file()
    ]
    if missing:
        parser.error("missing checkpoints:\n" + "\n".join(missing))
    checkpoints = {arm: path for arm, path in checkpoints.items() if path.is_file()}

    dataset = resolve_dataset(args.data_root)
    levels = normalize_levels(None)
    metrics = []
    coverages = []
    for arm, label, k, correction, group in (*VARIANTS, *ADDITIONAL_VARIANTS):
        if arm not in checkpoints:
            continue
        print(f"Evaluating {label}", flush=True)
        model, step = load_model(checkpoints[arm], device=args.device)
        if step != args.step:
            raise ValueError(
                f"{checkpoints[arm]} contains step {step}, expected {args.step}"
            )
        result = run_calibration(
            model, dataset, contexts=CONTEXTS, levels=levels, device=args.device
        )
        metadata = dict(
            arm=arm, variant=label, k_train=k, correction=correction, group=group
        )
        metrics.append(result.metrics.assign(**metadata))
        coverages.append(result.coverage.assign(**metadata))

    by_context = pd.concat(metrics, ignore_index=True)
    by_context = by_context.rename(
        columns={
            "point/cate/npehe": "npehe",
            "point/cate/r2_shape": "r2_shape",
            "cov90_pooled": "coverage90",
            "width90": "mean_width90",
        }
    )
    summary_columns = [
        "variant",
        "n_ctx",
        "npehe",
        "r2_shape",
        "crps",
        "coverage90",
        "mean_width90",
    ]
    summary = by_context.loc[by_context["n_ctx"].eq(max(CONTEXTS)), summary_columns]
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "performance_table.csv", index=False)
    by_context.to_csv(out / "metrics_by_context.csv", index=False)
    pd.concat(coverages, ignore_index=True).to_csv(out / "coverage.csv", index=False)
    provenance = {
        "dataset_root": str(dataset.root),
        "dataset_content_digest": dataset.content_digest,
        "episodes": dataset.n_episodes,
        "query_rows_per_context": int(by_context["eval_rows"].iloc[0]),
        "contexts": CONTEXTS,
        "checkpoint_step": args.step,
        "checkpoints": {arm: str(path.resolve()) for arm, path in checkpoints.items()},
        "device": args.device,
        "torch_version": torch.__version__,
    }
    (out / "run.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
