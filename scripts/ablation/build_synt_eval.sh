#!/usr/bin/env bash
# Build the realistic-mechanism validation set for the small ablation runs.
# Use a different NAME and SEED for the held-out test set.
# The realistic mechanisms require at least one covariate.

set -euo pipefail

# shellcheck source=scripts/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"

NAME=${NAME:-synt_val_ablation_512_v1}
OUT_ROOT=${OUT_ROOT:-data/synt_eval}

N_DATASETS=${N_DATASETS:-240}
N_MAX=${N_MAX:-512}
N_Q=${N_Q:-16}
K=${K:-10000}
SEED=${SEED:-20260807}
MEM_BUDGET=${MEM_BUDGET:-8000000000}

OUT="${OUT_ROOT}/${NAME}"

echo "Building realistic replicate set:"
echo "  out           = ${OUT}"
echo "  n_datasets    = ${N_DATASETS}   (spread evenly over the 12 mechanisms)"
echo "  n_max/n_q/K   = ${N_MAX}/${N_Q}/${K}   (rows per dataset: $((N_MAX + N_Q * K)))"
echo "  seed          = ${SEED}"
echo

python -m eval_datasets.synt_eval.build_proximal_datasets \
  --name "${NAME}" \
  --out_root "${OUT_ROOT}" \
  --n_datasets "${N_DATASETS}" \
  --n_max "${N_MAX}" \
  --n_q "${N_Q}" \
  --mc_cate "${K}" \
  --seed "${SEED}" \
  --mc_mem_budget "${MEM_BUDGET}" \
  --min_dim_U 1 --max_dim_U 3 \
  --min_dim_X 1 --max_dim_X 4 \
  --min_dim_W 1 --max_dim_W 4 \
  --min_dim_Z 1 --max_dim_Z 4

echo
echo "Done. Next:"
echo "  training:  --synt_eval_dir ${OUT}       (the ROOT, not its val/)"
echo "  benchmark: data.val_dir = ${OUT}/val"
