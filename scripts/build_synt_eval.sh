#!/usr/bin/env bash
# Build a PRODUCTION realistic-mechanism replicate set.
#
#   validation set (selection):  NAME=realistic_prod_v1 SEED=20260807 ./scripts/build_synt_eval.sh
#   held-out test set:           NAME=realistic_test_v1 SEED=<new>    ./scripts/build_synt_eval.sh
#
# Output: data/synt_eval/<NAME>/  ({val/, frozen_meta.pt, config.yaml, manifest.json};
# pass the ROOT to --synt_eval_dir, the val/ subdirectory to the benchmark harness).
#
# Why MC_CATE matters more than anything else here: at K = 1 the set carries
# realized potential outcomes, MU0/MU1 are draws of the ITE, and NO sidecar is
# written -- the trainer then refuses it outright (that refusal is the whole
# point: scoring a posterior against sampled ITEs rewards over-wide posteriors).
# K > 1 replicates each query row through the latent and noise draw, so MU0/MU1
# become Monte Carlo estimates of E[Y_a | X] and S2 their paired sampling
# variance, and the sidecar makes the set scorable as synt_point/* + synt_dist/*.
#
# N_DATASETS is generative here (unlike the prior set): the mechanism pool
# shuffle depends on it, so this set is NOT appendable. Neither are N_MAX, N_Q or
# MC_CATE. Everything is fixed by NAME; a different config needs a different NAME.
#
# The split is given as N_MAX (context rows) + N_Q (query rows), the same
# vocabulary as the prior set, because N_Q -- not the total row count -- is what
# trades against K: generation is CPU-only and single-threaded per dataset, and
# memory scales as N = N_MAX + N_Q*K rows for ONE dataset at a time. An
# unaffordable K raises rather than being silently capped (which would make
# target noise vary across datasets and machines).
#
# The defaults mirror data/prior_eval/frozen_prod_v1 exactly (n_max 2048, n_q 128,
# K 4000), so the two surfaces are read at the same context size and the same
# target noise. The builder accepts --seq_len/--prop_train as the older spelling
# of the same split, but not both spellings at once.

set -euo pipefail

# venv on PATH, cd to the repo root, WANDB_API_KEY if a key file exists.
# shellcheck source=scripts/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

NAME=${NAME:-synt_val_v1}
OUT_ROOT=${OUT_ROOT:-data/synt_eval}

N_DATASETS=${N_DATASETS:-480}
N_MAX=${N_MAX:-2048}
N_Q=${N_Q:-128}
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

python -m synt_eval.build_proximal_datasets \
  --name "${NAME}" \
  --out_root "${OUT_ROOT}" \
  --n_datasets "${N_DATASETS}" \
  --n_max "${N_MAX}" \
  --n_q "${N_Q}" \
  --mc_cate "${K}" \
  --seed "${SEED}" \
  --mc_mem_budget "${MEM_BUDGET}" \
  --min_dim_U 1 --max_dim_U 10 \
  --min_dim_X 1 --max_dim_X 10 \
  --min_dim_W 1 --max_dim_W 10 \
  --min_dim_Z 1 --max_dim_Z 10

echo
echo "Done. Next:"
echo "  training:  --synt_eval_dir ${OUT}       (the ROOT, not its val/)"
echo "  benchmark: data.val_dir = ${OUT}/val"
