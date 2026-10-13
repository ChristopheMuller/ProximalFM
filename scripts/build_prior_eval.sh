#!/usr/bin/env bash
# Build a PRODUCTION frozen prior-evaluation set.
#
#   validation set (selection):  NAME=frozen_prod_v1 SEED=20260806 ./scripts/build_prior_eval.sh
#   held-out test set:           NAME=frozen_test_v1 SEED=<new>    ./scripts/build_prior_eval.sh
#
# Output: data/prior_eval/<NAME>/  ({val/, frozen_meta.pt}; pass the ROOT, not val/).
#
# Sizing rationale (the part that is not obvious from the flags):
#
#   N_DATASETS  APPENDABLE. episode_seed hashes (seed, index), so raising it
#               later leaves episodes 0..N-1 bit-identical.
#   N_Q, K      NOT appendable. Both change seq_len / the RNG consumption order,
#               so changing either regenerates every episode and breaks
#               comparability with every number already logged. Over-provision
#               now; you cannot fix them later without a new set name.
#
# N_Q is what makes the PER-EPISODE statistics readable: the per-episode PIT-KS
# null band is ~1.36/sqrt(n_q). Pooled metrics gain much less from n_q than from
# n_datasets, because rows inside one episode share an SCM and a posterior width.
#
# Determinism is per-device: reproducible on the same GPU + torch build, NOT
# bit-identical to a cpu build. gen_device is recorded in frozen_meta.pt and is
# part of the content_digest.
#
# The test set must keep N_Q and K identical to the validation set (only SEED,
# and optionally N_DATASETS, change), so the two are comparable.

set -euo pipefail

# venv on PATH, cd to the repo root, WANDB_API_KEY if a key file exists.
# shellcheck source=scripts/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

NAME=${NAME:-prior_val_v1}
OUT_ROOT=${OUT_ROOT:-data/prior_eval}
DEVICE=${DEVICE:-cuda:0}

N_DATASETS=${N_DATASETS:-500}
N_MAX=${N_MAX:-2048}
N_Q=${N_Q:-128}
K=${K:-10000}
SEED=${SEED:-20260806}
MEM_BUDGET=${MEM_BUDGET:-8000000000}

MIN_DIM_U=${MIN_DIM_U:-1}
MAX_DIM_U=${MAX_DIM_U:-10}
MIN_DIM_W=${MIN_DIM_W:-1}
MAX_DIM_W=${MAX_DIM_W:-10}
MIN_DIM_Z=${MIN_DIM_Z:-1}
MAX_DIM_Z=${MAX_DIM_Z:-10}
MIN_DIM_X=${MIN_DIM_X:-1}
MAX_DIM_X=${MAX_DIM_X:-10}

OUT="${OUT_ROOT}/${NAME}"

echo "Building frozen prior set:"
echo "  out         = ${OUT}"
echo "  device      = ${DEVICE}"
echo "  n_datasets  = ${N_DATASETS}"
echo "  n_max/n_q/K = ${N_MAX}/${N_Q}/${K}   (rows per episode: $((N_MAX + N_Q * K)))"
echo "  dims U/X/W/Z = ${MIN_DIM_U}-${MAX_DIM_U}/${MIN_DIM_X}-${MAX_DIM_X}/${MIN_DIM_W}-${MAX_DIM_W}/${MIN_DIM_Z}-${MAX_DIM_Z}"
echo "  seed        = ${SEED}"
echo

python -m eval_datasets.prior_eval.build \
  --name "${NAME}" \
  --out_root "${OUT_ROOT}" \
  --gen_device "${DEVICE}" \
  --n_datasets "${N_DATASETS}" \
  --n_max "${N_MAX}" \
  --n_q "${N_Q}" \
  --mc_cate "${K}" \
  --seed "${SEED}" \
  --prior_mc_mem_budget "${MEM_BUDGET}" \
  --min_dim_U "${MIN_DIM_U}" \
  --max_dim_U "${MAX_DIM_U}" \
  --min_dim_W "${MIN_DIM_W}" \
  --max_dim_W "${MAX_DIM_W}" \
  --min_dim_Z "${MIN_DIM_Z}" \
  --max_dim_Z "${MAX_DIM_Z}" \
  --min_dim_X "${MIN_DIM_X}" \
  --max_dim_X "${MAX_DIM_X}" \
  --datasets_per_file 8 \
  --n_jobs 1

echo
echo "Done. Next:"
echo "  python -m eval_datasets.prior_eval.verify --dir ${OUT} --cate_vmin -5 --cate_vmax 5"
echo "  python -m eval_datasets.prior_eval.score  --dir ${OUT} --ckpt <ckpt>"
echo "  training: --prior_eval_dir ${OUT}   (the ROOT, not its val/)"
