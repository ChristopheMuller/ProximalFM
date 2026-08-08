#!/usr/bin/env bash
# ROADMAP A1 -- build the production frozen posterior-evaluation set on GPU.
#
# Sizing rationale (the part that is not obvious from the flags):
#
#   n_datasets  APPENDABLE. episode_seed hashes (seed, index), so raising it
#               later leaves episodes 0..N-1 bit-identical. Start at what you
#               can afford; extend when A2's stratified cuts need more.
#   n_q, K      NOT appendable. Both change seq_len / the RNG consumption
#               order, so changing either regenerates every episode and breaks
#               comparability with every number already logged. Over-provision
#               now; you cannot fix them later without a new set name.
#
# n_q is what makes the PER-EPISODE statistics readable: the per-episode PIT-KS
# null band is ~1.36/sqrt(n_q), so 64 -> 0.17 (unusable), 256 -> 0.085. Pooled
# metrics gain much less from n_q than from n_datasets, because rows inside one
# episode share an SCM and a posterior width -- the effective sample size for
# anything pooled is ~n_datasets, not n_datasets*n_q.
#
# Scoring cost is the counterweight to a big set: each episode re-encodes the
# full n_max context. Use --prior_eval_max_episodes to subsample during the run
# and score the whole set only at milestones.
#
# Determinism is per-device: this set is reproducible on the same GPU + torch
# build, and is NOT bit-identical to a cpu build. gen_device is recorded in
# frozen_meta.pt.

set -euo pipefail

OUT=${OUT:-data/posterior_eval/frozen_prod_v1}
DEVICE=${DEVICE:-cuda:0}

N_DATASETS=${N_DATASETS:-1000}
N_MAX=${N_MAX:-2048}
N_Q=${N_Q:-128}
K=${K:-4000}
SEED=${SEED:-20260806}
MEM_BUDGET=${MEM_BUDGET:-8000000000}

echo "Building frozen set:"
echo "  out         = ${OUT}"
echo "  device      = ${DEVICE}"
echo "  n_datasets  = ${N_DATASETS}"
echo "  n_max/n_q/K = ${N_MAX}/${N_Q}/${K}   (rows per episode: $((N_MAX + N_Q * K)))"
echo "  seed        = ${SEED}"
echo

source .venv/bin/activate

python -m posterior_eval.build \
  --out "${OUT}" \
  --gen_device "${DEVICE}" \
  --n_datasets "${N_DATASETS}" \
  --n_max "${N_MAX}" \
  --n_q "${N_Q}" \
  --mc_cate "${K}" \
  --seed "${SEED}" \
  --prior_mc_mem_budget "${MEM_BUDGET}" \
  --datasets_per_file 8 \
  --n_jobs 1

echo
echo "Done. Next:"
echo "  uv run python -m posterior_eval.score --dir ${OUT} --ckpt <ckpt>"
echo "  training: --prior_eval_dir ${OUT}  (the ROOT, not its val/)"
