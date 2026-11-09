#!/usr/bin/env bash
# Build the frozen validation set shared by the ablation runs.

set -euo pipefail

PRIOR_EVAL_DIR=${PRIOR_EVAL_DIR:-data/prior_eval/prior_val_ablation_u1_x2-3_w1-2_z1-2_v1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

NAME="$(basename "${PRIOR_EVAL_DIR}")" OUT_ROOT="$(dirname "${PRIOR_EVAL_DIR}")" \
    DEVICE=cpu PRIOR_TYPE=ablation_scm N_MAX=256 N_Q=16 K=10000 \
    SEED="${PRIOR_EVAL_SEED:-20260806}" N_DATASETS="${PRIOR_EVAL_N_DATASETS:-200}" \
    MIN_DIM_U=1 MAX_DIM_U=1 MIN_DIM_X=2 MAX_DIM_X=3 \
    MIN_DIM_W=1 MAX_DIM_W=2 MIN_DIM_Z=1 MAX_DIM_Z=2 \
    exec bash "$(dirname "${BASH_SOURCE[0]}")/../build_prior_eval.sh"
