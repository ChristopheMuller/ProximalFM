#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../_env.sh"

# Set these in the environment to keep training and calibration in sync.
RUN=${RUN:-ablation_slope/noise_aware_no_correction}
MAX_STEPS=${MAX_STEPS:-5000}
PRIOR_EVAL_DIR=${PRIOR_EVAL_DIR:-data/prior_eval/prior_val_ablation_slope_v1}
CALIBRATION_DEVICE=${CALIBRATION_DEVICE:-cpu}

checkpoint="$(printf 'step-%06d' "$((10#${MAX_STEPS}))")"

# Use the K=16 targets with the baseline likelihood and zero MC variance.
RUN="${RUN}" bash "${script_dir}/pretrain_ablation_default.sh" --mc_cate 16 "$@" \
    --mc_noise_correction False \
    --max_steps "${MAX_STEPS}" --save_perm_steps "${MAX_STEPS}" \
    --checkpoint_dir "${RUN}" --prior_eval_dir "${PRIOR_EVAL_DIR}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}" \
    python -m analysis.analysis_eval.posterior_calibration \
    --data-root "${PRIOR_EVAL_DIR}" --ckpt "${RUN}/${checkpoint}.ckpt" \
    --device "${CALIBRATION_DEVICE}" --contexts 64 128 256 \
    --out-dir "${RUN}/calibration/${checkpoint}" --force
