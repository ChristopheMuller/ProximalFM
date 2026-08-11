#!/usr/bin/env bash

set -euo pipefail

RUN=${RUN:-mc_run_7_NoiseAwareTuning}
CUDA=${CUDA:-0}
CKPT=${CKPT:-mc_run_6_TuneToOneHead/step-10000.ckpt}

PRIOR_EVAL_DIR=${PRIOR_EVAL_DIR:-data/prior_eval/frozen_prod_v1}
SYNT_EVAL_DIR=${SYNT_EVAL_DIR:-data/synt_eval/realistic_prod_v1}

MAX_STEPS=${MAX_STEPS:-10000}
LR=${LR:-1e-4}
HL_START_SIGMA=${HL_START_SIGMA:-0.5}
HL_WARMUP_SIGMA_STEPS=${HL_WARMUP_SIGMA_STEPS:-50}
HL_LINEAR_DECAY_STEPS=${HL_LINEAR_DECAY_STEPS:-150}
HL_FINAL_SIGMA=${HL_FINAL_SIGMA:-0.00625}
EVAL_INTERVAL=${EVAL_INTERVAL:-50}
PRIOR_MAX_EP=${PRIOR_MAX_EP:-1000}
SYNT_MAX_EP=${SYNT_MAX_EP:-480}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA}

# venv on PATH, cd to the repo root, WANDB_API_KEY if a key file exists.
# shellcheck source=scripts/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

# Preflight. The defaults above are where the sets are SUPPOSED to live; on this
# checkout the prior set has also been seen at data/posterior_eval/frozen_prod_v1.
for d in "${PRIOR_EVAL_DIR}" "${SYNT_EVAL_DIR}"; do
    if [[ ! -f "${d}/frozen_meta.pt" ]]; then
        echo "ERROR: '${d}' is not a usable eval set root (no frozen_meta.pt)." >&2
        echo "  Build it (./scripts/build_prior_eval.sh, ./scripts/build_synt_eval.sh)" >&2
        echo "  or point PRIOR_EVAL_DIR/SYNT_EVAL_DIR at the existing build." >&2
        exit 1
    fi
done

python -m proximalfm.train \
    --device cuda:0 --prior_device cuda:0 --dtype bfloat16 \
    --model_compile False \
    --max_steps "${MAX_STEPS}" --batch_size 64 --micro_batch_size 16 \
    --max_seq_len 2048 --min_train_size 0.5 \
    --prior_type mlp_scm \
    --min_dim_U 1 --max_dim_U 10 \
    --min_dim_W 1 --max_dim_W 10 \
    --min_dim_Z 1 --max_dim_Z 10 \
    --min_dim_X 1 --max_dim_X 10 \
    --mc_cate 50 --mc_targets_on_context False \
    --prior_mc_mem_budget 8000000000 \
    --gen_block_size 64 --gen_reuse_passes 1 \
    --train_dataloader_num_workers 0 --train_dataloader_prefetch_factor none \
    --lr "${LR}" --warmup_proportion -1 --warmup_steps 500 \
    --icl_num_blocks 12 \
    --loss_type cate_single_noise_aware --lambda_cate 0.5 \
    --nbins 1000 --cate_vmin -4 --cate_vmax 4 \
    --hl_start_sigma "${HL_START_SIGMA}" \
    --hl_warmup_sigma_steps "${HL_WARMUP_SIGMA_STEPS}" \
    --hl_linear_decay_steps "${HL_LINEAR_DECAY_STEPS}" \
    --hl_final_sigma "${HL_FINAL_SIGMA}" \
    --checkpoint_path "${CKPT}" --only_load_model True \
    --transfer_rules transfer_rules.yaml \
    --checkpoint_dir "${RUN}" --save_perm_every 2000 \
    --save_best True --best_metric synt_dist/crps --best_metric_mode min \
    --prior_eval_dir "${PRIOR_EVAL_DIR}" \
    --prior_eval_interval "${EVAL_INTERVAL}" --prior_eval_max_episodes "${PRIOR_MAX_EP}" \
    --synt_eval_dir "${SYNT_EVAL_DIR}" \
    --synt_eval_interval "${EVAL_INTERVAL}" --synt_eval_max_episodes "${SYNT_MAX_EP}" \
    --wandb_log True --wandb_name "${RUN}" --wandb_mode online
