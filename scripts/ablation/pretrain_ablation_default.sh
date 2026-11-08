#!/usr/bin/env bash

set -euo pipefail

RUN=${RUN:-ablation_slope}
DEVICE=${DEVICE:-mps}
MAX_STEPS=${MAX_STEPS:-5000}
N_MAX=${N_MAX:-256}
PRIOR_EVAL_DIR=${PRIOR_EVAL_DIR:-data/prior_eval/prior_val_ablation_slope_v1}
PRIOR_EVAL_INTERVAL=${PRIOR_EVAL_INTERVAL:-200}
OUTCOME_ENCODING=${OUTCOME_ENCODING:-treatment_gated}
OUTCOME_INJECTION=${OUTCOME_INJECTION:-both}

source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"

# The context sampler's upper bound is exclusive.
python -m proximalfm.train \
    --np_seed 1234 --torch_seed 1234 \
    --device "${DEVICE}" --prior_device cpu \
    --embed_dim 96 \
    --col_num_blocks 2 --row_num_blocks 2 --icl_num_blocks 4 \
    --col_nhead 4 --row_nhead 4 --icl_nhead 4 \
    --row_num_cls 4 --col_num_inds 128 \
    --outcome_encoding "${OUTCOME_ENCODING}" \
    --outcome_injection "${OUTCOME_INJECTION}" \
    --max_steps "${MAX_STEPS}" --batch_size 4 --micro_batch_size 2 \
    --batch_size_per_gp 2 --min_seq_len 64 --max_seq_len "$((N_MAX + 1))" --n_query 16 \
    --prior_type ablation_scm_slope \
    --min_dim_U 1 --max_dim_U 1 \
    --min_dim_X 1 --max_dim_X 1 \
    --min_dim_W 1 --max_dim_W 1 \
    --min_dim_Z 1 --max_dim_Z 1 \
    --mc_cate 250 --mc_targets_on_context False \
    --gen_block_size 4 --gen_reuse_passes 1 \
    --train_dataloader_num_workers 0 \
    --lr 1e-4 --warmup_proportion 0.1 \
    --loss_type noise_aware --lambda_cate 0.5 \
    --nbins 999 --cate_vmin -5 --cate_vmax 5 \
    --hl_start_sigma 0.3 --hl_warmup_sigma_steps 10 \
    --hl_linear_decay_steps 100 --hl_final_sigma_bins 2.0 \
    --prior_eval_dir "${PRIOR_EVAL_DIR}" --prior_eval_interval "${PRIOR_EVAL_INTERVAL}" \
    --checkpoint_dir "${RUN}" --save_best False --save_perm_steps "${MAX_STEPS}" \
    --wandb_name "${RUN}" \
    "$@"
