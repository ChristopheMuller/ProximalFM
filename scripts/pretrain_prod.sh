#!/usr/bin/env bash

set -euo pipefail

RUN=${RUN:-whole_1}
CUDA=${CUDA:-0}
CKPT=${CKPT:-ckpt_tabicl/tabicl-regressor-v2-20260212.ckpt}
TRANSFER=${TRANSFER:-ckpt_tabicl/transfer_rules.yaml}

PRIOR_EVAL_DIR=${PRIOR_EVAL_DIR:-data/prior_eval/prior_val_v1}
SYNT_EVAL_DIR=${SYNT_EVAL_DIR:-data/synt_eval/synt_val_v1}
SYNT_EVAL_INTERVAL=${SYNT_EVAL_INTERVAL:-100}
PRIOR_EVAL_INTERVAL=${PRIOR_EVAL_INTERVAL:-200}
PRIOR_MAX_EP=${PRIOR_MAX_EP:-500}
SYNT_MAX_EP=${SYNT_MAX_EP:-480}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA}

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

for d in "${PRIOR_EVAL_DIR}" "${SYNT_EVAL_DIR}"; do
    if [[ ! -f "${d}/frozen_meta.pt" ]]; then
        echo "ERROR: '${d}' is not a usable eval set root (no frozen_meta.pt)." >&2
        echo "  Build it (./scripts/build_prior_eval.sh, ./scripts/build_synt_eval.sh)" >&2
        echo "  or point PRIOR_EVAL_DIR/SYNT_EVAL_DIR at the existing build." >&2
        exit 1
    fi
done

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: backbone checkpoint '${CKPT}' not found (override with CKPT=)." >&2
    exit 1
fi

python -m proximalfm.train \
    --np_seed 1234 --torch_seed 1234 \
    --device cuda:0 --prior_device cuda:0 --dtype bfloat16 \
    --model_compile False \
    --max_steps 75000 --batch_size 64 --micro_batch_size 16 \
    --min_seq_len 64 \
    --max_seq_len 2048 \
    --log_seq_len False \
    --n_query 128 \
    --prior_type mlp_scm \
    --min_dim_U 1 --max_dim_U 10 \
    --min_dim_W 1 --max_dim_W 10 \
    --min_dim_Z 1 --max_dim_Z 10 \
    --min_dim_X 0 --max_dim_X 10 \
    --mc_cate 250 --mc_targets_on_context False \
    --prior_mc_mem_budget 8000000000 \
    --gen_block_size 64 --gen_reuse_passes 1 \
    --train_dataloader_num_workers 0 --train_dataloader_prefetch_factor none \
    --lr 1e-4 --warmup_proportion -1 --warmup_steps 2500 \
    --loss_type cate_single_noise_aware --lambda_cate 0.5 \
    --nbins 2000 --cate_vmin -5 --cate_vmax 5 \
    --hl_start_sigma 0.5 \
    --hl_warmup_sigma_steps 500 \
    --hl_linear_decay_steps 3000 \
    --hl_final_sigma_bins 2.0 \
    --checkpoint_path "${CKPT}" --only_load_model True \
    --transfer_rules "${TRANSFER}" \
    --checkpoint_dir "${RUN}" --save_perm_every 1000 \
    --save_best True --best_metric synt_dist/crps --best_metric_mode min \
    --prior_eval_dir "${PRIOR_EVAL_DIR}" \
    --prior_eval_interval "${PRIOR_EVAL_INTERVAL}" --prior_eval_max_episodes "${PRIOR_MAX_EP}" \
    --synt_eval_dir "${SYNT_EVAL_DIR}" \
    --synt_eval_interval "${SYNT_EVAL_INTERVAL}" --synt_eval_max_episodes "${SYNT_MAX_EP}" \
    --wandb_log True --wandb_name "${RUN}" --wandb_mode online \
    --save_perm_steps 1,50,100,200,300,500,750,1000,1500
