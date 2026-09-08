#!/usr/bin/env bash
# Post-tune whole_1 on larger in-context datasets (up to 10k context rows).
#
# This keeps the effective meta-batch and causal prior from pretrain_prod.sh,
# but uses smaller gradient-accumulation chunks plus activation recomputation
# for the longer context.  It is a fresh optimization run initialized from the
# complete trained ProximalFM checkpoint, not a resume of whole_1.
#
# Sequence lengths are sampled uniformly over [64, 10000).  This expands the
# original support instead of replacing it: about 20% of episodes remain below
# 2048 for rehearsal, while about 20% are at least 8000 for large-context
# adaptation.  Log-uniform is deliberately off because it would concentrate
# too much of this short post-tune at small contexts.  The fixed validation
# point is 10000 context rows (+ 128 query rows).
#
# MAX_STEPS=5000 is provisional until steps/second and validation overhead have
# been measured on the target GPU.  All runtime-sensitive choices below are
# environment-overridable without editing this production recipe.

set -euo pipefail

RUN=${RUN:-large_run}
CUDA=${CUDA:-0}
CKPT=${CKPT:-whole_1/best_model.ckpt}
TRANSFER=${TRANSFER:-large_run/transfer_rules_posttune.yml}

PRIOR_EVAL_DIR=${PRIOR_EVAL_DIR:-data/prior_eval/prior_val_large_v1}
SYNT_EVAL_DIR=${SYNT_EVAL_DIR:-data/synt_eval/synt_val_large_v1}
SYNT_EVAL_INTERVAL=${SYNT_EVAL_INTERVAL:-100}
PRIOR_EVAL_INTERVAL=${PRIOR_EVAL_INTERVAL:-200}
# In-training validation uses stable subsets to keep 10k-row evaluation from
# dominating wall time. Score all 500/480 episodes offline for final selection.
PRIOR_MAX_EP=${PRIOR_MAX_EP:-128}
SYNT_MAX_EP=${SYNT_MAX_EP:-120}

MAX_STEPS=${MAX_STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
MIN_SEQ_LEN=${MIN_SEQ_LEN:-64}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-10000}
LOG_SEQ_LEN=${LOG_SEQ_LEN:-False}
RECOMPUTE=${RECOMPUTE:-True}

# A lower peak LR protects the already-trained posterior during adaptation.
# Once the 10-hour step budget is measured, cosine decay will end at that
# budget; the short warmup is for the fresh AdamW state.
LR=${LR:-2e-5}
SCHEDULER=${SCHEDULER:-cosine_warmup}
WARMUP_STEPS=${WARMUP_STEPS:-100}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA}

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

for d in "${PRIOR_EVAL_DIR}" "${SYNT_EVAL_DIR}"; do
    if [[ ! -f "${d}/frozen_meta.pt" ]]; then
        echo "ERROR: '${d}' is not a usable large eval set root (no frozen_meta.pt)." >&2
        echo "  Build it with ./scripts/build_prior_eval_large.sh or" >&2
        echo "  ./scripts/build_synt_eval_large.sh, as appropriate." >&2
        exit 1
    fi
done

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: ProximalFM checkpoint '${CKPT}' not found (override with CKPT=)." >&2
    exit 1
fi

if [[ ! -f "${TRANSFER}" ]]; then
    echo "ERROR: post-tuning transfer rules '${TRANSFER}' not found." >&2
    exit 1
fi

python -m proximalfm.train \
    --np_seed 1234 --torch_seed 1234 \
    --device cuda:0 --prior_device cuda:0 --dtype bfloat16 \
    --model_compile False --recompute "${RECOMPUTE}" \
    --max_steps "${MAX_STEPS}" --batch_size "${BATCH_SIZE}" --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --min_seq_len "${MIN_SEQ_LEN}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --log_seq_len "${LOG_SEQ_LEN}" \
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
    --lr "${LR}" --scheduler "${SCHEDULER}" \
    --warmup_proportion -1 --warmup_steps "${WARMUP_STEPS}" \
    --loss_type noise_aware --lambda_cate 0.5 \
    --nbins 2000 --cate_vmin -5 --cate_vmax 5 \
    --hl_start_sigma 0.01 \
    --hl_warmup_sigma_steps 0 \
    --hl_linear_decay_steps 0 \
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
    --save_perm_steps 1,10,25,50,100,200,300,500,750,1000,1500
