#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"

arm=${ABLATION_ARM:-default}
case "${arm}" in
    default)            k=250; correction=True;  role=learned ;;
    k16_correction)     k=16;  correction=True;  role=learned ;;
    k16_no_correction)  k=16;  correction=False; role=learned ;;
    k250_no_correction) k=250; correction=False; role=learned ;;
    k1_no_correction)   k=1;   correction=False; role=learned ;;
    no_role_tags)       k=250; correction=True;  role=none ;;
    *) echo "Unknown ablation arm: ${arm}" >&2; exit 2 ;;
esac

SEED=${SEED:-1234}
DEVICE=${DEVICE:-mps}
MAX_STEPS=${MAX_STEPS:-15000}
RUN=${RUN:-ablation_scm/u1_x2-3_w1-2_z1-2/steps_${MAX_STEPS}/seed_${SEED}/${arm}}
PRIOR_EVAL_DIR=${PRIOR_EVAL_DIR:-data/prior_eval/prior_val_ablation_u1_x2-3_w1-2_z1-2_v1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

if [[ ! -s "${PRIOR_EVAL_DIR}/frozen_meta.pt" ]]; then
    echo "Missing validation set at ${PRIOR_EVAL_DIR}. Run scripts/ablation/build_ablation_data.sh first." >&2
    exit 1
fi

exec python -m proximalfm.train \
    --np_seed "${SEED}" --torch_seed "${SEED}" \
    --device "${DEVICE}" --prior_device cpu \
    --embed_dim 96 --col_num_blocks 2 --row_num_blocks 2 --icl_num_blocks 4 \
    --col_nhead 4 --row_nhead 4 --icl_nhead 4 --row_num_cls 4 --col_num_inds 128 \
    --role_embedding "${role}" --outcome_encoding treatment_gated --outcome_injection both \
    --max_steps "${MAX_STEPS}" --batch_size 4 --micro_batch_size 2 \
    --batch_size_per_gp 2 --min_seq_len 64 --max_seq_len 257 --n_query 16 \
    --prior_type ablation_scm \
    --min_dim_U 1 --max_dim_U 1 --min_dim_X 2 --max_dim_X 3 \
    --min_dim_W 1 --max_dim_W 2 --min_dim_Z 1 --max_dim_Z 2 \
    --mc_cate "${k}" --mc_noise_correction "${correction}" --mc_targets_on_context False \
    --gen_block_size 4 --gen_reuse_passes 1 --train_dataloader_num_workers 0 \
    --lr 1e-4 --warmup_proportion -1 --warmup_steps 500 \
    --loss_type noise_aware --lambda_cate 0.5 --nbins 999 --cate_vmin -5 --cate_vmax 5 \
    --hl_start_sigma 0.3 --hl_warmup_sigma_steps 10 \
    --hl_linear_decay_steps 100 --hl_final_sigma_bins 2.0 \
    --prior_eval_dir "${PRIOR_EVAL_DIR}" --prior_eval_interval "${PRIOR_EVAL_INTERVAL:-500}" \
    --prior_eval_max_episodes "${PRIOR_EVAL_MAX_EPISODES:-64}" \
    --checkpoint_dir "${RUN}" --save_best False \
    --save_temp_every 500 --max_checkpoints 1 --save_perm_every "${MAX_STEPS}" \
    --wandb_log True --wandb_name "${RUN}" \
    "$@"
