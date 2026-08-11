#!/usr/bin/env bash
# PRODUCTION fine-tune -- point-estimate checkpoint -> calibrated
# posterior over tau, on GPU.
#
#   CUDA=3 RUN=mc_run_7_NoiseAware ./scripts/finetune_prod.sh
#
# Prerequisites (both are set ROOTS, not their val/ subdirectories):
#   ./scripts/build_prior_eval.sh   -> data/prior_eval/frozen_prod_v1
#   ./scripts/build_synt_eval.sh    -> data/synt_eval/realistic_prod_v1
# The synt set MUST be a replicate build (K > 1): without the frozen_meta.pt
# sidecar the trainer refuses it at __init__, by design.
#
# The three choices that are not obvious from the flags
# -----------------------------------------------------
# GRID. --nbins is the head's output WIDTH, so changing it throws the
# pretrained decoder away (94 % load instead of 100 %). --cate_vmin/--cate_vmax
# are not parameters. Hence +-5 at nbins 1000: the head loads whole, the bin
# width grows 0.008 -> 0.010, and the 0.96 % of frozen-set targets that fell
# outside +-3 come back inside the grid. Do not change --nbins again mid-programme.
#
# SIGMA FLOOR. hl_final_sigma is a HARD floor on learnable posterior width:
# a perfect fit reports sqrt(sigma^2 + w^2/12). A flat mean_sd curve sitting on
# it is the SCHEDULE, not convergence -- which is why diag/sigma is logged on
# mean_sd's axis. Plot them together, always. The grid's own floor here is
# w/sqrt(12) ~ 0.0029, so the schedule binds for anything above ~0.003.
# HL_FINAL_SIGMA is the axis of the sigma sweep; keep everything else fixed across it.
#
# SELECTION (written down BEFORE any number exists). Rank on prior_dist/crps
# (proper, so it ranks; but it mixes calibration and sharpness and diagnoses
# neither). REJECT any candidate with:
#     prior_dist/sd_over_rmse outside [0.90, 1.15]
#     prior_dist/pit_ks/p90 above prior_dist/pit_ks/band (its own null band)
# sd_over_rmse is unusable AS a best_metric: 1.0 is a target, not a direction.
# Also watch prior_point/* against the pre-fine-tune checkpoint -- point accuracy
# degrading is the expected failure mode of a distributional objective.
#
# Reading order for every eval pass, not negotiable:
#   out_of_grid_frac -> sd_over_rmse -> pit_ks_pooled vs its band (then p90)
#   -> crps vs crps_over_oracle -> diag/sigma vs mean_sd -> prior_point/* vs synt_point/*
#
# Cost (still unmeasured at production sizes): every eval episode re-encodes
# the full n_max = 2048 context. *_max_episodes subsamples deterministically (a
# prefix of the set's episode order); score the full sets at milestones with
# `python -m prior_eval.score`, not at training cadence.

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
