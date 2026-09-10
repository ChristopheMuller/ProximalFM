#!/usr/bin/env bash
# Resume the interrupted 5k large-context post-tune.
#
# ``pretrain_prod_large.sh``'s default ``--resume auto`` gives a step checkpoint
# in RUN precedence over CKPT.  It therefore restores the model, AdamW state,
# cosine scheduler, and current step, rather than starting a new transfer from
# whole_1/best_model.ckpt.  Do not change the optimization-defining variables
# below when continuing an existing run.

set -euo pipefail

RUN=${RUN:-large_5k}
CUDA=${CUDA:-2}
TRANSFER=${TRANSFER:-large_run/transfer_rules_freezing.yml}

MAX_STEPS=${MAX_STEPS:-30000}
BATCH_SIZE=${BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
MIN_SEQ_LEN=${MIN_SEQ_LEN:-64}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-5000}
RECOMPUTE=${RECOMPUTE:-False}
LR=${LR:-2e-5}
SCHEDULER=${SCHEDULER:-cosine_warmup}
WARMUP_STEPS=${WARMUP_STEPS:-500}

shopt -s nullglob
checkpoints=("${RUN}"/step-*.ckpt)
if (( ${#checkpoints[@]} == 0 )); then
    echo "ERROR: no step-*.ckpt checkpoint found in '${RUN}'; refusing a cold start." >&2
    exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec env \
    RUN="${RUN}" CUDA="${CUDA}" TRANSFER="${TRANSFER}" \
    MAX_STEPS="${MAX_STEPS}" BATCH_SIZE="${BATCH_SIZE}" \
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}" MIN_SEQ_LEN="${MIN_SEQ_LEN}" \
    MAX_SEQ_LEN="${MAX_SEQ_LEN}" RECOMPUTE="${RECOMPUTE}" LR="${LR}" \
    SCHEDULER="${SCHEDULER}" WARMUP_STEPS="${WARMUP_STEPS}" \
    "${script_dir}/pretrain_prod_large.sh"
