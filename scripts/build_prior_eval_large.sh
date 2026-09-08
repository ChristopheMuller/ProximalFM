#!/usr/bin/env bash
# Build the large-context frozen-prior validation set used for post-tuning.
#
# The model's dataset size is the context size: N_MAX=10000 observed context
# rows, followed by the same fixed N_Q=128 query rows used in pre-training.
# Only query rows receive K Monte Carlo replicates, so generation memory scales
# as N_MAX + N_Q*K rather than (N_MAX + N_Q)*K.
#
# This is a thin preset over build_prior_eval.sh so its validation, output
# format, and production defaults cannot drift.  For a disposable smoke build,
# override both NAME and N_DATASETS, for example:
#
#   NAME=prior_val_large_smoke N_DATASETS=8 ./scripts/build_prior_eval_large.sh
#
# Prior episodes are appendable when every other setting remains identical.
# A different N_MAX, N_Q, K, device, or seed must use a different NAME.

set -euo pipefail

export NAME=${NAME:-prior_val_large_v1}
export N_DATASETS=${N_DATASETS:-500}
export N_MAX=${N_MAX:-10000}
export N_Q=${N_Q:-128}
export K=${K:-10000}
export SEED=${SEED:-20260909}
export MEM_BUDGET=${MEM_BUDGET:-8000000000}

exec "$(dirname "${BASH_SOURCE[0]}")/build_prior_eval.sh"
