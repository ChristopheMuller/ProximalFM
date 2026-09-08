#!/usr/bin/env bash
# Build the large-context realistic validation set used for post-tuning.
#
# The model's dataset size is the context size: N_MAX=10000 observed context
# rows, followed by the same fixed N_Q=128 query rows used in pre-training.
# Query rows are a batching axis and do not attend to one another.
#
# This is a thin preset over build_synt_eval.sh so its validation, output
# format, and production defaults cannot drift.  The realistic set is not
# appendable, so the default keeps all 480 episodes (40 per mechanism).  For a
# disposable smoke build, override both NAME and N_DATASETS, for example:
#
#   NAME=synt_val_large_smoke N_DATASETS=12 ./scripts/build_synt_eval_large.sh
#
# A different N_MAX, N_Q, K, or seed must use a different NAME.

set -euo pipefail

export NAME=${NAME:-synt_val_large_v1}
export N_DATASETS=${N_DATASETS:-480}
export N_MAX=${N_MAX:-10000}
export N_Q=${N_Q:-128}
export K=${K:-10000}
export SEED=${SEED:-20260908}
export MEM_BUDGET=${MEM_BUDGET:-8000000000}

exec "$(dirname "${BASH_SOURCE[0]}")/build_synt_eval.sh"
