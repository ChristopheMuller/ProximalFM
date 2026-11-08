#!/usr/bin/env bash
# Build the frozen prior validation set for the small ablation runs.
# Use a different NAME and SEED for the held-out test set.

set -euo pipefail

export NAME=${NAME:-prior_val_ablation_512_v1}
export DEVICE=${DEVICE:-cpu}
export PRIOR_TYPE=ablation_scm
export N_DATASETS=${N_DATASETS:-240}
export N_MAX=${N_MAX:-512}
export N_Q=${N_Q:-16}

export MIN_DIM_U=${MIN_DIM_U:-1}
export MAX_DIM_U=${MAX_DIM_U:-3}
export MIN_DIM_X=${MIN_DIM_X:-1}
export MAX_DIM_X=${MAX_DIM_X:-4}
export MIN_DIM_W=${MIN_DIM_W:-1}
export MAX_DIM_W=${MAX_DIM_W:-4}
export MIN_DIM_Z=${MIN_DIM_Z:-1}
export MAX_DIM_Z=${MAX_DIM_Z:-4}

exec bash "$(dirname "${BASH_SOURCE[0]}")/../build_prior_eval.sh"
