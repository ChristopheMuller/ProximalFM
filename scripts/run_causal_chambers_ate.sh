#!/usr/bin/env bash
# Full-context-prefix ATE learning curves on the Causal Chambers data.
#
# Override the defaults without editing tracked files, for example:
#   CONTEXT_SIZES="128 256" N_PERMUTATIONS=2 bash scripts/run_causal_chambers_ate.sh
#   OUT=data/results/cc_ate bash scripts/run_causal_chambers_ate.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG="${CONFIG:-benchmarks/causal_chambers/config_ate.yaml}"
CONTEXT_SIZES="${CONTEXT_SIZES:-128 256 512 1000}"
N_PERMUTATIONS="${N_PERMUTATIONS:-1}"
OUT="${OUT:-benchmarks/causal_chambers/results/ate_p2sls}"

python benchmarks/causal_chambers/run_ate.py \
  --config "${CONFIG}" \
  --context-sizes ${CONTEXT_SIZES} \
  --n-permutations "${N_PERMUTATIONS}" \
  --output-dir "${OUT}" \
  "$@"
