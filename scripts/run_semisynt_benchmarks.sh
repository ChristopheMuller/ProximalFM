#!/usr/bin/env bash
# Run the shared benchmark method configuration over the semisynthetic matrix.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

METHODS_CONFIG=${METHODS_CONFIG:-benchmarks/config_methods.yaml}
MATRIX_ROOT=${MATRIX_ROOT:-data/semisynt_eval/semisynt_val_v1}
CONTEXT_SIZES=(128 512)

run_case() {
  local case_name=$1
  python benchmarks/run.py \
    --config "${METHODS_CONFIG}" \
    --val-dir "${MATRIX_ROOT}/${case_name}/val" \
    --context-sizes "${CONTEXT_SIZES[@]}"
}

run_case linear_conf_low_proxy_low
run_case linear_conf_low_proxy_high
run_case linear_conf_high_proxy_low
run_case linear_conf_high_proxy_high
run_case nonlinear_conf_low_proxy_low
run_case nonlinear_conf_low_proxy_high
run_case nonlinear_conf_high_proxy_low
run_case nonlinear_conf_high_proxy_high
