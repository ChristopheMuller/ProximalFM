#!/usr/bin/env bash
# Run the selected methods on the high-confounding, low-proxy nonlinear
# semisynthetic case across the complete 128--8192 learning curve.

set -euo pipefail

# shellcheck source=scripts/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

METHODS_CONFIG=${METHODS_CONFIG:-benchmarks/config_methods_selection.yaml}
VAL_DIR=${VAL_DIR:-data/semisynt_eval/semisynt_val_v1/d-5/nonlinear_conf_high_proxy_low/val}
read -r -a CONTEXT_SIZES <<< "${CONTEXT_SIZES:-128 256 512 1024 2048 4096 8192}"

if [[ ! -f "$METHODS_CONFIG" ]]; then
  echo "Methods config not found: $METHODS_CONFIG" >&2
  exit 1
fi
if [[ ! -d "$VAL_DIR" ]]; then
  echo "Validation directory not found: $VAL_DIR" >&2
  exit 1
fi

COMMAND=(
  python benchmarks/run.py
  --config "$METHODS_CONFIG"
  --val-dir "$VAL_DIR"
  --context-sizes "${CONTEXT_SIZES[@]}"
)
if [[ "${FORCE:-0}" == "1" ]]; then
  COMMAND+=(--force)
fi

"${COMMAND[@]}"
