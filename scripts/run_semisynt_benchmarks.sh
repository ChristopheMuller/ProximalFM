#!/usr/bin/env bash
# Run the shared benchmark method configuration over every d-5 semi-synthetic
# case.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

METHODS_CONFIG=${METHODS_CONFIG:-benchmarks/config_methods.yaml}
DATA_ROOT=${DATA_ROOT:-data/semisynt_eval/semisynt_val_v1/d-5}
# Pass a space-separated environment string, e.g. CONTEXT_SIZES="128 512".
read -r -a CONTEXT_SIZE_VALUES <<< "${CONTEXT_SIZES:-128 512 1024 2048}"

# Materialize every case's val directory under d-5 before starting any runs.
CELL_PATHS=$(find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -type d -name val -print | sort)
if [[ -z "$CELL_PATHS" ]]; then
  echo "No benchmark cases found beneath: $DATA_ROOT" >&2
  exit 1
fi
while IFS= read -r val_dir; do
  python benchmarks/run.py \
    --config "${METHODS_CONFIG}" \
    --val-dir "$val_dir" \
    --context-sizes "${CONTEXT_SIZE_VALUES[@]}"
done <<< "$CELL_PATHS"
