#!/usr/bin/env bash
# Run the shared benchmark method configuration over the semisynthetic matrix.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

METHODS_CONFIG=${METHODS_CONFIG:-benchmarks/config_methods.yaml}
MATRIX_ROOT=${MATRIX_ROOT:-data/semisynt_eval/semisynt_val_v1}
# Pass a space-separated environment string, e.g. CONTEXT_SIZES="128 512".
read -r -a CONTEXT_SIZE_VALUES <<< "${CONTEXT_SIZES:-128 512 1024 2048}"

# Read manifest cells rather than assuming a particular dimension/regime grid.
# Materialize the list first so manifest errors stop the script before any runs.
CELL_PATHS=$(python - "$MATRIX_ROOT" <<'PYTHON'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
for cell in manifest["cases"]:
    print(root / f"d-{cell['dimension']}" / cell["case"] / "val")
PYTHON
)
while IFS= read -r val_dir; do
  python benchmarks/run.py \
    --config "${METHODS_CONFIG}" \
    --val-dir "$val_dir" \
    --context-sizes "${CONTEXT_SIZE_VALUES[@]}"
done <<< "$CELL_PATHS"
