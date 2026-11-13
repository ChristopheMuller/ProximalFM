#!/usr/bin/env bash
# Compare CATE accuracy and per-episode fit + predict time across the prepared
# widths in the nonlinear, high-confounding, low-proxy regime. Each d is the
# width of X, W, and Z individually (D = 3d).
#
# All five widths:
#   ./scripts/run_semisynt_dimension_benchmarks.sh
# A smaller sweep:
#   DIMENSIONS="5 10" CONTEXT_SIZES=512 \
#     ./scripts/run_semisynt_dimension_benchmarks.sh
# Set DRY_RUN=1 to print the commands or FORCE=1 to refresh cached predictions.
# Set OUTPUT_ROOT when comparing separate hardware or method configurations.

set -euo pipefail

# shellcheck source=scripts/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

DATA_ROOT=${DATA_ROOT:-data/semisynt_eval/semisynt_val_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/semisynt_dimension}
METHODS_CONFIG=${METHODS_CONFIG:-benchmarks/config_methods_dimension.yml}
case_name=nonlinear_conf_high_proxy_low
read -r -a dimension_values <<< "${DIMENSIONS:-1 2 3 10}"
read -r -a context_sizes <<< "${CONTEXT_SIZES:-128 512 1024 2048}"

if [[ ! -f "$METHODS_CONFIG" ]]; then
  echo "Methods config not found: $METHODS_CONFIG" >&2
  exit 1
fi

# Check all selected widths before starting the sweep.
shopt -s nullglob
for d in "${dimension_values[@]}"; do
  val_dir="$DATA_ROOT/d-$d/$case_name/val"
  batches=("$val_dir"/batch_*.pt)
  if [[ ${#batches[@]} -eq 0 ]]; then
    echo "No validation batches found in: $val_dir" >&2
    exit 1
  fi
done

# Set numerical thread limits before importing NumPy, PyTorch, or sklearn.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

for d in "${dimension_values[@]}"; do
  command=(
    python benchmarks/run.py
    --config "$METHODS_CONFIG"
    --val-dir "$DATA_ROOT/d-$d/$case_name/val"
    --output-dir "$OUTPUT_ROOT/d-$d/$case_name"
    --no-parallel
    --context-sizes "${context_sizes[@]}"
  )
  if [[ "${FORCE:-0}" == "1" ]]; then
    command+=(--force)
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    printf '\nRunning d-%s / %s\n' "$d" "$case_name"
    "${command[@]}"
  fi
done
