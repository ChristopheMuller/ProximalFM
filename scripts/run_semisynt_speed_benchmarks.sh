#!/usr/bin/env bash
# Measure per-episode runtime on all 36 datasets in one semi-synthetic regime.
#
# Episodes are run sequentially, while each individual method retains the
# intra-method parallelism declared in config_methods_speed.yaml (n_jobs=-1 by
# default). Results live outside the main benchmark's bench/ directory, so its
# prediction and timing caches can never be reused accidentally.

set -euo pipefail

# shellcheck source=scripts/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

METHODS_CONFIG=${METHODS_CONFIG:-benchmarks/config_methods_speed.yaml}
VAL_DIR=${VAL_DIR:-data/semisynt_eval/semisynt_val_v1/d-5/nonlinear_conf_high_proxy_low/val}
OUTPUT_DIR=${OUTPUT_DIR:-data/semisynt_eval/semisynt_val_v1/d-5/nonlinear_conf_high_proxy_low/bench_speed_sequential}
read -r -a CONTEXT_SIZE_VALUES <<< "${CONTEXT_SIZES:-128 512 1024 2048}"

if [[ ! -f "$METHODS_CONFIG" ]]; then
  echo "Methods config not found: $METHODS_CONFIG" >&2
  exit 1
fi
if [[ ! -d "$VAL_DIR" ]]; then
  echo "Validation directory not found: $VAL_DIR" >&2
  exit 1
fi

# Grid searches may use all cores through n_jobs=-1. Keep each job's numerical
# kernels single-threaded so joblib workers do not each start another full BLAS
# or OpenMP pool. These must be set before Python imports NumPy/torch/sklearn.
BLAS_THREADS=${BLAS_THREADS:-1}
export OMP_NUM_THREADS="$BLAS_THREADS"
export MKL_NUM_THREADS="$BLAS_THREADS"
export OPENBLAS_NUM_THREADS="$BLAS_THREADS"
export VECLIB_MAXIMUM_THREADS="$BLAS_THREADS"
export NUMEXPR_NUM_THREADS="$BLAS_THREADS"

COMMAND=(
  python benchmarks/run.py
  --config "$METHODS_CONFIG"
  --val-dir "$VAL_DIR"
  --output-dir "$OUTPUT_DIR"
  --no-parallel
  --context-sizes "${CONTEXT_SIZE_VALUES[@]}"
)
if [[ "${FORCE:-0}" == "1" ]]; then
  COMMAND+=(--force)
fi

echo "Running sequential semi-synthetic speed benchmark:"
echo "  methods config = $METHODS_CONFIG"
echo "  validation set = $VAL_DIR"
echo "  output          = $OUTPUT_DIR"
echo "  context sizes   = ${CONTEXT_SIZE_VALUES[*]}"
echo "  BLAS threads    = $BLAS_THREADS per job"
echo "  episode workers = 1 (--no-parallel)"
echo

"${COMMAND[@]}"
