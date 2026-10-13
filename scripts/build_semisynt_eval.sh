#!/usr/bin/env bash
# Build the complete 12-source semi-synthetic linearity/confounding/proxy
# evaluation matrix (eight cases per dimension).
#
#   validation set:  NAME=semisynt_val_v1  SEED=20260904 ./scripts/build_semisynt_eval.sh
#   held-out set:    NAME=semisynt_test_v1 SEED=<new>    ./scripts/build_semisynt_eval.sh
#
# Output: data/semisynt_eval/<NAME>/d-<d>/<CASE>/val. The matrix and each case have
# their own config and manifest. Point the benchmark at one CASE's
# directory's val/ subdirectory, not at the matrix root.
#
# The scientific matrix (families, confounding levels, proxy information,
# arrows, nonlinearities, and interactions) lives in CONFIG. This script only
# overrides build-level sizing and source-selection values.

set -euo pipefail

# venv on PATH and cd to the repository root. `uv run` is unavailable on the
# cluster, so this script deliberately invokes plain python.
# shellcheck source=scripts/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG=${CONFIG:-eval_datasets/semisynt_eval/config.yaml}
NAME=${NAME:-semisynt_val_v1}
OUT_ROOT=${OUT_ROOT:-data/semisynt_eval}
CSV=${CSV:-all}

N_CONTEXT=${N_CONTEXT:-8192}
N_QUERY=${N_QUERY:-256}
REPLICATES_PER_CSV=${REPLICATES_PER_CSV:-3}
read -r -a DIMENSION_VALUES <<< "${DIMENSIONS:-1 2 3 5 10}"
N_CALIBRATION=${N_CALIBRATION:-1000}
SPLIT_SEED=${SPLIT_SEED:-1234}
SEED=${SEED:-1234}

OUT="${OUT_ROOT}/${NAME}"

command=(
  python -m eval_datasets.semisynt_eval.build
  --config "${CONFIG}"
  --name "${NAME}"
  --out_root "${OUT_ROOT}"
  --csv "${CSV}"
  --n_context "${N_CONTEXT}"
  --n_query "${N_QUERY}"
  --replicates_per_csv "${REPLICATES_PER_CSV}"
  --dimensions "${DIMENSION_VALUES[@]}"
  --n_calibration "${N_CALIBRATION}"
  --split_seed "${SPLIT_SEED}"
  --seed "${SEED}"
)
if [[ -n "${DATASETS_PER_FILE:-}" ]]; then
  command+=(--datasets_per_file "${DATASETS_PER_FILE}")
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  command+=(--force)
fi

echo "Building semi-synthetic evaluation matrix:"
echo "  config                 = ${CONFIG}"
echo "  out                    = ${OUT}"
echo "  csv                    = ${CSV}"
echo "  n_context/n_query      = ${N_CONTEXT}/${N_QUERY}"
echo "  replicates/source/case = ${REPLICATES_PER_CSV}"
echo "  dimensions             = ${DIMENSION_VALUES[*]}"
echo "  calibration rows       = ${N_CALIBRATION}"
echo "  split_seed/seed        = ${SPLIT_SEED}/${SEED}"
echo

"${command[@]}"

echo
echo "Done. Each case is benchmark-ready at:"
echo "  ${OUT}/d-<d>/<case>/val"
echo "See ${OUT}/manifest.json for the case names and content digests."
