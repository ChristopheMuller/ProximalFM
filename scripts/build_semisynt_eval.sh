#!/usr/bin/env bash
# Build the complete semi-synthetic linear/nonlinear evaluation matrix.
#
#   validation set:  NAME=semisynt_val_v1  SEED=20260904 ./scripts/build_semisynt_eval.sh
#   held-out set:    NAME=semisynt_test_v1 SEED=<new>    ./scripts/build_semisynt_eval.sh
#
# Output: data/semisynt_eval/<NAME>/<CASE>/val. The matrix and each case have
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

CONFIG=${CONFIG:-semisynt_eval/config.yaml}
NAME=${NAME:-semisynt_val_v1}
OUT_ROOT=${OUT_ROOT:-data/semisynt_eval}
CSV=${CSV:-all}

N_CONTEXT=${N_CONTEXT:-2048}
N_QUERY=${N_QUERY:-256}
REPLICATES_PER_CSV=${REPLICATES_PER_CSV:-4}
N_FEATURES_X=${N_FEATURES_X:-5}
N_FEATURES_U=${N_FEATURES_U:-5}
SPLIT_SEED=${SPLIT_SEED:-1234}
SEED=${SEED:-1234}

OUT="${OUT_ROOT}/${NAME}"

command=(
  python -m semisynt_eval.build
  --config "${CONFIG}"
  --name "${NAME}"
  --out_root "${OUT_ROOT}"
  --csv "${CSV}"
  --n_context "${N_CONTEXT}"
  --n_query "${N_QUERY}"
  --replicates_per_csv "${REPLICATES_PER_CSV}"
  --n_features_x "${N_FEATURES_X}"
  --n_features_u "${N_FEATURES_U}"
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
echo "  n_features_x/u         = ${N_FEATURES_X}/${N_FEATURES_U}"
echo "  split_seed/seed        = ${SPLIT_SEED}/${SEED}"
echo

"${command[@]}"

echo
echo "Done. Each case is benchmark-ready at:"
echo "  ${OUT}/<case>/val"
echo "See ${OUT}/manifest.json for the case names and content digests."
