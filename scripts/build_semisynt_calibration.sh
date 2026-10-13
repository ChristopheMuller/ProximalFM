#!/usr/bin/env bash
# Build the focused semi-synthetic set for posterior-calibration evaluation.
#
# Design:
#   - d_X = d_U = d_W = d_Z = 5
#   - nonlinear mechanisms
#   - high latent confounding and low proxy informativeness
#   - all 12 bundled empirical CSV sources
#   - 10 independently seeded episodes per source
#   - 8192 context rows and 256 fixed query rows per episode
#
# Build the maximum context once. Evaluation should use the nested prefixes
# 128, 256, 512, 1024, 2048, 4096, and 8192 while retaining the same query
# rows. The final two prefixes test context extrapolation beyond pre-training.
#
# Example:
#   ./scripts/build_semisynt_calibration.sh
#
# Output:
#   data/semisynt_eval/<NAME>/d-5/nonlinear_conf_high_proxy_low/val

set -euo pipefail

# Activate the repository environment and cd to the repository root. The
# cluster has no `uv run`, so invoke plain python after sourcing this helper.
# shellcheck source=scripts/_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG=${CONFIG:-eval_datasets/semisynt_eval/config.yaml}
NAME=${NAME:-semisynt_posterior_calibration_v1}
OUT_ROOT=${OUT_ROOT:-data/semisynt_eval}
CSV=${CSV:-all}

CASE=nonlinear_conf_high_proxy_low
DIMENSION=5
N_CONTEXT=${N_CONTEXT:-8192}
N_QUERY=${N_QUERY:-256}
REPLICATES_PER_CSV=10
N_CALIBRATION=${N_CALIBRATION:-1000}
SPLIT_SEED=${SPLIT_SEED:-1234}
# Deliberately differs from the general builder's default so changing only the
# output name cannot silently reproduce its default episodes.
SEED=${SEED:-20260915}

OUT="${OUT_ROOT}/${NAME}"

command=(
  python -m eval_datasets.semisynt_eval.build
  --config "${CONFIG}"
  --name "${NAME}"
  --out_root "${OUT_ROOT}"
  --csv "${CSV}"
  --case "${CASE}"
  --dimensions "${DIMENSION}"
  --n_context "${N_CONTEXT}"
  --n_query "${N_QUERY}"
  --replicates_per_csv "${REPLICATES_PER_CSV}"
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

echo "Building focused semi-synthetic posterior-calibration set:"
echo "  config                 = ${CONFIG}"
echo "  out                    = ${OUT}"
echo "  csv                    = ${CSV}"
echo "  case                   = ${CASE}"
echo "  dimension              = ${DIMENSION}"
echo "  n_context/n_query      = ${N_CONTEXT}/${N_QUERY}"
echo "  context prefixes       = 128 256 512 1024 2048 4096 8192"
echo "  replicates/source      = ${REPLICATES_PER_CSV}"
echo "  calibration rows       = ${N_CALIBRATION}"
echo "  split_seed/seed        = ${SPLIT_SEED}/${SEED}"
echo

"${command[@]}"

echo
echo "Done. Calibration set:"
echo "  ${OUT}/d-${DIMENSION}/${CASE}/val"
echo "Use context prefixes 128 256 512 1024 2048 4096 8192 during posterior scoring."
