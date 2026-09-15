#!/usr/bin/env bash
set -euo pipefail

# Frozen d=5 prior slice for the posterior-calibration analysis.
#
# The default 400 episodes provide 102,400 query targets at each nested context
# size, while preserving 400 independent draws from the SCM prior.  K=4000 is
# already enough to make Monte Carlo error in tau_star negligible relative to
# posterior uncertainty; increasing K further costs considerably more than
# adding independent episodes.
#
# A cheap production-shaped pilot can be built first with:
#   N_DATASETS=120 NAME=prior_calibration_d5_v1 scripts/build_prior_calibration.sh
# Rerunning the default command with the same NAME, seed, device, and structural
# settings appends deterministically from 120 to 400 episodes.
#
# N_MAX, N_Q, K, and the dimensions are part of the frozen set definition.  Use
# a new NAME if changing any of them.  Builds are deterministic only on the
# same device type.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NAME="${NAME:-prior_calibration_d5_v1}"
export N_DATASETS="${N_DATASETS:-400}"
export N_MAX="${N_MAX:-2048}"
export N_Q="${N_Q:-256}"
export K="${K:-4000}"
export SEED="${SEED:-20260916}"
export MEM_BUDGET="${MEM_BUDGET:-8000000000}"

# Match the focused semi-synthetic experiment and leave context size as the
# sole designed axis in the main calibration figure.
export MIN_DIM_U=5
export MAX_DIM_U=5
export MIN_DIM_W=5
export MAX_DIM_W=5
export MIN_DIM_Z=5
export MAX_DIM_Z=5
export MIN_DIM_X=5
export MAX_DIM_X=5
export DEVICE="${DEVICE:-mps}"

exec "${SCRIPT_DIR}/build_prior_eval.sh"
