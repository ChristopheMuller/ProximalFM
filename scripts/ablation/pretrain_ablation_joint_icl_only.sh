#!/usr/bin/env bash

set -euo pipefail

ABLATION_ARM=joint_icl_only exec bash "$(dirname "${BASH_SOURCE[0]}")/pretrain_ablation_default.sh" "$@"
