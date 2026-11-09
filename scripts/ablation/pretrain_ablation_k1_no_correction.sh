#!/usr/bin/env bash

set -euo pipefail

ABLATION_ARM=k1_no_correction exec bash "$(dirname "${BASH_SOURCE[0]}")/pretrain_ablation_default.sh" "$@"
