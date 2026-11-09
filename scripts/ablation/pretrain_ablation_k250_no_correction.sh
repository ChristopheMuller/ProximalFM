#!/usr/bin/env bash

set -euo pipefail

ABLATION_ARM=k250_no_correction exec bash "$(dirname "${BASH_SOURCE[0]}")/pretrain_ablation_default.sh" "$@"
