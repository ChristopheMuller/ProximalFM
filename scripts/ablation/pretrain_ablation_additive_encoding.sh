#!/usr/bin/env bash

set -euo pipefail

ABLATION_ARM=additive_encoding exec bash "$(dirname "${BASH_SOURCE[0]}")/pretrain_ablation_default.sh" "$@"
