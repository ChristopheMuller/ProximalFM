#!/usr/bin/env bash

set -euo pipefail

ABLATION_ARM=no_role_tags exec bash "$(dirname "${BASH_SOURCE[0]}")/pretrain_ablation_default.sh" "$@"
