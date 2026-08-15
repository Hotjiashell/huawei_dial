#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "run_qwen2_5_3b_grpo.sh is deprecated; starting the Qwen3-4B recipe." >&2
exec "$SCRIPT_DIR/run_qwen3_4b_grpo.sh" "$@"
