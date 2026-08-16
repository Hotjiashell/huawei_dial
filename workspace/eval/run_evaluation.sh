#!/usr/bin/env bash
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${EVAL_CONFIG:-${EVAL_DIR}/.env}"

if [[ -f "${CONFIG_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${CONFIG_FILE}"
    set +a
fi

exec python3 "${EVAL_DIR}/evaluate.py" "$@"
