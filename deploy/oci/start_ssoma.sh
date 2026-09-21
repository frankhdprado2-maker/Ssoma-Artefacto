#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${SSOMA_ENV_FILE:-}" ]]; then
  set -a; source "$SSOMA_ENV_FILE"; set +a
fi
: "${SSOMA_APP_DIR:=/opt/ssoma/app}"
: "${SSOMA_VENV_DIR:=/opt/ssoma/venv}"
: "${SSOMA_DATA_DIR:=/var/lib/ssoma}"
[[ -n "${SSOMA_ACCESS_TOKEN:-}" ]] || { echo 'Set SSOMA_ACCESS_TOKEN in the external environment.' >&2; exit 2; }
[[ "${SSOMA_BIND:-127.0.0.1}" == '127.0.0.1' ]] || { echo 'OCI demo must bind loopback.' >&2; exit 2; }
[[ "${SSOMA_DEVICE:-cpu}" == cpu ]] || exit 2
export SSOMA_BIND=127.0.0.1 SSOMA_DEVICE=cpu
export YOLO_CONFIG_DIR="$SSOMA_DATA_DIR/ultralytics"
cd "$SSOMA_APP_DIR"
exec "$SSOMA_VENV_DIR/bin/python" -m deployment.adapter_v001.cli serve
