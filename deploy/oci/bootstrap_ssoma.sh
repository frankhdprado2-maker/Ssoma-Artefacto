#!/usr/bin/env bash
# Run only on an authorized Ubuntu 24.04 ARM64 VM. Does not open ports/start tunnels.
set -euo pipefail
[[ "$EUID" -eq 0 ]] || { echo 'Run with sudo on the authorized VM.' >&2; exit 2; }
[[ "$(uname -m)" == aarch64 ]] || { echo 'Native aarch64 target required.' >&2; exit 2; }
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]] || { echo 'Ubuntu 24.04 required.' >&2; exit 2; }
if [[ -f /etc/ssoma/ssoma.env ]]; then
  set -a; source /etc/ssoma/ssoma.env; set +a
fi
APP="${SSOMA_APP_DIR:-/opt/ssoma/app}"
VENV="${SSOMA_VENV_DIR:-/opt/ssoma/venv}"
DATA="${SSOMA_DATA_DIR:-/var/lib/ssoma}"
MODELS="${SSOMA_MODEL_DIR:-/opt/ssoma/models}"
UPLOADS="${SSOMA_UPLOAD_DIR:-$DATA/uploads}"
RESULTS="${SSOMA_RESULT_DIR:-$DATA/results}"
LOGS="${SSOMA_LOG_DIR:-/var/log/ssoma}"
for value in "$APP" "$VENV" "$DATA" "$MODELS" "$UPLOADS" "$RESULTS" "$LOGS"; do
  [[ "$value" == /* && "$value" != *[[:space:]%]* && "$value" != *\'* && "$value" != *\"* ]] || { echo 'Use absolute paths without whitespace, quotes or percent.' >&2; exit 2; }
done
[[ -f "$APP/deploy/oci/requirements.arm64.lock.txt" ]] || { echo 'Extract the portable app package first.' >&2; exit 2; }
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends python3.12 python3.12-venv ffmpeg libgl1 libglib2.0-0t64 ca-certificates curl
id ssoma >/dev/null 2>&1 || useradd --system --home-dir "$DATA" --shell /usr/sbin/nologin ssoma
install -d -m 0750 -o ssoma -g ssoma "$DATA" "$UPLOADS" "$RESULTS" "$LOGS"
install -d -m 0755 "$MODELS"
install -d -m 0750 -o root -g ssoma /etc/ssoma
[[ -x "$VENV/bin/python" ]] || python3.12 -m venv "$VENV"
"$VENV/bin/python" -m pip install --only-binary=:all: --require-hashes -r "$APP/deploy/oci/requirements.arm64.lock.txt"
"$VENV/bin/python" -m pip check
if [[ ! -e /etc/ssoma/ssoma.env ]]; then
  install -m 0640 -o root -g ssoma "$APP/deploy/oci/ssoma.env.example" /etc/ssoma/ssoma.env
  export APP VENV DATA MODELS UPLOADS RESULTS LOGS
  "$VENV/bin/python" - <<'PY'
import os
from pathlib import Path
p=Path('/etc/ssoma/ssoma.env');text=p.read_text()
for name,value in [('SSOMA_APP_DIR','APP'),('SSOMA_VENV_DIR','VENV'),('SSOMA_DATA_DIR','DATA'),('SSOMA_MODEL_DIR','MODELS'),('SSOMA_UPLOAD_DIR','UPLOADS'),('SSOMA_RESULT_DIR','RESULTS'),('SSOMA_LOG_DIR','LOGS')]:
    text='\n'.join(name+'='+os.environ[value] if line.startswith(name+'=') else line for line in text.splitlines())+'\n'
p.write_text(text)
PY
fi
# Existing configuration/secrets are never overwritten on rerun.
set -a; source /etc/ssoma/ssoma.env; set +a
export YOLO_CONFIG_DIR="$SSOMA_DATA_DIR/ultralytics"
install -m 0644 "$APP/deploy/oci/ssoma.service" /etc/systemd/system/ssoma.service
install -d -m 0755 /etc/systemd/system/ssoma.service.d
cat > /etc/systemd/system/ssoma.service.d/paths.conf <<EOF
[Service]
WorkingDirectory=$SSOMA_APP_DIR
ExecStart=
ExecStart=/bin/bash $SSOMA_APP_DIR/deploy/oci/start_ssoma.sh
ReadWritePaths=
ReadWritePaths=$SSOMA_DATA_DIR $SSOMA_UPLOAD_DIR $SSOMA_RESULT_DIR $SSOMA_LOG_DIR
EOF
systemctl daemon-reload
cd "$SSOMA_APP_DIR"
runuser -u ssoma --preserve-environment -- "$SSOMA_VENV_DIR/bin/python" deploy/oci/verify_arm64.py --models --output "$SSOMA_DATA_DIR/arm64_verification.json"
echo 'Bootstrap ready. Configure external token, then explicitly start ssoma.service. No service/tunnel started.'
