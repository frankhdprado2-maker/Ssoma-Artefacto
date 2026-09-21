#!/usr/bin/env bash
# Explicit manual launch only; never invoked by bootstrap/systemd.
set -euo pipefail
: "${SSOMA_ENV_FILE:=/etc/ssoma/ssoma.env}"
set -a; source "$SSOMA_ENV_FILE"; set +a
[[ -n "${SSOMA_ACCESS_TOKEN:-}" ]] || { echo 'Public demo requires an external access token.' >&2; exit 2; }
[[ "${SSOMA_BIND:-127.0.0.1}" == 127.0.0.1 ]] || exit 2
: "${SSOMA_VENV_DIR:=/opt/ssoma/venv}"
: "${SSOMA_APP_DIR:=/opt/ssoma/app}"
: "${SSOMA_DATA_DIR:=/var/lib/ssoma}"
"$SSOMA_VENV_DIR/bin/python" - "$SSOMA_DATA_DIR/oci_ready.json" "$SSOMA_APP_DIR/config/deploy_model_registry_v001.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
try:
    ready=json.loads(Path(sys.argv[1]).read_text())
    assert ready['ARM64_RUNTIME_VERIFIED'] and ready['semantic_fixture_equivalence']
    assert ready['registry_sha256']==hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
except Exception:
    raise SystemExit('Successful native ARM64 benchmark required before a public tunnel.')
PY
command -v cloudflared >/dev/null || { echo 'Install the verified official Linux ARM64 cloudflared binary first.' >&2; exit 2; }
PORT="${PORT:-8000}"; [[ "$PORT" =~ ^[0-9]+$ ]] || exit 2
curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null
# Check that the actual running app rejects anonymous sensitive access.
[[ "$(curl --silent --output /dev/null --write-out '%{http_code}' "http://127.0.0.1:$PORT/api/analyses")" == 401 ]] || { echo 'Running application is not token protected.' >&2; exit 2; }
LOG_DIR="${SSOMA_LOG_DIR:-/var/log/ssoma}"
umask 077; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/quick_tunnel_$(date -u +%Y%m%dT%H%M%SZ).log"
URL_FILE="$LOG_DIR/quick_tunnel_url.txt"
# Do not accidentally load a preexisting named-tunnel configuration.
for config in "$HOME/.cloudflared/config.yml" "$HOME/.cloudflared/config.yaml" /etc/cloudflared/config.yml /etc/cloudflared/config.yaml /usr/local/etc/cloudflared/config.yml /usr/local/etc/cloudflared/config.yaml; do
  [[ ! -f "$config" ]] || { echo 'Existing cloudflared config: use a clean demo environment.' >&2; exit 2; }
done
# Anonymous quick tunnel uses no account.
# URL appears once in the log and URL file; no application token is logged.
cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" 2>&1 | tee "$LOG" | \
  awk -v out="$URL_FILE" '{print; if(match($0,/https:\/\/[-a-z0-9]+\.trycloudflare\.com/)) {print substr($0,RSTART,RLENGTH) > out; close(out)} fflush()}'
