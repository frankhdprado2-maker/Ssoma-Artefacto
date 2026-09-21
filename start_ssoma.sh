#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "${SSOMA_PYTHON:-}" ]]; then PY="$SSOMA_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"
else PY=python3; fi
"$PY" -c 'import sys,torch,uvicorn; assert sys.version_info[:2] == (3,12)' || { echo 'Instale Python 3.12 y requirements.deploy.txt en un venv; configure SSOMA_PYTHON si es externo.' >&2; exit 2; }
if [[ "${1:-}" == --self-check ]]; then exec "$PY" scripts/self_check.py; fi
exec "$PY" scripts/run_runtime.py
