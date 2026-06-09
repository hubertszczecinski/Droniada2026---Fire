#!/usr/bin/env bash
# Uczenie kolorów z migawek Test.mov + weryfikacja GT (tylko migawki).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv_yolo/bin/python"
if [[ ! -x "$PY" ]]; then PY=python3; fi
exec "$PY" scripts/train_snapshot_colors.py "$@"
