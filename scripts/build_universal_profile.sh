#!/usr/bin/env bash
# Profil kolorystyczny: *2 + nag5 + Test.mov (lub własne --recording).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv_yolo/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

WEIGHTS="${ROOT}/runs/pose/droniada_real_finetune/weights/best.pt"
if [[ -f "$WEIGHTS" ]]; then
  export DRONIADA_YOLO_POSE_WEIGHTS="$WEIGHTS"
fi

exec "$PY" scripts/build_universal_profile.py "$@"
