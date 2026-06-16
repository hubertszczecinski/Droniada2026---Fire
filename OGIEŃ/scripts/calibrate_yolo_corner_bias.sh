#!/usr/bin/env bash
# Policz i zapisz korekcję rogów YOLO-Pose (pred−GT na panel_labels).
# Live ładuje wynik automatycznie przy DRONIADA_YOLO_BIAS=1.
#
# Użycie:
#   ./scripts/calibrate_yolo_corner_bias.sh
#   ./scripts/calibrate_yolo_corner_bias.sh runs/pose/droniada_real_finetune/weights/best.pt
#   ./scripts/calibrate_yolo_corner_bias.sh --report-only
#
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x .venv_yolo/bin/python ]]; then
  PY="$(pwd)/.venv_yolo/bin/python"
else
  echo "Brak .venv_yolo — utwórz venv z ultralytics + opencv"
  exit 127
fi

WEIGHTS="${DRONIADA_YOLO_POSE_WEIGHTS:-runs/pose/droniada_real_finetune/weights/best.pt}"
if [[ $# -gt 0 && "$1" != --* && -f "$1" ]]; then
  WEIGHTS="$1"
  shift
fi

export DRONIADA_YOLO_POSE_WEIGHTS="$WEIGHTS"
export DRONIADA_YOLO_BIAS="${DRONIADA_YOLO_BIAS:-1}"

echo "=== YOLO corner bias ==="
echo "Wagi: $WEIGHTS"
if [[ $# -gt 0 ]]; then
  "$PY" -m release.yolo_corner_bias --weights "$WEIGHTS" "$@"
else
  "$PY" -m release.yolo_corner_bias --weights "$WEIGHTS"
fi
