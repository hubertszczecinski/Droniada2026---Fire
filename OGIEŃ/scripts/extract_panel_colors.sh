#!/usr/bin/env bash
# Ekstrakcja kolorów kartek z nagrania panelu (layout JSON + wideo).
#
# Przykład:
#   cp config/panel_color_layout.example.json config/panel_color_layout.json
#   # uzupełnij row/col/color
#   ./scripts/extract_panel_colors.sh --video dataset/my_capture/panel_calib.mov
#
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

VIDEO="${DRONIADA_PANEL_VIDEO:-}"
LAYOUT="${DRONIADA_PANEL_LAYOUT:-config/panel_color_layout.json}"
EXTRA=()

if [[ -n "$VIDEO" && "$#" -eq 0 ]]; then
  set -- --video "$VIDEO" --layout "$LAYOUT"
elif [[ "$#" -eq 1 && "$1" != --* ]]; then
  set -- --video "$1" --layout "$LAYOUT"
fi

echo "=== Ekstrakcja kolorów z nagrania panelu ==="
echo "Python: $PY"
echo ""

exec "$PY" scripts/extract_panel_colors.py "${EXTRA[@]}" "$@"
