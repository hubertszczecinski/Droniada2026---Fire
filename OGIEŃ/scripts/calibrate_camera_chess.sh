#!/usr/bin/env bash
# Kalibracja kamery szachownicą → config/camera_calibration.npz
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv_yolo/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

CHESS_DIR="${CHESS_DIR:-calibration_chess}"
PATTERN_COLS="${PATTERN_COLS:-9}"
PATTERN_ROWS="${PATTERN_ROWS:-6}"
SQUARE_MM="${SQUARE_MM:-25.0}"
CAMERA="${CAMERA:-1}"
ROTATE="${ROTATE:-180}"

echo "=== 1/2 Zbieranie zdjęć (kamera=$CAMERA, obrót=${ROTATE}°) ==="
echo "    Wzorzec: ${PATTERN_COLS}x${PATTERN_ROWS} wewnętrznych narożników, kwadrat ${SQUARE_MM} mm"
echo ""
"$PY" -m pipelines.capture_chess_calibration \
  --camera "$CAMERA" \
  --rotate "$ROTATE" \
  --pattern-cols "$PATTERN_COLS" \
  --pattern-rows "$PATTERN_ROWS" \
  --out-dir "$CHESS_DIR"

echo ""
echo "=== 2/2 Obliczanie K, dist → config/camera_calibration.npz ==="
"$PY" -m pipelines.calibrate_camera \
  --images "${CHESS_DIR}/*.jpg" \
  --pattern-cols "$PATTERN_COLS" \
  --pattern-rows "$PATTERN_ROWS" \
  --square-mm "$SQUARE_MM" \
  --out config/camera_calibration.npz

echo ""
echo "Gotowe. Live i PnP użyją config/camera_calibration.npz automatycznie (gdy plik istnieje)."
