#!/usr/bin/env bash
# Stanowisko testowe wieczorne: modul A (pozycja + ustawienie panelu) + modul B (CXY latch).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv_yolo/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

WEIGHTS="${ROOT}/runs/pose/droniada_real_finetune/weights/best.pt"
if [[ ! -f "$WEIGHTS" ]]; then
  echo "Brak wag YOLO: $WEIGHTS"
  echo "Ustaw DRONIADA_YOLO_POSE_WEIGHTS lub wytrenuj model."
  exit 1
fi

export DRONIADA_YOLO_POSE_WEIGHTS="$WEIGHTS"
export DRONIADA_YOLO_BIAS="${DRONIADA_YOLO_BIAS:-1}"

mkdir -p "${ROOT}/dataset/results"

CAM="${DRONIADA_CAMERA:-1}"
LOG="${ROOT}/dataset/results/live_bench_$(date +%Y%m%d_%H%M%S).log"

echo "=== Droniada LIVE BENCH (A + B) ==="
echo "Kamera: $CAM  (nadpisz: DRONIADA_CAMERA=0)"
echo "Log:    $LOG"
echo ""
echo "Okna:"
echo "  droniada_dashboard     — wszystko w jednym: A+B+dane+migawki"
echo "  (stary tryb bez --dashboard: osobne okno zatrzasku)"
echo ""
echo "Klawisze: q=wyjscie  s=zlap CXY  r=reset latch"
echo "Zamknij QuickTime / inne apki uzywajace kamery."
echo ""

exec "$PY" -m release.run_live_panel \
  --bench \
  --camera "$CAM" \
  --rotate 180 \
  --camera-profile tarot_t10x_2a:wide \
  --cxy-latch-dir dataset/debug_cxy_latch \
  --log-file "$LOG" \
  "$@"
