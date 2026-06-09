#!/usr/bin/env bash
# Test modułu A+B na Jetsonie — headless + MJPEG :8088
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export DRONIADA_DEVICE="${DRONIADA_DEVICE:-0}"
export DRONIADA_YOLO_BIAS="${DRONIADA_YOLO_BIAS:-1}"
export DRONIADA_YOLO_POSE_WEIGHTS="${DRONIADA_YOLO_POSE_WEIGHTS:-${ROOT}/runs/pose/droniada_real_finetune/weights/best.pt}"
export DRONIADA_CAMERA="${DRONIADA_CAMERA:-0}"

WEIGHTS="$DRONIADA_YOLO_POSE_WEIGHTS"
if [[ ! -f "$WEIGHTS" ]]; then
  echo "Brak wag YOLO: $WEIGHTS"
  exit 1
fi

mkdir -p "${ROOT}/dataset/results" "${ROOT}/dataset/live_dashboard"
LOG="${ROOT}/dataset/results/jetson_ab_$(date +%Y%m%d_%H%M%S).log"
MJPEG_PORT="${DRONIADA_MJPEG_PORT:-8088}"

echo "=== Test A+B (Jetson) ==="
echo "Kamera: ${DRONIADA_CAMERA}  MJPEG: http://$(hostname -I 2>/dev/null | awk '{print $1}'):${MJPEG_PORT}/"
echo "Log: $LOG"
echo ""

PREVIEW_ARGS=(--headless --mjpeg-port "${MJPEG_PORT}")
if [[ -n "${DISPLAY:-}" && "${DRONIADA_GUI:-}" == "1" ]]; then
  PREVIEW_ARGS=(--preview)
fi

exec python3 -m release.run_live_panel \
  --dashboard \
  --module-a \
  --cxy-latch \
  --corner-mode yolo_pose \
  --camera "$DRONIADA_CAMERA" \
  --rotate 180 \
  --camera-profile tarot_t10x_2a:wide \
  --preview-width 1280 \
  --cxy-stable-frames 5 \
  --no-debug \
  --interval-ms 350 \
  --log-file "$LOG" \
  "${PREVIEW_ARGS[@]}" \
  "$@"
