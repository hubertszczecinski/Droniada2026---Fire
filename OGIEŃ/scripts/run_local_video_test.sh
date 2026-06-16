#!/usr/bin/env bash
# Lokalny test na pliku wideo (Mac) — ten sam pipeline co Jetson + panel WWW.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv_yolo/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

VIDEO_REL="${1:-dataset/my_capture/Test.mov}"
VIDEO="${VIDEO_REL}"
if [[ "$VIDEO" != /* ]]; then
  VIDEO="${ROOT}/${VIDEO_REL}"
fi
if [[ ! -f "$VIDEO" ]]; then
  echo "Brak pliku: ${VIDEO}" >&2
  exit 1
fi

WEIGHTS="${ROOT}/runs/pose/droniada_real_finetune/weights/best.pt"
if [[ ! -f "$WEIGHTS" ]]; then
  echo "Brak wag YOLO: ${WEIGHTS}" >&2
  exit 1
fi

export DRONIADA_YOLO_POSE_WEIGHTS="${DRONIADA_YOLO_POSE_WEIGHTS:-$WEIGHTS}"
export DRONIADA_YOLO_BIAS="${DRONIADA_YOLO_BIAS:-1}"
export DRONIADA_MISSION_PANELS="${DRONIADA_MISSION_PANELS:-A,B,C}"
export DRONIADA_SNAPSHOTS_PER_PANEL="${DRONIADA_SNAPSHOTS_PER_PANEL:-5}"
unset DRONIADA_PANEL_LAYOUT
if [[ -f "${ROOT}/config/card_colors.json" ]]; then
  export DRONIADA_CARD_COLORS="${ROOT}/config/card_colors.json"
  echo "Kolory kartek: config/card_colors.json (kalibracja zawodowa)"
fi
export OPENCV_VIDEOIO_PRIORITY_LIST="${DRONIADA_OPENCV_VIDEOIO:-FFMPEG,GSTREAMER,V4L2}"

HOST_PORT="${DRONIADA_HOST_PORT:-8088}"
CONTROL_PORT="${DRONIADA_CONTROL_PORT:-$((HOST_PORT + 1))}"
export DRONIADA_STREAM_SOURCE="${DRONIADA_STREAM_SOURCE:-dashboard}"
export DRONIADA_STREAM_WIDTH="${DRONIADA_STREAM_WIDTH:-1280}"
LOOP="${DRONIADA_VIDEO_LOOP:-1}"
LOG="${ROOT}/dataset/results/local_video_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${ROOT}/dataset/results" "${ROOT}/dataset/live_dashboard"

EXTRA_LOOP=""
if [[ "$LOOP" != "1" ]]; then
  EXTRA_LOOP="--no-loop"
fi

echo "=== Droniada — test lokalny (wideo) ==="
echo "Plik:  ${VIDEO_REL}"
echo "Wagi:  ${WEIGHTS}"
echo "Podgląd (ramki):  http://127.0.0.1:${HOST_PORT}/"
echo "Sterowanie:       http://127.0.0.1:${CONTROL_PORT}/"
echo "Panele misji:     ${DRONIADA_MISSION_PANELS:-A,B,C}"
echo "Log:   ${LOG}"
echo "Stop:  Ctrl+C"
echo ""

exec "$PY" -m release.run_live_panel \
  --dashboard \
  --module-a \
  --cxy-latch \
  --corner-mode yolo_pose \
  --video "$VIDEO" \
  $EXTRA_LOOP \
  --rotate 180 \
  --camera-profile tarot_t10x_2a:wide \
  --preview-width 1280 \
  --cxy-latch-dir dataset/debug_cxy_latch \
  --smooth-alpha "${DRONIADA_SMOOTH_ALPHA:-0.28}" \
  --hold-frames "${DRONIADA_HOLD_FRAMES:-14}" \
  --interval-ms "${DRONIADA_INTERVAL_MS:-350}" \
  --web-port "$HOST_PORT" \
  --web-control-port "$CONTROL_PORT" \
  --web-host 127.0.0.1 \
  --mission-panels "${DRONIADA_MISSION_PANELS:-A,B,C}" \
  --snapshots-per-panel "${DRONIADA_SNAPSHOTS_PER_PANEL:-5}" \
  --snapshot-competition-min-votes "${DRONIADA_SNAPSHOT_COMP_MIN_VOTES:-2}" \
  --snapshot-competition-min-ratio "${DRONIADA_SNAPSHOT_COMP_MIN_RATIO:-0.0}" \
  --headless \
  --no-debug \
  --log-file "$LOG"
