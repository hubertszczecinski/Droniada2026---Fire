#!/usr/bin/env bash
# Test na Jetsonie z pliku wideo — ten sam pipeline co zawody + pełny dashboard na :8088.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export WS_ROOT_PATH="${WS_ROOT_PATH:-$ROOT}"
HOST_PORT="${DRONIADA_HOST_PORT:-8088}"
CONTROL_PORT="${DRONIADA_CONTROL_PORT:-8089}"
WEIGHTS="${ROOT}/runs/pose/droniada_real_finetune/weights/best.pt"

VIDEO_ARG="${1:-videos/Test.mov}"
if [[ "$VIDEO_ARG" == /* ]]; then
  VIDEO_HOST="$VIDEO_ARG"
  VIDEO_REL="${VIDEO_ARG#${ROOT}/}"
  VIDEO_CONTAINER="/ws/${VIDEO_REL#${ROOT}/}"
  if [[ "$VIDEO_CONTAINER" == /ws/* ]]; then
    :
  else
    VIDEO_CONTAINER="/ws/${VIDEO_ARG#/}"
  fi
else
  VIDEO_REL="$VIDEO_ARG"
  VIDEO_HOST="${ROOT}/${VIDEO_REL}"
  VIDEO_CONTAINER="/ws/${VIDEO_REL}"
fi

# Po sync_jetson_video.sh plik ląduje w videos/ na hoście Jetsona.
if [[ ! -f "${VIDEO_HOST}" && "$VIDEO_ARG" == videos/* ]]; then
  VIDEO_HOST="${ROOT}/${VIDEO_ARG}"
fi
if [[ ! -f "${VIDEO_HOST}" && -f "${ROOT}/videos/$(basename "$VIDEO_ARG")" ]]; then
  VIDEO_HOST="${ROOT}/videos/$(basename "$VIDEO_ARG")"
  VIDEO_CONTAINER="/ws/videos/$(basename "$VIDEO_ARG")"
fi
if [[ ! -f "${VIDEO_HOST}" && -f "${ROOT}/dataset/my_capture/$(basename "$VIDEO_ARG")" ]]; then
  VIDEO_HOST="${ROOT}/dataset/my_capture/$(basename "$VIDEO_ARG")"
  VIDEO_CONTAINER="/ws/dataset/my_capture/$(basename "$VIDEO_ARG")"
fi

echo "=== Droniada — test wideo (konfiguracja zawodowa) ==="
echo "Plik: ${VIDEO_CONTAINER}"

if [[ ! -f "${VIDEO_HOST}" ]]; then
  echo "Brak pliku na hoście: ${VIDEO_HOST}" >&2
  echo "Na Macu: DRONIADA_JETSON_PASS=sknr ./scripts/sync_jetson_video.sh dataset/my_capture/Test.mov" >&2
  exit 1
fi

if ! docker image inspect droniada-vision:latest >/dev/null 2>&1; then
  echo "Brak obrazu droniada-vision:latest." >&2
  exit 1
fi

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Brak wag YOLO: ${WEIGHTS}" >&2
  exit 1
fi

unset DRONIADA_VIDEO_DEVICE DRONIADA_CAMERA_DEVICE
export DRONIADA_VIDEO_FILE="${VIDEO_CONTAINER}"
export DRONIADA_VIDEO_LOOP="${DRONIADA_VIDEO_LOOP:-1}"
export DRONIADA_HOST_PORT="${HOST_PORT}"
export DRONIADA_CONTROL_PORT="${CONTROL_PORT}"
export DRONIADA_MISSION_PANELS="${DRONIADA_MISSION_PANELS:-A,B,C}"
export DRONIADA_SNAPSHOTS_PER_PANEL="${DRONIADA_SNAPSHOTS_PER_PANEL:-5}"
export DRONIADA_USE_GST_CAPTURE=0
export DRONIADA_STREAM_PASSTHROUGH=0
# Pełny dashboard (parametry, miniatury, raport) na /stream.mjpg — jak lokalny run_local_video_test.sh
export DRONIADA_STREAM_SOURCE="${DRONIADA_STREAM_SOURCE:-dashboard}"
export DRONIADA_STREAM_WIDTH="${DRONIADA_STREAM_WIDTH:-1280}"
export DRONIADA_PREVIEW_WIDTH="${DRONIADA_PREVIEW_WIDTH:-1280}"
export DRONIADA_STREAM_INTERVAL_S="${DRONIADA_STREAM_INTERVAL_S:-0.033}"
export DRONIADA_VIS_PREVIEW_INTERVAL_S="${DRONIADA_VIS_PREVIEW_INTERVAL_S:-0.033}"
export DRONIADA_STREAM_DRAW_STATUS="${DRONIADA_STREAM_DRAW_STATUS:-1}"
export DRONIADA_SMOOTH_ALPHA="${DRONIADA_SMOOTH_ALPHA:-0.28}"
export DRONIADA_HOLD_FRAMES="${DRONIADA_HOLD_FRAMES:-14}"
export DRONIADA_INTERVAL_MS="${DRONIADA_INTERVAL_MS:-350}"
export DRONIADA_VIDEO_LOOP="${DRONIADA_VIDEO_LOOP:-1}"
# Analiza zsynchronizowana z tempem wideo (nie przewijaj 100× szybciej niż źródło).
export DRONIADA_ASYNC_ANALYSIS="${DRONIADA_ASYNC_ANALYSIS:-1}"
export DRONIADA_OPENCV_VIDEOIO="${DRONIADA_OPENCV_VIDEOIO:-FFMPEG,GSTREAMER,V4L2}"

echo "Pętla: ${DRONIADA_VIDEO_LOOP} (0 = jeden przebieg, 1 = zapętlenie)"
echo "Stream: ${DRONIADA_STREAM_SOURCE} · draw_status=${DRONIADA_STREAM_DRAW_STATUS}"
echo "Wagi: ${WEIGHTS}"
if [[ -f "${ROOT}/config/card_colors.json" ]]; then
  export DRONIADA_CARD_COLORS="/ws/config/card_colors.json"
  echo "Kolory kartek: config/card_colors.json"
fi

docker compose -f docker-compose.jetson.yml up -d --force-recreate
sleep 12

if docker ps --filter name=droniada_vision --format '{{.Status}}' | grep -qi up; then
  IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
  echo ""
  if docker logs droniada_vision 2>&1 | grep -qE 'source=video|Panel WWW:|vid_'; then
    echo "OK — kontener odtwarza wideo + panel WWW."
  else
    echo "UWAGA: sprawdź logi." >&2
    docker logs droniada_vision 2>&1 | tail -15
    exit 1
  fi
  echo "Podgląd:    http://${IP:-<jetson-ip>}:${HOST_PORT}/"
  echo "Sterowanie: http://${IP:-<jetson-ip>}:${CONTROL_PORT}/"
  echo "Powrót do kamery: ./scripts/jetson_competition_start.sh"
  docker logs droniada_vision 2>&1 | tail -12
else
  echo "Kontener nie wystartował." >&2
  docker logs droniada_vision 2>&1 | tail -15
  exit 1
fi
