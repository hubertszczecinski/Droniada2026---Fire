#!/usr/bin/env bash
# Test na Jetsonie z pliku wideo zamiast kamery USB (panel WWW na :8088).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export WS_ROOT_PATH="${WS_ROOT_PATH:-$ROOT}"
HOST_PORT="${DRONIADA_HOST_PORT:-8088}"
WEIGHTS="${ROOT}/runs/pose/droniada_real_finetune/weights/best.pt"

VIDEO_REL="${1:-dataset/my_capture/Droniada_nag5.mov}"
VIDEO_HOST="${ROOT}/${VIDEO_REL}"
VIDEO_CONTAINER="/ws/${VIDEO_REL}"

echo "=== Droniada — test wideo ==="
echo "Plik: ${VIDEO_CONTAINER}"

if [[ ! -f "${VIDEO_HOST}" ]]; then
  echo "Brak pliku na hoście: ${VIDEO_HOST}" >&2
  echo "Na Macu: ./scripts/sync_jetson_video.sh ${VIDEO_REL}" >&2
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

unset DRONIADA_VIDEO_DEVICE
export DRONIADA_VIDEO_FILE="${VIDEO_CONTAINER}"
export DRONIADA_VIDEO_LOOP="${DRONIADA_VIDEO_LOOP:-1}"

echo "Pętla: ${DRONIADA_VIDEO_LOOP} (0 = jeden przebieg, 1 = zapętlenie)"
echo "Wagi: ${WEIGHTS}"
if [[ -f "${ROOT}/config/card_colors.json" ]]; then
  export DRONIADA_CARD_COLORS="/ws/config/card_colors.json"
  echo "Kolory kartek: config/card_colors.json"
else
  echo "Kolory kartek: domyślne HSV (kalibracja: ./scripts/calibrate_card_colors.sh)"
fi

docker compose -f docker-compose.jetson.yml up -d --force-recreate
sleep 10

if docker ps --filter name=droniada_vision --format '{{.Status}}' | grep -qi up; then
  IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
  echo ""
  if docker logs droniada_vision 2>&1 | grep -qE 'Panel WWW:|source=video|vid_000001'; then
    echo "OK — kontener odtwarza wideo + panel WWW."
  else
    echo "UWAGA: sprawdź logi." >&2
    docker logs droniada_vision 2>&1 | tail -12
    exit 1
  fi
  echo "Panel WWW:  http://${IP:-<jetson-ip>}:${HOST_PORT}/"
  echo "Powrót do kamery: ./scripts/jetson_offline_start.sh"
  docker logs droniada_vision 2>&1 | tail -10
else
  echo "Kontener nie wystartował." >&2
  docker logs droniada_vision 2>&1 | tail -15
  exit 1
fi
