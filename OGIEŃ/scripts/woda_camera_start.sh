#!/usr/bin/env bash
# Start podglądu woda (:8087) — ten sam tor kamery co panel :8088 (OpenCV V4L2 + MJPEG).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export WS_ROOT_PATH="${WS_ROOT_PATH:-$ROOT}"
export WODA_HTTP_PORT="${WODA_HTTP_PORT:-8087}"
export WODA_WS_URL="${WODA_WS_URL:-ws://127.0.0.1:8765}"
export WODA_OUTPUT_DIR="${WODA_OUTPUT_DIR:-/migawka-woda}"
export WODA_CAMERA_FOURCC="${WODA_CAMERA_FOURCC:-MJPG}"
export WODA_CAMERA_BRIGHTNESS="${WODA_CAMERA_BRIGHTNESS:-60}"
export WODA_ROTATE="${WODA_ROTATE:-180}"
export WODA_STREAM_WIDTH="${WODA_STREAM_WIDTH:-960}"
export WODA_CAMERA_WIDTH="${WODA_CAMERA_WIDTH:-1920}"
export WODA_CAMERA_HEIGHT="${WODA_CAMERA_HEIGHT:-1080}"

# Detekcja USB Video + MJPEG (jak jetson_offline_start.sh).
if [[ -z "${WODA_VIDEO_DEVICE:-}" ]] && command -v v4l2-ctl >/dev/null 2>&1; then
  mapfile -t USB_VIDEO_DEVS < <(
    v4l2-ctl --list-devices 2>/dev/null | awk '
      /USB Video/ { in_usb=1; next }
      /^[^ \t]/ { in_usb=0 }
      in_usb && /\/dev\/video/ { print $1 }
    '
  )
  pick_dev=""
  for dev in "${USB_VIDEO_DEVS[@]}" /dev/video0; do
    [[ -e "$dev" ]] || continue
    if v4l2-ctl -d "$dev" --list-formats-ext 2>/dev/null | grep -q MJPG; then
      pick_dev="$dev"
      break
    fi
  done
  [[ -n "$pick_dev" ]] && export WODA_VIDEO_DEVICE="$pick_dev"
fi
export WODA_VIDEO_DEVICE="${WODA_VIDEO_DEVICE:-/dev/video0}"

if command -v v4l2-ctl >/dev/null 2>&1; then
  v4l2-ctl -d "${WODA_VIDEO_DEVICE}" \
    --set-fmt-video="width=${WODA_CAMERA_WIDTH},height=${WODA_CAMERA_HEIGHT},pixelformat=${WODA_CAMERA_FOURCC}" \
    2>/dev/null || true
  v4l2-ctl -d "${WODA_VIDEO_DEVICE}" --set-ctrl="brightness=${WODA_CAMERA_BRIGHTNESS}" 2>/dev/null || true
fi

echo sknr | sudo -S mkdir -p "$WODA_OUTPUT_DIR" 2>/dev/null || sudo mkdir -p "$WODA_OUTPUT_DIR"
echo sknr | sudo -S chmod 1777 "$WODA_OUTPUT_DIR" 2>/dev/null || true

echo "=== Migawka woda (OpenCV V4L2 jak :8088) ==="
echo "Kamera host: ${WODA_VIDEO_DEVICE} → kontener /dev/video0"
echo "Format:      ${WODA_CAMERA_FOURCC} ${WODA_CAMERA_WIDTH}x${WODA_CAMERA_HEIGHT} brightness=${WODA_CAMERA_BRIGHTNESS} rotate=${WODA_ROTATE}"
echo "Podgląd:     http://$(hostname -I | awk '{print $1}'):${WODA_HTTP_PORT}/"
echo "WS trigger:  ${WODA_WS_URL}"
echo "Zapis:       ${WODA_OUTPUT_DIR}/"

docker compose -f docker-compose.woda.yml up --build -d

sleep 5
if [[ -x "$ROOT/scripts/woda_probe_mjpeg.py" ]]; then
  echo
  echo "=== Probe kamery (std>8 = OK, std≈0 = brak sygnału HDMI) ==="
  docker compose -f docker-compose.woda.yml exec -T woda_camera \
    python3 /ws/scripts/woda_probe_mjpeg.py /dev/video0 2>/dev/null | tail -2 || true
fi

echo
echo "Test migawki: curl -X POST http://127.0.0.1:${WODA_HTTP_PORT}/api/snapshot"
