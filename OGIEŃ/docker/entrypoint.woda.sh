#!/usr/bin/env bash
set -euo pipefail

cd /ws
export PYTHONPATH="/ws:${PYTHONPATH:-}"
export OPENCV_VIDEOIO_PRIORITY_LIST="${OPENCV_VIDEOIO_PRIORITY_LIST:-V4L2}"

mkdir -p "${WODA_OUTPUT_DIR:-/migawka-woda}"

if [[ "${1:-}" == "woda-camera" ]]; then
  shift
  echo "[woda] kamera=${WODA_CAMERA_DEVICE:-/dev/video0}"
  echo "[woda] podgląd http://<jetson>:${WODA_HTTP_PORT:-8087}/"
  echo "[woda] migawki → ${WODA_OUTPUT_DIR:-/migawka-woda}"
  echo "[woda] trigger WS ${WODA_WS_URL:-ws://127.0.0.1:8765} (eventy: ${WODA_SNAPSHOT_ON:-hold_started,...})"
  exec python3 -m release.woda_camera_panel "$@"
fi

exec "$@"
