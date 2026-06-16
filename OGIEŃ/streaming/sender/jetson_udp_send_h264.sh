#!/usr/bin/env bash
# Przykład nadajnika: kamera Jetson → UDP RTP H.264 do Gigabyte.
# Dostosuj device i IP docelowe w config/stream.env lub poniżej.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=gigabyte/lib_stream.sh
source "${ROOT}/gigabyte/lib_stream.sh"
load_stream_env "${ROOT}/config/stream.env"

DEST="${UDP_DEST_HOST:-192.168.100.249}"
PORT="${UDP_DEST_PORT:-5600}"
DEVICE="${CAMERA_DEVICE:-/dev/video0}"
WIDTH="${CAMERA_WIDTH:-1280}"
HEIGHT="${CAMERA_HEIGHT:-720}"
FPS="${CAMERA_FPS:-30}"
BITRATE="${SEND_BITRATE_KBPS:-4000}"

echo "Nadajnik UDP RTP H.264 → ${DEST}:${PORT}"
echo "Kamera: ${DEVICE} ${WIDTH}x${HEIGHT}@${FPS}"

# x264enc na Jetsonie (software lub omxh264enc jeśli masz — tu uniwersalny x264)
gst-launch-1.0 -e \
  v4l2src device="${DEVICE}" ! \
  video/x-raw,width="${WIDTH}",height="${HEIGHT}",framerate="${FPS}"/1 ! \
  videoconvert ! video/x-raw,format=I420 ! \
  x264enc tune=zerolatency speed-preset=ultrafast bitrate="${BITRATE}" key-int-max=30 ! \
  video/x-h264,profile=baseline ! rtph264pay config-interval=1 pt=96 ! \
  udpsink host="${DEST}" port="${PORT}" sync=false async=false
