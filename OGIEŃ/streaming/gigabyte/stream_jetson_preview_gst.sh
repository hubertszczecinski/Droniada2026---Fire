#!/usr/bin/env bash
# Twitch: podgląd Jetson (MJPEG) → RTMP — wersja GStreamer (bez ffmpeg).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=gigabyte/lib_stream.sh
source "${ROOT}/gigabyte/lib_stream.sh"
load_stream_env "${ROOT}/config/stream.env"

JETSON_HOST="${JETSON_PREVIEW_HOST:-192.168.100.200}"
JETSON_PORT="${JETSON_PREVIEW_PORT:-8088}"
PREVIEW_URL="${JETSON_PREVIEW_URL:-http://${JETSON_HOST}:${JETSON_PORT}/stream.mjpg}"

RTMP_URL="$(rtmp_publish_url)" || exit 1

echo "=== Stream podglądu → Twitch (GStreamer) ==="
echo "Źródło: ${PREVIEW_URL}"
if [[ -n "${TWITCH_CHANNEL_VISION:-}" ]]; then
  echo "Oglądaj: https://www.twitch.tv/${TWITCH_CHANNEL_VISION}"
fi

HEALTH_URL="${JETSON_PREVIEW_HEALTH_URL:-http://${JETSON_HOST}:${JETSON_PORT}/live.jpg}"

for i in $(seq 1 45); do
  if curl -sf --max-time 2 -o /dev/null "${HEALTH_URL}" 2>/dev/null; then
    echo "OK — Jetson podgląd dostępny (${HEALTH_URL})"
    break
  fi
  [[ "$i" -eq 45 ]] && { echo "Brak ${HEALTH_URL}" >&2; exit 1; }
  sleep 2
done

# Twitch często nie pokazuje NA ŻYWO bez ścieżki audio — dodajemy ciche AAC.
AAC_ENC="avenc_aac"
if ! gst-inspect-1.0 "${AAC_ENC}" >/dev/null 2>&1; then
  AAC_ENC="voaacenc"
fi

STREAM_FPS="${DRONIADA_TWITCH_FPS:-25}"
STREAM_W="${DRONIADA_TWITCH_WIDTH:-1280}"
STREAM_H="${DRONIADA_TWITCH_HEIGHT:-720}"
BITRATE="${X264_BITRATE_KBPS:-3000}"

if gst-inspect-1.0 nvh264enc >/dev/null 2>&1; then
  echo "Enkoder wideo: nvh264enc (NVENC)"
  H264_ENC="nvh264enc preset=low-latency-hq rc-mode=cbr bitrate=${BITRATE} zerolatency=true gop-size=${STREAM_FPS} ! video/x-h264,profile=baseline ! h264parse"
elif gst-inspect-1.0 vaapih264enc >/dev/null 2>&1; then
  echo "Enkoder wideo: vaapih264enc"
  H264_ENC="vaapih264enc rate-control=cbr bitrate=${BITRATE} keyframe-period=${STREAM_FPS} ! h264parse"
else
  echo "Enkoder wideo: x264enc (software)"
  H264_ENC="x264enc tune=zerolatency speed-preset=superfast bitrate=${BITRATE} key-int-max=${STREAM_FPS} bframes=0 ! video/x-h264,profile=baseline ! h264parse"
fi

exec gst-launch-1.0 -e \
  souphttpsrc location="${PREVIEW_URL}" blocksize=1048576 do-timestamp=true ! \
  multipartdemux ! jpegparse ! jpegdec ! videoconvert ! \
  videoscale ! video/x-raw,width="${STREAM_W}",height="${STREAM_H}" ! \
  videorate drop-only=false max-rate="${STREAM_FPS}" ! \
  video/x-raw,framerate="${STREAM_FPS}"/1 ! \
  ${H264_ENC} ! \
  queue max-size-buffers=4 leaky=downstream ! mux. \
  flvmux name=mux streamable=true ! \
  queue max-size-buffers=4 leaky=downstream ! \
  rtmpsink location="${RTMP_URL}" sync=false async=false \
  audiotestsrc is-live=true wave=silence ! audioconvert ! audioresample ! \
  "${AAC_ENC}" bitrate=64000 ! queue max-size-buffers=2 leaky=downstream ! mux.
