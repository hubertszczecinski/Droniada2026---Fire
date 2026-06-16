#!/usr/bin/env bash
# Test klucza Twitch BEZ UDP — wysyła kolorowe paski na kanał (~15 s).
# Jeśli to działa, problem jest po stronie nadajnika UDP, nie Twitcha.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=gigabyte/lib_stream.sh
source "${ROOT}/gigabyte/lib_stream.sh"
load_stream_env "${ROOT}/config/stream.env"

RTMP_URL="$(rtmp_publish_url)" || exit 1
DUR="${1:-30}"

echo "=== Test Twitch (videotestsrc → RTMP) ==="
echo "Ingest: ${TWITCH_RTMP_URL}/***"
echo "Czas: ${DUR}s"
if [[ -n "${TWITCH_CHANNEL_VISION:-}" ]]; then
  echo ""
  echo ">>> OGLĄDAJ TERAZ: https://www.twitch.tv/${TWITCH_CHANNEL_VISION}"
  echo ">>> (otwórz w przeglądarce ZANIM test się skończy)"
  echo ""
fi

AAC_ENC="avenc_aac"
if ! gst-inspect-1.0 "${AAC_ENC}" >/dev/null 2>&1; then
  AAC_ENC="voaacenc"
fi

gst-launch-1.0 -e \
  videotestsrc is-live=true pattern=smpte ! \
  video/x-raw,width=1280,height=720,framerate=30/1 ! \
  videoconvert ! video/x-raw,format=I420 ! \
  x264enc tune=zerolatency speed-preset=veryfast bitrate="${X264_BITRATE_KBPS}" key-int-max=60 ! \
  video/x-h264,profile=baseline ! h264parse ! queue ! mux. \
  flvmux name=mux streamable=true ! rtmpsink location="${RTMP_URL}" \
  audiotestsrc is-live=true wave=silence ! audioconvert ! audioresample ! \
  "${AAC_ENC}" bitrate=64000 ! queue ! mux. \
  &
PID=$!
sleep "${DUR}"
kill "${PID}" 2>/dev/null || true
wait "${PID}" 2>/dev/null || true
echo "Koniec testu."
