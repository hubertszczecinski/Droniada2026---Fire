#!/usr/bin/env bash
# Twitch: podgląd Droniada z Jetsona (MJPEG :8088/stream.mjpg) → RTMP.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=gigabyte/lib_stream.sh
source "${ROOT}/gigabyte/lib_stream.sh"
load_stream_env "${ROOT}/config/stream.env"

JETSON_HOST="${JETSON_PREVIEW_HOST:-192.168.100.200}"
JETSON_PORT="${JETSON_PREVIEW_PORT:-8088}"
PREVIEW_URL="${JETSON_PREVIEW_URL:-http://${JETSON_HOST}:${JETSON_PORT}/stream.mjpg}"
BR="${STREAM_PREVIEW_BITRATE_KBPS:-${X264_BITRATE_KBPS:-3000}}"

RTMP_URL="$(rtmp_publish_url)" || exit 1

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Brak ffmpeg — sudo apt install ffmpeg" >&2
  exit 1
fi

echo "=== Stream podglądu → Twitch ==="
echo "Źródło: ${PREVIEW_URL}"
echo "Twitch: ${TWITCH_RTMP_URL}/***"
if [[ -n "${TWITCH_CHANNEL_VISION:-}" ]]; then
  echo "Oglądaj: https://www.twitch.tv/${TWITCH_CHANNEL_VISION}"
fi
echo "Ctrl+C = stop"
echo ""

HEALTH_URL="${JETSON_PREVIEW_HEALTH_URL:-http://${JETSON_HOST}:${JETSON_PORT}/live.jpg}"

# Czekaj aż Jetson wystawi podgląd (max ~90 s)
for i in $(seq 1 45); do
  if curl -sf --max-time 2 -o /dev/null "${HEALTH_URL}" 2>/dev/null; then
    echo "OK — Jetson podgląd dostępny (${HEALTH_URL})"
    break
  fi
  if [[ "$i" -eq 45 ]]; then
    echo "BŁĄD: brak ${HEALTH_URL} — uruchom najpierw Droniada na Jetsonie (:8088)" >&2
    exit 1
  fi
  sleep 2
done

FPS="${DRONIADA_TWITCH_FPS:-25}"
exec ffmpeg -nostdin -loglevel warning \
  -fflags nobuffer -flags low_delay \
  -f mjpeg -reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 \
  -i "${PREVIEW_URL}" \
  -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=${FPS}" \
  -c:v libx264 -preset superfast -tune zerolatency \
  -b:v "${BR}k" -maxrate "$((BR + 500))k" -bufsize "$((BR * 2))k" \
  -g "${FPS}" -pix_fmt yuv420p \
  -f flv "${RTMP_URL}"
