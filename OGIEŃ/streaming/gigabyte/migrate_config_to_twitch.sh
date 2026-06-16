#!/usr/bin/env bash
# Jednorazowa migracja starego stream.env (YouTube) → Twitch.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV="${ROOT}/config/stream.env"
EXAMPLE="${ROOT}/config/stream.env.example"
BACKUP="${ENV}.bak.$(date +%Y%m%d_%H%M%S)"

if [[ ! -f "${ENV}" ]]; then
  cp "${EXAMPLE}" "${ENV}"
  echo "Utworzono ${ENV} z szablonu Twitch."
  exit 0
fi

# shellcheck disable=SC1090
source "${ENV}" 2>/dev/null || true
OLD_KEY="${TWITCH_STREAM_KEY:-${YOUTUBE_STREAM_KEY:-}}"
OLD_RELAY="${RTMP_RELAY:-${YOUTUBE_RELAY:-0}}"

cp "${ENV}" "${BACKUP}"
echo "Kopia zapasowa: ${BACKUP}"

cat > "${ENV}" <<EOF
# Droniada lastmile — Twitch (zmigrowano z YouTube $(date +%Y-%m-%d))

UDP_PORT=${UDP_PORT:-5600}
RTP_CODEC=${RTP_CODEC:-H264}
RTP_PAYLOAD=${RTP_PAYLOAD:-96}
UDP_BIND=${UDP_BIND:-0.0.0.0}
LOCAL_DISPLAY=${LOCAL_DISPLAY:-1}
DISPLAY_SYNC=${DISPLAY_SYNC:-false}

RTMP_RELAY=${OLD_RELAY}
TWITCH_STREAM_KEY=${OLD_KEY}
TWITCH_RTMP_URL=rtmp://live.twitch.tv/app

X264_BITRATE_KBPS=${X264_BITRATE_KBPS:-3000}
X264_KEYINT=${X264_KEYINT:-60}

TWITCH_CHANNEL_VISION=${TWITCH_CHANNEL_VISION:-}
TWITCH_CHANNEL_QGC=${TWITCH_CHANNEL_QGC:-}

UDP_DEST_HOST=${UDP_DEST_HOST:-192.168.100.249}
UDP_DEST_PORT=${UDP_DEST_PORT:-5600}
EOF

echo "OK — ${ENV} ustawiony pod Twitch."
if [[ -z "${OLD_KEY}" ]]; then
  echo "Wklej klucz: ./gigabyte/configure_twitch.sh"
else
  echo "Klucz zachowany (${#OLD_KEY} znaków). Sprawdź: ./gigabyte/preflight.sh"
fi
