#!/usr/bin/env bash
# Konfiguracja Twitch — interaktywnie lub: ./configure_twitch.sh live_TWOJ_KLUCZ [nazwa_kanalu]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV="${ROOT}/config/stream.env"
EXAMPLE="${ROOT}/config/stream.env.example"

if [[ ! -f "${ENV}" ]]; then
  cp "${EXAMPLE}" "${ENV}"
fi

# shellcheck disable=SC1090
source "${ENV}" 2>/dev/null || true

CLI_KEY="${1:-}"
CLI_CHANNEL="${2:-}"

echo "=== Konfiguracja Twitch (lastmile) ==="
echo ""
echo "Klucz: https://dashboard.twitch.tv/settings/stream → Klucz podstawowy streamu"
echo ""

CURRENT="${TWITCH_STREAM_KEY:-${YOUTUBE_STREAM_KEY:-}}"

if [[ -n "${CLI_KEY}" ]]; then
  CURRENT="${CLI_KEY}"
  RELAY=1
  CHANNEL="${CLI_CHANNEL:-${TWITCH_CHANNEL_VISION:-}}"
  echo "Tryb: argument z linii poleceń (relay=włączony)"
else
  if [[ -n "${CURRENT}" ]]; then
    echo "Obecny klucz: ustawiony (${#CURRENT} znaków). Enter = zostaw."
  else
    echo "Obecny klucz: BRAK"
  fi
  read -r -p "Wklej TWITCH_STREAM_KEY (live_...): " NEW_KEY
  if [[ -n "${NEW_KEY}" ]]; then
    CURRENT="${NEW_KEY}"
  fi
  if [[ -z "${CURRENT}" ]]; then
    echo ""
    echo "Brak klucza. Użyj jednej komendy (wklej swój live_...):" >&2
    echo "  ./gigabyte/configure_twitch.sh 'live_TWOJ_KLUCZ_TUTAJ'" >&2
    exit 1
  fi
  read -r -p "Włączyć relay na Twitch? [Y/n]: " RELAY_ANS
  RELAY=1
  [[ "${RELAY_ANS}" =~ ^[Nn] ]] && RELAY=0
  read -r -p "Nazwa kanału Twitch (opcjonalnie): " CHANNEL
  CHANNEL="${CHANNEL:-${TWITCH_CHANNEL_VISION:-}}"
fi

if [[ -z "${CURRENT}" ]]; then
  echo "Brak klucza — przerwano." >&2
  exit 1
fi

UDP_PORT="${UDP_PORT:-5600}"
RTP_CODEC="${RTP_CODEC:-H264}"
LOCAL_DISPLAY="${LOCAL_DISPLAY:-1}"

cat > "${ENV}" <<EOF
# Droniada lastmile — Twitch

UDP_PORT=${UDP_PORT}
RTP_CODEC=${RTP_CODEC}
RTP_PAYLOAD=${RTP_PAYLOAD:-96}
UDP_BIND=${UDP_BIND:-0.0.0.0}
LOCAL_DISPLAY=${LOCAL_DISPLAY}
DISPLAY_SYNC=${DISPLAY_SYNC:-false}

RTMP_RELAY=${RELAY}
TWITCH_STREAM_KEY=${CURRENT}
TWITCH_RTMP_URL=rtmp://live.twitch.tv/app

X264_BITRATE_KBPS=${X264_BITRATE_KBPS:-3000}
X264_KEYINT=${X264_KEYINT:-60}

TWITCH_CHANNEL_VISION=${CHANNEL}
TWITCH_CHANNEL_QGC=${TWITCH_CHANNEL_QGC:-}

UDP_DEST_HOST=${UDP_DEST_HOST:-192.168.100.249}
UDP_DEST_PORT=${UDP_DEST_PORT:-5600}
EOF

echo ""
echo "Zapisano ${ENV} (klucz: ${#CURRENT} znaków, RTMP_RELAY=${RELAY})"
"${ROOT}/gigabyte/preflight.sh"
