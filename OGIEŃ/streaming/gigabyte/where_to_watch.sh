#!/usr/bin/env bash
# Wypisz gdzie oglądać stream + jak przetestować.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=gigabyte/lib_stream.sh
source "${ROOT}/gigabyte/lib_stream.sh"
load_stream_env "${ROOT}/config/stream.env"

CHANNEL="${TWITCH_CHANNEL_VISION:-}"
if [[ -z "${CHANNEL}" ]]; then
  echo "Brak TWITCH_CHANNEL_VISION w config/stream.env"
  echo "Ustaw nazwę kanału (login z twitch.tv/NAZWA):"
  echo "  ./gigabyte/configure_twitch.sh 'live_KLUCZ' TWOJA_NAZWA"
  exit 1
fi

echo "=== Gdzie oglądać stream ==="
echo ""
echo "  Kanał (gdy NADAWANIE działa):"
echo "    https://www.twitch.tv/${CHANNEL}"
echo ""
echo "  Podgląd u siebie (Creator Dashboard):"
echo "    https://dashboard.twitch.tv/u/${CHANNEL}/stream-manager"
echo ""
if [[ -n "${TWITCH_STREAM_KEY:-}" ]]; then
  echo "  Klucz streamu: OK (${#TWITCH_STREAM_KEY} znaków)"
else
  echo "  Klucz streamu: BRAK — ./gigabyte/configure_twitch.sh 'live_...'"
fi
echo "  Relay RTMP:    RTMP_RELAY=${RTMP_RELAY}"
echo ""
echo "=== Test (60 s kolorowych pasków) ==="
echo "  1. Otwórz w drugiej karcie: https://www.twitch.tv/${CHANNEL}"
echo "  2. Uruchom tutaj: ./gigabyte/test_twitch_key.sh 60"
echo "  3. Odśwież Twitch po ~10 s — powinna być plakietka NA ŻYWO"
echo ""
