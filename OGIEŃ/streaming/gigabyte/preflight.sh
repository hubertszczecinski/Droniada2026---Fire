#!/usr/bin/env bash
# Sprawdzenie config + pluginów przed startem.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=gigabyte/lib_stream.sh
source "${ROOT}/gigabyte/lib_stream.sh"

echo "=== Lastmile preflight ==="
print_stream_config_summary "${ROOT}/config/stream.env"
echo ""

MISS=0
for plug in udpsrc rtph264depay avdec_h264 x264enc flvmux rtmpsink; do
  if ! gst-inspect-1.0 "${plug}" >/dev/null 2>&1; then
    echo "BRAK pluginu GStreamer: ${plug} — uruchom ./gigabyte/install_deps.sh"
    MISS=1
  fi
done

load_stream_env "${ROOT}/config/stream.env"
if [[ "${RTMP_RELAY}" == "1" ]]; then
  if [[ -z "${TWITCH_STREAM_KEY:-}" ]]; then
    echo "BŁĄD: RTMP_RELAY=1 ale brak TWITCH_STREAM_KEY"
    echo "  → ./gigabyte/configure_twitch.sh"
    MISS=1
  fi
  if [[ "${TWITCH_RTMP_URL:-}" == *youtube* ]]; then
    echo "BŁĄD: ingest wskazuje YouTube zamiast Twitch"
    echo "  → ./gigabyte/migrate_config_to_twitch.sh"
    MISS=1
  fi
  if [[ "${TWITCH_STREAM_KEY:-}" == live_* ]] && [[ "${TWITCH_RTMP_URL:-}" != *twitch* ]]; then
    echo "BŁĄD: klucz Twitch (live_...) ale URL nie jest twitch.tv"
    MISS=1
  fi
fi

if [[ "${RTMP_RELAY}" != "1" ]] && [[ "${LOCAL_DISPLAY}" != "1" ]]; then
  echo "BŁĄD: RTMP_RELAY=0 i LOCAL_DISPLAY=0 — nic nie wystartuje"
  MISS=1
fi

if [[ "${MISS}" == "1" ]]; then
  exit 1
fi
echo "OK — możesz uruchomić ./gigabyte/start_lastmile.sh"
echo "Test samego klucza Twitch (bez UDP): ./gigabyte/test_twitch_key.sh"
