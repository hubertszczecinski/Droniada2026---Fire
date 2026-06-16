#!/usr/bin/env bash
# Wspólne funkcje pipeline GStreamer (source + env).
set -euo pipefail

_LASTMILE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TWITCH_RTMP_DEFAULT="rtmp://live.twitch.tv/app"

load_stream_env() {
  local env_file="${1:-${_LASTMILE_ROOT}/config/stream.env}"
  if [[ -f "${env_file}" ]]; then
    # shellcheck disable=SC1090
    set -a
    source "${env_file}"
    set +a
  elif [[ -f "${_LASTMILE_ROOT}/config/stream.env.example" ]]; then
    # shellcheck disable=SC1090
    set -a
    source "${_LASTMILE_ROOT}/config/stream.env.example"
    set +a
  fi
  UDP_PORT="${UDP_PORT:-5600}"
  UDP_BIND="${UDP_BIND:-0.0.0.0}"
  RTP_CODEC="${RTP_CODEC:-H264}"
  RTP_PAYLOAD="${RTP_PAYLOAD:-96}"
  LOCAL_DISPLAY="${LOCAL_DISPLAY:-1}"
  DISPLAY_SYNC="${DISPLAY_SYNC:-false}"
  X264_BITRATE_KBPS="${X264_BITRATE_KBPS:-3000}"
  X264_KEYINT="${X264_KEYINT:-60}"

  # Relay: nowa nazwa RTMP_RELAY lub stara YOUTUBE_RELAY
  if [[ -n "${RTMP_RELAY:-}" ]]; then
    :
  elif [[ -n "${YOUTUBE_RELAY:-}" ]]; then
    RTMP_RELAY="${YOUTUBE_RELAY}"
  else
    RTMP_RELAY=0
  fi

  # Klucz: TWITCH_STREAM_KEY lub stary YOUTUBE_STREAM_KEY (często wklejony do złej zmiennej)
  if [[ -n "${TWITCH_STREAM_KEY:-}" ]]; then
    :
  elif [[ -n "${YOUTUBE_STREAM_KEY:-}" ]]; then
    TWITCH_STREAM_KEY="${YOUTUBE_STREAM_KEY}"
  else
    TWITCH_STREAM_KEY=""
  fi

  # URL ingest — domyślnie Twitch; NIGDY nie używaj youtube.com jeśli nie ustawiono jawnie TWITCH_RTMP_URL
  if [[ -n "${TWITCH_RTMP_URL:-}" ]]; then
    TWITCH_RTMP_URL="${TWITCH_RTMP_URL%/}"
  elif [[ -n "${YOUTUBE_RTMP_BASE:-}" ]] && [[ "${YOUTUBE_RTMP_BASE}" != *youtube* ]]; then
    TWITCH_RTMP_URL="${YOUTUBE_RTMP_BASE%/}"
  else
    TWITCH_RTMP_URL="${TWITCH_RTMP_DEFAULT}"
  fi
}

rtmp_publish_url() {
  if [[ -z "${TWITCH_STREAM_KEY:-}" ]]; then
    echo "[lastmile] Brak TWITCH_STREAM_KEY w config/stream.env" >&2
    echo "[lastmile] Uruchom: ./gigabyte/configure_twitch.sh" >&2
    return 1
  fi
  if [[ "${TWITCH_RTMP_URL}" == *youtube* ]]; then
    echo "[lastmile] BŁĄD: TWITCH_RTMP_URL wskazuje YouTube — ustaw rtmp://live.twitch.tv/app" >&2
    return 1
  fi
  echo "${TWITCH_RTMP_URL}/${TWITCH_STREAM_KEY}"
}

print_stream_config_summary() {
  load_stream_env "${1:-${_LASTMILE_ROOT}/config/stream.env}"
  echo "  UDP odbiór:     ${UDP_BIND}:${UDP_PORT} (${RTP_CODEC})"
  echo "  Podgląd lokalny: LOCAL_DISPLAY=${LOCAL_DISPLAY}"
  echo "  Twitch relay:   RTMP_RELAY=${RTMP_RELAY}"
  if [[ "${RTMP_RELAY}" == "1" ]]; then
    if [[ -n "${TWITCH_STREAM_KEY:-}" ]]; then
      echo "  Klucz Twitch:   ustawiony (${#TWITCH_STREAM_KEY} znaków)"
    else
      echo "  Klucz Twitch:   BRAK"
    fi
    echo "  Ingest RTMP:    ${TWITCH_RTMP_URL}"
  fi
  if [[ -f "${_LASTMILE_ROOT}/config/stream.env" ]] && grep -q 'youtube.com' "${_LASTMILE_ROOT}/config/stream.env" 2>/dev/null; then
    echo "  UWAGA: stream.env zawiera youtube.com — uruchom ./gigabyte/migrate_config_to_twitch.sh"
  fi
}

rtp_caps() {
  local codec="${1:-H264}"
  local payload="${2:-96}"
  case "${codec^^}" in
    H265|HEVC)
      echo "application/x-rtp,media=video,encoding-name=H265,payload=${payload}"
      ;;
    *)
      echo "application/x-rtp,media=video,encoding-name=H264,payload=${payload}"
      ;;
  esac
}

build_display_sink() {
  local sync="${DISPLAY_SYNC:-false}"
  if [[ -n "${DISPLAY:-}" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    echo "autovideosink sync=${sync}"
  else
    echo "fakesink sync=${sync}"
  fi
}
