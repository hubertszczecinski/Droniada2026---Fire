#!/usr/bin/env bash
# Gigabyte: podgląd Jetson → Twitch (w tle). Log: /tmp/droniada_twitch_preview.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${DRONIADA_TWITCH_LOG:-/tmp/droniada_twitch_preview.log}"
PIDFILE="${DRONIADA_TWITCH_PID:-/tmp/droniada_twitch_preview.pid}"

if [[ -f "${PIDFILE}" ]] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
  echo "Stream już działa (pid $(cat "${PIDFILE}")). Stop: ./gigabyte/stop_twitch_preview.sh"
  exit 0
fi

if command -v ffmpeg >/dev/null 2>&1; then
  STREAM_CMD=("${ROOT}/gigabyte/stream_jetson_preview_to_twitch.sh")
else
  STREAM_CMD=("${ROOT}/gigabyte/stream_jetson_preview_gst.sh")
fi

nohup "${STREAM_CMD[@]}" >>"${LOG}" 2>&1 &
echo $! > "${PIDFILE}"
echo "Start stream → Twitch (pid $(cat "${PIDFILE}"))"
echo "Log: ${LOG}"
"${ROOT}/gigabyte/where_to_watch.sh"
