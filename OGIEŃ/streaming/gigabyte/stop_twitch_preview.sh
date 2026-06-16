#!/usr/bin/env bash
PIDFILE="${DRONIADA_TWITCH_PID:-/tmp/droniada_twitch_preview.pid}"
if [[ -f "${PIDFILE}" ]]; then
  kill "$(cat "${PIDFILE}")" 2>/dev/null || true
  rm -f "${PIDFILE}"
  echo "Zatrzymano stream Twitch."
else
  pkill -f 'stream_jetson_preview' 2>/dev/null || true
  echo "Brak pidfile — wysłano pkill do stream_jetson_preview*"
fi
