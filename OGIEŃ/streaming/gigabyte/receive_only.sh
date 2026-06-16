#!/usr/bin/env bash
# Tylko odbiór i podgląd (bez Twitch).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export RTMP_RELAY=0
export LOCAL_DISPLAY=1
exec "${ROOT}/gigabyte/start_lastmile.sh"
