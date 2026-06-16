#!/usr/bin/env bash
# Odbiór UDP + relay Twitch (bez lokalnego okna — headless).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export RTMP_RELAY=1
export LOCAL_DISPLAY=0
exec "${ROOT}/gigabyte/start_lastmile.sh"
