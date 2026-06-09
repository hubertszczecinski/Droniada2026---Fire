#!/usr/bin/env bash
# Zawody + raporty preset (skróty 1/2/3 na :8089). Wymaga config/preset_reports.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export WS_ROOT_PATH="${WS_ROOT_PATH:-$ROOT}"
export DRONIADA_REPORT_MODE=preset
export DRONIADA_PRESET_REPORTS="${DRONIADA_PRESET_REPORTS:-${ROOT}/config/preset_reports.json}"
if [[ ! -f "${DRONIADA_PRESET_REPORTS}" ]]; then
  echo "Brak ${DRONIADA_PRESET_REPORTS}" >&2
  echo "Skopiuj: cp config/preset_reports.example.json config/preset_reports.json i uzupełnij." >&2
  exit 1
fi
export DRONIADA_MISSION_PANELS="${DRONIADA_MISSION_PANELS:-A,B,C}"
export DRONIADA_SNAPSHOTS_PER_PANEL="${DRONIADA_SNAPSHOTS_PER_PANEL:-5}"
export DRONIADA_HOST_PORT="${DRONIADA_HOST_PORT:-8088}"
export DRONIADA_CONTROL_PORT="${DRONIADA_CONTROL_PORT:-8089}"
exec ./scripts/jetson_offline_start.sh
