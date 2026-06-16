#!/usr/bin/env bash
# Sync kodu Mac → Jetson + restart zawodów (migawki jak lokalny test wideo).
# Użycie:
#   DRONIADA_JETSON_PASS=sknr ./scripts/jetson_competition_deploy.sh
#   DRONIADA_JETSON_PASS=sknr ./scripts/jetson_competition_deploy.sh --no-start
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_HOST="${DRONIADA_JETSON_HOST:-sknr@192.168.100.200}"
REMOTE_DIR="${DRONIADA_JETSON_DIR:-acoustics/droniada}"
PASS="${DRONIADA_JETSON_PASS:-}"
START=1
if [[ "${1:-}" == "--no-start" ]]; then
  START=0
fi

echo "=== Droniada — deploy zawodów na Jetson ==="
echo "Host: ${REMOTE_HOST}"
echo ""

DRONIADA_JETSON_HOST="${REMOTE_HOST}" DRONIADA_JETSON_PASS="${PASS}" \
  "${ROOT}/scripts/sync_to_jetson.sh"

if [[ "${START}" -eq 0 ]]; then
  echo ""
  echo "Sync OK (bez restartu). Na Jetsonie:"
  echo "  cd ~/${REMOTE_DIR} && export WS_ROOT_PATH=\$(pwd)"
  echo "  DRONIADA_FORCE_RECREATE=1 ./scripts/jetson_competition_start.sh"
  exit 0
fi

SSH=(ssh -o StrictHostKeyChecking=accept-new)
if [[ -n "${PASS}" ]] && command -v sshpass >/dev/null 2>&1; then
  SSH=(sshpass -p "${PASS}" ssh -o StrictHostKeyChecking=accept-new)
fi

echo ""
echo "→ Restart kontenera na Jetsonie…"
"${SSH[@]}" "${REMOTE_HOST}" \
  "cd ~/${REMOTE_DIR} && export WS_ROOT_PATH=\$(pwd) && DRONIADA_FORCE_RECREATE=1 ./scripts/jetson_competition_start.sh"
