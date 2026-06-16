#!/usr/bin/env bash
# Tunel SSH: dashboard Jetson na localhost (gdy laptop nie jest w SKNR_LAN).
#
# Na laptopie (Mac), w osobnym terminalu:
#   DRONIADA_JETSON_PASS=sknr ./scripts/jetson_dashboard_tunnel.sh
# Potem w przeglądarce:
#   http://127.0.0.1:8088/   — podgląd
#   http://127.0.0.1:8089/   — sterowanie
set -euo pipefail
HOST="${DRONIADA_JETSON_HOST:-sknr@192.168.100.200}"
PASS="${DRONIADA_JETSON_PASS:-}"
PORT="${DRONIADA_HOST_PORT:-8088}"
CTRL="${DRONIADA_CONTROL_PORT:-8089}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -N -L "127.0.0.1:${PORT}:127.0.0.1:${PORT}" -L "127.0.0.1:${CTRL}:127.0.0.1:${CTRL}")

echo "Tunel SSH → Jetson (${HOST})"
echo "  Podgląd:    http://127.0.0.1:${PORT}/"
echo "  Sterowanie: http://127.0.0.1:${CTRL}/"
echo "Zostaw ten terminal otwarty (Ctrl+C = koniec tunelu)."
echo ""

if [[ -n "${PASS}" ]] && command -v sshpass >/dev/null 2>&1; then
  exec sshpass -p "${PASS}" ssh "${SSH_OPTS[@]}" "${HOST}"
else
  exec ssh "${SSH_OPTS[@]}" "${HOST}"
fi
