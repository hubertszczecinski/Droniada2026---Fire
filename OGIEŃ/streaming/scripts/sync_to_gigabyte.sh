#!/usr/bin/env bash
# Rsync modułu lastmile na Gigabyte (robot@192.168.100.249).
# Nie nadpisuje config/stream.env jeśli już istnieje na celu.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_HOST="${GIGABYTE_HOST:-robot@192.168.100.249}"
REMOTE_DIR="${GIGABYTE_DIR:-droniada_lastmile}"

echo "→ ${REMOTE_HOST}:${REMOTE_DIR}/"
rsync -avz --checksum \
  --exclude 'config/stream.env' \
  -e "ssh -o StrictHostKeyChecking=accept-new" \
  "${ROOT}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo ""
echo "Na Gigabyte:"
echo "  ssh ${REMOTE_HOST}"
echo "  cd ~/${REMOTE_DIR}"
echo "  cp -n config/stream.env.example config/stream.env   # pierwszy raz"
echo "  # TWITCH_STREAM_KEY — patrz TWITCH_SETUP.md"
echo "  chmod +x gigabyte/*.sh sender/*.sh"
echo "  ./gigabyte/install_deps.sh"
echo "  ./gigabyte/start_lastmile.sh"
