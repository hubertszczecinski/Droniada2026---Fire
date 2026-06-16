#!/usr/bin/env bash
# Synchronizacja wyłącznie ~/acoustics/droniada na Jetson (bez innych katalogów acoustics).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_HOST="${DRONIADA_JETSON_HOST:-sknr@192.168.100.200}"
REMOTE_DIR="${DRONIADA_JETSON_DIR:-acoustics/droniada}"
PASS="${DRONIADA_JETSON_PASS:-}"

echo "→ ${REMOTE_HOST}:${REMOTE_DIR}/"

RSYNC_SSH="ssh -o StrictHostKeyChecking=accept-new"
if [[ -n "$PASS" ]] && command -v sshpass >/dev/null 2>&1; then
  RSYNC_SSH="sshpass -p ${PASS} ssh -o StrictHostKeyChecking=accept-new"
fi

rsync -avz \
  -e "$RSYNC_SSH" \
  --exclude '.git' \
  --exclude 'venv/' \
  --exclude '.venv/' \
  --exclude '.venv_yolo/' \
  --exclude 'env/' \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude 'Regulamin*.pdf' \
  --exclude '*.pdf' \
  --exclude 'yolo*.pt' \
  --exclude 'live_debug/' \
  --exclude 'reports/' \
  --exclude 'logs/' \
  --exclude 'dataset/' \
  --exclude 'runs/' \
  --exclude 'filmy/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude '.cursor/' \
  "$ROOT/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo ""
echo "Na Jetsonie (zawody — kamera w kontenerze CUDA):"
echo "  ssh ${REMOTE_HOST}"
echo "  cd ~/${REMOTE_DIR}"
echo "  export WS_ROOT_PATH=\$(pwd)"
echo "  ./scripts/jetson_competition_start.sh"
echo "  → podgląd http://<jetson-ip>:\${DRONIADA_HOST_PORT:-8088}/"
echo "  → sterowanie http://<jetson-ip>:\${DRONIADA_CONTROL_PORT:-8089}/"
