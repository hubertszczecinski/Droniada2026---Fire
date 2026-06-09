#!/usr/bin/env bash
# Wag YOLO + bias JSON na Jetson (bez datasetu / treningu).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_HOST="${DRONIADA_JETSON_HOST:-sknr@192.168.100.200}"
REMOTE_DIR="${DRONIADA_JETSON_DIR:-acoustics/droniada}"
PASS="${DRONIADA_JETSON_PASS:-}"
WEIGHTS="${DRONIADA_WEIGHTS_LOCAL:-$ROOT/runs/pose/droniada_real_finetune/weights/best.pt}"
BIAS="${DRONIADA_BIAS_LOCAL:-$ROOT/module_panel/data/yolo_corner_bias.json}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Brak wag: $WEIGHTS" >&2
  exit 1
fi

RSYNC_SSH="ssh -o StrictHostKeyChecking=accept-new"
if [[ -n "$PASS" ]] && command -v sshpass >/dev/null 2>&1; then
  RSYNC_SSH="sshpass -p ${PASS} ssh -o StrictHostKeyChecking=accept-new"
fi

echo "→ ${REMOTE_HOST}:${REMOTE_DIR}/runs/pose/droniada_real_finetune/weights/"
rsync -avz -e "$RSYNC_SSH" "$WEIGHTS" \
  "${REMOTE_HOST}:${REMOTE_DIR}/runs/pose/droniada_real_finetune/weights/best.pt"

if [[ -f "$BIAS" ]]; then
  echo "→ bias JSON"
  rsync -avz -e "$RSYNC_SSH" "$BIAS" \
    "${REMOTE_HOST}:${REMOTE_DIR}/module_panel/data/yolo_corner_bias.json"
fi

echo "Gotowe. Na Jetsonie: export WS_ROOT_PATH=~/acoustics/droniada && ./scripts/docker_jetson_up.sh"
