#!/usr/bin/env bash
# Kopiuj plik(i) wideo na Jetson (dataset/my_capture nie idzie w sync_to_jetson).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_HOST="${DRONIADA_JETSON_HOST:-sknr@192.168.100.200}"
REMOTE_DIR="${DRONIADA_JETSON_DIR:-acoustics/droniada}"
PASS="${DRONIADA_JETSON_PASS:-}"
VIDEO="${1:-dataset/my_capture/Droniada_nag5.mov}"

if [[ "$VIDEO" != /* ]]; then
  VIDEO="${ROOT}/${VIDEO}"
fi
if [[ ! -f "$VIDEO" ]]; then
  echo "Brak pliku: ${VIDEO}" >&2
  exit 1
fi

REL="${VIDEO#${ROOT}/}"
DEST_DIR="$(dirname "$REL")"

RSYNC_SSH="ssh -o StrictHostKeyChecking=accept-new"
if [[ -n "$PASS" ]] && command -v sshpass >/dev/null 2>&1; then
  RSYNC_SSH="sshpass -p ${PASS} ssh -o StrictHostKeyChecking=accept-new"
fi

REMOTE_PATH="${REMOTE_DIR#~/}"
REMOTE_BASE="~/${REMOTE_PATH}"
REMOTE_VIDEOS="${REMOTE_BASE}/videos"

echo "→ ${REMOTE_HOST}:${REMOTE_BASE}/${DEST_DIR}/ (lub ${REMOTE_VIDEOS}/)"
$RSYNC_SSH "${REMOTE_HOST}" "mkdir -p ${REMOTE_BASE}/${DEST_DIR} ${REMOTE_VIDEOS} 2>/dev/null || mkdir -p ${REMOTE_VIDEOS}"

DEST="${REMOTE_HOST}:${REMOTE_BASE}/${REL}"
if [[ "${REL}" == dataset/my_capture/* ]]; then
  BASENAME="$(basename "$VIDEO")"
  DEST="${REMOTE_HOST}:${REMOTE_VIDEOS}/${BASENAME}"
  echo "Uwaga: dataset/ na Jetsonie bywa root-owned — kopiuję do videos/${BASENAME}"
fi

rsync -avz --progress \
  -e "$RSYNC_SSH" \
  "$VIDEO" "${DEST}"

echo ""
if [[ "${REL}" == dataset/my_capture/* ]]; then
  BASENAME="$(basename "$VIDEO")"
  echo "OK: /ws/videos/${BASENAME} w kontenerze"
  echo "Start testu: ./scripts/jetson_video_start.sh videos/${BASENAME}"
else
  echo "OK: /ws/${REL} w kontenerze"
  echo "Start testu: ./scripts/jetson_video_start.sh ${REL}"
fi
