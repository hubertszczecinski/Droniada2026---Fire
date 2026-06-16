#!/usr/bin/env bash
# Kalibracja kolorów na Jetsonie (ten sam wynik co na Macu).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export WS_ROOT_PATH="${WS_ROOT_PATH:-$ROOT}"

FOLDER="${1:-config/competition_cards}"
OUT="${2:-config/card_colors.json}"
NOTES=()
if [[ -n "${DRONIADA_CALIB_NOTES:-}" ]]; then
  NOTES=(--notes "${DRONIADA_CALIB_NOTES}")
fi

echo "=== Kalibracja kolorów (Jetson) ==="
echo "Wejście: ${FOLDER}  →  ${OUT}"

if docker ps --filter name=droniada_vision --format '{{.Names}}' | grep -q droniada_vision; then
  docker exec droniada_vision python3 /ws/scripts/calibrate_card_colors.py \
    --folder "/ws/${FOLDER}" \
    --out "/ws/${OUT}" \
    "${NOTES[@]}"
elif [[ -x "${ROOT}/.venv_yolo/bin/python" ]]; then
  "${ROOT}/.venv_yolo/bin/python" scripts/calibrate_card_colors.py \
    --folder "$FOLDER" --out "$OUT" "${NOTES[@]}"
else
  python3 scripts/calibrate_card_colors.py \
    --folder "$FOLDER" --out "$OUT" "${NOTES[@]}"
fi

echo ""
echo "Restart panelu: ./scripts/jetson_video_start.sh  lub  ./scripts/jetson_offline_start.sh"
