#!/usr/bin/env bash
# Kalibracja kolorów kartek przed zawodami (≈2 min).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv_yolo/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

FOLDER="${1:-config/competition_cards}"
OUT="${2:-config/card_colors.json}"
NOTES="${DRONIADA_CALIB_NOTES:-}"

ARGS=(--folder "$FOLDER" --out "$OUT")
if [[ -n "$NOTES" ]]; then
  ARGS+=(--notes "$NOTES")
fi

echo "=== Kalibracja kolorów kartek ==="
echo "Wejście: $FOLDER  (6 plików: CZERWONA.jpg … POMARANCZOWA.jpg)"
echo "Wyjście: $OUT"
echo ""

exec "$PY" scripts/calibrate_card_colors.py "${ARGS[@]}"
