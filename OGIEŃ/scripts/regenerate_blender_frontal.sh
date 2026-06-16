#!/usr/bin/env bash
# Regeneracja datasetu Blender — tylko widoki z przodu (bez tyłu i czystego boku).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x /Applications/Blender.app/Contents/MacOS/Blender ]]; then
  BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
elif command -v blender >/dev/null 2>&1; then
  BLENDER=blender
else
  echo "Nie znaleziono Blendera (macOS: Blender.app lub blender w PATH)" >&2
  exit 1
fi

export DRONIADA_FRESH="${DRONIADA_FRESH:-1}"
export DRONIADA_TARGET_IMAGES="${DRONIADA_TARGET_IMAGES:-400}"
export DRONIADA_NUM_SCENES="${DRONIADA_NUM_SCENES:-120}"
export DRONIADA_ORBIT_AZIMUTHS="${DRONIADA_ORBIT_AZIMUTHS:--22,-14,-7,0,7,14,22}"
export DRONIADA_MIN_FRONT_DOT="${DRONIADA_MIN_FRONT_DOT:-0.72}"
export DRONIADA_MIN_PANEL_SPAN_PX="${DRONIADA_MIN_PANEL_SPAN_PX:-200}"
export DRONIADA_MIN_PANEL_AREA_FRAC="${DRONIADA_MIN_PANEL_AREA_FRAC:-0.06}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "Blender: $BLENDER"
echo "TARGET_IMAGES=$DRONIADA_TARGET_IMAGES NUM_SCENES=$DRONIADA_NUM_SCENES"
echo "ORBIT_AZIMUTHS=$DRONIADA_ORBIT_AZIMUTHS"
echo "MIN_FRONT_DOT=$DRONIADA_MIN_FRONT_DOT MIN_PANEL_SPAN_PX=$DRONIADA_MIN_PANEL_SPAN_PX"

exec "$BLENDER" --background --python generate_dataset_blender.py
