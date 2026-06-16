#!/usr/bin/env bash
# Szybki restart kamery z parametrami trackera (bez przebudowy obrazu).
#
# Przykłady:
#   ./scripts/jetson_camera_tune.sh
#   ./scripts/jetson_camera_tune.sh 0.45 24 280
#   DRONIADA_SMOOTH_ALPHA=0.5 DRONIADA_HOLD_FRAMES=22 ./scripts/jetson_camera_tune.sh
#
# smooth_alpha  0.05–1.0  wyżej = szybsze skoki ramki (kamera USB)
# hold_frames     ile klatek trzymać ostatni quad gdy YOLO na chwilę zgubi panel
# interval_ms     co ile ms odpalać pełną analizę YOLO (niżej = częściej, obciąża GPU)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export WS_ROOT_PATH="${WS_ROOT_PATH:-$ROOT}"

if [[ -n "${1:-}" ]]; then
  export DRONIADA_SMOOTH_ALPHA="$1"
fi
if [[ -n "${2:-}" ]]; then
  export DRONIADA_HOLD_FRAMES="$2"
fi
if [[ -n "${3:-}" ]]; then
  export DRONIADA_INTERVAL_MS="$3"
fi
if [[ -n "${4:-}" ]]; then
  export DRONIADA_TRACKER_GOOD_REPROJ="$4"
fi

unset DRONIADA_VIDEO_FILE DRONIADA_VIDEO_LOOP

echo "=== Tuning kamery ==="
echo "  DRONIADA_SMOOTH_ALPHA=${DRONIADA_SMOOTH_ALPHA:-<domyślnie 0.38>}"
echo "  DRONIADA_HOLD_FRAMES=${DRONIADA_HOLD_FRAMES:-<domyślnie 20>}"
echo "  DRONIADA_INTERVAL_MS=${DRONIADA_INTERVAL_MS:-<domyślnie 300>}"
echo "  DRONIADA_TRACKER_GOOD_REPROJ=${DRONIADA_TRACKER_GOOD_REPROJ:-<domyślnie 28>}"
echo ""
echo "Wskazówki:"
echo "  • ramka znika → podnieś HOLD_FRAMES (np. 24–30)"
echo "  • za wolno reaguje → podnieś SMOOTH_ALPHA (np. 0.45–0.55)"
echo "  • za drga → obniż SMOOTH_ALPHA (np. 0.22–0.30)"
echo "  • wewnętrzne rogi → obniż TRACKER_GOOD_REPROJ nie pomoże; spróbuj wyższy alpha + hold"
echo ""

exec ./scripts/jetson_offline_start.sh
