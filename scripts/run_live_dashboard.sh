#!/usr/bin/env bash
# Pełny panel live: moduł A + B + sidebar + migawki (niski reproj).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv_yolo/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

WEIGHTS="${ROOT}/runs/pose/droniada_real_finetune/weights/best.pt"
if [[ ! -f "$WEIGHTS" ]]; then
  echo "Brak wag YOLO: $WEIGHTS"
  exit 1
fi

export DRONIADA_YOLO_POSE_WEIGHTS="$WEIGHTS"
export DRONIADA_YOLO_BIAS="${DRONIADA_YOLO_BIAS:-1}"

mkdir -p "${ROOT}/dataset/results" "${ROOT}/dataset/live_dashboard"

CAM="${DRONIADA_CAMERA:-1}"
LOG="${ROOT}/dataset/results/live_dashboard_$(date +%Y%m%d_%H%M%S).log"

echo "=== Droniada PANEL LIVE (A + B + migawki) ==="
echo "Kamera: $CAM"
echo "Log:    $LOG"
echo ""
echo "Okno: droniada_dashboard"
echo "  lewo: kamera (zielony=A, żółty=B)"
echo "  środek: panel 10×10"
echo "  prawo: wszystkie dane A+B"
echo "  dół: galeria migawek (reproj <= 15 px)"
echo ""
echo "Po sesji: dataset/live_dashboard/session_*/index.html"
echo "Klawisze: q=wyjście  s=zatrzask CXY  r=reset"
echo ""

exec "$PY" -m release.run_live_panel \
  --dashboard \
  --module-a \
  --cxy-latch \
  --corner-mode yolo_pose \
  --camera "$CAM" \
  --rotate 180 \
  --camera-profile tarot_t10x_2a:wide \
  --preview-width 1800 \
  --cxy-latch-dir dataset/debug_cxy_latch \
  --snapshot-max-reproj 15 \
  --no-debug \
  --interval-ms 350 \
  --log-file "$LOG" \
  "$@"
