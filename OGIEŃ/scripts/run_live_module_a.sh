#!/usr/bin/env bash
# Live test modulu A (YOLO-Pose + PnP) — tylko kamera, bez modulu B.
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

mkdir -p "${ROOT}/dataset/results"

CAM="${DRONIADA_CAMERA:-1}"
LOG="${ROOT}/dataset/results/live_module_a_$(date +%Y%m%d_%H%M%S).log"

echo "=== Droniada MODUL A (kamera) ==="
echo "Kamera: $CAM  (nadpisz: DRONIADA_CAMERA=0)"
echo "Log:    $LOG"
echo ""
echo "Okno: droniada_live — zielony trapez + odleglosc + stojak + roll/pitch/yaw"
echo "Klawisz q = wyjscie"
echo "Zamknij QuickTime / inne apki z kamera."
echo ""

exec "$PY" -m release.run_live \
  --mode pose \
  --preview \
  --camera "$CAM" \
  --rotate 180 \
  --camera-profile tarot_t10x_2a:wide \
  --interval-ms 400 \
  --log-file "$LOG" \
  "$@"
