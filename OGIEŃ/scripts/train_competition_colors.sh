#!/usr/bin/env bash
# Trening kolorów przed zawodami: *2 od organizatora + Test/Test2/Test3 (ten sam panel).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv_yolo/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

WEIGHTS="${ROOT}/runs/pose/droniada_real_finetune/weights/best.pt"
if [[ -f "$WEIGHTS" ]]; then
  export DRONIADA_YOLO_POSE_WEIGHTS="$WEIGHTS"
fi

echo "=== Trening kolorów (Test panel + competition_cards) ==="
"$PY" scripts/train_competition_colors.py "$@"

export DRONIADA_CARD_COLORS="${ROOT}/config/card_colors.json"
echo ""
echo "Profil aktywny: config/card_colors.json"
echo "Test live:"
echo "  ./scripts/run_local_video_test.sh dataset/my_capture/Test.mov"
