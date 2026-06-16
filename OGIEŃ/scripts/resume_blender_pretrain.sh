#!/usr/bin/env bash
# Dokończenie pretrainu Blender od best.pt (po przerwaniu), NIE od yolo11n-pose.pt.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x .venv_yolo/bin/python ]]; then
  PY="$(pwd)/.venv_yolo/bin/python"
else
  echo "Brak .venv_yolo"
  exit 127
fi

PRE="runs/pose/droniada_blender_pretrain/weights/best.pt"
if [[ ! -f "$PRE" ]]; then
  echo "Brak $PRE — pełny pretrain: ./scripts/train_yolo_two_stage.sh 1"
  exit 1
fi

EPOCHS="${EPOCHS_BLENDER_REMAIN:-34}"
LR0="${LR0_BLENDER_RESUME:-0.001}"
echo "Kontynuacja: $PRE → +${EPOCHS} epok (lr0=$LR0), ten sam katalog runs/pose/droniada_blender_pretrain"

exec "$PY" -m release.eval_yolo_pose_corners \
  --train \
  --no-post-train-eval \
  --data-yaml dataset/droniada_pose_blender/droniada_pose_blender.yaml \
  --train-weights "$PRE" \
  --train-name droniada_blender_pretrain \
  --epochs "$EPOCHS" \
  --lr0 "$LR0" \
  --imgsz "${IMGSZ:-640}"
