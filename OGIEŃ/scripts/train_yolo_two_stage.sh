#!/usr/bin/env bash
# Etap 0: eksport Blender + real. Etap 1: pretrain Blender. Etap 2: fine-tune real.
# Domyślnie YOLO11n-Pose (Ultralytics 8.4+). Na Jetsonie możesz: YOLO_MODEL=yolov8n-pose.pt
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

if [[ -x .venv_yolo/bin/python ]]; then
  PY="$(pwd)/.venv_yolo/bin/python"
elif [[ -x .venv/bin/python ]] && .venv/bin/python -c "import cv2" 2>/dev/null; then
  PY="$(pwd)/.venv/bin/python"
else
  echo "Brak .venv_yolo/bin/python — utwórz: python3 -m venv .venv_yolo && pip install ultralytics opencv-python-headless"
  exit 127
fi

YOLO_MODEL="${YOLO_MODEL:-yolo11n-pose.pt}"
echo "Python: $PY"
echo "Model:  $YOLO_MODEL"
STAGE="${1:-all}"

if [[ "$STAGE" == "0" || "$STAGE" == "all" ]]; then
  echo "=== Etap 0: eksport datasetów YOLO-Pose ==="
  "$PY" -m release.export_blender_yolo_pose --dataset dataset --out dataset/droniada_pose_blender
  "$PY" -m release.export_yolo_pose_dataset --out dataset/droniada_pose
fi

if [[ "$STAGE" == "1" || "$STAGE" == "all" ]]; then
  if [[ ! -f dataset/droniada_pose_blender/droniada_pose_blender.yaml ]]; then
    echo "Brak dataset/droniada_pose_blender — uruchom: $0 0"
    exit 1
  fi
  echo "=== Etap 1: pretrain Blender ==="
  "$PY" -m release.eval_yolo_pose_corners \
    --train \
    --data-yaml dataset/droniada_pose_blender/droniada_pose_blender.yaml \
    --train-weights "$YOLO_MODEL" \
    --train-name droniada_blender_pretrain \
    --epochs "${EPOCHS_BLENDER:-100}" \
    --imgsz "${IMGSZ:-640}"
fi

if [[ "$STAGE" == "2" || "$STAGE" == "all" ]]; then
  PRE="${PRETRAIN_WEIGHTS:-runs/pose/droniada_blender_pretrain/weights/best.pt}"
  if [[ ! -f "$PRE" ]]; then
    echo "Brak wag pretrain: $PRE — najpierw: ./scripts/train_yolo_two_stage.sh 1"
    exit 1
  fi
  if [[ ! -f dataset/droniada_pose/droniada_pose.yaml ]]; then
    echo "Eksport realnych etykiet..."
    "$PY" -m release.export_yolo_pose_dataset --out dataset/droniada_pose
  fi
  echo "=== Etap 2: fine-tune real ==="
  "$PY" -m release.eval_yolo_pose_corners \
    --train \
    --data-yaml dataset/droniada_pose/droniada_pose.yaml \
    --train-weights "$PRE" \
    --train-name droniada_real_finetune \
    --epochs "${EPOCHS_REAL:-60}" \
    --lr0 "${LR0_REAL:-0.001}" \
    --imgsz "${IMGSZ:-640}"
  echo "=== Ewaluacja na panel_labels ==="
  "$PY" -m release.eval_yolo_pose_corners \
    --weights runs/pose/droniada_real_finetune/weights/best.pt \
    --mode yolo
  echo "=== Kalibracja biasu rogów (panel_labels → yolo_corner_bias.json) ==="
  "$PY" -m release.yolo_corner_bias \
    --weights runs/pose/droniada_real_finetune/weights/best.pt
  echo "Wagi finalne: runs/pose/droniada_real_finetune/weights/best.pt"
  echo "Live: export DRONIADA_YOLO_POSE_WEIGHTS=\$(pwd)/runs/pose/droniada_real_finetune/weights/best.pt"
  echo "      export DRONIADA_YOLO_BIAS=1  # korekcja z module_panel/data/yolo_corner_bias.json"
fi
