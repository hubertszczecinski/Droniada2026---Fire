#!/usr/bin/env bash
# Moduł A: kalibracja ustawienia panelu + testy regresji na Blenderze.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv_yolo/bin/python"
if [[ ! -x "$PY" ]]; then PY=python3; fi

echo "=== Moduł A — kalibracja (Blender) ==="
"$PY" -m pipelines.calibrate_panel_stand --dataset dataset

echo ""
echo "=== Moduł A — ewaluacja ==="
"$PY" -m pipelines.eval_module_a_blender --dataset dataset \
  --out dataset/results/eval_module_a_blender.json

echo ""
echo "=== Moduł A — testy regresji ==="
"$PY" -m unittest tests.test_module_a_blender -v
