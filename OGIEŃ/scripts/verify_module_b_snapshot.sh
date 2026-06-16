#!/usr/bin/env bash
# Weryfikacja migawki modułu B (regresja + manifest).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv_yolo/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

echo "=== Migawka modułu B ==="
echo "Dokumentacja: docs/MODULE_B_SNAPSHOT.md"
echo "Manifest:     release/data/module_b_snapshot.json"
echo ""

if [[ -f "${ROOT}/.git/HEAD" ]]; then
  echo "Git: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?') $(git -C "$ROOT" log -1 --oneline 2>/dev/null || true)"
  echo ""
fi

MISS=0
for p in \
  "dataset/my_capture/Droniada_nag3.mov" \
  "runs/pose/droniada_real_finetune/weights/best.pt" \
  "release/data/module_b_snapshot.json" \
  "docs/MODULE_B_SNAPSHOT.md"
do
  if [[ -e "$ROOT/$p" ]]; then
    echo "OK  $p"
  else
    echo "BRAK $p"
    MISS=$((MISS + 1))
  fi
done
echo ""

echo "=== Testy regresji ==="
"$PY" -m unittest discover -s tests -p 'test_*.py' -v
echo ""
echo "=== Testy OK ==="
echo ""
echo "Live (nag3 + zatrzask):"
echo "  export DRONIADA_YOLO_POSE_WEIGHTS=\"\$(pwd)/runs/pose/droniada_real_finetune/weights/best.pt\""
echo "  $PY -m release.run_live_panel \\"
echo "    --video dataset/my_capture/Droniada_nag3.mov --rotate 180 \\"
echo "    --corner-mode yolo_pose --cxy-latch"
echo ""
if [[ "$MISS" -gt 0 ]]; then
  echo "Uwaga: brakuje $MISS plików (testy mogły być pominięte przez skip)."
fi
