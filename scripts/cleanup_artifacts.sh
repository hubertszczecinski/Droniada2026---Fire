#!/usr/bin/env bash
# Usuwa stare debugi, nagrania testowe i sesje live_debug (~3+ GB). Nie rusza dataset/images, wag YOLO, my_capture.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

removed=0
rm_path() {
  if [[ -e "$1" ]]; then
    echo "  usuwam: $1"
    rm -rf "$1"
    removed=$((removed + 1))
  fi
}

echo "=== Czyszczenie artefaktow testowych ==="

# Sesje live_debug (regenerowalne: --debug-dir live_debug)
rm_path "${ROOT}/live_debug"

# Stare galerie ewaluacji line_grid / frontal
for d in \
  dataset/debug_frontal_all \
  dataset/debug_frontal_errors \
  dataset/debug_frontal_reliable \
  dataset/debug_line_grid \
  dataset/debug_line_grid_108 \
  dataset/debug_line_grid_v2 \
  dataset/debug_line_grid_v3 \
  dataset/debug_module_b \
  dataset/debug_module_b_panel
do
  rm_path "${ROOT}/$d"
done

# Stare nagrania / logi testowe (zostawiamy debug_cxy_latch — ostatni latch)
for f in \
  dataset/debug_nag3_cxy_latch.mp4 \
  dataset/debug_nag3_cxy_latch.log \
  dataset/debug_nag3_latch_run.log \
  dataset/debug_yolo_nag2_smooth.mp4 \
  dataset/debug_yolo_nag3_live.mp4 \
  dataset/debug_yolo_nag3_smooth.mp4 \
  dataset/debug_yolo_nag3_frame_30pct.jpg \
  dataset/debug_yolo_nag3_frame_50pct.jpg \
  dataset/debug_yolo_nag3_frame_75pct.jpg
do
  rm_path "${ROOT}/$f"
done

rm_path "${ROOT}/release/CORNERS_EXPERIMENTS.jsonl"

# Duze miniatury z treningu YOLO (zostawiamy weights/best.pt)
while IFS= read -r -d '' f; do
  rm_path "$f"
done < <(find "${ROOT}/runs/pose" -type f \( -name 'train_batch*.jpg' -o -name 'val_batch*.jpg' -o -name '*.png' \) ! -path '*/weights/*' -print0 2>/dev/null || true)

echo ""
echo "Gotowe. Usunieto ${removed} sciezek."
echo "Zostalo: dataset/images, my_capture, runs/pose/*/weights, debug_cxy_latch"
