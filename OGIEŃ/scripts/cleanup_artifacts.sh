#!/usr/bin/env bash
# Remove regenerable artifacts (sessions, debug, logs, eval JSON). Keeps dataset/images and YOLO weights.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

removed=0
rm_path() {
  if [[ -e "$1" ]]; then
    echo "  removing: $1"
    rm -rf "$1"
    removed=$((removed + 1))
  fi
}

echo "=== Cleaning artifacts ==="

rm_path "${ROOT}/live_debug"
rm_path "${ROOT}/logs"
rm_path "${ROOT}/reports"
rm_path "${ROOT}/.pytest_cache"

find "${ROOT}/dataset/live_dashboard" -maxdepth 1 -type d \( -name 'session_*' -o -name 'nag5_*' \) 2>/dev/null | while read -r d; do
  rm_path "$d"
done

for f in "${ROOT}"/dataset/results/*; do
  [[ -e "$f" ]] || continue
  base="$(basename "$f")"
  case "$base" in
    .gitkeep) continue ;;
    roi_anchor_ab*) rm_path "$f" ;;
    *.json|*.jsonl|*.csv|*.log|*.mp4) rm_path "$f" ;;
  esac
done

for d in \
  dataset/debug_test3 \
  dataset/debug_grid_overlap \
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

rm -f "${ROOT}"/dataset/results/local_video_*.log 2>/dev/null || true

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

while IFS= read -r -d '' f; do
  rm_path "$f"
done < <(find "${ROOT}/runs/pose" -type f \( -name 'train_batch*.jpg' -o -name 'val_batch*.jpg' -o -name '*.png' \) ! -path '*/weights/*' -print0 2>/dev/null || true)

echo ""
echo "Done. Removed ${removed} path(s)."
echo "Kept: dataset/images, my_capture, runs/pose/*/weights, debug_cxy_latch"
