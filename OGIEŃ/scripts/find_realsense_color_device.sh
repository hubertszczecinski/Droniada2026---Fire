#!/usr/bin/env bash
# Find RealSense RGB (YUYV) V4L2 node supporting 1280x720.
set -euo pipefail

WIDTH="${1:-1280}"
HEIGHT="${2:-720}"

has_yuyv_mode() {
  local dev="$1"
  v4l2-ctl -d "$dev" --list-formats-ext 2>/dev/null \
    | grep -A50 "'YUYV'" | grep -q "Size: Discrete ${WIDTH}x${HEIGHT}"
}

candidates=()
if [[ -d /dev/v4l/by-id ]]; then
  while IFS= read -r link; do
    [[ -n "$link" ]] || continue
    dev="/dev/$(basename "$(readlink -f "$link")")"
    candidates+=("$dev")
  done < <(ls -1 /dev/v4l/by-id/*RealSense*video-index* 2>/dev/null || true)
fi

for dev in "${candidates[@]}" /dev/video*; do
  [[ -c "$dev" ]] || continue
  if has_yuyv_mode "$dev"; then
    echo "$dev"
    exit 0
  fi
done

echo "No YUYV ${WIDTH}x${HEIGHT} device found" >&2
exit 1
