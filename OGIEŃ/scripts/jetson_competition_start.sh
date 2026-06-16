#!/usr/bin/env bash
# Start na zawody — kamera + panel 8088 (podgląd) + 8089 (sterowanie) + misja A,B,C.
# Wymaga: obraz droniada-vision:latest, best.pt, sync kodu z Maca.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export WS_ROOT_PATH="${WS_ROOT_PATH:-$ROOT}"
export DRONIADA_MISSION_PANELS="${DRONIADA_MISSION_PANELS:-A,B,C}"
export DRONIADA_SNAPSHOTS_PER_PANEL="${DRONIADA_SNAPSHOTS_PER_PANEL:-5}"
export DRONIADA_HOST_PORT="${DRONIADA_HOST_PORT:-8088}"
export DRONIADA_CONTROL_PORT="${DRONIADA_CONTROL_PORT:-8089}"
# Autonomia (edge-trigger na :8765): hold_started / hold_stopped — NIE speed.
# hold_started = zbieraj migawki; hold_stopped = raport z migawek + pauza analizy.
export DRONIADA_AUTONOMY_WS_URL="${DRONIADA_AUTONOMY_WS_URL:-ws://127.0.0.1:8765}"
export DRONIADA_AUTONOMY_HOLD_PAUSE_S="${DRONIADA_AUTONOMY_HOLD_PAUSE_S:-8}"
# Osobny kanał na speed drona (jeśli orchestrator ma inny port/URL):
# export DRONIADA_WS_URL=ws://127.0.0.1:9xxx
# Niższy próg powierzchni panelu przy 640×480 MJPEG:
export DRONIADA_PANEL_MIN_AREA_FRAC="${DRONIADA_PANEL_MIN_AREA_FRAC:-0.04}"
# Rozdzielczość: jetson_offline_start.sh wybiera najwyższe MJPEG (docelowo 1920×1080 po HDMI).
# Docker headless: GStreamer+obrót 180° pada (brak EGL) — OpenCV V4L2 jest szybsze i stabilniejsze.
export DRONIADA_USE_GST_CAPTURE="${DRONIADA_USE_GST_CAPTURE:-0}"
export DRONIADA_GST_HW_JPEG="${DRONIADA_GST_HW_JPEG:-0}"
export DRONIADA_STREAM_PASSTHROUGH="${DRONIADA_STREAM_PASSTHROUGH:-0}"
# Płynny podgląd WWW (~30 fps): vis-preview + MJPEG encoder (nie czeka na YOLO co 300 ms).
export DRONIADA_STREAM_SOURCE="${DRONIADA_STREAM_SOURCE:-vis}"
export DRONIADA_STREAM_WIDTH="${DRONIADA_STREAM_WIDTH:-640}"
export DRONIADA_STREAM_INTERVAL_S="${DRONIADA_STREAM_INTERVAL_S:-0.033}"
export DRONIADA_VIS_PREVIEW_INTERVAL_S="${DRONIADA_VIS_PREVIEW_INTERVAL_S:-0.033}"
# Luźne progi migawek (zawody / lot) — łatwiej zapisać klatkę przy panelu w kadrze.
export DRONIADA_SNAPSHOT_MIN_GRID_OVERLAP="${DRONIADA_SNAPSHOT_MIN_GRID_OVERLAP:-0.45}"
export DRONIADA_SNAPSHOT_MIN_GRID_OVERLAP_UNRELIABLE="${DRONIADA_SNAPSHOT_MIN_GRID_OVERLAP_UNRELIABLE:-0.68}"
export DRONIADA_SNAPSHOT_MIN_STABLE="${DRONIADA_SNAPSHOT_MIN_STABLE:-1}"
export DRONIADA_SNAPSHOT_MAX_REPROJ="${DRONIADA_SNAPSHOT_MAX_REPROJ:-20}"
export DRONIADA_SNAPSHOT_MAX_REPROJ_A="${DRONIADA_SNAPSHOT_MAX_REPROJ_A:-22}"
export DRONIADA_SNAPSHOT_MIN_WARP_COVERAGE="${DRONIADA_SNAPSHOT_MIN_WARP_COVERAGE:-0.25}"
export DRONIADA_SNAPSHOT_COMP_MIN_VOTES="${DRONIADA_SNAPSHOT_COMP_MIN_VOTES:-2}"
export DRONIADA_SNAPSHOT_COMP_MIN_RATIO="${DRONIADA_SNAPSHOT_COMP_MIN_RATIO:-0.0}"
export DRONIADA_SNAPSHOT_REQUIRE_YOLO_CORNERS="${DRONIADA_SNAPSHOT_REQUIRE_YOLO_CORNERS:-1}"
export DRONIADA_SNAPSHOT_MIN_STAND_CONFIDENCE="${DRONIADA_SNAPSHOT_MIN_STAND_CONFIDENCE:-0.45}"
export DRONIADA_ASYNC_ANALYSIS="${DRONIADA_ASYNC_ANALYSIS:-1}"
# Raporty z pliku + skróty 1/2/3 (A/B/C) na :8089 — przygotuj config/preset_reports.json:
# export DRONIADA_REPORT_MODE=preset
exec ./scripts/jetson_offline_start.sh
