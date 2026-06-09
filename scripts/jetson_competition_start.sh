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
export DRONIADA_PANEL_MIN_AREA_FRAC="${DRONIADA_PANEL_MIN_AREA_FRAC:-0.02}"
# Rozdzielczość: jetson_offline_start.sh wybiera najwyższe MJPEG (docelowo 1920×1080 po HDMI).
# GStreamer tee: HW MJPEG decode (nvjpegdec) + passthrough na :8088 (bez cv2.imencode).
export DRONIADA_USE_GST_CAPTURE="${DRONIADA_USE_GST_CAPTURE:-1}"
export DRONIADA_GST_HW_JPEG="${DRONIADA_GST_HW_JPEG:-1}"
export DRONIADA_STREAM_PASSTHROUGH="${DRONIADA_STREAM_PASSTHROUGH:-1}"
# Raporty z pliku + skróty 1/2/3 (A/B/C) na :8089 — przygotuj config/preset_reports.json:
# export DRONIADA_REPORT_MODE=preset
exec ./scripts/jetson_offline_start.sh
