#!/usr/bin/env bash
# Zależności GStreamer na Gigabyte (Ubuntu 22.04, bez NVENC).
set -euo pipefail

echo "=== Lastmile Gigabyte — instalacja GStreamer ==="

if ! command -v apt-get >/dev/null 2>&1; then
  echo "Oczekiwano Ubuntu/Debian z apt." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav \
  gstreamer1.0-x \
  gstreamer1.0-gl \
  gstreamer1.0-alsa \
  libx264-dev

echo ""
echo "Sprawdzenie pluginów:"
gst-inspect-1.0 udpsrc rtph264depay rtph265depay avdec_h264 avdec_h265 x264enc rtmpsink flvmux 2>/dev/null | head -20 || true
echo ""
echo "OK. Uruchom: ./gigabyte/start_lastmile.sh"
