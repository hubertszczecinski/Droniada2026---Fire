#!/usr/bin/env bash
# Gigabyte: odbiór UDP RTP + podgląd + opcjonalny relay Twitch (RTMP).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=gigabyte/lib_stream.sh
source "${ROOT}/gigabyte/lib_stream.sh"
load_stream_env "${ROOT}/config/stream.env"

echo "=== Droniada Lastmile (Gigabyte) ==="
print_stream_config_summary "${ROOT}/config/stream.env"
echo ""

if ! "${ROOT}/gigabyte/preflight.sh"; then
  exit 1
fi
echo "Ctrl+C = stop"
echo ""

CAPS="$(rtp_caps "${RTP_CODEC}" "${RTP_PAYLOAD}")"
DISPLAY_SINK="$(build_display_sink)"
IS_H265=0
[[ "${RTP_CODEC^^}" == "H265" || "${RTP_CODEC^^}" == "HEVC" ]] && IS_H265=1

if [[ "${RTMP_RELAY}" == "1" ]]; then
  RTMP_URL="$(rtmp_publish_url)"
  echo "Relay → ${TWITCH_RTMP_URL}/***"
  if [[ "${LOCAL_DISPLAY}" == "1" ]]; then
    if [[ "${IS_H265}" == "1" ]]; then
      gst-launch-1.0 -e \
        udpsrc address="${UDP_BIND}" port="${UDP_PORT}" caps="${CAPS}" ! \
        rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! tee name=t \
        t. ! queue ! ${DISPLAY_SINK} \
        t. ! queue max-size-buffers=4 leaky=downstream ! videoconvert ! \
          video/x-raw,format=I420 ! \
          x264enc tune=zerolatency speed-preset=veryfast bitrate="${X264_BITRATE_KBPS}" key-int-max="${X264_KEYINT}" ! \
          video/x-h264,profile=baseline ! h264parse ! flvmux streamable=true ! \
          rtmpsink location="${RTMP_URL}"
    else
      gst-launch-1.0 -e \
        udpsrc address="${UDP_BIND}" port="${UDP_PORT}" caps="${CAPS}" ! \
        rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! tee name=t \
        t. ! queue ! ${DISPLAY_SINK} \
        t. ! queue max-size-buffers=4 leaky=downstream ! videoconvert ! \
          video/x-raw,format=I420 ! \
          x264enc tune=zerolatency speed-preset=veryfast bitrate="${X264_BITRATE_KBPS}" key-int-max="${X264_KEYINT}" ! \
          video/x-h264,profile=baseline ! h264parse ! flvmux streamable=true ! \
          rtmpsink location="${RTMP_URL}"
    fi
  else
    if [[ "${IS_H265}" == "1" ]]; then
      gst-launch-1.0 -e \
        udpsrc address="${UDP_BIND}" port="${UDP_PORT}" caps="${CAPS}" ! \
        rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! queue ! videoconvert ! \
          video/x-raw,format=I420 ! \
          x264enc tune=zerolatency speed-preset=veryfast bitrate="${X264_BITRATE_KBPS}" key-int-max="${X264_KEYINT}" ! \
          video/x-h264,profile=baseline ! h264parse ! flvmux streamable=true ! \
          rtmpsink location="${RTMP_URL}"
    else
      gst-launch-1.0 -e \
        udpsrc address="${UDP_BIND}" port="${UDP_PORT}" caps="${CAPS}" ! \
        rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! queue ! videoconvert ! \
          video/x-raw,format=I420 ! \
          x264enc tune=zerolatency speed-preset=veryfast bitrate="${X264_BITRATE_KBPS}" key-int-max="${X264_KEYINT}" ! \
          video/x-h264,profile=baseline ! h264parse ! flvmux streamable=true ! \
          rtmpsink location="${RTMP_URL}"
    fi
  fi
else
  [[ "${LOCAL_DISPLAY}" == "1" ]] || { echo "Włącz LOCAL_DISPLAY lub RTMP_RELAY" >&2; exit 1; }
  if [[ "${IS_H265}" == "1" ]]; then
    gst-launch-1.0 -e \
      udpsrc address="${UDP_BIND}" port="${UDP_PORT}" caps="${CAPS}" ! \
      rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! ${DISPLAY_SINK}
  else
    gst-launch-1.0 -e \
      udpsrc address="${UDP_BIND}" port="${UDP_PORT}" caps="${CAPS}" ! \
      rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! ${DISPLAY_SINK}
  fi
fi
