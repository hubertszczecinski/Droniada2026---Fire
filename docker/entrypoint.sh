#!/usr/bin/env bash
set -euo pipefail

cd /ws

export PYTHONPATH="/ws:${PYTHONPATH:-}"
export OPENCV_VIDEOIO_PRIORITY_LIST="${OPENCV_VIDEOIO_PRIORITY_LIST:-V4L2}"
export DRONIADA_DEVICE="${DRONIADA_DEVICE:-0}"
export DRONIADA_CAMERA="${DRONIADA_CAMERA:-0}"
export DRONIADA_CAMERA_DEVICE="${DRONIADA_CAMERA_DEVICE:-/dev/video0}"
export DRONIADA_CAMERA_FOURCC="${DRONIADA_CAMERA_FOURCC:-MJPG}"
export DRONIADA_CAMERA_WIDTH="${DRONIADA_CAMERA_WIDTH:-1920}"
export DRONIADA_CAMERA_HEIGHT="${DRONIADA_CAMERA_HEIGHT:-1080}"
export DRONIADA_CAMERA_BRIGHTNESS="${DRONIADA_CAMERA_BRIGHTNESS:-60}"
export DRONIADA_YOLO_BIAS="${DRONIADA_YOLO_BIAS:-1}"
export DRONIADA_YOLO_POSE_WEIGHTS="${DRONIADA_YOLO_POSE_WEIGHTS:-/ws/runs/pose/droniada_real_finetune/weights/best.pt}"
if [[ -z "${DRONIADA_CARD_COLORS:-}" && -f /ws/config/card_colors.json ]]; then
  export DRONIADA_CARD_COLORS=/ws/config/card_colors.json
elif [[ -n "${DRONIADA_CARD_COLORS:-}" && ! -f "${DRONIADA_CARD_COLORS}" ]]; then
  _cc_base="$(basename "${DRONIADA_CARD_COLORS}")"
  if [[ -f "/ws/config/${_cc_base}" ]]; then
    export DRONIADA_CARD_COLORS="/ws/config/${_cc_base}"
  fi
fi

if [[ ! -f "${DRONIADA_YOLO_POSE_WEIGHTS}" ]]; then
  echo "[entrypoint] Brak wag YOLO: ${DRONIADA_YOLO_POSE_WEIGHTS}" >&2
  echo "[entrypoint] Zamontuj repo w /ws (runs/pose/.../best.pt)." >&2
  exit 1
fi

if [[ "${1:-}" == "droniada-vision" ]]; then
  shift
  LOG="${DRONIADA_LOG_FILE:-/ws/dataset/results/jetson_live.log}"
  mkdir -p "$(dirname "$LOG")" /ws/dataset/live_dashboard /ws/dataset/debug_cxy_latch

  WEB_PORT="${DRONIADA_WEB_PORT:-8088}"
  WEB_CONTROL_PORT="${DRONIADA_WEB_CONTROL_PORT:-8089}"
  WEB_BIND="${DRONIADA_WEB_BIND:-0.0.0.0}"
  if [[ "${DRONIADA_WEB_ENABLED:-1}" == "0" ]]; then
    WEB_PORT=0
  fi

  VIDEO_ARGS=()
  if [[ -n "${DRONIADA_VIDEO_FILE:-}" ]]; then
    if [[ ! -f "${DRONIADA_VIDEO_FILE}" ]]; then
      echo "[entrypoint] Brak pliku wideo: ${DRONIADA_VIDEO_FILE}" >&2
      exit 1
    fi
    VIDEO_ARGS=(--video "${DRONIADA_VIDEO_FILE}")
    if [[ "${DRONIADA_VIDEO_LOOP:-1}" != "1" ]]; then
      VIDEO_ARGS+=(--no-loop)
    fi
    echo "[entrypoint] source=video ${DRONIADA_VIDEO_FILE} loop=${DRONIADA_VIDEO_LOOP:-1}"
    export OPENCV_VIDEOIO_PRIORITY_LIST="${DRONIADA_OPENCV_VIDEOIO:-FFMPEG,GSTREAMER,V4L2}"
  else
    echo "[entrypoint] camera=${DRONIADA_CAMERA_DEVICE} ${DRONIADA_CAMERA_WIDTH}x${DRONIADA_CAMERA_HEIGHT} fourcc=${DRONIADA_CAMERA_FOURCC} brightness=${DRONIADA_CAMERA_BRIGHTNESS}"
    if [[ -n "${DRONIADA_CAP_PIPELINE:-}" ]]; then
      echo "[entrypoint] capture=GStreamer (DRONIADA_CAP_PIPELINE)"
    fi
  fi
  echo "[entrypoint] YOLO weights=${DRONIADA_YOLO_POSE_WEIGHTS}"
  if [[ -n "${DRONIADA_CARD_COLORS:-}" && -f "${DRONIADA_CARD_COLORS}" ]]; then
    echo "[entrypoint] card colors=${DRONIADA_CARD_COLORS}"
  else
    echo "[entrypoint] card colors=domyślny HSV (brak /ws/config/card_colors.json)"
  fi
  if [[ "${WEB_PORT}" != "0" ]]; then
    echo "[entrypoint] podgląd WWW: http://<jetson-ip>:${DRONIADA_HOST_PORT:-8088}/"
    echo "[entrypoint] sterowanie: http://<jetson-ip>:${DRONIADA_CONTROL_PORT:-8089}/ (w kontenerze ${WEB_BIND}:${WEB_PORT}+${WEB_CONTROL_PORT})"
  else
    echo "[entrypoint] panel WWW wyłączony (DRONIADA_WEB_ENABLED=0)"
  fi
  if [[ -n "${DRONIADA_WS_URL:-}" ]]; then
    echo "[entrypoint] WebSocket out (speed): ${DRONIADA_WS_URL}"
  fi
  if [[ -n "${DRONIADA_AUTONOMY_WS_URL:-}" ]]; then
    echo "[entrypoint] WebSocket in (hold_started/stopped): ${DRONIADA_AUTONOMY_WS_URL} · raport po stop, pauza analizy=${DRONIADA_AUTONOMY_HOLD_PAUSE_S:-8}s"
  fi

  if [[ -n "${DRONIADA_VIDEO_FILE:-}" ]]; then
    SMOOTH_ALPHA="${DRONIADA_SMOOTH_ALPHA:-0.28}"
    HOLD_FRAMES="${DRONIADA_HOLD_FRAMES:-14}"
    INTERVAL_MS="${DRONIADA_INTERVAL_MS:-350}"
  else
    # Kamera USB: szybsze skoki do dobrych klatek + dłuższy hold przy zawieszeniach V4L2.
    SMOOTH_ALPHA="${DRONIADA_SMOOTH_ALPHA:-0.38}"
    HOLD_FRAMES="${DRONIADA_HOLD_FRAMES:-20}"
    INTERVAL_MS="${DRONIADA_INTERVAL_MS:-300}"
  fi
  TRACKER_GOOD="${DRONIADA_TRACKER_GOOD_REPROJ:-28}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  echo "[entrypoint] tracker: smooth_alpha=${SMOOTH_ALPHA} hold_frames=${HOLD_FRAMES} interval_ms=${INTERVAL_MS} good_reproj=${TRACKER_GOOD}"
  echo "[entrypoint] capture: GST=${DRONIADA_USE_GST_CAPTURE:-1} HW_JPEG=${DRONIADA_GST_HW_JPEG:-1} passthrough=${DRONIADA_STREAM_PASSTHROUGH:-1}"
  echo "[entrypoint] stream: /stream.mjpg=surowy MJPEG z kamery (GStreamer), /live.jpg=overlay+dashboard"

  REPORT_MODE="${DRONIADA_REPORT_MODE:-live}"
  PRESET_REPORTS="${DRONIADA_PRESET_REPORTS:-/ws/config/preset_reports.json}"
  REPORT_ARGS=()
  if [[ "${REPORT_MODE}" == "preset" ]]; then
    REPORT_ARGS=(--report-mode preset --preset-reports "${PRESET_REPORTS}")
    echo "[entrypoint] report_mode=preset file=${PRESET_REPORTS} (skróty 1/2/3 = A/B/C na :8089)"
  else
    echo "[entrypoint] report_mode=live (raport z CXY / edycja)"
  fi

  GUI_ARGS=(--headless --no-debug)
  if [[ "${WEB_PORT}" != "0" ]]; then
    GUI_ARGS+=(--web-port "${WEB_PORT}" --web-control-port "${WEB_CONTROL_PORT}" --web-host "${WEB_BIND}")
  fi
  if [[ -n "${DISPLAY:-}" && "${DRONIADA_GUI:-}" == "1" ]]; then
    echo "[entrypoint] DISPLAY=${DISPLAY} + DRONIADA_GUI=1 → okno OpenCV"
    GUI_ARGS=(--preview --no-debug)
    if [[ "${WEB_PORT}" != "0" ]]; then
      GUI_ARGS+=(--web-port "${WEB_PORT}" --web-control-port "${WEB_CONTROL_PORT}" --web-host "${WEB_BIND}")
    fi
  fi

  exec python3 -m release.run_live_panel \
    --dashboard \
    --module-a \
    --cxy-latch \
    --corner-mode yolo_pose \
    --camera "${DRONIADA_CAMERA}" \
    --rotate "${DRONIADA_ROTATE:-180}" \
    --camera-profile "${DRONIADA_CAMERA_PROFILE:-tarot_t10x_2a:wide}" \
    --preview-width "${DRONIADA_PREVIEW_WIDTH:-1280}" \
    --cxy-latch-dir /ws/dataset/debug_cxy_latch \
    --snapshot-max-reproj "${DRONIADA_SNAPSHOT_MAX_REPROJ:-15}" \
    --cxy-stable-frames "${DRONIADA_CXY_STABLE_FRAMES:-5}" \
    --no-debug \
    --interval-ms "${INTERVAL_MS}" \
    --smooth-alpha "${SMOOTH_ALPHA}" \
    --hold-frames "${HOLD_FRAMES}" \
    --tracker-good-reproj "${TRACKER_GOOD}" \
    --mission-panels "${DRONIADA_MISSION_PANELS:-A,B,C}" \
    --snapshots-per-panel "${DRONIADA_SNAPSHOTS_PER_PANEL:-5}" \
    --snapshot-competition-min-votes "${DRONIADA_SNAPSHOT_COMP_MIN_VOTES:-3}" \
    --snapshot-competition-min-ratio "${DRONIADA_SNAPSHOT_COMP_MIN_RATIO:-0.5}" \
    "${GUI_ARGS[@]}" \
    "${REPORT_ARGS[@]}" \
    --log-file "$LOG" \
    "${VIDEO_ARGS[@]}" \
    "$@"
fi

exec "$@"
