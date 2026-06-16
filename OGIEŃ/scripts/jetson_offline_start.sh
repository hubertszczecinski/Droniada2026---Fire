#!/usr/bin/env bash
# Start Droniada na Jetsonie BEZ internetu (obraz Docker już zbudowany lokalnie).
# Uruchamiaj po zalogowaniu SSH w sieci konkursowej.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export WS_ROOT_PATH="${WS_ROOT_PATH:-$ROOT}"
HOST_PORT="${DRONIADA_HOST_PORT:-8088}"
CONTROL_PORT="${DRONIADA_CONTROL_PORT:-8089}"
WEIGHTS="${ROOT}/runs/pose/droniada_real_finetune/weights/best.pt"

echo "=== Droniada — start offline ==="
echo "Katalog: ${ROOT}"

COMPOSE_FILE="${ROOT}/docker-compose.jetson.yml"

if docker ps -a --filter name=droniada_vision --format '{{.Names}}' 2>/dev/null | grep -q droniada_vision; then
  echo "Zatrzymuję stary kontener (zwalniam kamerę USB)…"
  docker compose -f "${COMPOSE_FILE}" down 2>/dev/null || true
  sleep 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Brak docker. Zainstaluj / uruchom Docker przed startem." >&2
  exit 1
fi

if ! docker image inspect droniada-vision:latest >/dev/null 2>&1; then
  echo "Brak obrazu droniada-vision:latest." >&2
  echo "Zbuduj go RAZ w sieci z internetem (na Macu sync + na Jetsonie):" >&2
  echo "  cd ~/acoustics/droniada && export WS_ROOT_PATH=\$(pwd) && docker compose -f docker-compose.jetson.yml build" >&2
  exit 1
fi

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Brak wag YOLO: ${WEIGHTS}" >&2
  echo "Na Macu (sieć z internetem): ./scripts/sync_jetson_weights.sh" >&2
  exit 1
fi

VIDEO_DEV="${DRONIADA_VIDEO_DEVICE:-}"
TARGET_W="${DRONIADA_CAMERA_WIDTH:-1920}"
TARGET_H="${DRONIADA_CAMERA_HEIGHT:-1080}"
USB_VIDEO_DEVS=()
if [[ -z "${VIDEO_DEV}" ]]; then
  if command -v v4l2-ctl >/dev/null 2>&1; then
    mapfile -t USB_VIDEO_DEVS < <(
      v4l2-ctl --list-devices 2>/dev/null | awk '
        /USB Video/ { in_usb=1; next }
        /^[^ \t]/ { in_usb=0 }
        in_usb && /\/dev\/video/ { print $1 }
      '
    )
  fi
  mjpg_sizes_for_dev() {
    local dev="$1"
    v4l2-ctl -d "$dev" --list-formats-ext 2>/dev/null | awk '
      /MJPG/ { in_mjpg=1; next }
      /^\t\[[0-9]+\]:/ && !/MJPG/ { in_mjpg=0 }
      in_mjpg && /Size: Discrete/ {
        gsub(/.*Discrete /, "")
        print
      }
    '
  }
  dev_has_mjpg() {
    local dev="$1"
    mjpg_sizes_for_dev "$dev" | grep -q .
  }
  pick_from_dev() {
    local dev="$1"
    [[ -e "$dev" ]] || return 1
    while IFS= read -r size; do
      [[ -z "$size" ]] && continue
      w="${size%%x*}"
      h="${size##*x}"
      [[ "$w" =~ ^[0-9]+$ && "$h" =~ ^[0-9]+$ ]] || continue
      px=$(( w * h ))
      if [[ "$w" -eq "$TARGET_W" && "$h" -eq "$TARGET_H" ]]; then
        best_dev="$dev"
        best_w="$w"
        best_h="$h"
        best_px="$px"
        has_target=1
        return 0
      fi
      if [[ "$has_target" -eq 0 && "$px" -gt "$best_px" ]]; then
        best_dev="$dev"
        best_w="$w"
        best_h="$h"
        best_px="$px"
      fi
    done < <(mjpg_sizes_for_dev "$dev" || true)
    return 1
  }
  best_dev="" best_w=0 best_h=0 best_px=0 has_target=0
  if [[ ${#USB_VIDEO_DEVS[@]} -gt 0 ]]; then
    for dev in "${USB_VIDEO_DEVS[@]}"; do
      dev_has_mjpg "$dev" || continue
      pick_from_dev "$dev" || true
      [[ "$has_target" -eq 1 ]] && break
    done
  fi
  if [[ -z "$best_dev" ]]; then
    for dev in /dev/video*; do
      [[ -e "$dev" ]] || continue
      dev_has_mjpg "$dev" || continue
      pick_from_dev "$dev" || true
      [[ "$has_target" -eq 1 ]] && break
    done
  fi
  if [[ -z "$best_dev" ]]; then
    echo "BŁĄD: brak urządzenia USB Video z MJPEG." >&2
    echo "  MacroSilicon: obraz jest na /dev/video0 (MJPEG)." >&2
    echo "  /dev/video1 to tylko metadane UVC — nie da się z niego czytać klatek." >&2
    echo "  Sprawdź HDMI z gimbala Tarot → przechwytywacz USB." >&2
    exit 1
  fi
  VIDEO_DEV="$best_dev"
  export DRONIADA_CAMERA_FOURCC="${DRONIADA_CAMERA_FOURCC:-MJPG}"
  if [[ "$has_target" -eq 1 ]]; then
    export DRONIADA_CAMERA_WIDTH="${TARGET_W}"
    export DRONIADA_CAMERA_HEIGHT="${TARGET_H}"
  else
    # Nadpisz 1920×1080 z jetson_competition_start — urządzenie ma tylko mniejsze MJPEG.
    export DRONIADA_CAMERA_WIDTH="${best_w}"
    export DRONIADA_CAMERA_HEIGHT="${best_h}"
    echo "UWAGA: brak MJPEG ${TARGET_W}x${TARGET_H} — używam ${best_w}x${best_h} na ${VIDEO_DEV}." >&2
    echo "  Sprawdź HDMI z gimbala Tarot → przechwytywacz USB (MacroSilicon)." >&2
  fi
fi
# vis = płynny MJPEG (kamera + overlay YOLO ~30 fps); dashboard = pełny panel na /live.jpg
export DRONIADA_STREAM_SOURCE="${DRONIADA_STREAM_SOURCE:-vis}"
export DRONIADA_STREAM_WIDTH="${DRONIADA_STREAM_WIDTH:-640}"
export DRONIADA_STREAM_INTERVAL_S="${DRONIADA_STREAM_INTERVAL_S:-0.033}"
export DRONIADA_VIS_PREVIEW_INTERVAL_S="${DRONIADA_VIS_PREVIEW_INTERVAL_S:-0.033}"
export DRONIADA_VIDEO_DEVICE="${VIDEO_DEV}"
# W kontenerze kamera jest zawsze pod /dev/video0 (mapowanie w docker-compose).
export DRONIADA_CAMERA_DEVICE=/dev/video0
export DRONIADA_CAMERA=0
unset DRONIADA_VIDEO_FILE DRONIADA_VIDEO_LOOP
if command -v v4l2-ctl >/dev/null 2>&1; then
  v4l2-ctl -d "${VIDEO_DEV}" \
    --set-fmt-video="width=${DRONIADA_CAMERA_WIDTH:-1920},height=${DRONIADA_CAMERA_HEIGHT:-1080},pixelformat=${DRONIADA_CAMERA_FOURCC:-MJPG}" \
    2>/dev/null || true
  br="${DRONIADA_CAMERA_BRIGHTNESS:-60}"
  v4l2-ctl -d "${VIDEO_DEV}" --set-ctrl="brightness=${br}" 2>/dev/null || true
fi
if [[ ${#USB_VIDEO_DEVS[@]} -gt 0 ]]; then
  echo "USB Video: ${USB_VIDEO_DEVS[*]} → host ${VIDEO_DEV} → kontener ${DRONIADA_CAMERA_DEVICE}"
  if [[ " ${USB_VIDEO_DEVS[*]} " == *" /dev/video1 "* ]]; then
    echo "  (video1 = metadane UVC; przechwytywanie tylko z video0 gdy ma MJPEG)"
  fi
fi
echo "Kamera: ${DRONIADA_CAMERA_DEVICE} (${DRONIADA_CAMERA_FOURCC:-MJPG} ${DRONIADA_CAMERA_WIDTH:-1920}x${DRONIADA_CAMERA_HEIGHT:-1080}) [host: ${VIDEO_DEV}]"
if command -v python3 >/dev/null 2>&1 && [[ -e "${VIDEO_DEV}" ]]; then
  _probe="$(python3 - "$VIDEO_DEV" "${DRONIADA_CAMERA_WIDTH}" "${DRONIADA_CAMERA_HEIGHT}" <<'PY' 2>/dev/null || true
import sys
import cv2
dev, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
if not cap.isOpened():
    print("brak dostępu (zajęte?)")
    raise SystemExit(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
ok, frame = False, None
for _ in range(15):
    ok, frame = cap.read()
    if ok and frame is not None and float(frame.std()) > 8.0:
        break
cap.release()
if not ok or frame is None:
    print("brak klatki testowej")
else:
    std = float(frame.std())
    print(f"test {frame.shape[1]}x{frame.shape[0]} std={std:.1f}", end="")
    if std < 8.0:
        print(" — SZARY EKRAN? Brak sygnału HDMI z gimbala.")
    else:
        print(" — sygnał OK")
PY
)"
  if [[ -n "${_probe}" ]]; then
    echo "Sygnał: ${_probe}"
    if [[ "${_probe}" == *"SZARY EKRAN"* || "${_probe}" == *"std="* ]]; then
      case "${_probe}" in
        *std=0.0*|*std=0.1*|*std=0.2*|*std=0.3*|*std=0.4*|*std=0.5*|*std=0.6*|*std=0.7*|*std=0.8*)
          echo "UWAGA: niski kontrast klatki (std) — poczekaj na stabilny HDMI albo podnieś jasność." >&2
          ;;
      esac
    fi
  fi
fi
# Host zwalnia V4L2 po probe — krótka pauza zanim kontener otworzy /dev/video0.
sleep 2
if [[ -n "${DRONIADA_INTERVAL_MS:-}" || -n "${DRONIADA_VIS_PREVIEW_INTERVAL_S:-}" ]]; then
  echo "Tune: INTERVAL_MS=${DRONIADA_INTERVAL_MS:-300} VIS_PREVIEW_S=${DRONIADA_VIS_PREVIEW_INTERVAL_S:-0.12} (→ kontener Docker)"
fi
echo "Wagi:   ${WEIGHTS} ($(du -h "${WEIGHTS}" | awk '{print $1}'))"
if [[ -n "${DRONIADA_SMOOTH_ALPHA:-}" || -n "${DRONIADA_HOLD_FRAMES:-}" ]]; then
  echo "Tracker: alpha=${DRONIADA_SMOOTH_ALPHA:-auto} hold=${DRONIADA_HOLD_FRAMES:-auto} interval=${DRONIADA_INTERVAL_MS:-auto}ms"
else
  echo "Tracker: domyślny profil kamery (alpha=0.38 hold=20) — tune: ./scripts/jetson_camera_tune.sh 0.45 24"
fi
if [[ -f "${ROOT}/config/card_colors.json" ]]; then
  export DRONIADA_CARD_COLORS="/ws/config/card_colors.json"
  echo "Kolory kartek: config/card_colors.json"
else
  echo "Kolory kartek: domyślne HSV (kalibracja: ./scripts/calibrate_card_colors.sh)"
fi

echo "Start kontenera (mapowanie ${VIDEO_DEV} → /dev/video0 w kontenerze)…"
if [[ "${DRONIADA_FORCE_RECREATE:-0}" == "1" ]]; then
  docker compose -f "${COMPOSE_FILE}" up -d --force-recreate
else
  docker compose -f "${COMPOSE_FILE}" up -d
fi
sleep 5

if docker ps --filter name=droniada_vision --format '{{.Status}}' | grep -qi up; then
  IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
  echo ""
  sleep 8
  if docker logs droniada_vision 2>&1 | grep -qE 'Podgląd WWW:|kamera:|live_panel.*8088|Panel WWW:|Migawki:'; then
    echo "OK — kontener działa (podgląd MJPEG na porcie ${HOST_PORT})."
    echo "Migawki (host): ${ROOT}/dataset/live_dashboard/session_*/panels/<A|B|C>/snapshots/"
  else
    echo "UWAGA: kontener Up, ale brak logów kamery/panelu." >&2
    docker logs droniada_vision 2>&1 | tail -10
    exit 1
  fi
  echo "Podgląd (tylko IP LAN — laptop musi być w tej samej sieci, np. SKNR_LAN):"
  _printed=0
  for _ip in $(hostname -I 2>/dev/null); do
    [[ "${_ip}" == 127.* ]] && continue
    # Pomiń mosty Docker (172.16–172.31) — z laptopa i tak ERR_ADDRESS_UNREACHABLE.
    [[ "${_ip}" =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\. ]] && continue
    echo "  http://${_ip}:${HOST_PORT}/"
    _printed=1
  done
  if [[ "${_printed}" -eq 0 ]]; then
    echo "  http://${IP:-<jetson-ip>}:${HOST_PORT}/"
  fi
  echo "Sterowanie: http://<jetson-ip>:${CONTROL_PORT}/"
  echo ""
  echo "Jeśli przeglądarka: ERR_ADDRESS_UNREACHABLE — laptop NIE widzi Jetsona w sieci."
  echo "  • Podłącz Wi‑Fi SKNR_LAN (adres laptopa też 192.168.100.x), albo"
  echo "  • Na Macu: DRONIADA_JETSON_PASS=sknr ./scripts/jetson_dashboard_tunnel.sh"
  echo "    potem http://127.0.0.1:${HOST_PORT}/ (tunel SSH)"
  echo "Logi:       docker logs -f droniada_vision"
  echo "Stop:       docker compose -f docker-compose.jetson.yml down"
  if command -v curl >/dev/null 2>&1; then
    _code="$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 4 "http://127.0.0.1:${HOST_PORT}/" 2>/dev/null || echo 000)"
    _api="$(curl -s --connect-timeout 4 "http://127.0.0.1:${HOST_PORT}/api/state" 2>/dev/null | head -c 80 || true)"
    if [[ "${_code}" == "200" ]]; then
      echo "WWW OK: port ${HOST_PORT} odpowiada (HTTP ${_code}) — otwórz jeden z adresów powyżej z laptopa w tej samej sieci (SKNR_LAN)."
      if [[ -n "${_api}" ]]; then
        echo "  API: ${_api}…"
      fi
    else
      echo "UWAGA: brak odpowiedzi HTTP na porcie ${HOST_PORT} (kod ${_code})." >&2
      echo "  Sprawdź: docker logs droniada_vision | tail -30" >&2
    fi
  fi
  docker logs droniada_vision 2>&1 | tail -8
else
  echo "Kontener nie wystartował. Sprawdź: docker logs droniada_vision" >&2
  exit 1
fi
