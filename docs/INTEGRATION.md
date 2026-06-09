# Integracja misji (WebSocket + GStreamer)

Wspólny rdzeń wizji: `release/run_live_panel` (YOLO-Pose, moduł A/B, CXY latch).  
Dashboard WWW i WebSocket to **opcjonalne** warstwy operatorskie / integracyjne.

## Mac (development)

```bash
python3 -m release.run_live_panel \
  --dashboard --module-a --cxy-latch --corner-mode yolo_pose \
  --camera 0 --web-port 8088 --headless
```

Trening YOLO i dataset **tylko lokalnie** — nie synchronizuj `dataset/` ani `runs/` całego drzewa na Jetson.

**Podgląd na zawodach:** `:8088` — kamera + przełączanie raportów A/B/C (broadcast); `:8089` — panel operatora (skróty **1/2/3/0**). Teksty z `config/preset_reports.json`. Moduł [`lastmile/`](../lastmile/README.md) (Twitch/Gigabyte) — **nieużywany** w obecnym workflow.

## Jetson (runtime)

1. `./scripts/sync_to_jetson.sh` — kod + config (bez dataset/runs)
2. `./scripts/sync_jetson_weights.sh` — `best.pt` + opcjonalnie `yolo_corner_bias.json`
3. Na urządzeniu: `export WS_ROOT_PATH=~/acoustics/droniada && ./scripts/docker_jetson_up.sh`
4. Panel: **http://&lt;jetson-ip&gt;:8088/**

## Zmienne środowiskowe (kontener)

| Zmienna | Opis |
|---------|------|
| `DRONIADA_CAP_PIPELINE` | Pipeline GStreamer → OpenCV `CAP_GSTREAMER` (np. `shmsrc` z symulatora) |
| `DRONIADA_WS_URL` | `ws://host:port/path` — wysyłka `speed` do orchestratora |
| `DRONIADA_AUTONOMY_WS_URL` | `ws://host:port` — odbiór `hold_started` / `hold_stopped` (patrz niżej) |
| `DRONIADA_AUTONOMY_HOLD_PAUSE_S` | Fallback pauzy tylko gdy host **nie** poda `timeout` w `hold_started` |
| `DRONIADA_REPORT_MODE` | `live` (domyślnie) lub `preset` — raporty z `config/preset_reports.json`, skróty **1/2/3** na `:8089` |
| `DRONIADA_PRESET_REPORTS` | Ścieżka do JSON z raportami A/B/C (tryb preset) |
| `DRONIADA_WEB_ENABLED=0` | Lot bez HTTP (tylko log + opcjonalnie WS) |
| `DRONIADA_YOLO_POSE_WEIGHTS` | Ścieżka do `best.pt` w `/ws` |

CLI: `--cap-pipeline '...'` ustawia `DRONIADA_CAP_PIPELINE` przed startem kamery.

## Payload WebSocket (szkic)

```json
{
  "frame_id": "cam_000042",
  "timestamp_ms": 1717420000000,
  "panels": [{
    "track_id": "panel_A",
    "panel_id": "A",
    "orientation": {"roll": 0, "pitch": 90, "yaw": 0},
    "distance": {"x": 0, "y": 0, "z": 2.4},
    "size": {"x": 0.35, "y": 0.28},
    "processed": [{"grid_row": 3, "grid_col": 3, "color": "pomarańczowa"}],
    "meta": {"reliable": true, "latch_locked": false}
  }]
}
```

`processed` jest `null`, dopóki brak `reliable` lub zatrzaśniętego CXY (gdy latch aktywny).

## Wątki CPU (Jetson)

Pipeline rozłożony na rdzenie (domyślnie `DRONIADA_ASYNC_ANALYSIS=1`):

| Wątek | Rola |
|-------|------|
| `droniada-cam` | Odczyt V4L2 / GStreamer |
| `droniada-vis-preview` | Overlay + bufor streamu (~30 fps) |
| `droniada-mjpeg-enc` | Enkod JPEG → `/stream.mjpg` |
| `droniada-yolo` | YOLO + moduł B + CXY (kolejka=1, drop starych) |
| HTTP | `ThreadingHTTPServer` (8088/8089) |
| główna pętla | Orkiestracja, dashboard, migawki (lekka) |

`OMP_NUM_THREADS=1` — każdy worker nie zajmuje wszystkich rdzeni naraz.

Wyłącz async YOLO (debug): `DRONIADA_ASYNC_ANALYSIS=0`

## GStreamer MJPEG (Jetson)

Domyślnie na zawody (`DRONIADA_USE_GST_CAPTURE=1`):

```
v4l2src (MJPEG) ──tee──► jpegparse ──► HTTP /stream.mjpg  (passthrough, bez cv2.imencode)
                  └──► nvjpegdec ──► BGR ──► YOLO
```

| Zmienna | Domyślnie | Znaczenie |
|---------|-----------|-----------|
| `DRONIADA_USE_GST_CAPTURE` | `1` | GStreamer zamiast OpenCV V4L2 |
| `DRONIADA_GST_HW_JPEG` | `1` | `nvjpegdec`/`nvjpegenc` na Jetsonie |
| `DRONIADA_STREAM_PASSTHROUGH` | `1` | Surowy JPEG kamery na `:8088/stream.mjpg` |

Overlay YOLO jest na **`/live.jpg`** (odświeżany po analizie). Stream MJPEG = płynna kamera jak lokalnie.

Wyłącz passthrough (software encode overlay na stream): `DRONIADA_STREAM_PASSTHROUGH=0`

## Podgląd WWW (8088) — zacinanie YOLO

MJPEG (`/stream.mjpg`) domyślnie używa `DRONIADA_STREAM_SOURCE=vis`: kamera + overlay YOLO odświeżany **między** inferencjami (~30 fps z cache), a YOLO pełne działa co `interval_ms` (~300 ms). Pełny dashboard (moduł A/B, inset) jest na `/live.jpg` (~3 fps).

| Przyczyna zacinania | Rozwiązanie |
|---------------------|-------------|
| `STREAM_SOURCE=dashboard` — stream tylko po YOLO | Ustaw `vis` (domyślnie na Jetsonie) |
| YOLO ~300 ms na klatkę | Normalne; overlay między klatkami z cache |
| Szeroki MJPEG 1280 px | `DRONIADA_STREAM_WIDTH=960` |

## Zdarzenia autonomii (inbound WebSocket)

Host ROS2 wysyła zdarzenia **edge-triggered** (jedna wiadomość na start/stop, bez strumienia odliczania). Ustaw `DRONIADA_AUTONOMY_WS_URL=ws://host:8765`.

| Zdarzenie | Działanie wizji |
|-----------|-----------------|
| `hold_started` | Start holdu — **zbieranie migawek** (detekcja włączona), lokalny timer na `timeout` s z hosta |
| `hold_stopped` | Koniec holdu — raport z **migawek (CXY)** na **:8088**, zerowanie migawek, **pauza analizy** 8 s (`DRONIADA_AUTONOMY_HOLD_PAUSE_S`) |
| `hold_expired` | Lokalny timer wygasł bez `hold_stopped` — to samo co wiersz wyżej |

```json
{"event": "hold_started", "timeout": 5.0}
{"event": "hold_stopped", "timeout": 0.0}
```

Port **8765** to wyłącznie te zdarzenia (nie `speed`). Klient (`IntegrationWsSubscriber`) utrzymuje lokalny timer holdu: nowy `hold_started` preempuje poprzedni; `hold_stopped` anuluje timer i wyzwala raport. `timeout` w `hold_started` = czas trwania holdu u autonomii; pauza analizy po raporcie = osobno (domyślnie 8 s).

**Test lokalny (mock hosta):**

```bash
python3 scripts/mock_autonomy_ws.py --port 8765
# drugi terminal:
DRONIADA_AUTONOMY_WS_URL=ws://127.0.0.1:8765 ./scripts/run_local_video_test.sh dataset/my_capture/Test3.mov
# w mocku: Enter = hold_started, s = hold_stopped
```

## Workflow z Piotrem

- **Mac:** UI, raporty, kontrakt WS — szybka iteracja.
- **Jetson:** ten sam obraz Docker, inne env (kamera UVC, `DRONIADA_WS_URL` do orchestratora, `DRONIADA_WEB_ENABLED=0` w locie).
