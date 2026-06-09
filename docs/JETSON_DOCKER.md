# Jetson — kontener z pełnym panelem live

Detekcja (YOLO-Pose + moduł A/B + migawki) działa **w kontenerze** `droniada_vision`.  
Podgląd w przeglądarce z Maca/LAN: **pełny dashboard** (nie tylko MJPEG).

**Gdzie rozwijać:** dashboard, raporty i WebSocket — na **Macu** (ten sam kod). Na **Jetsonie** tylko ten sam obraz Docker + env (`DRONIADA_CAP_PIPELINE`, `DRONIADA_WS_URL`, opcjonalnie `DRONIADA_WEB_ENABLED=0` w locie). Trening YOLO zostaje na macOS.

Szczegóły integracji: [INTEGRATION.md](INTEGRATION.md).

## Sync kodu (Mac → Jetson, tylko `~/acoustics/droniada`)

```bash
export DRONIADA_JETSON_PASS='…'   # opcjonalnie
./scripts/sync_to_jetson.sh
./scripts/sync_jetson_weights.sh   # best.pt (+ bias JSON)
```

## Start na Jetsonie

```bash
cd ~/acoustics/droniada
export WS_ROOT_PATH="$(pwd)"
./scripts/docker_jetson_up.sh
```

### Sieć konkursowa (bez internetu)

Wszystko musi być **wcześniej** na Jetsonie: kod w `~/acoustics/droniada`, obraz `droniada-vision:latest`, `runs/pose/.../best.pt`.

**Na Macu** (tylko gdy masz połączenie z Jetsonem — dowolna sieć):

```bash
export DRONIADA_JETSON_HOST=sknr@192.168.100.210   # IP w sieci konkursowej
export DRONIADA_JETSON_PASS=sknr                     # jeśli używasz sshpass
./scripts/sync_to_jetson.sh
./scripts/sync_jetson_weights.sh
```

**Na Jetsonie** (SSH, bez internetu — nie rób `docker compose build`):

```bash
cd ~/acoustics/droniada
export WS_ROOT_PATH="$(pwd)"
./scripts/jetson_offline_start.sh
```

Port 80 zamiast 8088 (jeśli otwierasz `http://192.168.100.210/` bez portu):

```bash
export DRONIADA_HOST_PORT=80
./scripts/jetson_offline_start.sh
# → http://192.168.100.210/
```

**Przydatne komendy na miejscu:**

| Akcja | Komenda |
|-------|---------|
| Status | `docker ps \| grep droniada` |
| Logi live | `docker logs -f droniada_vision` |
| Restart | `docker compose -f docker-compose.jetson.yml restart` |
| Stop | `docker compose -f docker-compose.jetson.yml down` |
| Kamera | `v4l2-ctl --list-devices` → ustaw `DRONIADA_VIDEO_DEVICE=/dev/videoN` |

**Checklist przed wyjazdem (sieć z internetem):**

1. `./scripts/sync_to_jetson.sh` + `sync_jetson_weights.sh`
2. Na Jetsonie raz: `docker compose -f docker-compose.jetson.yml build` (pobiera zależności do obrazu)
3. Test: `./scripts/jetson_offline_start.sh` → panel w przeglądarce
4. `docker image ls droniada-vision` — obraz musi istnieć offline

## Panel w przeglądarce

| Port | Adres | Rola |
|------|--------|------|
| Podgląd | **http://&lt;jetson-ip&gt;:8088/** | Dashboard live (moduł A/B, kratka, parametry); raporty A/B/C nakładane po lewej (skróty na :8089) |
| Sterowanie | **http://&lt;jetson-ip&gt;:8089/** | Skróty **1/2/3** dodają raport A/B/C na :8088 (kumulują się), **0** czyści wszystkie |

Przy `DRONIADA_HOST_PORT=80` podgląd może być pod **http://&lt;jetson-ip&gt;/** (bez portu).

Raporty broadcast: `config/preset_reports.json` (szablon: `config/preset_reports.example.json`). Start: `./scripts/jetson_competition_start.sh` lub `./scripts/jetson_restart.sh`.

Na :8089 dodatkowo: parametry modułu A/B, migawki, status analizy YOLO.

## Kamera USB (UVC)

Domyślnie w `docker-compose.jetson.yml`:

- `/dev/video1` (USB UVC na tym Jetsonie; jeśli inny indeks: `DRONIADA_VIDEO_DEVICE=/dev/video2`)
- **MJPEG** — rozdzielczość zależy od kamery (często 640×480 lub 1920×1080 po `v4l2-ctl`)
- `DRONIADA_CAMERA_BRIGHTNESS=60` (fabryczne -11 = prawie czarny obraz)

Zmiana jasności:

```bash
DRONIADA_CAMERA_BRIGHTNESS=80 docker compose -f docker-compose.jetson.yml up -d
```

Inny port na hoście:

```bash
DRONIADA_HOST_PORT=9090 docker compose -f docker-compose.jetson.yml up -d
# → http://<jetson>:9090/
```

## Lot bez WWW (tylko WS / log)

```bash
DRONIADA_WEB_ENABLED=0 DRONIADA_WS_URL=ws://192.168.1.10:9000/vision \
  docker compose -f docker-compose.jetson.yml up -d
```

## GStreamer zamiast `/dev/video0`

Ustaw pipeline (np. symulator):

```bash
DRONIADA_CAP_PIPELINE='shmsrc socket-path=/tmp/cam.sock ! ... ! appsink' \
  docker compose -f docker-compose.jetson.yml up -d
```
