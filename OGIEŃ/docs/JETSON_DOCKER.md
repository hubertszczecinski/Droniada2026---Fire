# Jetson - live panel Docker container

Detection (YOLO-Pose + modules A/B + snapshots) runs in container `droniada_vision`.  
Browser preview from Mac/LAN: **full dashboard** (not MJPEG only).

**Where to develop:** dashboard, reports, WebSocket - on **Mac** (same code). On **Jetson** only the Docker image + env (`DRONIADA_CAP_PIPELINE`, `DRONIADA_WS_URL`, optional `DRONIADA_WEB_ENABLED=0` in flight). YOLO training stays on macOS.

See [INTEGRATION.md](INTEGRATION.md).

## Sync code (Mac → Jetson, `~/acoustics/droniada`)

```bash
export DRONIADA_JETSON_PASS='…'   # optional
./scripts/sync_to_jetson.sh
./scripts/sync_jetson_weights.sh   # best.pt (+ bias JSON)
```

## Start on Jetson

```bash
cd ~/acoustics/droniada
export WS_ROOT_PATH="$(pwd)"
./scripts/docker_jetson_up.sh
```

### Competition network (no internet)

Everything must already be on the Jetson: code in `~/acoustics/droniada`, image `droniada-vision:latest`, `runs/pose/.../best.pt`.

**On Mac** (when connected to Jetson):

```bash
export DRONIADA_JETSON_HOST=sknr@192.168.100.210
export DRONIADA_JETSON_PASS=sknr
./scripts/sync_to_jetson.sh
./scripts/sync_jetson_weights.sh
```

**On Jetson** (SSH, offline - do not `docker compose build`):

```bash
cd ~/acoustics/droniada
export WS_ROOT_PATH="$(pwd)"
./scripts/jetson_offline_start.sh
```

Port 80 instead of 8088:

```bash
export DRONIADA_HOST_PORT=80
./scripts/jetson_offline_start.sh
# → http://192.168.100.210/
```

**On-site commands:**

| Action | Command |
|--------|---------|
| Status | `docker ps \| grep droniada` |
| Live logs | `docker logs -f droniada_vision` |
| Restart | `docker compose -f docker-compose.jetson.yml restart` |
| Stop | `docker compose -f docker-compose.jetson.yml down` |
| Camera | `v4l2-ctl --list-devices` → `DRONIADA_VIDEO_DEVICE=/dev/videoN` |

**Pre-trip checklist (with internet):**

1. `./scripts/sync_to_jetson.sh` + `sync_jetson_weights.sh`
2. On Jetson once: `docker compose -f docker-compose.jetson.yml build`
3. Test: `./scripts/jetson_offline_start.sh` → panel in browser
4. `docker image ls droniada-vision` - image must exist offline

## Browser panel

| Port | URL | Role |
|------|-----|------|
| Preview | **http://&lt;jetson-ip&gt;:8088/** | Live dashboard (A/B, grid, params); reports A/B/C overlaid (keys on :8089) |
| Operator | **http://&lt;jetson-ip&gt;:8089/** | Keys **1/2/3** add report A/B/C on :8088 (stack), **0** clears all |

With `DRONIADA_HOST_PORT=80`, preview may be **http://&lt;jetson-ip&gt;/** (no port).

Broadcast reports: `config/preset_reports.json`. Start: `./scripts/jetson_competition_start.sh` or `./scripts/jetson_restart.sh`.

On :8089: module A/B params, snapshots, YOLO analysis status.

## Water view + snapshots (:8087)

Separate lightweight container (no YOLO) - camera preview and snapshot save on WebSocket trigger `:8765`.

| Port | URL | Role |
|------|-----|------|
| Water preview | **http://&lt;jetson-ip&gt;:8087/** | MJPEG + snapshot gallery under stream |
| Trigger | **ws://&lt;jetson-ip&gt;:8765** | `hold_started`, `hold_stopped`, `snapshot`, … |

Disk path on Jetson: **`/migawka-woda/snapshot_YYYYMMDD_HHMMSS_mmm.jpg`**

```bash
cd ~/acoustics/droniada
export WS_ROOT_PATH="$(pwd)"
./scripts/woda_camera_start.sh
```

Manual snapshot (no WS):

```bash
curl -X POST http://127.0.0.1:8087/api/snapshot
```

Mock autonomy (Enter = `hold_started` → snapshot):

```bash
python3 scripts/mock_autonomy_ws.py --port 8765
```

Env: `WODA_VIDEO_DEVICE`, `WODA_WS_URL`, `WODA_SNAPSHOT_ON`, `WODA_OUTPUT_DIR`.

## USB camera (UVC)

Defaults in `docker-compose.jetson.yml`:

- `/dev/video1` (USB UVC; override: `DRONIADA_VIDEO_DEVICE=/dev/video2`)
- **MJPEG** - resolution depends on camera (often 640×480 or 1920×1080 after `v4l2-ctl`)
- `DRONIADA_CAMERA_BRIGHTNESS=60` (factory -11 = nearly black image)

```bash
DRONIADA_CAMERA_BRIGHTNESS=80 docker compose -f docker-compose.jetson.yml up -d
```

Different host port:

```bash
DRONIADA_HOST_PORT=9090 docker compose -f docker-compose.jetson.yml up -d
```

## Flight without WWW (WS / log only)

```bash
DRONIADA_WEB_ENABLED=0 DRONIADA_WS_URL=ws://192.168.1.10:9000/vision \
  docker compose -f docker-compose.jetson.yml up -d
```

## GStreamer instead of `/dev/video0`

```bash
DRONIADA_CAP_PIPELINE='shmsrc socket-path=/tmp/cam.sock ! ... ! appsink' \
  docker compose -f docker-compose.jetson.yml up -d
```
