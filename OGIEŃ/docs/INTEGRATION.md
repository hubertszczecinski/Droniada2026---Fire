# Mission integration (WebSocket + GStreamer)

Shared vision core: `release/run_live_panel` (YOLO-Pose, modules A/B, CXY latch).  
Web dashboard and WebSocket are **optional** operator / integration layers.

## Mac (development)

```bash
python3 -m release.run_live_panel \
  --dashboard --module-a --cxy-latch --corner-mode yolo_pose \
  --camera 0 --web-port 8088 --headless
```

Train YOLO and keep the full `dataset/` / `runs/` tree on **Mac only** - do not sync the whole dataset to Jetson.

**Competition preview:** `:8088` - camera + broadcast reports A/B/C; `:8089` - operator panel (keys **1/2/3/0**). Texts from `config/preset_reports.json`. Historical `streaming/` Twitch relay is **not** used in the 2026 workflow.

## Jetson (runtime)

1. `./scripts/sync_to_jetson.sh` - code + config (no dataset/runs)
2. `./scripts/sync_jetson_weights.sh` - `best.pt` + optional `yolo_corner_bias.json`
3. On device: `export WS_ROOT_PATH=~/acoustics/droniada && ./scripts/docker_jetson_up.sh`
4. Browser: **http://&lt;jetson-ip&gt;:8088/**

## Environment variables (container)

| Variable | Description |
|----------|-------------|
| `DRONIADA_CAP_PIPELINE` | GStreamer pipeline → OpenCV `CAP_GSTREAMER` (e.g. `shmsrc` from simulator) |
| `DRONIADA_WS_URL` | `ws://host:port/path` - send `speed` to orchestrator |
| `DRONIADA_AUTONOMY_WS_URL` | `ws://host:port` - receive `hold_started` / `hold_stopped` |
| `DRONIADA_AUTONOMY_HOLD_PAUSE_S` | Analysis pause fallback when host omits `timeout` in `hold_started` |
| `DRONIADA_REPORT_MODE` | `live` (default) or `preset` - reports from `config/preset_reports.json`, keys **1/2/3** on `:8089` |
| `DRONIADA_PRESET_REPORTS` | Path to preset JSON (preset mode) |
| `DRONIADA_WEB_ENABLED=0` | Flight without HTTP (log + optional WS only) |
| `DRONIADA_YOLO_POSE_WEIGHTS` | Path to `best.pt` in `/ws` |

CLI: `--cap-pipeline '...'` sets `DRONIADA_CAP_PIPELINE` before camera start.

## WebSocket payload (sketch)

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
    "processed": [{"grid_row": 3, "grid_col": 3, "color": "pomaranczowa"}],
    "meta": {"reliable": true, "latch_locked": false}
  }]
}
```

`processed` is `null` until geometry is reliable or CXY is latched (when latch is active).

## CPU threads (Jetson)

Pipeline spread across cores (default `DRONIADA_ASYNC_ANALYSIS=1`):

| Thread | Role |
|--------|------|
| `droniada-cam` | V4L2 / GStreamer read |
| `droniada-vis-preview` | Overlay + stream buffer (~30 fps) |
| `droniada-mjpeg-enc` | JPEG encode → `/stream.mjpg` |
| `droniada-yolo` | YOLO + module B + CXY (queue=1, drop stale) |
| HTTP | `ThreadingHTTPServer` (8088/8089) |
| main loop | Orchestration, dashboard, snapshots (light) |

`OMP_NUM_THREADS=1` - workers do not grab all cores at once.

Disable async YOLO (debug): `DRONIADA_ASYNC_ANALYSIS=0`

## GStreamer MJPEG (Jetson)

Default for competition (`DRONIADA_USE_GST_CAPTURE=1`):

```
v4l2src (MJPEG) ──tee──► jpegparse ──► HTTP /stream.mjpg  (passthrough, no cv2.imencode)
                  └──► nvjpegdec ──► BGR ──► YOLO
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `DRONIADA_USE_GST_CAPTURE` | `1` | GStreamer instead of OpenCV V4L2 |
| `DRONIADA_GST_HW_JPEG` | `1` | `nvjpegdec`/`nvjpegenc` on Jetson |
| `DRONIADA_STREAM_PASSTHROUGH` | `1` | Raw camera JPEG on `:8088/stream.mjpg` |

YOLO overlay on **`/live.jpg`** (refreshed after each analysis). MJPEG stream = smooth camera like local preview.

Disable passthrough (software overlay on stream): `DRONIADA_STREAM_PASSTHROUGH=0`

## Web preview (8088) - YOLO stutter

MJPEG (`/stream.mjpg`) defaults to `DRONIADA_STREAM_SOURCE=vis`: camera + cached YOLO overlay between inferences (~30 fps), full YOLO every `interval_ms` (~300 ms). Full dashboard (modules A/B, inset) on `/live.jpg` (~3 fps).

| Cause | Fix |
|-------|-----|
| `STREAM_SOURCE=dashboard` - stream waits for YOLO | Set `vis` (Jetson default) |
| YOLO ~300 ms/frame | Expected; overlay uses cache between frames |
| Wide MJPEG 1280 px | `DRONIADA_STREAM_WIDTH=960` |

## Autonomy events (inbound WebSocket)

ROS2 host sends **edge-triggered** events (one message per start/stop). Set `DRONIADA_AUTONOMY_WS_URL=ws://host:8765`.

| Event | Vision action |
|-------|----------------|
| `hold_started` | Start hold - **collect snapshots** (detection on), local timer for host `timeout` s |
| `hold_stopped` | End hold - report from **snapshots (CXY)** on **:8088**, clear snapshots, **pause analysis** 8 s (`DRONIADA_AUTONOMY_HOLD_PAUSE_S`) |
| `hold_expired` | Local timer expired without `hold_stopped` - same as row above |

```json
{"event": "hold_started", "timeout": 5.0}
{"event": "hold_stopped", "timeout": 0.0}
```

Port **8765** is for these events only (not `speed`). Client (`IntegrationWsSubscriber`) keeps a local hold timer: new `hold_started` preempts the previous one; `hold_stopped` cancels timer and triggers report.

**Local test (mock host):**

```bash
python3 scripts/mock_autonomy_ws.py --port 8765
# second terminal:
DRONIADA_AUTONOMY_WS_URL=ws://127.0.0.1:8765 ./scripts/run_local_video_test.sh dataset/my_capture/Test3.mov
# in mock: Enter = hold_started, s = hold_stopped
```

## Workflow split

- **Mac:** UI, reports, WS contract - fast iteration.
- **Jetson:** same Docker image, different env (UVC camera, `DRONIADA_WS_URL` to orchestrator, `DRONIADA_WEB_ENABLED=0` in flight).
