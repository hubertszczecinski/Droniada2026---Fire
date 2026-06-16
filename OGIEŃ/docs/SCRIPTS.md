# Scripts — pipeline, corners, colours, Jetson, water

Reference for scripts used in daily workflow. Run from `OGIEŃ/`:

```bash
cd OGIEŃ
```

### 1) Live pipeline and video tests

- `./scripts/run_live_dashboard.sh` — full live dashboard: A+B, grid/colours, snapshots, HTTP/WS operator layer.
- `./scripts/run_live_bench.sh` — shortcut (same as dashboard with bench defaults).
- `./scripts/run_local_video_test.sh <video.mov>` — simplest local end-to-end test on a file.

```bash
./scripts/run_local_video_test.sh dataset/my_capture/Test.mov
```

### 2) Corners — YOLO-Pose and geometry

- `./scripts/calibrate_yolo_corner_bias.sh` — YOLO corner bias (pred−GT on `panel_labels`) → `module_panel/data/yolo_corner_bias.json`.
- `python3 -m release.run_corner_tune ...` — corner detection preview/tuning (same logic as live).
- `./scripts/train_yolo_two_stage.sh [0|1|2|all]` — two-stage YOLO-Pose (Blender export → pretrain → real finetune).

### 3) Card colours

- `./scripts/calibrate_card_colors.sh` — from `config/competition_cards/*.png`.
- `./scripts/extract_panel_colors.sh --video <panel_calib.mov>` — colours from panel recording.
- `./scripts/train_competition_colors.sh` — competition colour classifier.
- `./scripts/train_snapshot_colors.sh` — tuning for snapshot lighting.
- `./scripts/jetson_calibrate_colors.sh` — colour calibration wrapper on Jetson.
- `./scripts/build_universal_profile.sh` — merge profiles into one universal file.

### 4) Jetson — sync and competition start

- `./scripts/sync_to_jetson.sh` — code sync (no heavy dataset).
- `./scripts/sync_jetson_weights.sh` — `best.pt` + bias JSON.
- `./scripts/docker_jetson_up.sh` — start `droniada_vision` container.
- `./scripts/jetson_offline_start.sh` — offline competition start (no `docker compose build`).
- `./scripts/jetson_competition_start.sh` — competition snapshot thresholds + full stack.
- `./scripts/jetson_restart.sh`, `./scripts/jetson_stop.sh` — restart / stop.
- `./scripts/jetson_dashboard_tunnel.sh` — port tunnel helper.
- `./scripts/test_ab_jetson.sh` — module A/B smoke test on Jetson.

### 5) Cleanup

- `./scripts/cleanup_artifacts.sh` — removes generated debug/live galleries and eval JSON; keeps training images and weights.

### 6) Water (separate view + snapshots)

- `./scripts/woda_camera_start.sh` — lightweight water camera preview + snapshots on API/WS trigger.
- `python3 scripts/mock_autonomy_ws.py --port 8765` — mock autonomy host (`hold_started` / `hold_stopped`).
