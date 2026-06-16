# OGIEŃ - live panel vision (Droniada 2026)

Python vision stack for the OGIEŃ stage: **module A** (pose, distance, panel stand) + **module B** (10×10 grid, card colours), YOLO-Pose corners, CXY latch, snapshots, and web dashboard (`:8088` / `:8089`).

**Team SKN Robotycy × KINO** - qualified as one of two teams at the Fire & Water qualifier and **won the OGIEŃ competition** with this code.

Dependencies: `pip install -r requirements.txt` (NumPy + OpenCV headless).  
Optional: **Blender** (bundled Python) for synthetic dataset generation.

## Documentation

| Doc | Topic |
|-----|--------|
| [`docs/INTEGRATION.md`](docs/INTEGRATION.md) | Autonomy WebSocket + GStreamer |
| [`docs/JETSON_DOCKER.md`](docs/JETSON_DOCKER.md) | Jetson Docker runtime |
| [`docs/COMPETITION_REPORT.md`](docs/COMPETITION_REPORT.md) | Competition console report format |
| [`docs/MODULE_A.md`](docs/MODULE_A.md), [`docs/MODULE_B_SNAPSHOT.md`](docs/MODULE_B_SNAPSHOT.md) | Modules A / B |
| [`docs/LIVE_BENCH.md`](docs/LIVE_BENCH.md) | Live bench / dashboard |
| [`docs/SCRIPTS.md`](docs/SCRIPTS.md) | Script reference |

## Dataset

Scripts expect `--dataset` (default `dataset/`). Images, labels, and weights are gitignored — generate synthetic data with Blender or train on your own captures (see **Blender dataset** below).

```
dataset/
├── images/           # img_0.png, img_1.png, …
├── labels_yolo/      # YOLO det: class cx cy w h
├── labels_raport/    # GT report lines (parsed by pipeline_competition)
└── labels_pose/      # intrinsics, camera, panel, model_to_camera_opencv (Blender GT)
```

`module_panel/data/` holds calibration files (e.g. `angle_linear_rmat.json`, `yolo_corner_bias.json`).

## Videos (local tests)

Do not commit `.mov` files. Place test recordings at:

- `dataset/my_capture/<name>.mov` - `run_local_video_test.sh`, `release.run_video`
- `python3 -m release.run_live_panel --video <panel.mov>` - module B on a file

Demo screen recording (source for README GIF): `docs/demo.mov` (local, gitignored).

## Module A - `module_pose`

**Input:** `pose_from_image(image_bgr, yolo_det=None, k=None, dist=None, …)` or `pose_from_paths(...)`.

**Output:** `PoseResult` → `to_dict()` with `ok`, `confidence`, `rvec`/`tvec`, `corners_px`, Euler angles, distance to panel centre, `meta.reproj_mean_px`, etc.

## Module B - `module_panel`

**Input:** `analyze_panel_image(image_bgr, yolo_det, k, dist, xy_mode=..., angle_source=..., ...)`

**Output:** `PanelAnalyzeResult` - `predictions` (grid cells + colours), `warped_bgr`, `report_angle_deg`, `meta` (`xy_mode`, `grid_xy_reliable`, …).

Single-image pipeline:

```bash
python3 -m pipelines.full_run --image dataset/images/img_0.png
python3 -m pipelines.full_run --image dataset/images/img_0.png --competition
```

## CLI overview

| Command | Purpose |
|---------|---------|
| `python3 pipeline_competition.py --dataset ./dataset` | Legacy benchmark variants → `reports/` |
| `python3 -m pipelines.eval_module_b --dataset dataset` | Module B metrics vs `labels_raport` |
| `./scripts/run_live_dashboard.sh` | Full live dashboard A+B - [`docs/LIVE_BENCH.md`](docs/LIVE_BENCH.md) |
| `python3 -m release.run_live --mode pose --preview` | Live module A - [`docs/MODULE_A.md`](docs/MODULE_A.md) |
| `python3 -m release.run_live_panel --video … --corner-mode yolo_pose --cxy-latch` | Live module B - [`docs/MODULE_B_SNAPSHOT.md`](docs/MODULE_B_SNAPSHOT.md) |
| `./scripts/cleanup_artifacts.sh` | Remove sessions, eval JSON, debug galleries (keeps training images + weights) |
| `./scripts/run_local_video_test.sh <video.mov>` | Local end-to-end test (`:8088` / `:8089`) |

## Blender dataset

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python generate_dataset_blender.py
DRONIADA_FRESH=1 ./scripts/regenerate_blender_frontal.sh
```

YOLO-Pose training:

```bash
./scripts/train_yolo_two_stage.sh all
export DRONIADA_YOLO_POSE_WEIGHTS="$(pwd)/runs/pose/droniada_real_finetune/weights/best.pt"
```

## Release / live

```bash
python3 -m release.run_pose --dataset dataset
python3 -m release.run_panel --dataset dataset --angle-source rmat_linear
```

Mac camera (close QuickTime first):

```bash
python3 -m release.run_live --mode both --interval-ms 500 --preview
python3 -m release.run_live_panel --video dataset/my_capture/Droniada_nag3.mov --rotate 180 \
  --corner-mode yolo_pose --cxy-latch
```

Alignment / corner tuning: `release.run_alignment`, `release.run_corner_tune`, `pipelines.eval_corner_methods`.

## Python imports

Project root must be on `sys.path` (CLI scripts add it automatically):

```python
import sys
sys.path.insert(0, "/path/to/Droniada/OGIEŃ")
from module_pose.api import pose_from_paths
from module_panel.analyze import analyze_panel_image
```
