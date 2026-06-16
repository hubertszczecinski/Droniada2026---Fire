# Module A - drone pose + panel stand (2026-05-27)

Module A handles **approach geometry**: panel corners, PnP, distance, drone angles relative to the panel, and **stand category** (horizontal / tilted / vertical).

Module B (close-range CXY) is **not** the source of panel stand - it receives category from A or from the mission.

---

## Live architecture (real footage)

Module A has a **separate PnP path** but shares the **same corner detector** as module B:

```text
Camera → YOLO-Pose (1× per frame when bench A+B)
              ├→ Module B: warp + CXY + latch
              └→ Module A: corner canon → PnP → d, roll/pitch/yaw + stand
```

| Layer | Module A | Module B |
|-------|----------|----------|
| Corners | `module_pose/yolo_pose_bridge.py` → `detect_corners_yolo_pose` | `release/live_corners.py` (`corner_mode=yolo_pose`) |
| Weights / bias | `DRONIADA_YOLO_POSE_WEIGHTS`, `yolo_corner_bias.json` | same |
| Stabilisation | EMA/hold from `live_corners` | same |
| Next | `pose_from_corners` → PnP | warp + HSV CXY |

With `./scripts/run_live_bench.sh`, YOLO runs **once** - corners go to both modules (`shared_yolo` in `run_live_panel.py`).

API:

```python
from module_pose import pose_from_yolo_pose, pose_from_corners

res = pose_from_yolo_pose(bgr, k=k, dist=dist)
res = pose_from_corners(bgr, corners_px, k=k, dist=dist, base_method='yolo_pose')
```

`PoseConfig(corner_source='yolo_pose')` - default in live; CV fallback when YOLO returns no corners.

---

## Integration output (flight / shmsrc)

```python
from release.pose_runtime import PoseConfig, PoseRuntime

rt = PoseRuntime(PoseConfig(corner_source='yolo_pose'))
po = rt.process_bgr(bgr, 'frame_001', k=k, dist=dist)
d = po.to_integration_dict(panel_id='A')
```

`report_angle_deg` (0 / 45 / 90) = **panel on stand**, not drone flight angle. Viewing angles → `roll_deg`, `pitch_deg`, `yaw_deg`.

---

## Stand classification (3 classes)

| Category | Report angle | Label (PL rules) |
|----------|--------------|------------------|
| `horizontal` | 0° | poziomy |
| `45_deg` | 45° | poziomy-przechylony |
| `vertical` | 90° | pionowy |

Algorithm (`module_pose/panel_stand.py`):

1. **Linear calibration** on PnP + quad geometry features  
2. Weights: `module_pose/data/panel_stand_linear.json` (`pipelines.calibrate_panel_stand`)  
3. Fallback: rotation-matrix thresholds (`rmat_theta`) when calibration file missing

---

## Calibration and tests (Blender)

```bash
chmod +x scripts/verify_module_a_blender.sh
./scripts/verify_module_a_blender.sh
```

```bash
python3 -m pipelines.calibrate_panel_stand --dataset dataset
python3 -m pipelines.eval_module_a_blender --dataset dataset --out dataset/results/eval_module_a_blender.json
python3 -m unittest tests.test_module_a_blender -v
```

### Reference metrics (156 Blender frames, frontal bank)

| Metric | Value |
|--------|-------|
| Pose OK | 100% |
| Stand (3 classes) | ~80% |
| Vertical | ~94% |
| Distance error (median) | ~1.3 m |

Weaker case: **horizontal vs 45°** (~62–74%) - similar appearance; improve with more real hall frames + per-stand corner labels.

---

## Files

| File | Role |
|------|------|
| `module_pose/yolo_pose_bridge.py` | A→B bridge: same YOLO-Pose + tracker |
| `module_pose/api.py` | `pose_from_yolo_pose`, `pose_from_corners`, PnP |
| `module_pose/panel_stand.py` | Stand classifier + `to_integration_dict` |
| `module_pose/types.py` | `PoseResult.to_integration_dict()` |
| `release/pose_runtime.py` | Live runtime |
| `pipelines/calibrate_panel_stand.py` | Blender training |
| `pipelines/eval_module_a_blender.py` | JSON report |
| `tests/test_module_a_blender.py` | Regression |
