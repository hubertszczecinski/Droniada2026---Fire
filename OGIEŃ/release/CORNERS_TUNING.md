# Outer corner tuning (module B)

Goal: **median error &lt; 50 px** on 73 labels (`dataset/panel_labels/*_r180.json`).

## Work loop

```bash
python3 -m release.eval_outer_tune --note "change description"
python3 -m release.eval_corner_distance
```

Log JSONL: `release/CORNERS_EXPERIMENTS.jsonl`

## Findings (session 2026-05-20)

### Baseline (before fixes)

| Metric | Value |
|--------|-------|
| PICK median | **101.5 px** |
| ORACLE median | **96.7 px** |
| GAP median | 0 px (max **501 px** on one frame) |

**Main issue: wrong candidate choice**, not missing candidates (oracle→pick gap).

### Cause #1: full-frame penalty on line_grid

On close frames (`nag3_f000120`, `f000450`) `align_lg_img_panel` has **GT error ~200 px** but `rank_score` ~820 due to full-frame / axis-aligned penalties. Meanwhile `align_hsv_panel` / `align_white_grid` win ranking with **GT error 600–700 px**.

### Cause #2: white_grid in fast mode

`probe_outer_corner_candidates(fast=True)` added `align_white_grid` — on `f000450` picks 699 px instead of morph/line_grid ~198 px.

### exp1–exp6c (median pick down to **89.7 px**)

- Removed `white_grid` from fast probe; relaxed geometry penalties for trusted alignments; ensemble/cluster fixes; border vs line_grid heuristics.
- Reverted exp9–11 (regression to ~180 px). Kept **exp6c** state.

### YOLOv8n-Pose (Sim2Real) — 2026-05-22

| Method | All 73 labels | Val only (10 hold-out) |
|--------|---------------|------------------------|
| **CV** (`outer_corners`) | med **89.7** px | med **134** px |
| **YOLO zero-shot** (COCO) | 0/73 detections | — |
| **YOLO fine-tune** (63 train / 10 val, 60 ep.) | med **69.0** px | med **69.4** px |

Production path is now **YOLO-Pose + bias** (`corner_mode=yolo_pose`).

```bash
.venv_yolo/bin/python -m release.export_yolo_pose_dataset
.venv_yolo/bin/python -m release.eval_yolo_pose_corners --train --epochs 60
```

### Next steps (toward 50 px)

1. TL bias snap / line_grid anchor.
2. Better probe candidates (oracle med ~88 px still 2× target).
3. Stricter `grid_structure_score`.
