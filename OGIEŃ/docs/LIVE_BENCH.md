# Live panel - module A + B + snapshots

Single window with all telemetry and automatic gallery of low-reprojection frames.

## Run (camera)

```bash
chmod +x scripts/run_live_dashboard.sh
./scripts/run_live_dashboard.sh
```

Shortcut (`--bench` enables `--dashboard`):

```bash
./scripts/run_live_bench.sh
```

Module A only (no B):

```bash
./scripts/run_live_module_a.sh
```

## Window layout `droniada_dashboard`

| Area | Content |
|------|---------|
| **Left** | Camera: green trapezoid = module A, yellow + grid = module B |
| **Centre** | 10×10 panel with CXY labels |
| **Right** | Sidebar: distance, stand, roll/pitch/yaw, reliable, reproj, CXY list, latch |
| **Bottom** | Snapshot thumbnails (best reproj) |

## Snapshots (automatic)

Saved when all of:

- module B: `reliable=YES`
- reproj B ≤ **15 px** (default; `--snapshot-max-reproj 12`)
- module A: OK (when `--module-a` enabled)
- **2** consecutive good frames (`--snapshot-min-stable`)

Files in `dataset/live_dashboard/session_YYYYMMDD_HHMMSS/`:

| File | Description |
|------|-------------|
| `snapshots/{frame}_dashboard.jpg` | Full dashboard at capture time |
| `snapshots/{frame}_snapshot.json` | Module A + B dict + meta |
| `index.html` | Gallery after session (open in browser) |
| `index.json` | Machine-readable index |

After session:

```bash
open dataset/live_dashboard/session_*/index.html
```

CXY latch (manual `s` or auto): also `dataset/debug_cxy_latch/`.

## Keys

| Key | Action |
|-----|--------|
| `q` | Quit (+ generates `index.html`) |
| `s` | Manual CXY latch |
| `r` | Reset latch |

## Test recording

```bash
./scripts/run_live_dashboard.sh --video dataset/my_capture/Droniada_nag3.mov --no-loop --max-frames 40
```

## Parameters

```bash
./scripts/run_live_dashboard.sh \
  --snapshot-max-reproj 12 \
  --snapshot-max 10 \
  --preview-width 1800
```

Other camera: `DRONIADA_CAMERA=0 ./scripts/run_live_dashboard.sh`
