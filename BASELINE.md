# Baseline modułu B — `line_grid` v3

Stan zapisany jako punkt odniesienia po ewaluacji na datasecie (frontal, `reliable-only` v2).

## Metryki (orbit step 0)

| Tryb | Klatki reliable | CXY kart |
|------|-----------------|----------|
| `grid_geom_white` | 18 | 81.9% |
| **`line_grid` v3** | **22** | **93.2%** |

Kategorie (reliable): 45° **100%**, poziome **93%**, pionowe **89%**.

## Pliki

- `module_geom/line_grid.py` — rogi (img_panel / black_panel / warp) + wybór XY
- `module_panel/analyze.py` — integracja `line_grid`
- `dataset/results/eval_frontal_line_grid_v3.json`
- `dataset/debug_line_grid_v3/`

## Live

```bash
python3 -m release.run_live_panel --preview --camera 1 --rotate 180 --xy-mode line_grid
```

Opcjonalnie: `--require-reliable`, `--log-file live_panel.log`, `--save-dir live_captures`
