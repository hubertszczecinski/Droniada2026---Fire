# Tuning zewnętrznych rogów (moduł B)

Cel: **mediana błędu < 50 px** na 73 etykietach (`dataset/panel_labels/*_r180.json`).

## Pętla pracy

```bash
python3 -m release.eval_outer_tune --note "opis zmiany"
python3 -m release.eval_corner_distance
```

Log JSONL: `release/CORNERS_EXPERIMENTS.jsonl`

## Wnioski (sesja 2026-05-20)

### Baseline (przed poprawkami)

| Metryka | Wartość |
|---------|---------|
| PICK mediana | **101.5 px** |
| ORACLE mediana | **96.7 px** |
| GAP mediana | 0 px (max **501 px** na 1 klatce) |

**Główny problem nie brak kandydatów, tylko zły wybór** (gap oracle→pick).

### Przyczyna #1: kara „pełny kadr” na line_grid

Na klatkach z bliska (`nag3_f000120`, `f000450`) `align_lg_img_panel` ma **błąd GT ~200 px**, ale `rank_score` ~820 przez:

- `w_frac ≈ 1.0` → kara +400
- `axis_aligned` → +220
- brak skew → +110

Tymczasem `align_hsv_panel` / `align_white_grid` mają niższy reproj PnP (~130–150) i wygrywają ranking (**błąd GT 600–700 px**).

### Przyczyna #2: white_grid w trybie fast

`probe_outer_corner_candidates(fast=True)` dodawał `align_white_grid` — na `f000450` wybór 699 px zamiast morph/line_grid ~198 px.

### Hipotezy do dalszych iteracji

1. **grid_structure_score = 1.0** na wielu złych kandydatach — słaba dyskryminacja; rozważyć ostrzejszą metrykę lub wagę konsensusu.
2. Oracle mediana 97 px >> cel 50 px — nawet idealny wybór z probe wymaga **lepszych kandydatów** (line_grid refine, border_scan+LSD fuse).
3. Bias TL: dx−62, dy−77 — systematyczne przesunięcie line_grid; snap do przecięć linii LSD.

### Zmiany exp1 (mediana pick 101.5 px)

- Usunięto `white_grid` z fast probe.
- `_filter_rows_for_pick`: odrzucenie weak align gdy jest trusted (lg/border_scan).
- `_score_candidate_row`: relax kar geom dla trusted + cap reproj; +320 white_grid, +280 hsv.
- Efekt: `f000450` 699→199 px; `f000120` 623→203 px; gap max 501→119 px.

### Zmiany exp3 (mediana pick 101.5 px, średnia 126.6 px)

- `border_scan` przed ensemble gdy split≥300 px lub (split≥220 i reproj lg ≥ reproj border).
- Nie wybieraj border gdy najlepszy `lg_*` ma reproj ≪ border (np. `lg_warp2`).
- `f000345`/`f000045`: ~303→~189 px (border zamiast złego lg).

### Zmiany exp5b–exp6c (mediana pick **89.7 px**, gap max **53 px**)

- **Ensemble:** nie wymuszaj 3 kandydatów spoza klastra 72 px (koniec „border+lg” 241 px).
- **warp2** tier 4, bonus −155; promuj najlepszy `lg_*` w ordered.
- **`_pick_alt_when_lg_border_far`:** split≥300 px → wybór border/morph (reproj 22–145), bez grid_outer/dark_blob; wyjątek gdy lg reproj <22 (PnP mylący).
- Efekt: `f000015` 295→190, `nag2_f000360` 362→128 (morph), mediana 101.5→**89.7**.

### exp9–11 (cofnięte — regresja)

- `snap_border`, `border_tight`, `replace fullframe lg` — mediana skakała do ~180 px.
- `refine=True` — bez zmiany mediany (~89.7 px).
- Zostaje stan **exp6c**.

### Test YOLOv8n-Pose (Sim2Real / fine-tuning) — 2026-05-22

Środowisko: `.venv_yolo` + `release/export_yolo_pose_dataset.py`, `release/eval_yolo_pose_corners.py`.

| Metoda | Wszystkie 73 etykiety | Tylko val (10 klatek hold-out) |
|--------|----------------------|--------------------------------|
| **CV** (`outer_corners`) | med **89.7** px | med **134** px |
| **YOLO zero-shot** (COCO `yolov8n-pose.pt`) | 0/73 detekcji | — |
| **YOLO fine-tune** (63 train / 10 val, 60 ep.) | med **69.0** px | med **69.4** px |

Wnioski:
- Pre-trained COCO **nie wykrywa** panelu — fine-tuning na 4 kropkach jest konieczny.
- Na tym samym zbiorze YOLO **bije CV o ~20 px** mediany (in-sample); na val też ~69 vs ~134 px CV.
- **Ryzyko:** 73 zdjęcia to za mało na pewną generalizację (nowe nagranie, mockup, trawa); val=10 jest małe.
- **Mockup 100–200 zdjęć** + Blender — sensowna ścieżka Sim2Real zgodna z propozycją innego modelu.
- Live: YOLO ~kilka–kilkanaście ms/klatkę na CPU (M3); CV ~sekundy (probe wielu metod).

```bash
.venv_yolo/bin/python -m release.export_yolo_pose_dataset
.venv_yolo/bin/python -m release.eval_yolo_pose_corners --train --epochs 60
.venv_yolo/bin/python -m release.eval_yolo_pose_corners --weights runs/pose/droniada_panel_corners/weights/best.pt --mode both
```

### Następne kroki (do celu 50 px)

1. Bias TL (dx−128): snap lewej krawędzi / line_grid anchor.
2. Lepsze kandydaty w probe (oracle med ~88 px — nadal 2× za wysoko).
3. Ostrzejsza `grid_structure_score` (nie 1.0 dla złych quadów).
