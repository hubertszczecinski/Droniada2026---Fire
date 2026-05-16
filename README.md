
- Zależności: `pip install -r requirements.txt` (NumPy + OpenCV headless).
- Opcjonalnie **Blender** z wbudowanym Pythonem do generowania syntetycznego datasetu.

## Dataset

Wszystkie skrypty zakładają względem `--dataset` (domyślnie `dataset/`) - wysłałem Piotr Tobie na Teams link do MEGA

```
dataset/
├── images/           # img_0.png, img_1.png, …
├── labels_yolo/      # img_N.txt — YOLO: class cx cy w h
├── labels_raport/    # img_N.txt — linie GT raportu (tekst parsowany przez pipeline_competition)
└── labels_pose/      # img_N.json — intrinsics, camera, panel, model_to_camera_opencv (GT z Blendera)
```

Katalog `reports/` może zawierać wyniki ewaluacji; `module_panel/data/` — np. kalibracja kąta (`angle_linear_rmat.json`).

## Moduł pozycji — `module_pose`

**Wejście (API):**

- `pose_from_image(image_bgr, yolo_det=None, k=None, dist=None, …)` — obraz BGR, opcjonalnie lista detekcji YOLO `(cls, cx, cy, w, h)`, macierz kamery i dystorsja.
- `pose_from_paths(image_path, yolo_path=None, pose_gt_json_path=None)` — ścieżki do plików; jeśli podasz JSON z `intrinsics`, użyje ich zamiast domyślnych.

**Wyjście:** obiekt `PoseResult`; do JSON użyj `result.to_dict()`:

- Zawsze: `ok`, `confidence`, `method`, `meta`.
- Przy sukcesie m.in.: `rvec`, `tvec`, `corners_px`, kąty (`euler_cam_deg`, `roll_deg` / `pitch_deg` / `yaw_deg`, `panel_orientation_vs_drone`), odległość i wektor do środka panelu w układzie kamery (`distance_camera_to_panel_center_m`, `panel_center_in_camera_m`).
- `meta` zawiera m.in. `reproj_mean_px`, informacje o wybranych narożnikach i kandydatach PnP.

Przy błędzie: `ok: false`, w `meta` np. `reason`: `no_corners`, `pnp_failed_all_candidates`, `no_image`.

## Moduł panelu — `module_panel`

**Wejście (główna funkcja):**

`analyze_panel_image(image_bgr, yolo_det, k, dist, xy_mode=..., angle_source=..., json_report_angle_deg=..., angle_calibration_path=...)`

- Obraz BGR, detekcje YOLO (ta sama lista co w `pipeline_competition.load_yolo`), kalibracja kamery.
- `xy_mode`: `grid_geom` lub `grid_geom_white` (korekta białego narożnika na zunifikowanym prostokącie).
- `angle_source`: skąd bierze się kąt raportu — m.in. `json` (z etykiety), `rmat_linear` / `rmat_theta` (z oszacowanej rotacji + opcjonalna kalibracja), `geom`, `pnp`.

**Wyjście:** `PanelAnalyzeResult`:

- `predictions` — lista `{'x': kolumna, 'y': wiersz, 'color': nazwa}` (komórki 1…10; kolor jak w `CLASS_TO_COLOR` w `pipeline_competition`).
- `warped_bgr`, `homography` — prostokąt panelu i homografia (numpy, nie serializowane w `full_run` JSON).
- `report_angle_deg`, `panel_angle_category`.
- `meta`: m.in. `xy_mode`, `angle_source`, `reproj_mean_px`, `pnp_ok`, `corner_source`, `grid_xy_reliable`; przy braku narożników `err`: `no_corners`.

Pełny przebieg na jednym obrazie — `pipelines.full_run`

Uruchomienie z katalogu głównego repozytorium:

```bash
python3 -m pipelines.full_run --image dataset/images/img_0.png
```

- Domyślnie szuka `labels_yolo/<stem>.txt`; przy braku własnego `--pose-json` próbuje `labels_pose/<stem>.json` (poza trybem `--competition`).
- `--competition` — kąt z `rmat_linear`, JSON pozy tylko jeśli podasz `--pose-json` (np. do intrinsics).

Przykładowe flagi:

```bash
python3 -m pipelines.full_run --image dataset/images/img_0.png --competition
python3 -m pipelines.full_run --image dataset/images/img_0.png --angle-source rmat_linear --xy-mode grid_geom_white
python3 -m pipelines.full_run --image dataset/images/img_0.png --angle-calibration module_panel/data/angle_linear_rmat.json
```

**JSON na stdout** (skrót pól):

- `ok` — `true`, gdy moduł pozycji zwrócił narożniki (`pose.ok`) i obraz się wczytał; wtedy wywołano też analizę panelu. Szczegóły błędów modułu B są w `panel.analyze_meta` (np. `err`).
- `pose` — słownik z `PoseResult.to_dict()`.
- `predictions` — lista komórek/kolorów z modułu panelu.
- `report_lines` — gotowe linie raportu (tekst).
- `panel` — `id`, `report_angle_deg`, `panel_angle_category`, `analyze_meta`.
- `flight_hints` — `module_a_reproj_mean_px`, `module_b_reproj_mean_px`, `module_b_grid_xy_reliable`, `trust_module_b_xy`.

## Inne skrypty (CLI)

| Moduł | Komenda | Opis |
|--------|---------|------|
| Konkurs starych pipeline’ów | `python3 pipeline_competition.py --dataset ./dataset` | Benchmark wielu wariantów z `pipeline_competition`; wyniki w `reports/` (JSON/CSV/JSONL). |
| Ewaluacja modułu B | `python3 -m pipelines.eval_module_b --dataset dataset` | Metryki kart / kolor / XY / kąt względem `labels_raport`. |
| Widoki przednie vs reszta | `python3 -m pipelines.eval_frontal_views --dataset dataset` | Metryki A (pose) i B (panel); opcja `--compare-non-frontal`. |
| Błąd rotacji vs GT | `python3 -m pipelines.eval_pose_angular --dataset dataset` | Porównanie `model_to_camera_opencv` z estymatą. |
| Kalibracja kąta raportu | `python3 -m pipelines.calibrate_report_angle --dataset dataset` | Zapis wagi do `module_panel/data/angle_linear_rmat.json` (ścieżka `--out`). |

## Generowanie datasetu (Blender)

Skrypt `generate_dataset_blender.py` jest pisany pod interpreter Blendera. Na początku pliku jest **`BASE_DIR`** — ustaw na ścieżkę do tego repozytorium na swojej maszynie (lub użyj zmiennych środowiskowych tam, gdzie są czytane, np. `DRONIADA_DATASET_SUBDIR`).

Przykład:

```bash
blender --background --python generate_dataset_blender.py
```

Szybki test (mniej scen, jeśli zaimplementowane w skrypcie): np. `DRONIADA_QUICK_TEST=1` (patrz stałe w pliku).

Wygenerowane pliki trafiają do `dataset/` (obrazy + `labels_yolo`, `labels_raport`, `labels_pose`).

## Import w kodzie

Katalog główny projektu musi być na `sys.path` (skrypty w `pipelines/` dodają go automatycznie). Z zewnątrz:

```python
import sys
sys.path.insert(0, "/ścieżka/do/Droniada")
from module_pose.api import pose_from_paths
from module_panel.analyze import analyze_panel_image
```
