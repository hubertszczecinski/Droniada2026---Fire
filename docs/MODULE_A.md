# Moduł A — pozycja drona + ustawienie panelu (2026-05-27)

Moduł A odpowiada za **geometrię podejścia**: rogi panelu, PnP, odległość, kąty drona względem panelu oraz **kategorię stojaka** (poziomy / poziomy‑przechylony / pionowy).

Moduł B (CXY z bliska) **nie** jest źródłem ustawienia panelu — dostaje kategorię z A lub z misji.

---

## Architektura live (real)

Moduł A ma **oddzielną ścieżkę PnP**, ale **ten sam detektor rogów** co moduł B:

```text
Kamera → YOLO-Pose (1× na klatkę przy bench A+B)
              ├→ Moduł B: warp + CXY + latch
              └→ Moduł A: kanonizacja rogów → PnP → d, roll/pitch/yaw + stojak
```

| Warstwa | Moduł A | Moduł B |
|---------|---------|---------|
| Rogi | `module_pose/yolo_pose_bridge.py` → `detect_corners_yolo_pose` | `release/live_corners.py` (`corner_mode=yolo_pose`) |
| Wagi / bias | `DRONIADA_YOLO_POSE_WEIGHTS`, `yolo_corner_bias.json` | to samo |
| Stabilizacja | EMA/hold z `live_corners` | to samo |
| Dalej | `pose_from_corners` → PnP | warp + HSV CXY |

Przy `./scripts/run_live_bench.sh` YOLO jest wołane **raz** — rogi trafiają do obu modułów (`shared_yolo` w `run_live_panel.py`).

API:

```python
from module_pose import pose_from_yolo_pose, pose_from_corners

# samodzielnie (np. run_live --mode pose)
res = pose_from_yolo_pose(bgr, k=k, dist=dist)

# współdzielone rogi z B
res = pose_from_corners(bgr, corners_px, k=k, dist=dist, base_method='yolo_pose')
```

`PoseConfig(corner_source='yolo_pose')` — domyślnie w live; fallback na CV gdy YOLO nie zwróci rogów.

---

## Wyjście integracyjne (pod shmsrc / lot)

```python
from release.pose_runtime import PoseConfig, PoseRuntime

rt = PoseRuntime(PoseConfig(corner_source='yolo_pose'))
po = rt.process_bgr(bgr, 'frame_001', k=k, dist=dist)
d = po.to_integration_dict(panel_id='A')
# ok, distance_camera_to_panel_center_m, roll/pitch/yaw_deg,
# report_angle_deg, panel_angle_category, panel_stand_label_pl,
# stand_confidence, reproj_mean_px, method
```

`report_angle_deg` (0 / 45 / 90) = **ustawienie panelu na stojaku**, nie kąt lotu drona. Kąt patrzenia → `roll_deg`, `pitch_deg`, `yaw_deg`.

---

## Klasyfikacja ustawienia (3 klasy)

| Kategoria | Kąt raportu | PL |
|-----------|-------------|-----|
| `horizontal` | 0° | poziomy |
| `45_deg` | 45° | poziomy-przechylony |
| `vertical` | 90° | pionowy |

Algorytm (`module_pose/panel_stand.py`):

1. **Kalibracja liniowa** na cechach PnP + geometrii quadu (`wh`, proporcje, area) — 3 klasy  
2. Wagi: `module_pose/data/panel_stand_linear.json` (generuj: `pipelines.calibrate_panel_stand`)  
3. Fallback: progi na macierzy rotacji (`rmat_theta`), gdy brak pliku kalibracji

---

## Kalibracja i testy (Blender)

```bash
# trening klasyfikatora + ewaluacja + testy regresji
chmod +x scripts/verify_module_a_blender.sh
./scripts/verify_module_a_blender.sh
```

Osobno:

```bash
python3 -m pipelines.calibrate_panel_stand --dataset dataset
python3 -m pipelines.eval_module_a_blender --dataset dataset --out dataset/results/eval_module_a_blender.json
python3 -m unittest tests.test_module_a_blender -v
```

### Metryki referencyjne (156 klatek Blender, frontal bank)

| Metryka | Wartość |
|---------|---------|
| Pose OK | 100% |
| Ustawienie panelu (3 klasy) | ~80% |
| Pionowy | ~94% |
| Błąd odległości (mediana) | ~1.3 m |

Słabszy punkt: **poziomy vs 45°** (~62–74%) — podobny wygląd w kadrze; poprawa: więcej realnych klatek z hali + ewentualnie YOLO rogów per stand.

---

## Pliki

| Plik | Rola |
|------|------|
| `module_pose/yolo_pose_bridge.py` | Most A→B: ten sam YOLO-Pose + tracker |
| `module_pose/api.py` | `pose_from_yolo_pose`, `pose_from_corners`, PnP |
| `module_pose/panel_stand.py` | Klasyfikator stojaka + `to_integration_dict` |
| `module_pose/types.py` | `PoseResult.to_integration_dict()` |
| `release/pose_runtime.py` | Runtime live |
| `pipelines/calibrate_panel_stand.py` | Trening na Blenderze |
| `pipelines/eval_module_a_blender.py` | Raport JSON |
| `tests/test_module_a_blender.py` | Regresja |
