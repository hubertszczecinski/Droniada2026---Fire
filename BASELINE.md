# Baseline modułu B

**Aktualna migawka (live YOLO + CXY latch):** [`docs/MODULE_B_SNAPSHOT.md`](docs/MODULE_B_SNAPSHOT.md) (2026-05-20)

**Format raportu online (regulamin 2026):** [`docs/COMPETITION_REPORT.md`](docs/COMPETITION_REPORT.md)

**Jetson (Docker, pełny panel :8088):** [`docs/JETSON_DOCKER.md`](docs/JETSON_DOCKER.md)

Weryfikacja:

```bash
chmod +x scripts/verify_module_b_snapshot.sh
./scripts/verify_module_b_snapshot.sh
```

---

## Szybki start (produkcja)

```bash
export DRONIADA_YOLO_POSE_WEIGHTS="$(pwd)/runs/pose/droniada_real_finetune/weights/best.pt"
.venv_yolo/bin/python -m release.run_live_panel \
  --video dataset/my_capture/Droniada_nag3.mov --rotate 180 \
  --corner-mode yolo_pose --cxy-latch
```

Po zatrzaśnięciu: okno `droniada_cxy_zatrzask`, pliki w `dataset/debug_cxy_latch/`.

---

## Starszy baseline CV (archiwum)

Opis ścieżki `align_hybrid` + żółty trapez + `line_grid` bez YOLO latch — nadal w kodzie (`--corner-mode roi_hybrid` itd.), ale **nie** jest już główną ścieżką z tej migawki.

Szczegóły historyczne: commit `2929f7f` — „Ustal baseline modułu B: line_grid v3 i live test na kamerze.”
