# Panel live — moduł A + B + migawki

Jeden ekran z wszystkimi danymi i automatyczną galerią klatek o niskim reproj.

## Uruchomienie (kamera)

```bash
chmod +x scripts/run_live_dashboard.sh
./scripts/run_live_dashboard.sh
```

Skrót (to samo — `--bench` włącza `--dashboard`):

```bash
./scripts/run_live_bench.sh
```

Tylko moduł A (bez B):

```bash
./scripts/run_live_module_a.sh
```

## Układ okna `droniada_dashboard`

| Strefa | Zawartość |
|--------|-----------|
| **Lewo** | Kamera: zielony trapez = moduł A, żółty + siatka = moduł B |
| **Środek** | Panel 10×10 z etykietami CXY |
| **Prawo** | Sidebar: odległość, stojak, roll/pitch/yaw, reliable, reproj, lista CXY, zatrzask |
| **Dół** | Miniaturki migawek (najlepsze reproj) |

## Migawki (automatyczne)

Zapis gdy jednocześnie:

- moduł B: `reliable=TAK`
- reproj B ≤ **15 px** (domyślnie; `--snapshot-max-reproj 12`)
- moduł A: OK (gdy włączony `--module-a`)
- **2** kolejne dobre klatki (`--snapshot-min-stable`)

Pliki w `dataset/live_dashboard/session_YYYYMMDD_HHMMSS/`:

| Plik | Opis |
|------|------|
| `snapshots/{klatka}_dashboard.jpg` | Pełny panel w momencie zapisu |
| `snapshots/{klatka}_snapshot.json` | Dict modułu A + B + meta |
| `index.html` | Galeria po zakończeniu sesji (otwórz w przeglądarce) |
| `index.json` | Ten sam spis maszynowo |

Po sesji:

```bash
open dataset/live_dashboard/session_*/index.html
```

Zatrzask CXY (ręczny `s` lub auto): dodatkowo `dataset/debug_cxy_latch/`.

## Klawisze

| Klawisz | Akcja |
|---------|--------|
| `q` | Wyjście (+ generuje `index.html`) |
| `s` | Ręczny zatrzask CXY |
| `r` | Reset zatrzasku |

## Nagranie testowe

```bash
./scripts/run_live_dashboard.sh --video dataset/my_capture/Droniada_nag3.mov --no-loop --max-frames 40
```

## Parametry

```bash
./scripts/run_live_dashboard.sh \
  --snapshot-max-reproj 12 \
  --snapshot-max 10 \
  --preview-width 1800
```

Inna kamera: `DRONIADA_CAMERA=0 ./scripts/run_live_dashboard.sh`
