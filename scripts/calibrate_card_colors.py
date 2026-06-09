#!/usr/bin/env python3
"""Kalibracja kolorów kartek przed zawodami — 6 zdjęć → config/card_colors.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import pipeline_competition as pc

from release.card_color_profile import (
    _DEFAULT_PATH,
    _collect_calibration_files,
    _pick_calibration_paths,
    build_profile_from_folder,
    color_names,
    load_active_profile,
    sample_median_hsv,
)
from release.live_card_detect import _classify_patch_bgr


def _verify_folder(folder: Path) -> int:
    fails = 0
    prof = load_active_profile()
    calib = prof.meta.get('calibration_samples') or {}
    by_color = _collect_calibration_files(folder)
    for name in color_names():
        cls_id = int(pc.COLOR_TO_CLASS[name])
        listed = calib.get(name)
        if listed:
            paths = [folder / str(item['file']) for item in listed]
        else:
            paths = _pick_calibration_paths(by_color.get(name) or [])
        if not paths:
            print(f'  SKIP {name}: brak pliku')
            fails += 1
            continue
        for path in paths:
            if not path.is_file():
                print(f'  SKIP {name}: brak {path.name}')
                continue
            bgr = cv2.imread(str(path))
            if bgr is None:
                print(f'  FAIL {name}: nie wczytano {path}')
                fails += 1
                continue
            h, w = bgr.shape[:2]
            crop = bgr[h // 5 : 4 * h // 5, w // 5 : 4 * w // 5]
            meta = _classify_patch_bgr(crop, grid_row=5, grid_col=5)
            got = meta[0] if meta else None
            ok = got == cls_id
            label = pc.CLASS_TO_COLOR.get(got, '?') if got is not None else 'brak'
            mark = 'OK' if ok else 'FAIL'
            print(f'  {mark} {name}: oczekiwano {name}, jest {label}  ({path.name})')
            if not ok:
                fails += 1
    return fails


def _capture_from_camera(camera: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(int(camera))
    if not cap.isOpened():
        raise SystemExit(f'Nie otwarto kamery {camera}')
    print('Ustaw kartkę na ciemnym tle, wypełnij środek kadru.')
    print('Enter = zapis, q = przerwij\n')
    for name in color_names():
        while True:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit('Brak klatki z kamery')
            preview = frame.copy()
            h, w = preview.shape[:2]
            cv2.rectangle(preview, (w // 5, h // 5), (4 * w // 5, 4 * h // 5), (0, 255, 0), 2)
            cv2.putText(
                preview, f'Kartka: {name}  [Enter=zapis q=stop]',
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
            )
            cv2.imshow('droniada_calib', preview)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit('Przerwano')
            if key in (13, 10):
                crop = frame[h // 5 : 4 * h // 5, w // 5 : 4 * w // 5]
                path = out_dir / f'{name}.jpg'
                cv2.imwrite(str(path), crop)
                med = sample_median_hsv(crop)
                print(f'  zapisano {path.name}  HSV≈{med}')
                break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Kalibracja kolorów kartek (zawody): 6 zdjęć → config/card_colors.json',
    )
    ap.add_argument(
        '--folder', '-f',
        type=Path,
        help='Katalog ze zdjęciami: CZERWONA.jpg, ZIELONA.jpg, …',
    )
    ap.add_argument(
        '--camera', '-c',
        type=int,
        metavar='N',
        help='Zamiast folderu: zrób 6 zdjęć z kamery na miejscu',
    )
    ap.add_argument(
        '--capture-dir',
        type=Path,
        default=_ROOT / 'config' / 'competition_cards',
        help='Gdzie zapisać zdjęcia z --camera (domyślnie config/competition_cards)',
    )
    ap.add_argument(
        '--out', '-o',
        type=Path,
        default=_DEFAULT_PATH,
        help=f'Plik profilu JSON (domyślnie {_DEFAULT_PATH.relative_to(_ROOT)})',
    )
    ap.add_argument('--notes', default='', help='Notatka (np. "Zawody Kraków 2026")')
    ap.add_argument(
        '--verify',
        action='store_true',
        help='Sprawdź klasyfikację na zdjęciach (wymaga --folder lub istniejącego profilu)',
    )
    args = ap.parse_args()

    if args.camera is not None:
        _capture_from_camera(args.camera, args.capture_dir)
        args.folder = args.capture_dir

    if args.verify:
        folder = args.folder or args.capture_dir
        if not folder.is_dir():
            raise SystemExit(f'Brak folderu do weryfikacji: {folder}')
        import os
        if args.out.is_file():
            os.environ['DRONIADA_CARD_COLORS'] = str(args.out)
        load_active_profile(force=True)
        print(f'Weryfikacja ({folder}):')
        fails = _verify_folder(folder)
        raise SystemExit(1 if fails else 0)

    if args.folder is None:
        ap.print_help()
        print('\nPrzykład przed zawodami:')
        print('  1. Zrób 6 zdjęć kartek w tym samym świetle co hangar')
        print('  2. ./scripts/calibrate_card_colors.sh config/competition_cards')
        print('  3. Restart panelu (profil ładuje się z config/card_colors.json)')
        raise SystemExit(2)

    profile, samples = build_profile_from_folder(args.folder)
    profile.save(
        args.out,
        source=str(args.folder.resolve()),
        notes=args.notes,
        samples=samples,
        calibration_samples=profile.meta.get('calibration_samples'),
    )
    src = profile.meta.get('source_files') or {}
    calib = profile.meta.get('calibration_samples') or {}
    print(f'Zapisano profil: {args.out}')
    print('Pliki źródłowe (główne):')
    for name in color_names():
        print(f'  {name:14s}  ← {src.get(name, "?")}')
    print('Próbki kalibracji (centroidy):')
    for name in color_names():
        parts = calib.get(name) or []
        if len(parts) > 1:
            detail = ', '.join(f'{p["file"]}(H={p["h"]:.0f})' for p in parts)
            print(f'  {name:14s}  {detail}')
        else:
            s = samples[name]
            print(f'  {name:14s}  H={s["h"]:.0f}  S={s["s"]:.0f}  V={s["v"]:.0f}')
    print('Mediana HSV (zawody / najjaśniejsza próbka):')
    for name in color_names():
        s = samples[name]
        print(f'  {name:14s}  H={s["h"]:.0f}  S={s["s"]:.0f}  V={s["v"]:.0f}')
    print()
    print('Na zawodach:')
    print('  • profil jest w config/card_colors.json — restart panelu wystarczy')
    print('  • lub: export DRONIADA_CARD_COLORS=config/card_colors.json')
    print('  • test: python scripts/calibrate_card_colors.py --verify -f', args.folder)


if __name__ == '__main__':
    main()
