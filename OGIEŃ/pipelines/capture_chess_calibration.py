"""Zbieranie zdjęć szachownicy z kamery (ten sam obrót/indeks co live)."""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from release.camera_source import CameraConfig, CameraSource
from release.transform import apply_rotate


def main() -> None:
    ap = argparse.ArgumentParser(description='Zdjęcia szachownicy do kalibracji kamery')
    ap.add_argument('--camera', type=int, default=1, help='indeks kamery (Mac: często 0 lub 1)')
    ap.add_argument('--width', type=int, default=0)
    ap.add_argument('--height', type=int, default=0)
    ap.add_argument('--rotate', type=int, default=180, choices=[0, 90, 180, 270])
    ap.add_argument('--pattern-cols', type=int, default=9, help='wewnętrzne narożniki poziomo')
    ap.add_argument('--pattern-rows', type=int, default=6, help='wewnętrzne narożniki pionowo')
    ap.add_argument('--out-dir', default='calibration_chess')
    ap.add_argument('--min-valid', type=int, default=15, help='sugerowana liczba kadrów z wykrytą szachownicą')
    args = ap.parse_args()

    pattern = (args.pattern_cols, args.pattern_rows)
    os.makedirs(args.out_dir, exist_ok=True)
    existing = [f for f in os.listdir(args.out_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    idx = len(existing)

    print('Kalibracja szachownicą')
    print(f'  wzorzec (wewn. narożniki): {pattern[0]} x {pattern[1]}')
    print(f'  katalog: {args.out_dir}/')
    print(f'  kamera={args.camera}  obrót={args.rotate}°')
    print()
    print('Sterowanie:')
    print('  spacja — zapisz kadr (tylko gdy szachownica wykryta)')
    print('  s      — zapisz kadr bez wymogu wykrycia (awaryjnie)')
    print('  q      — koniec')
    print()
    print(f'Cel: co najmniej ~{args.min_valid} różnych ujęć (pochylenia, odległości, miejsca w kadrze).')
    print('Zamknij QuickTime — kamera może być tylko w jednym programie.')
    print()

    saved_valid = 0
    with CameraSource(CameraConfig(device_index=args.camera, width=args.width, height=args.height)) as cam:
        while True:
            ok, bgr, _fid = cam.read()
            if not ok or bgr is None:
                continue
            if args.rotate:
                bgr = apply_rotate(bgr, args.rotate)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern, None)
            vis = bgr.copy()
            if found:
                cv2.drawChessboardCorners(vis, pattern, corners, found)
                status = 'SZACHOWNICA OK — spacja = zapisz'
                color = (0, 220, 0)
            else:
                status = 'brak szachownicy — ustaw kartę w kadrze'
                color = (0, 140, 255)
            cv2.putText(
                vis, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f'zapisane OK: {saved_valid}  |  plikow w folderze: {idx}',
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow('droniada_chess_calib', vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key in (ord(' '), ord('s')):
                force = key == ord('s')
                if found or force:
                    path = os.path.join(args.out_dir, f'chess_{idx:04d}.jpg')
                    cv2.imwrite(path, bgr)
                    idx += 1
                    if found:
                        saved_valid += 1
                    tag = 'OK' if found else 'bez detekcji'
                    print(f'  zapisano {path} ({tag}), lacznie OK={saved_valid}')
                else:
                    print('  pominięto — najpierw ustaw szachownicę w kadrze (albo s = wymuś)')

    cv2.destroyAllWindows()
    print()
    print(f'Gotowe: {idx} plików w {args.out_dir}/ ({saved_valid} z wykrytą szachownicą)')
    if saved_valid < args.min_valid:
        print(f'Uwaga: mało kadrów ({saved_valid} < {args.min_valid}) — dodaj więcej ujęć przed kalibracją.')
    else:
        print('Możesz uruchomić:')
        print(
            f'  .venv_yolo/bin/python -m pipelines.calibrate_camera '
            f"--images '{args.out_dir}/*.jpg' "
            f'--pattern-cols {args.pattern_cols} --pattern-rows {args.pattern_rows}',
        )


if __name__ == '__main__':
    main()
