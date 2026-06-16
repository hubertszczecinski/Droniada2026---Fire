"""Calibrate camera from chessboard images (cv2.calibrateCamera) and save NPZ."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from module_geom.camera import save_calibration

def main() -> None:
    ap = argparse.ArgumentParser(description='Kalibracja kamery szachownicą -> config/camera_calibration.npz')
    ap.add_argument('--images', default='calibration_chess/*.jpg', help='glob ścieżek do zdjęć szachownicy')
    ap.add_argument('--pattern-cols', type=int, default=9, help='liczba wewnętrznych narożników w poziomie')
    ap.add_argument('--pattern-rows', type=int, default=6, help='liczba wewnętrznych narożników w pionie')
    ap.add_argument('--square-mm', type=float, default=25.0)
    ap.add_argument('--out', default='config/camera_calibration.npz')
    args = ap.parse_args()
    pattern_size = (args.pattern_cols, args.pattern_rows)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= float(args.square_mm)
    obj_points = []
    img_points = []
    image_size = None
    paths = sorted(glob.glob(args.images))
    if not paths:
        print(json.dumps({'ok': False, 'reason': 'no_images', 'glob': args.images}))
        return
    skipped: list[str] = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            skipped.append(path)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ok:
            skipped.append(path)
            continue
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
        )
        obj_points.append(objp)
        img_points.append(corners)
        image_size = (gray.shape[1], gray.shape[0])
    if len(obj_points) < 3 or image_size is None:
        print(
            json.dumps(
                {
                    'ok': False,
                    'reason': 'not_enough_valid_frames',
                    'used': len(obj_points),
                    'skipped': len(skipped),
                    'hint': 'sprawdź --pattern-cols/--pattern-rows (wewnętrzne narożniki)',
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    rms, k, dist, _rvecs, _tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    save_calibration(
        args.out,
        k,
        dist,
        meta={
            'rms': float(rms),
            'frames': len(obj_points),
            'pattern': list(pattern_size),
            'square_mm': float(args.square_mm),
        },
    )
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    print(
        json.dumps(
            {
                'ok': True,
                'out': args.out,
                'rms': round(float(rms), 4),
                'frames': len(obj_points),
                'skipped': len(skipped),
                'image_size': list(image_size),
                'fx': round(fx, 2),
                'fy': round(fy, 2),
                'cx': round(cx, 2),
                'cy': round(cy, 2),
                'dist': [round(float(x), 5) for x in dist.reshape(-1)],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
