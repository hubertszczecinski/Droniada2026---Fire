#!/usr/bin/env python3
"""Porównaj detekcję z ręcznymi etykietami — średni błąd rogów w px."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from release.panel_labels import corner_errors_px, iter_labels, load_label
from release.transform import apply_rotate


def _detect_yellow(bgr: np.ndarray) -> tuple:
    """Ten sam pipeline co live (align_hybrid), bez kalibracji pikselowej."""
    from release.live_corners import acquire_corners_for_live
    from module_pose.api import default_intrinsics

    corners, lbl, meta, rows = acquire_corners_for_live(
        bgr, *default_intrinsics(bgr.shape[:2]), corner_mode='align_hybrid',
    )
    if corners is None:
        return None, lbl, rows if rows else meta.get('candidate_rows', [])
    return np.asarray(corners, dtype=np.float32), str(lbl), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels-dir', default=None)
    ap.add_argument('--rotate', type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument('--save-overlay', action='store_true')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    paths = iter_labels(args.labels_dir)
    if not paths:
        print('Brak plików JSON w dataset/panel_labels — najpierw annotate_corners.py')
        return

    errs: List[float] = []
    per_corner: List[List[float]] = [[], [], [], []]
    report: List[Dict[str, Any]] = []

    for lp in paths:
        try:
            data = load_label(lp)
        except (KeyError, json.JSONDecodeError, OSError):
            print(f'Pominięto (zły JSON): {os.path.basename(lp)}')
            continue
        img_path = str(data.get('image_path', ''))
        if not img_path:
            continue
        if not os.path.isfile(img_path):
            print(f'Pominięto (brak pliku): {img_path}')
            continue
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        if args.rotate or int(data.get('rotate_deg', 0)):
            bgr = apply_rotate(bgr, args.rotate or int(data.get('rotate_deg', 0)))

        gt = data['yellow_corners']
        pred, lbl, _rows = _detect_yellow(bgr)
        if pred is None:
            print(f'{os.path.basename(img_path)}: BRAK detekcji')
            continue
        mean_e, each = corner_errors_px(pred, gt)
        errs.append(mean_e)
        for i, e in enumerate(each):
            per_corner[i].append(e)
        report.append({
            'image': os.path.basename(img_path),
            'mean_px': mean_e,
            'per_corner_px': each,
            'label': lbl,
        })
        print(f'{os.path.basename(img_path)}: mean={mean_e:.1f}px  TL={each[0]:.0f} TR={each[1]:.0f} BR={each[2]:.0f} BL={each[3]:.0f}  ({lbl})')

        if args.save_overlay:
            from release.panel_labels import draw_label_overlay

            out_dir = args.out_dir or os.path.join(
                os.path.dirname(lp), 'eval_overlay',
            )
            os.makedirs(out_dir, exist_ok=True)
            vis = draw_label_overlay(bgr, yellow=gt, pred=pred, blue_roi=data.get('blue_roi'))
            cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), vis)

    if errs:
        names = ('TL', 'TR', 'BR', 'BL')
        print('---')
        print(f'N={len(errs)}  średni błąd={np.mean(errs):.1f}px  mediana={np.median(errs):.1f}px')
        for i, name in enumerate(names):
            if per_corner[i]:
                print(f'  {name}: średnia={np.mean(per_corner[i]):.1f}px')


if __name__ == '__main__':
    main()
