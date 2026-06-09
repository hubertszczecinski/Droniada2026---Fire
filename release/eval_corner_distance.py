#!/usr/bin/env python3
"""Błąd TL/TR/BR/BL: pomarańczowy (pred) vs żółty (GT) na etykietach."""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import pipeline_competition as pc

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from release.panel_labels import CORNER_NAMES, corner_errors_px, iter_labels, load_label
from release.live_corners import acquire_corners_for_live
from module_pose.api import default_intrinsics


def _eval_all(labels_dir: str | None, corner_mode: str = 'align_hybrid') -> None:
    per = {n: [] for n in CORNER_NAMES}
    bias = {n: [] for n in CORNER_NAMES}
    fail = 0

    for lp in iter_labels(labels_dir):
        data = load_label(lp)
        img = cv2.imread(str(data['image_path']))
        if img is None:
            continue
        pred, lbl, _, _ = acquire_corners_for_live(
            img, *default_intrinsics(img.shape[:2]), corner_mode=corner_mode,
        )
        if pred is None:
            fail += 1
            print(f'{os.path.basename(data["image_path"])}: BRAK  ({lbl})')
            continue
        gt = pc.order_points(data['yellow_corners'])
        pred = pc.order_points(pred.astype(np.float32))
        mean_e, each = corner_errors_px(pred, gt)
        for i, name in enumerate(CORNER_NAMES):
            per[name].append(each[i])
            bias[name].append(pred[i] - gt[i])
        parts = ' '.join(f'{n}={each[i]:.0f}' for i, n in enumerate(CORNER_NAMES))
        print(f'{os.path.basename(data["image_path"])}: mean={mean_e:.1f}px  {parts}  ({lbl})')

    n_ok = len(per['TL'])
    print('---')
    print(f'Klatek OK={n_ok}  BRAK={fail}')
    if not n_ok:
        return
    all_err = [e for name in CORNER_NAMES for e in per[name]]
    print(f'Średnia 4 rogi: mean={np.mean(all_err):.1f}px  mediana={np.median(all_err):.1f}px')
    for name in CORNER_NAMES:
        a = np.array(per[name], dtype=np.float32)
        b = np.array(bias[name], dtype=np.float32)
        print(
            f'  {name}: err śr={a.mean():.1f} med={np.median(a):.1f} p90={np.percentile(a, 90):.1f}'
            f'  | bias pred−GT: dx={b[:, 0].mean():+.1f} dy={b[:, 1].mean():+.1f}'
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels-dir', default=None)
    ap.add_argument('--corner-mode', default='align_hybrid',
                    choices=('align_hybrid', 'outer_corners'))
    args = ap.parse_args()
    _eval_all(args.labels_dir, corner_mode=args.corner_mode)


if __name__ == '__main__':
    main()
