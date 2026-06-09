"""Kalibracja klasyfikatora ustawienia panelu (moduł A) na dataset Blender."""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pipeline_competition as pc
from module_pose.api import intrinsics_from_pose_json, load_pose_gt_json, pose_from_paths
from module_pose.panel_stand import (
    save_stand_calibration,
    stand_feature_vector,
    train_stand_classifier,
)
from module_pose.pnp_panel import solve_panel_pose


def _collect_samples(dataset: str, max_images: int = 0):
    img_dir = os.path.join(dataset, 'images')
    yd = os.path.join(dataset, 'labels_yolo')
    pd = os.path.join(dataset, 'labels_pose')
    X_rows = []
    y_cats = []
    stems = []
    for name in sorted(os.listdir(pd)):
        if not name.endswith('.json'):
            continue
        stem = name.replace('.json', '')
        img_p = os.path.join(img_dir, f'{stem}.png')
        pj = os.path.join(pd, name)
        yp = os.path.join(yd, f'{stem}.txt')
        if not os.path.isfile(img_p):
            continue
        gt = load_pose_gt_json(pj)
        if not gt:
            continue
        panel = gt.get('panel') or {}
        cat = panel.get('panel_angle_category')
        if cat not in ('horizontal', '45_deg', 'vertical'):
            continue
        pose = pose_from_paths(
            img_p,
            yolo_path=yp if os.path.isfile(yp) else None,
            pose_gt_json_path=pj,
        )
        if not pose.ok or pose.corners_px is None or pose.rvec is None:
            continue
        rmat, _ = cv2.Rodrigues(pose.rvec)
        img = cv2.imread(img_p)
        h, w = img.shape[:2]
        reproj = float((pose.meta or {}).get('reproj_mean_px', 999.0))
        X_rows.append(stand_feature_vector(rmat, reproj, pose.corners_px, (h, w)))
        y_cats.append(str(cat))
        stems.append(stem)
        if max_images > 0 and len(stems) >= max_images:
            break
    if not X_rows:
        raise RuntimeError('brak próbek — sprawdź dataset/images + labels_pose')
    return np.stack(X_rows, axis=0), y_cats, stems


def main() -> None:
    ap = argparse.ArgumentParser(description='Kalibracja ustawienia panelu (moduł A) na Blenderze')
    ap.add_argument('--dataset', default='dataset')
    ap.add_argument('--max-images', type=int, default=0)
    ap.add_argument(
        '--out',
        default=os.path.join(_ROOT, 'module_pose', 'data', 'panel_stand_linear.json'),
    )
    args = ap.parse_args()
    X, y_cats, stems = _collect_samples(args.dataset, max_images=args.max_images)
    cal = train_stand_classifier(X, y_cats)
    cal['dataset'] = os.path.abspath(args.dataset)
    cal['sample_stems'] = stems[:20]
    out = save_stand_calibration(cal, args.out)
    pred = np.argmax(X @ np.asarray(cal['W'], dtype=np.float64).T, axis=1)
    from module_pose.panel_stand import STAND_CATEGORIES
    per_cat = {c: {'n': 0, 'ok': 0} for c in STAND_CATEGORIES}
    for yi, cat in zip(pred, y_cats):
        per_cat[cat]['n'] += 1
        if STAND_CATEGORIES[yi] == cat:
            per_cat[cat]['ok'] += 1
    summary = {
        'out': out,
        'n_samples': len(y_cats),
        'train_acc_linear': cal['train_acc_linear'],
        'per_category': {
            c: round(100.0 * v['ok'] / max(1, v['n']), 1) for c, v in per_cat.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
