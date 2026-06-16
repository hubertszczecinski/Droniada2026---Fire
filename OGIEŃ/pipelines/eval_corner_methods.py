"""Benchmark corner detectors on synthetic dataset (rotation error vs GT)."""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import pipeline_competition as pc
from module_pose.api import (
    canonicalize_corners_by_white_anchor,
    default_intrinsics,
    detect_corners_black_panel,
    detect_corners_panel,
    gather_panel_quad_candidates,
    intrinsics_from_pose_json,
    score_panel_quad,
)
from module_pose.grid_corners import detect_corners_white_grid
from module_pose.pnp_panel import solve_panel_pose
from module_pose.refine_corners import refine_panel_corners_uniform_grid

DetectorFn = Callable[[np.ndarray], Optional[np.ndarray]]

def _angle_deg_from_rot_error(r_err: np.ndarray) -> float:
    tr = float(np.trace(r_err))
    c = float(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def _eval_method(
    name: str,
    fn: DetectorFn,
    img: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    r_gt: np.ndarray,
) -> Dict[str, Any]:
    corners = fn(img)
    if corners is None:
        return {'method': name, 'found': False}
    c_can, _ = canonicalize_corners_by_white_anchor(img, pc.order_points(corners))
    ok, rvec, _tvec, reproj = solve_panel_pose(c_can, k, dist, refine_lm=True)
    if not ok or rvec is None:
        return {'method': name, 'found': True, 'pnp_ok': False, 'reproj_px': float(reproj)}
    r_est, _ = cv2.Rodrigues(rvec)
    ang = _angle_deg_from_rot_error(r_est @ r_gt.T)
    h, w = img.shape[:2]
    area = float(cv2.contourArea(corners.astype(np.float32))) / float(h * w)
    return {
        'method': name,
        'found': True,
        'pnp_ok': True,
        'reproj_px': float(reproj),
        'angle_err_deg': float(ang),
        'area_ratio': float(area),
    }

def _detect_canny_only(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(gray, 40, 130)
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    img_area = float(image_bgr.shape[0] * image_bgr.shape[1])
    from module_pose.api import _quads_from_binary_mask
    quads = _quads_from_binary_mask(edge, img_area, min_area_frac=0.04, max_area_frac=0.52)
    if not quads:
        return None
    k, dist = default_intrinsics(image_bgr.shape)
    return max(quads, key=lambda q: score_panel_quad(image_bgr, q, k, dist))

def _detect_grid_refine(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    rough = detect_corners_white_grid(image_bgr)
    if rough is None:
        rough = detect_corners_black_panel(image_bgr)
    if rough is None:
        return None
    refined = refine_panel_corners_uniform_grid(image_bgr, rough)
    return refined if refined is not None else rough

def _detect_panel_scored(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    k, dist = default_intrinsics(image_bgr.shape)
    return detect_corners_panel(image_bgr, k, dist)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset')
    ap.add_argument('--max-images', type=int, default=200)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    methods: List[Tuple[str, DetectorFn]] = [
        ('panel_scored', _detect_panel_scored),
        ('white_grid', detect_corners_white_grid),
        ('grid_refine', _detect_grid_refine),
        ('black_panel', detect_corners_black_panel),
        ('thresh85', pc.detect_corners_img),
        ('canny', _detect_canny_only),
    ]
    agg: Dict[str, List[Dict[str, Any]]] = {n: [] for n, _ in methods}
    n_skip = 0
    for i in range(args.max_images):
        stem = f'img_{i}'
        img_p = os.path.join(args.dataset, 'images', f'{stem}.png')
        pose_p = os.path.join(args.dataset, 'labels_pose', f'{stem}.json')
        if not (os.path.isfile(img_p) and os.path.isfile(pose_p)):
            continue
        with open(pose_p, 'r', encoding='utf-8') as f:
            gt = json.load(f)
        gt_rot = (gt.get('model_to_camera_opencv') or {}).get('rotation_3x3')
        if gt_rot is None:
            n_skip += 1
            continue
        r_gt = np.asarray(gt_rot, dtype=np.float64).reshape(3, 3)
        img = cv2.imread(img_p)
        if img is None:
            continue
        k, dist = default_intrinsics(img.shape)
        if gt.get('intrinsics'):
            k, dist = intrinsics_from_pose_json(gt)
        for name, fn in methods:
            agg[name].append(_eval_method(name, fn, img, k, dist, r_gt))
    summary: Dict[str, Any] = {'dataset': args.dataset, 'max_images': args.max_images, 'skipped_no_gt': n_skip, 'methods': {}}
    for name, rows in agg.items():
        found = [r for r in rows if r.get('found')]
        pnp_ok = [r for r in found if r.get('pnp_ok')]
        angles = [r['angle_err_deg'] for r in pnp_ok]
        reprojs = [r['reproj_px'] for r in pnp_ok]
        areas = [r['area_ratio'] for r in pnp_ok]
        summary['methods'][name] = {
            'n': len(rows),
            'found_pct': 100.0 * len(found) / max(1, len(rows)),
            'pnp_ok_pct': 100.0 * len(pnp_ok) / max(1, len(rows)),
            'angle_err_deg_median': float(np.median(angles)) if angles else None,
            'angle_err_deg_mean': float(np.mean(angles)) if angles else None,
            'reproj_px_median': float(np.median(reprojs)) if reprojs else None,
            'area_ratio_median': float(np.median(areas)) if areas else None,
        }
    ranked = sorted(
        summary['methods'].items(),
        key=lambda kv: (
            kv[1]['pnp_ok_pct'],
            -(kv[1]['angle_err_deg_median'] or 999.0),
            kv[1]['reproj_px_median'] or 999.0,
        ),
        reverse=True,
    )
    summary['ranking'] = [n for n, _ in ranked]
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text)

if __name__ == '__main__':
    main()
