from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import numpy as np

import pipeline_competition as pc
from module_panel.analyze import analyze_panel_image
from module_panel.reliability import DEFAULT_ALLOWED_ORBIT_STEPS, FRONTAL_ORBIT_BANK
from module_pose.api import intrinsics_from_pose_json, load_pose_gt_json


def _parse_orbit_steps(raw: Optional[str]) -> Optional[Set[int]]:
    if raw is None or raw.strip().lower() in ('', 'any', 'all'):
        return None
    if raw.strip().lower() in ('frontal', 'frontal_bank'):
        return FRONTAL_ORBIT_BANK
    return {int(x.strip()) for x in raw.split(',') if x.strip() != ''}


def multiset_cxy(gt_list: List[Dict], pred_list: List[Dict]) -> Tuple[int, int]:
    gt_cxy = [(g['color'], g['x'], g['y']) for g in gt_list]
    pred_cxy = [(p['color'], p['x'], p['y']) for p in pred_list]
    b_left = list(pred_cxy)
    c = 0
    for x in gt_cxy:
        if x in b_left:
            c += 1
            b_left.remove(x)
    return (c, len(gt_list))


def xy_given_color_oracle(gt_list: List[Dict], pred_list: List[Dict]) -> Tuple[int, int]:
    by_color: Dict[str, List[Dict]] = {}
    for p in pred_list:
        by_color.setdefault(p['color'], []).append(p)
    ok = 0
    for g in gt_list:
        arr = by_color.get(g['color'], [])
        hit = next((i for i, p in enumerate(arr) if p['x'] == g['x'] and p['y'] == g['y']), None)
        if hit is not None:
            ok += 1
            arr.pop(hit)
    return (ok, len(gt_list))


def _eval_loop(
    *,
    base: str,
    max_images: int,
    xy_mode: str,
    angle_source: str,
    calibration: Optional[str],
    camera_calib: Optional[str],
    reliable_only: bool,
    legacy_reliable: bool,
    allowed_orbit_steps: Optional[Set[int]],
    min_homography_inliers: int,
    max_reproj_px: float,
) -> Dict[str, Any]:
    img_dir = os.path.join(base, 'images')
    yd = os.path.join(base, 'labels_yolo')
    pd = os.path.join(base, 'labels_pose')
    rd = os.path.join(base, 'labels_raport')

    n = 0
    n_orbit_filtered = 0
    n_reliable = 0
    sum_gt_cards = 0
    sum_cxy = 0
    sum_oracle_xy = 0
    angle_ok = 0
    cat_ok = 0
    n_cat = 0
    by_panel_cat: Dict[str, Dict[str, int]] = {}

    for i in range(max_images):
        stem = f'img_{i}'
        ip = os.path.join(img_dir, f'{stem}.png')
        yp = os.path.join(yd, f'{stem}.txt')
        pj = os.path.join(pd, f'{stem}.json')
        rp = os.path.join(rd, f'{stem}.txt')
        if not os.path.isfile(ip) or not os.path.isfile(yp):
            continue
        gt = pc.load_gt(rp)
        if not gt:
            continue

        gt_data = load_pose_gt_json(pj) if os.path.isfile(pj) else None
        if allowed_orbit_steps is not None and gt_data:
            if allowed_orbit_steps is FRONTAL_ORBIT_BANK:
                from module_panel.reliability import pose_is_frontal_orbit_bank
                if not pose_is_frontal_orbit_bank(gt_data):
                    n_orbit_filtered += 1
                    continue
            else:
                step = (gt_data.get('camera') or {}).get('orbit_step_index')
                if step is not None and int(step) not in allowed_orbit_steps:
                    n_orbit_filtered += 1
                    continue

        img = cv2.imread(ip)
        det = pc.load_yolo(yp)
        if isinstance(gt_data, dict) and gt_data.get('intrinsics') is not None:
            k, dist = intrinsics_from_pose_json(gt_data)
        else:
            h, w = img.shape[:2]
            k = np.array([[1000.0, 0.0, w / 2.0], [0.0, 1000.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            dist = np.zeros((4, 1), dtype=np.float32)

        json_angle = None
        if angle_source == 'json' and gt_data:
            json_angle = int(
                gt_data.get('panel', {}).get(
                    'panel_skew_report_deg',
                    gt_data.get('panel', {}).get('business_angle_xy_deg', 0),
                )
            )

        res = analyze_panel_image(
            img,
            det,
            k=k,
            dist=dist,
            xy_mode=xy_mode,
            angle_source=angle_source,
            json_report_angle_deg=json_angle,
            angle_calibration_path=calibration,
            camera_calib_path=camera_calib,
            pose_json=gt_data if isinstance(gt_data, dict) else None,
            allowed_orbit_steps=allowed_orbit_steps,
            min_homography_inliers=min_homography_inliers,
            max_reproj_px=max_reproj_px,
        )

        reliable_flag = bool(res.meta.get('grid_xy_reliable_legacy' if legacy_reliable else 'grid_xy_reliable', False))
        if reliable_only and not reliable_flag:
            continue
        if reliable_flag:
            n_reliable += 1

        gt_angle = int(gt[0]['angle_deg'])
        cxy_hit, n_gt = multiset_cxy(gt, res.predictions)
        sum_cxy += cxy_hit
        sum_gt_cards += n_gt
        angle_ok += int(res.report_angle_deg == gt_angle)

        cat_gt = (gt_data or {}).get('panel', {}).get('panel_angle_category') if gt_data else None
        if cat_gt:
            n_cat += 1
            cat_ok += int(res.panel_angle_category == cat_gt)
            bucket = by_panel_cat.setdefault(str(cat_gt), {'images': 0, 'cxy_hits': 0, 'cards': 0})
            bucket['images'] += 1
            bucket['cxy_hits'] += cxy_hit
            bucket['cards'] += n_gt

        n += 1

    out: Dict[str, Any] = {
        'images': n,
        'reliable_images': n_reliable,
        'orbit_filtered_out': n_orbit_filtered,
        'card_cxy_acc_pct': round(100.0 * sum_cxy / max(1, sum_gt_cards), 2),
        'angle_acc_pct': round(100.0 * angle_ok / max(1, n), 2),
        'category_vs_pose_json_pct': round(100.0 * cat_ok / max(1, n_cat), 2) if n_cat else None,
        'xy_mode': xy_mode,
        'angle_source': angle_source,
        'reliable_only': bool(reliable_only),
        'legacy_reliable': bool(legacy_reliable),
        'allowed_orbit_steps': sorted(allowed_orbit_steps) if allowed_orbit_steps is not None else None,
        'min_homography_inliers': min_homography_inliers,
        'max_reproj_px': max_reproj_px,
        'total_gt_cards': sum_gt_cards,
        'total_cxy_hits': sum_cxy,
        'by_panel_category': {
            k: {
                'images': v['images'],
                'cxy_acc_pct': round(100.0 * v['cxy_hits'] / max(1, v['cards']), 2),
            }
            for k, v in sorted(by_panel_cat.items())
        },
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Ewaluacja modułu B (kolor + X,Y + kąt)')
    ap.add_argument('--dataset', default='dataset')
    ap.add_argument('--max-images', type=int, default=600)
    ap.add_argument('--angle-source', default='rmat_linear')
    ap.add_argument('--xy-mode', default='grid_geom_white', choices=['grid_geom', 'grid_geom_white', 'warp_grid', 'geom_grid', 'line_grid'])
    ap.add_argument('--camera-calib', default=None)
    ap.add_argument('--oracle-color-xy', action='store_true')
    ap.add_argument('--reliable-only', action='store_true', help='tylko grid_xy_reliable v2 (reproj + inliers + orbit)')
    ap.add_argument('--legacy-reliable', action='store_true', help='z --reliable-only: stary filtr reproj<=8 tylko')
    ap.add_argument('--orbit-steps', default='frontal', help='frontal (=0), 0,11, any')
    ap.add_argument('--min-homography-inliers', type=int, default=12)
    ap.add_argument('--max-reproj-px', type=float, default=8.0)
    ap.add_argument('--calibration', default=None)
    ap.add_argument('--out', default=None, help='zapis JSON (np. dataset/results/eval_frontal.json)')
    ap.add_argument('--compare-modes', default=None, help='lista trybów po przecinku, np. grid_geom_white,geom_grid')
    args = ap.parse_args()

    orbit_steps = _parse_orbit_steps(args.orbit_steps)
    if args.orbit_steps.strip().lower() == 'default':
        orbit_steps = DEFAULT_ALLOWED_ORBIT_STEPS

    base = args.dataset
    modes = [args.xy_mode]
    if args.compare_modes:
        modes = [m.strip() for m in args.compare_modes.split(',') if m.strip()]

    results: Dict[str, Any] = {
        'dataset': base,
        'orbit_steps': sorted(orbit_steps) if orbit_steps is not None else None,
        'filters': {
            'reliable_only': bool(args.reliable_only),
            'legacy_reliable': bool(args.legacy_reliable),
            'min_homography_inliers': args.min_homography_inliers,
            'max_reproj_px': args.max_reproj_px,
        },
        'modes': {},
    }

    for mode in modes:
        results['modes'][mode] = _eval_loop(
            base=base,
            max_images=args.max_images,
            xy_mode=mode,
            angle_source=args.angle_source,
            calibration=args.calibration,
            camera_calib=args.camera_calib,
            reliable_only=args.reliable_only,
            legacy_reliable=args.legacy_reliable,
            allowed_orbit_steps=orbit_steps,
            min_homography_inliers=args.min_homography_inliers,
            max_reproj_px=args.max_reproj_px,
        )

    if len(modes) == 1:
        payload = results['modes'][modes[0]]
        payload['orbit_steps'] = results['orbit_steps']
        payload['filters'] = results['filters']
    else:
        payload = results

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text)


if __name__ == '__main__':
    main()
