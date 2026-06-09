"""Ewaluacja modułu A na dataset Blender (ustawienie panelu + odległość + orientacja drona)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from module_pose.api import load_pose_gt_json, pose_from_paths


def _gt_distance_m(pose_json: Dict[str, Any]) -> float:
    t = (pose_json.get('model_to_camera_opencv') or {}).get('translation_m')
    if not t:
        return float('nan')
    return float(np.linalg.norm(np.asarray(t, dtype=np.float64)))


def eval_module_a_blender(dataset: str, max_images: int = 0) -> Dict[str, Any]:
    img_dir = os.path.join(dataset, 'images')
    pd = os.path.join(dataset, 'labels_pose')
    yd = os.path.join(dataset, 'labels_yolo')
    rows: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(pd)):
        if not name.endswith('.json'):
            continue
        stem = name.replace('.json', '')
        if max_images > 0 and len(rows) >= max_images:
            break
        img_p = os.path.join(img_dir, f'{stem}.png')
        pj = os.path.join(pd, name)
        yp = os.path.join(yd, f'{stem}.txt')
        if not os.path.isfile(img_p):
            continue
        gt = load_pose_gt_json(pj)
        if not gt:
            continue
        panel = gt.get('panel') or {}
        cat_gt = str(panel.get('panel_angle_category', ''))
        if cat_gt not in ('horizontal', '45_deg', 'vertical'):
            continue
        gt_dist = _gt_distance_m(gt)
        pose = pose_from_paths(
            img_p,
            yolo_path=yp if os.path.isfile(yp) else None,
            pose_gt_json_path=pj,
        )
        d = pose.to_dict()
        pred_dist = float(d.get('distance_camera_to_panel_center_m', float('nan')))
        rows.append({
            'stem': stem,
            'pose_ok': bool(pose.ok),
            'category_gt': cat_gt,
            'category_pred': str(pose.panel_angle_category),
            'category_ok': bool(pose.ok and pose.panel_angle_category == cat_gt),
            'angle_gt': int(panel.get('panel_skew_report_deg', panel.get('business_angle_xy_deg', 0))),
            'angle_pred': int(pose.report_angle_deg),
            'angle_ok': bool(pose.ok and int(pose.report_angle_deg) == int(panel.get('panel_skew_report_deg', 0))),
            'distance_gt_m': gt_dist,
            'distance_pred_m': pred_dist,
            'distance_err_m': abs(pred_dist - gt_dist) if pose.ok and np.isfinite(gt_dist) else float('nan'),
            'reproj_px': float((pose.meta or {}).get('reproj_mean_px', float('nan'))),
            'stand_confidence': float(pose.stand_confidence),
            'stand_source': str((pose.meta or {}).get('stand_source', '')),
            'integration': pose.to_integration_dict(panel_id=str(panel.get('id', ''))),
        })

    ok_rows = [r for r in rows if r['pose_ok']]
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for r in ok_rows:
        by_cat.setdefault(r['category_gt'], []).append(r)

    def _acc(sub: List[Dict[str, Any]], key: str) -> float:
        if not sub:
            return 0.0
        return float(np.mean([bool(x[key]) for x in sub]))

    dist_errs = [r['distance_err_m'] for r in ok_rows if np.isfinite(r['distance_err_m'])]
    return {
        'dataset': os.path.abspath(dataset),
        'n_total': len(rows),
        'n_pose_ok': len(ok_rows),
        'pose_ok_pct': round(100.0 * len(ok_rows) / max(1, len(rows)), 2),
        'stand_category_acc_pct': round(100.0 * _acc(ok_rows, 'category_ok'), 2),
        'stand_angle_acc_pct': round(100.0 * _acc(ok_rows, 'angle_ok'), 2),
        'distance_err_median_m': round(float(np.median(dist_errs)), 3) if dist_errs else None,
        'distance_err_mean_m': round(float(np.mean(dist_errs)), 3) if dist_errs else None,
        'by_category': {
            cat: {
                'n': len(sub),
                'category_acc_pct': round(100.0 * _acc(sub, 'category_ok'), 2),
                'distance_err_median_m': round(
                    float(np.median([x['distance_err_m'] for x in sub if np.isfinite(x['distance_err_m'])])),
                    3,
                ) if sub else None,
            }
            for cat, sub in sorted(by_cat.items())
        },
        'samples': rows[:12],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Ewaluacja modułu A na Blenderze')
    ap.add_argument('--dataset', default='dataset')
    ap.add_argument('--max-images', type=int, default=0)
    ap.add_argument('--out', default=None, help='np. dataset/results/eval_module_a_blender.json')
    args = ap.parse_args()
    report = eval_module_a_blender(args.dataset, max_images=args.max_images)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if args.out:
        out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w', encoding='utf-8') as fh:
            fh.write(text)


if __name__ == '__main__':
    main()
