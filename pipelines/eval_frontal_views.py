from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import pipeline_competition as pc
from module_panel.analyze import analyze_panel_image
from module_panel.angle_from_pose import CATEGORY_BY_ANGLE
from module_pose.api import intrinsics_from_pose_json, load_pose_gt_json, pose_from_paths

def angle_deg_from_rot_error(r_err: np.ndarray) -> float:
    tr = float(np.trace(r_err))
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def multiset_cxy(gt_list: List[Dict], pred_list: List[Dict]) -> Tuple[int, int]:
    gt_cxy = [(g['color'], g['x'], g['y']) for g in gt_list]
    pr_cxy = [(p['color'], p['x'], p['y']) for p in pred_list]
    bb = list(pr_cxy)
    c = 0
    for x in gt_cxy:
        if x in bb:
            c += 1
            bb.remove(x)
    return (c, len(gt_cxy))

def multiset_col(gt_list: List[Dict], pred_list: List[Dict]) -> Tuple[int, int]:
    gt = [g['color'] for g in gt_list]
    pr = [p['color'] for p in pred_list]
    bb = list(pr)
    c = 0
    for x in gt:
        if x in bb:
            c += 1
            bb.remove(x)
    return (c, len(gt))

def _metrics_for_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {'n': 0}
    return {'n': len(rows), 'pose_ok_pct': float(100.0 * np.mean([r['pose_ok'] for r in rows])), 'reproj_mean': float(np.nanmean([r['reproj'] for r in rows])), 'reproj_median': float(np.nanmedian([r['reproj'] for r in rows])), 'A_angle_err_mean_deg': float(np.nanmean([r['A_ang'] for r in rows])), 'A_angle_err_median_deg': float(np.nanmedian([r['A_ang'] for r in rows])), 'A_dist_err_mean_m': float(np.nanmean([r['A_dist_err'] for r in rows])), 'A_dist_err_median_m': float(np.nanmedian([r['A_dist_err'] for r in rows])), 'B_cxy_acc_mean_pct': float(np.mean([r['cxy_pct'] for r in rows])), 'B_color_acc_mean_pct': float(np.mean([r['col_pct'] for r in rows])), 'B_angle_acc_pct': float(100.0 * np.mean([r['ang_ok'] for r in rows])), 'B_category_acc_pct': float(100.0 * np.mean([r['cat_ok'] for r in rows])), 'grid_xy_reliable_pct': float(100.0 * np.mean([r['grid_rel'] for r in rows]))}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset')
    ap.add_argument('--max-images', type=int, default=600)
    ap.add_argument('--xy-mode', default='grid_geom_white', choices=['grid_geom', 'grid_geom_white'])
    ap.add_argument('--angle-source', default='rmat_linear')
    ap.add_argument('--compare-non-frontal', action='store_true')
    args = ap.parse_args()
    base = args.dataset
    img_dir = os.path.join(base, 'images')
    yolo_dir = os.path.join(base, 'labels_yolo')
    pose_dir = os.path.join(base, 'labels_pose')
    rep_dir = os.path.join(base, 'labels_raport')
    frontal_rows: List[Dict[str, Any]] = []
    other_rows: List[Dict[str, Any]] = []
    for i in range(args.max_images):
        stem = f'img_{i}'
        ip = os.path.join(img_dir, f'{stem}.png')
        yp = os.path.join(yolo_dir, f'{stem}.txt')
        pp = os.path.join(pose_dir, f'{stem}.json')
        rp = os.path.join(rep_dir, f'{stem}.txt')
        if not all((os.path.isfile(p) for p in (ip, yp, pp, rp))):
            continue
        gt_pose = load_pose_gt_json(pp)
        if not isinstance(gt_pose, dict):
            continue
        step = (gt_pose.get('camera') or {}).get('orbit_step_index')
        if step is None:
            continue
        if not args.compare_non_frontal and int(step) != 0:
            continue
        m2c = gt_pose.get('model_to_camera_opencv') or {}
        gt_rot = m2c.get('rotation_3x3')
        gt_t = m2c.get('translation_m') or m2c.get('translation_vec3')
        pnl = gt_pose.get('panel') or {}
        cat = pnl.get('panel_angle_category', '?')
        pose = pose_from_paths(ip, yolo_path=yp, pose_gt_json_path=pp)
        img = cv2.imread(ip)
        det = pc.load_yolo(yp)
        gt_cards = pc.load_gt(rp) or []
        if isinstance(gt_pose, dict) and gt_pose.get('intrinsics') is not None:
            k, dist = intrinsics_from_pose_json(gt_pose)
        else:
            h, w = img.shape[:2]
            k = np.array([[1000.0, 0.0, w / 2.0], [0.0, 1000.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            dist = np.zeros((4, 1), dtype=np.float32)
        pan = analyze_panel_image(img, det, k=k, dist=dist, xy_mode=args.xy_mode, angle_source=args.angle_source)
        a_ang = float('nan')
        if pose.ok and pose.rvec is not None and (gt_rot is not None):
            r_est, _ = cv2.Rodrigues(pose.rvec)
            r_gt = np.asarray(gt_rot, dtype=np.float64).reshape(3, 3)
            a_ang = angle_deg_from_rot_error(r_est @ r_gt.T)
        a_dist_err = float('nan')
        if pose.ok and pose.tvec is not None and (gt_t is not None):
            d_est = float(np.linalg.norm(np.asarray(pose.tvec, dtype=np.float64).reshape(3)))
            d_gt = float(np.linalg.norm(np.asarray(gt_t, dtype=np.float64).reshape(3)))
            a_dist_err = abs(d_est - d_gt)
        cxy_h, n_gt = multiset_cxy(gt_cards, pan.predictions)
        col_h, _n2 = multiset_col(gt_cards, pan.predictions)
        gt_angle = int(gt_cards[0]['angle_deg']) if gt_cards else 0
        cat_gt = pnl.get('panel_angle_category') or CATEGORY_BY_ANGLE.get(gt_angle, 'horizontal')
        row: Dict[str, Any] = {'image': stem, 'orbit_step_index': int(step), 'panel_category': cat, 'reproj': float(pan.meta.get('reproj_mean_px', float('nan'))), 'A_ang': a_ang, 'A_dist_err': a_dist_err, 'cxy_pct': 100.0 * cxy_h / n_gt if n_gt else 0.0, 'col_pct': 100.0 * col_h / n_gt if n_gt else 0.0, 'ang_ok': int(pan.report_angle_deg == gt_angle), 'cat_ok': int(pan.panel_angle_category == cat_gt), 'pose_ok': int(bool(pose.ok)), 'grid_rel': int(bool(pan.meta.get('grid_xy_reliable', False))), 'corner_src': str(pan.meta.get('corner_source', ''))}
        if step == 0:
            frontal_rows.append(row)
        else:
            other_rows.append(row)
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for r in frontal_rows:
        by_cat.setdefault(str(r['panel_category']), []).append(r)
    out: Dict[str, Any] = {'dataset': base, 'max_images': args.max_images, 'xy_mode': args.xy_mode, 'angle_source': args.angle_source, 'frontal': {'n': len(frontal_rows), 'overall': _metrics_for_rows(frontal_rows), 'by_panel_category': {k: _metrics_for_rows(v) for k, v in sorted(by_cat.items())}}}
    if args.compare_non_frontal:
        out['non_frontal'] = {'n': len(other_rows), 'overall': _metrics_for_rows(other_rows)}
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
