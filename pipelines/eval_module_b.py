from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import cv2
import numpy as np
import pipeline_competition as pc
from module_panel.analyze import analyze_panel_image
from module_pose.api import intrinsics_from_pose_json, load_pose_gt_json

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset')
    ap.add_argument('--max-images', type=int, default=600)
    ap.add_argument('--angle-source', default='rmat_linear')
    ap.add_argument('--xy-mode', default='grid_geom_white', choices=['grid_geom', 'grid_geom_white'])
    ap.add_argument('--oracle-color-xy', action='store_true')
    ap.add_argument('--calibration', default=None)
    args = ap.parse_args()
    base = args.dataset
    img_dir = os.path.join(base, 'images')
    yd = os.path.join(base, 'labels_yolo')
    pd = os.path.join(base, 'labels_pose')
    rd = os.path.join(base, 'labels_raport')
    n = 0
    sum_gt_cards = 0
    sum_cxy = 0
    sum_oracle_xy = 0
    angle_ok = 0
    cat_ok = 0
    n_cat = 0
    for i in range(args.max_images):
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
        img = cv2.imread(ip)
        det = pc.load_yolo(yp)
        gt_data = load_pose_gt_json(pj) if os.path.isfile(pj) else None
        if isinstance(gt_data, dict) and gt_data.get('intrinsics') is not None:
            k, dist = intrinsics_from_pose_json(gt_data)
        else:
            h, w = img.shape[:2]
            k = np.array([[1000.0, 0.0, w / 2.0], [0.0, 1000.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            dist = np.zeros((4, 1), dtype=np.float32)
        json_angle = None
        if args.angle_source == 'json' and gt_data:
            json_angle = int(gt_data.get('panel', {}).get('panel_skew_report_deg', gt_data.get('panel', {}).get('business_angle_xy_deg', 0)))
        res = analyze_panel_image(img, det, k=k, dist=dist, xy_mode=args.xy_mode, angle_source=args.angle_source, json_report_angle_deg=json_angle, angle_calibration_path=args.calibration)
        gt_angle = int(gt[0]['angle_deg'])
        cxy_hit, n_gt = multiset_cxy(gt, res.predictions)
        sum_cxy += cxy_hit
        sum_gt_cards += n_gt
        if args.oracle_color_xy:
            ox, _ = xy_given_color_oracle(gt, res.predictions)
            sum_oracle_xy += ox
        angle_ok += int(res.report_angle_deg == gt_angle)
        cat_gt = (gt_data or {}).get('panel', {}).get('panel_angle_category') if gt_data else None
        if cat_gt:
            n_cat += 1
            cat_ok += int(res.panel_angle_category == cat_gt)
        n += 1
    out = {'images': n, 'card_cxy_acc_pct': round(100.0 * sum_cxy / max(1, sum_gt_cards), 2), 'angle_acc_pct': round(100.0 * angle_ok / max(1, n), 2), 'category_vs_pose_json_pct': round(100.0 * cat_ok / max(1, n_cat), 2) if n_cat else None, 'xy_mode': args.xy_mode, 'angle_source': args.angle_source, 'total_gt_cards': sum_gt_cards, 'total_cxy_hits': sum_cxy}
    if args.oracle_color_xy:
        out['oracle_color_xy_acc_pct'] = round(100.0 * sum_oracle_xy / max(1, sum_gt_cards), 2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
