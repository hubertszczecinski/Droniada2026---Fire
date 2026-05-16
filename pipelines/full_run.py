from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, Optional
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import cv2
import numpy as np
import pipeline_competition as pc
from module_panel.analyze import analyze_panel_image
from module_panel.report import predictions_to_report_lines
from module_pose.api import intrinsics_from_pose_json, load_pose_gt_json, pose_from_paths

def run_one(image_path: str, yolo_path: str, pose_json_path: Optional[str]=None, panel_id: str='A', *, competition: bool=False, angle_source: str='json', xy_mode: str='grid_geom_white', angle_calibration_path: Optional[str]=None) -> Dict[str, Any]:
    auto_pose = os.path.normpath(os.path.join(os.path.dirname(image_path), '..', 'labels_pose', os.path.basename(image_path).replace('.png', '.json')))
    if competition:
        effective_pose = pose_json_path if pose_json_path and os.path.isfile(pose_json_path) else None
    else:
        effective_pose = pose_json_path if pose_json_path and os.path.isfile(pose_json_path) else None
        if effective_pose is None and os.path.isfile(auto_pose):
            effective_pose = auto_pose
    pose = pose_from_paths(image_path, yolo_path=yolo_path, pose_gt_json_path=effective_pose)
    if not pose.ok or pose.corners_px is None:
        return {'ok': False, 'pose': pose.to_dict(), 'predictions': [], 'report_lines': [], 'panel': {}}
    img = cv2.imread(image_path)
    if img is None:
        return {'ok': False, 'pose': pose.to_dict(), 'predictions': [], 'report_lines': [], 'reason': 'no_image'}
    det = pc.load_yolo(yolo_path)
    h, w = img.shape[:2]
    gt_data = load_pose_gt_json(effective_pose) if effective_pose else None
    if isinstance(gt_data, dict) and gt_data.get('intrinsics') is not None:
        k, dist = intrinsics_from_pose_json(gt_data)
    else:
        k = np.array([[1000.0, 0.0, w / 2.0], [0.0, 1000.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        dist = np.zeros((4, 1), dtype=np.float32)
    src_angle = angle_source
    json_angle: Optional[int] = None
    if angle_source == 'json' and gt_data:
        json_angle = int(gt_data.get('panel', {}).get('panel_skew_report_deg', gt_data.get('panel', {}).get('business_angle_xy_deg', 0)))
    elif angle_source == 'json' and (not gt_data):
        src_angle = 'rmat_linear'
    if not competition and gt_data:
        panel_id = gt_data.get('panel', {}).get('id', panel_id)
    pan = analyze_panel_image(img, det, k=k, dist=dist, xy_mode=xy_mode, angle_source=src_angle, json_report_angle_deg=json_angle, angle_calibration_path=angle_calibration_path)
    lines = predictions_to_report_lines(panel_id, pan.report_angle_deg, pan.predictions)
    pose_d = pose.to_dict()
    reproj_a = float((pose.meta or {}).get('reproj_mean_px', float('nan')))
    reproj_b = float(pan.meta.get('reproj_mean_px', float('nan')))
    grid_ok = bool(pan.meta.get('grid_xy_reliable', False))
    return {'ok': True, 'pose': pose_d, 'predictions': pan.predictions, 'report_lines': lines, 'panel': {'id': panel_id, 'report_angle_deg': pan.report_angle_deg, 'panel_angle_category': pan.panel_angle_category, 'analyze_meta': pan.meta}, 'flight_hints': {'module_a_reproj_mean_px': reproj_a, 'module_b_reproj_mean_px': reproj_b, 'module_b_grid_xy_reliable': grid_ok, 'trust_module_b_xy': grid_ok}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--yolo', default=None)
    ap.add_argument('--pose-json', default=None)
    ap.add_argument('--competition', action='store_true')
    ap.add_argument('--panel-id', default='A')
    ap.add_argument('--angle-source', default=None, choices=['json', 'rmat_linear', 'rmat_theta', 'geom', 'pnp'])
    ap.add_argument('--xy-mode', default='grid_geom_white', choices=['grid_geom', 'grid_geom_white'])
    ap.add_argument('--angle-calibration', default=None)
    args = ap.parse_args()
    stem = os.path.splitext(os.path.basename(args.image))[0]
    base = os.path.dirname(args.image)
    yolo = args.yolo or os.path.normpath(os.path.join(base, '..', 'labels_yolo', f'{stem}.txt'))
    angle_source = args.angle_source
    if angle_source is None:
        if args.competition:
            angle_source = 'rmat_linear'
        else:
            auto_p = os.path.normpath(os.path.join(base, '..', 'labels_pose', f'{stem}.json'))
            angle_source = 'json' if os.path.isfile(auto_p) else 'rmat_linear'
    out = run_one(args.image, yolo, pose_json_path=args.pose_json, panel_id=args.panel_id, competition=args.competition, angle_source=angle_source, xy_mode=args.xy_mode, angle_calibration_path=args.angle_calibration)
    print(json.dumps(out, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
