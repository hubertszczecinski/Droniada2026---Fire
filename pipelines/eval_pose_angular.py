from __future__ import annotations
import argparse
import json
import os
import sys
from typing import List
import cv2
import numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from module_pose.api import pose_from_paths

def angle_deg_from_rot_error(r_err: np.ndarray) -> float:
    tr = float(np.trace(r_err))
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset')
    ap.add_argument('--max-images', type=int, default=600)
    args = ap.parse_args()
    ds = args.dataset
    img_dir = os.path.join(ds, 'images')
    yolo_dir = os.path.join(ds, 'labels_yolo')
    pose_dir = os.path.join(ds, 'labels_pose')
    errs: List[float] = []
    missing_gt = 0
    pose_fail = 0
    for i in range(args.max_images):
        stem = f'img_{i}'
        img_p = os.path.join(img_dir, f'{stem}.png')
        yolo_p = os.path.join(yolo_dir, f'{stem}.txt')
        pose_p = os.path.join(pose_dir, f'{stem}.json')
        if not (os.path.isfile(img_p) and os.path.isfile(yolo_p) and os.path.isfile(pose_p)):
            continue
        with open(pose_p, 'r', encoding='utf-8') as f:
            gt = json.load(f)
        gt_rot = (gt.get('model_to_camera_opencv') or {}).get('rotation_3x3')
        if gt_rot is None:
            missing_gt += 1
            continue
        r_gt = np.asarray(gt_rot, dtype=np.float64).reshape(3, 3)
        res = pose_from_paths(img_p, yolo_path=yolo_p, pose_gt_json_path=pose_p)
        if not res.ok or res.rvec is None:
            pose_fail += 1
            continue
        r_est, _ = cv2.Rodrigues(res.rvec)
        r_err = r_est @ r_gt.T
        errs.append(angle_deg_from_rot_error(r_err))
    if not errs:
        print(json.dumps({'ok': False, 'reason': 'no_gt_rotation', 'missing_gt_rotation_count': missing_gt}, ensure_ascii=False, indent=2))
        return
    arr = np.asarray(errs, dtype=np.float64)
    out = {'ok': True, 'samples': int(arr.size), 'pose_fail_count': int(pose_fail), 'missing_gt_rotation_count': int(missing_gt), 'angle_error_deg_mean': float(arr.mean()), 'angle_error_deg_median': float(np.median(arr)), 'angle_error_deg_p90': float(np.percentile(arr, 90)), 'angle_error_deg_p95': float(np.percentile(arr, 95))}
    print(json.dumps(out, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
