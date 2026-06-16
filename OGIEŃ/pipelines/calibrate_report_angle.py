from __future__ import annotations
import argparse
import json
import math
import os
import sys
import cv2
import numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import pipeline_competition as pc
from module_pose.api import intrinsics_from_pose_json, load_pose_gt_json, pose_from_paths
from module_pose.pnp_panel import solve_panel_pose
ANGLE_CLASSES = np.array([0, 45, 90], dtype=np.int32)

def _feat(rmat: np.ndarray, reproj: float) -> np.ndarray:
    return np.concatenate([rmat.reshape(-1)[:9], [reproj / 25.0], [1.0]]).astype(np.float64)

def _fit_linear(X: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    W = []
    for c in range(3):
        t = (y_idx == c).astype(np.float64)
        beta, *_ = np.linalg.lstsq(X, t, rcond=None)
        W.append(beta)
    return np.stack(W, axis=0)

def _eval_theta(Rs: list, y_idx: np.ndarray, t0: float, t1: float) -> float:
    correct = 0
    for rmat, yi in zip(Rs, y_idx):
        c = float(np.clip(abs(rmat[2, 2]), 0.0, 1.0))
        th = math.degrees(math.acos(c))
        if th < t0:
            p = 0
        elif th < t1:
            p = 45
        else:
            p = 90
        if p == int(ANGLE_CLASSES[yi]):
            correct += 1
    return correct / max(1, len(y_idx))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset')
    ap.add_argument('--out', default=os.path.join(_ROOT, 'module_panel', 'data', 'angle_linear_rmat.json'))
    args = ap.parse_args()
    base = args.dataset
    img_dir = os.path.join(base, 'images')
    yd = os.path.join(base, 'labels_yolo')
    pd = os.path.join(base, 'labels_pose')
    rd = os.path.join(base, 'labels_raport')
    X_rows = []
    y_list = []
    Rs = []
    for name in sorted(os.listdir(yd)):
        if not name.endswith('.txt'):
            continue
        stem = name.replace('.txt', '')
        img_p = os.path.join(img_dir, f'{stem}.png')
        pj = os.path.join(pd, f'{stem}.json')
        rp = os.path.join(rd, f'{stem}.txt')
        if not os.path.isfile(img_p) or not os.path.isfile(pj):
            continue
        gt = pc.load_gt(rp)
        if not gt:
            continue
        gt_angle = int(gt[0]['angle_deg'])
        if gt_angle not in (0, 45, 90):
            continue
        y_i = int(np.where(ANGLE_CLASSES == gt_angle)[0][0])
        data = load_pose_gt_json(pj)
        if not data or 'intrinsics' not in data:
            continue
        k, dist = intrinsics_from_pose_json(data)
        det = pc.load_yolo(os.path.join(yd, name))
        pose = pose_from_paths(img_p, yolo_path=os.path.join(yd, name), pose_gt_json_path=pj)
        if not pose.ok or pose.corners_px is None:
            continue
        corners = pose.corners_px
        ok, rvec, _t, reproj = solve_panel_pose(corners, k, dist)
        if not ok or rvec is None:
            continue
        rmat, _ = cv2.Rodrigues(rvec)
        X_rows.append(_feat(rmat, reproj))
        y_list.append(y_i)
        Rs.append(rmat)
    X = np.stack(X_rows, axis=0)
    y_idx = np.array(y_list, dtype=np.int32)
    W = _fit_linear(X, y_idx)
    pred = np.argmax(X @ W.T, axis=1)
    acc_lin = float(np.mean(pred == y_idx))
    best_t0, best_t1, best_acc = (25.0, 55.0, 0.0)
    for t0 in np.linspace(15, 45, 16):
        for t1 in np.linspace(35, 75, 17):
            if t1 <= t0 + 3:
                continue
            a = _eval_theta(Rs, y_idx, t0, t1)
            if a > best_acc:
                best_acc, best_t0, best_t1 = (a, t0, t1)
    out = {'version': 1, 'W': W.tolist(), 'feature_dim': int(W.shape[1]), 'theta_t01': float(best_t0), 'theta_t12': float(best_t1), 'train_acc_linear': round(acc_lin, 5), 'train_acc_theta': round(best_acc, 5), 'n_samples': int(len(y_idx))}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
