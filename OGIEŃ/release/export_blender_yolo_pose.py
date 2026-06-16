#!/usr/bin/env python3
"""Eksport dataset/images + labels_pose (Blender GT 3D) → YOLO-Pose (4 kropki TL..BL)."""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pipeline_competition as pc
import yaml

from module_pose.api import intrinsics_from_pose_json, load_pose_gt_json
from module_pose.pnp_panel import PANEL_OBJECT_PTS
from release.export_yolo_pose_dataset import _yolo_pose_line

_ROOT = Path(__file__).resolve().parents[1]


def _pose_is_frontal_export(pose_data: dict, *, max_abs_azimuth_deg: float = 60.0) -> bool:
    """Filtr widoków z przodu (bez tyłu/boku) na podstawie metadanych Blendera."""
    cam = pose_data.get('camera') or {}
    if cam.get('accepted') is False:
        return False
    if str(cam.get('placement', '')).endswith('rejected'):
        return False
    az = cam.get('orbit_azimuth_offset_deg', cam.get('orbit_azimuth_deg'))
    if az is None:
        return True
    try:
        return abs(float(az)) <= float(max_abs_azimuth_deg)
    except (TypeError, ValueError):
        return True


def corners_px_from_pose_json(pose_data: dict) -> np.ndarray | None:
    m = pose_data.get('model_to_camera_opencv') or {}
    R = np.asarray(m.get('rotation_3x3'), dtype=np.float64)
    t = np.asarray(m.get('translation_m'), dtype=np.float64).reshape(3)
    if R.shape != (3, 3):
        return None
    k, _dist = intrinsics_from_pose_json(pose_data)
    pts = []
    for p in PANEL_OBJECT_PTS:
        pc_cam = R @ p.reshape(3) + t
        if pc_cam[2] <= 1e-06:
            return None
        u = k[0, 0] * pc_cam[0] / pc_cam[2] + k[0, 2]
        v = k[1, 1] * pc_cam[1] / pc_cam[2] + k[1, 2]
        pts.append([u, v])
    return pc.order_points(np.asarray(pts, dtype=np.float32))


def export_blender_dataset(
    dataset_root: Path,
    out_root: Path,
    *,
    val_frac: float = 0.15,
    seed: int = 42,
    max_abs_azimuth_deg: float = 60.0,
) -> Path:
    img_dir = dataset_root / 'images'
    pose_dir = dataset_root / 'labels_pose'
    stems = sorted(p.stem for p in img_dir.glob('img_*.png'))
    pairs: list[tuple[str, Path, Path]] = []
    skipped = 0
    for stem in stems:
        ip = img_dir / f'{stem}.png'
        pp = pose_dir / f'{stem}.json'
        if not pp.is_file():
            skipped += 1
            continue
        pose = load_pose_gt_json(str(pp))
        if pose is None:
            skipped += 1
            continue
        if not _pose_is_frontal_export(pose, max_abs_azimuth_deg=max_abs_azimuth_deg):
            skipped += 1
            continue
        corners = corners_px_from_pose_json(pose)
        if corners is None:
            skipped += 1
            continue
        pairs.append((stem, ip, pp))

    random.Random(seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_frac)) if pairs else 0
    val_set = {s for s, _, _ in pairs[:n_val]}
    train_set = {s for s, _, _ in pairs[n_val:]}

    for split, subset in (('train', train_set), ('val', val_set)):
        out_img = out_root / 'images' / split
        out_lbl = out_root / 'labels' / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        for stem, ip, pp in pairs:
            if stem not in subset:
                continue
            pose = load_pose_gt_json(str(pp))
            corners = corners_px_from_pose_json(pose)
            if corners is None:
                continue
            bgr = cv2.imread(str(ip))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            shutil.copy2(ip, out_img / f'{stem}.jpg')
            line = _yolo_pose_line(corners, w, h)
            (out_lbl / f'{stem}.txt').write_text(line + '\n', encoding='utf-8')

    yaml_path = out_root / 'droniada_pose_blender.yaml'
    cfg = {
        'path': str(out_root.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'panel_grid'},
        'kpt_shape': [4, 3],
        'flip_idx': [],
    }
    yaml_path.write_text(yaml.dump(cfg, default_flow_style=False), encoding='utf-8')
    print(
        f'blender export: total={len(pairs)} train={len(train_set)} val={len(val_set)} '
        f'skipped={skipped} -> {out_root}'
    )
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=str(_ROOT / 'dataset'))
    ap.add_argument('--out', default=str(_ROOT / 'dataset' / 'droniada_pose_blender'))
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--max-azimuth-deg', type=float, default=60.0)
    args = ap.parse_args()
    export_blender_dataset(
        Path(args.dataset),
        Path(args.out),
        val_frac=args.val_frac,
        max_abs_azimuth_deg=args.max_azimuth_deg,
    )


if __name__ == '__main__':
    main()
