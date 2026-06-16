#!/usr/bin/env python3
"""Eksport dataset/panel_labels → YOLO-Pose (4 kropki: TL, TR, BR, BL)."""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pipeline_competition as pc
import yaml

from release.panel_labels import iter_labels, load_label

_ROOT = Path(__file__).resolve().parents[1]
KPT_NAMES = ['TL', 'TR', 'BR', 'BL']


def _yolo_pose_line(corners: np.ndarray, w: int, h: int) -> str:
    q = pc.order_points(corners.astype(np.float32))
    xs = q[:, 0] / float(w)
    ys = q[:, 1] / float(h)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    bw = max(1e-6, x_max - x_min)
    bh = max(1e-6, y_max - y_min)
    parts = ['0', f'{cx:.6f}', f'{cy:.6f}', f'{bw:.6f}', f'{bh:.6f}']
    for x, y in zip(xs, ys):
        parts.extend([f'{x:.6f}', f'{y:.6f}', '2'])
    return ' '.join(parts)


def export_dataset(
    out_root: Path,
    labels_dir: str | None,
    val_frac: float = 0.15,
    seed: int = 42,
    *,
    skip_propagated: bool = False,
) -> Path:
    paths = []
    for lp in iter_labels(labels_dir):
        if skip_propagated:
            import json
            notes = str(json.load(open(lp, encoding='utf-8')).get('notes', ''))
            if notes.startswith('propagated:'):
                continue
        paths.append(lp)
    random.Random(seed).shuffle(paths)
    n_val = max(1, int(len(paths) * val_frac))
    val_set = set(paths[:n_val])
    train_set = set(paths[n_val:])

    for split, subset in (('train', train_set), ('val', val_set)):
        img_dir = out_root / 'images' / split
        lbl_dir = out_root / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for lp in sorted(subset):
            data = load_label(lp)
            ip = data.get('image_path')
            if not ip:
                continue
            src = Path(ip)
            if not src.is_file():
                continue
            yc = data.get('yellow_corners')
            if yc is None:
                continue
            stem = src.stem
            dst_img = img_dir / f'{stem}.jpg'
            shutil.copy2(src, dst_img)
            w, h = data['image_size']
            line = _yolo_pose_line(np.asarray(yc), int(w), int(h))
            (lbl_dir / f'{stem}.txt').write_text(line + '\n', encoding='utf-8')

    yaml_path = out_root / 'droniada_pose.yaml'
    cfg = {
        'path': str(out_root.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'panel_grid'},
        'kpt_shape': [4, 3],
        'flip_idx': [],
    }
    yaml_path.write_text(yaml.dump(cfg, default_flow_style=False), encoding='utf-8')
    print(f'export: train={len(train_set)} val={len(val_set)} -> {out_root}')
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(_ROOT / 'dataset' / 'droniada_pose'))
    ap.add_argument('--labels-dir', default=None)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument(
        '--skip-propagated',
        action='store_true',
        help='Pomiń JSON z notes propagated: (tylko ręczne keyframe)',
    )
    args = ap.parse_args()
    export_dataset(
        Path(args.out),
        args.labels_dir,
        val_frac=args.val_frac,
        skip_propagated=args.skip_propagated,
    )


if __name__ == '__main__':
    main()
