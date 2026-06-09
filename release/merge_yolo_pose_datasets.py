#!/usr/bin/env python3
"""Scala dwa zbiory YOLO-Pose (np. Blender + real) — val tylko z real."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]


def merge_datasets(
    blender_root: Path,
    real_root: Path,
    out_root: Path,
    *,
    prefix_blender: str = 'bl_',
    prefix_real: str = 'rl_',
) -> Path:
    for split in ('train', 'val'):
        (out_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    def _copy_tree(src: Path, split: str, prefix: str) -> int:
        n = 0
        for img in sorted((src / 'images' / split).glob('*')):
            if img.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
                continue
            stem = f'{prefix}{img.stem}'
            lbl = src / 'labels' / split / f'{img.stem}.txt'
            if not lbl.is_file():
                continue
            ext = '.jpg' if img.suffix.lower() != '.png' else img.suffix
            shutil.copy2(img, out_root / 'images' / split / f'{stem}{ext}')
            shutil.copy2(lbl, out_root / 'labels' / split / f'{stem}.txt')
            n += 1
        return n

    n_bl_tr = _copy_tree(blender_root, 'train', prefix_blender)
    n_rl_tr = _copy_tree(real_root, 'train', prefix_real)
    n_rl_va = _copy_tree(real_root, 'val', prefix_real)

    yaml_path = out_root / 'droniada_pose_mixed.yaml'
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
        f'merge -> {out_root}\n'
        f'  train: blender={n_bl_tr} real={n_rl_tr} total={n_bl_tr + n_rl_tr}\n'
        f'  val:   real only={n_rl_va}'
    )
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--blender', default=str(_ROOT / 'dataset' / 'droniada_pose_blender'))
    ap.add_argument('--real', default=str(_ROOT / 'dataset' / 'droniada_pose'))
    ap.add_argument('--out', default=str(_ROOT / 'dataset' / 'droniada_pose_mixed'))
    args = ap.parse_args()
    merge_datasets(Path(args.blender), Path(args.real), Path(args.out))


if __name__ == '__main__':
    main()
