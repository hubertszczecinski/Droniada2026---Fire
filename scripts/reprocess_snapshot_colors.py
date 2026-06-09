#!/usr/bin/env python3
"""Przelicz kolory na zapisanych migawkach (GT + grid_lines z JSON)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2

from release.snapshot_cell_color import (
    detect_gt_cells_on_warped,
    gt_cells_from_config,
    load_test_mov_gt,
)
from release.card_color_profile import load_active_profile
from module_panel.competition_report import predictions_to_report_lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('session_dir', type=Path)
    ap.add_argument('--gt-config', type=Path, default=_ROOT / 'config' / 'test_mov_gt.json')
    ap.add_argument('--profile', type=Path, default=_ROOT / 'config' / 'card_colors.json')
    ap.add_argument('--video', type=Path, default=None)
    ap.add_argument('--rotate', type=int, default=None)
    args = ap.parse_args()

    cfg = load_test_mov_gt(args.gt_config)
    if not cfg:
        raise SystemExit(f'Brak GT: {args.gt_config}')
    video = args.video or (_ROOT / str(cfg.get('video', 'dataset/my_capture/Test.mov')))
    rotate = int(args.rotate if args.rotate is not None else cfg.get('rotate_deg', 180))
    cells = gt_cells_from_config(cfg)

    os.environ['DRONIADA_CARD_COLORS'] = str(args.profile.resolve())
    os.environ.setdefault(
        'DRONIADA_YOLO_POSE_WEIGHTS',
        str(_ROOT / 'runs/pose/droniada_real_finetune/weights/best.pt'),
    )
    load_active_profile(force=True)

    from release.yolo_pose_live import detect_corners_yolo_pose
    from module_panel.warp import warp_panel_rect
    from release.snapshot_cell_color import frame_index_from_id

    snap_dir = args.session_dir / 'snapshots'
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f'Brak wideo: {video}')

    n_ok = 0
    for json_path in sorted(snap_dir.glob('*_snapshot.json')):
        data = json.loads(json_path.read_text(encoding='utf-8'))
        pan = data.get('pan_meta') or {}
        glx, gly = pan.get('grid_lines_x'), pan.get('grid_lines_y')
        if not glx or not gly:
            continue
        fi = frame_index_from_id(str(data.get('frame_id', '')))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fi))
        ok, frame = cap.read()
        if not ok:
            continue
        if rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        corners, _, _ = detect_corners_yolo_pose(frame)
        if corners is None:
            continue
        warped, _ = warp_panel_rect(frame, corners)
        raw = detect_gt_cells_on_warped(warped, glx, gly, cells)
        preds = [{'x': int(p['x']), 'y': int(p['y']), 'color': str(p['color'])} for p in raw]
        panel_id = str((data.get('module_b') or {}).get('panel_id', 'A'))
        angle = int((data.get('module_b') or {}).get('report_angle_deg', 0))
        lines = predictions_to_report_lines(panel_id, angle, preds)
        mb = dict(data.get('module_b') or {})
        mb['live_predictions'] = preds
        mb['live_report_lines'] = lines
        data['module_b'] = mb
        data['color_detect'] = {'mode': 'on_snapshot_gt', 'n_cards': len(preds)}
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        labels = [f"{c.get('user_label')}={next((p['color'] for p in preds if p['y']==c['row'] and p['x']==c['col']), '—')}" for c in cells]
        print(f"{data['frame_id']}: {', '.join(labels)}")
        n_ok += 1
    cap.release()
    print(f'Zaktualizowano {n_ok} migawek.')


if __name__ == '__main__':
    main()
