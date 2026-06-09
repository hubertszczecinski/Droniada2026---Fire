#!/usr/bin/env python3
"""Jedna klatka (kamera lub plik) → folder live_debug do analizy."""
from __future__ import annotations

import argparse
import os
import sys

import cv2

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from release.run_live_panel import (
    _compose_preview,
    _default_intrinsics,
    _draw_panel_overlay,
    _draw_status_block,
    _draw_warped_inset,
    _log,
)
from module_panel.analyze import analyze_panel_image
from module_panel.report import predictions_to_report_lines
from module_panel.warp import warp_panel_rect
from release.live_card_detect import detect_cards_live
from release.live_corners import probe_all_corner_candidates
from release.live_debug import draw_candidates_board, new_session_dir, save_live_frame_bundle
from release.transform import apply_rotate


def process_frame(bgr, fid: str, session_dir: str, *, xy_mode: str, panel_id: str, no_color: bool) -> None:
    h, w = bgr.shape[:2]
    k, dist = _default_intrinsics(w, h)
    cand_rows = probe_all_corner_candidates(bgr, k, dist)
    if cand_rows:
        cand_rows[0]['chosen'] = True
        corners_px = cand_rows[0]['corners']
        corner_src = str(cand_rows[0]['label'])
        corner_meta = {k: v for k, v in cand_rows[0].items() if k != 'corners'}
    else:
        corners_px, corner_src, corner_meta = None, 'none', {}

    det = []
    warped_dets = []
    if corners_px is not None:
        warped, h_mat = warp_panel_rect(bgr, corners_px)
        if not no_color:
            det, warped_dets = detect_cards_live(bgr, corners_px, h_mat, warped)
        pan = analyze_panel_image(
            bgr, det, k=k, dist=dist, xy_mode=xy_mode, corners_px=corners_px, corner_source=corner_src,
        )
        warped = pan.warped_bgr
        preds = list(pan.predictions)
        pan_meta = pan.meta
        reliable = bool(pan.meta.get('grid_xy_reliable', False))
        reproj = float(pan.meta.get('reproj_mean_px', 999))
        xy_back = str(pan.meta.get('xy_backend_selected', '?'))
        angle = int(pan.report_angle_deg)
        category = str(pan.panel_angle_category)
    else:
        warped = bgr.copy()
        preds = []
        pan_meta = {'err': 'no_corners'}
        reliable, reproj, xy_back, angle, category = False, 999.0, '-', 0, 'horizontal'

    vis = bgr.copy()
    _draw_panel_overlay(vis, corners_px, det, preds, ok_corners=corners_px is not None)
    _draw_status_block(vis, reliable=reliable, corner_source=corner_src, xy_backend=xy_back,
                       reproj=reproj, n_cards=len(preds), angle=angle, category=category)
    thumb = _draw_warped_inset(warped, preds, warped_dets)
    board = draw_candidates_board(bgr, cand_rows, winner_label=corner_src)
    record = {
        'frame_id': fid,
        'corner_source': corner_src,
        'reproj_mean_px': reproj,
        'grid_structure_score': corner_meta.get('grid_structure_score'),
        'panel_interior_score': corner_meta.get('panel_interior_score'),
        'grid_xy_reliable': reliable,
        'predictions': preds,
        'candidates': [{k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in c.items()} for c in cand_rows],
    }
    save_live_frame_bundle(session_dir, fid, raw_bgr=bgr, overlay_bgr=_compose_preview(vis, thumb),
                          warped_bgr=warped, candidates_board=board, record=record)
    _log(None, f'zapisano {session_dir}/{fid} corner={corner_src} reproj={reproj:.1f}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default=None, help='test na pliku zamiast kamery')
    ap.add_argument('--camera', type=int, default=1)
    ap.add_argument('--rotate', type=int, default=180)
    ap.add_argument('--debug-dir', default='live_debug')
    ap.add_argument('--xy-mode', default='line_grid')
    ap.add_argument('--no-color-detect', action='store_true')
    args = ap.parse_args()
    root = args.debug_dir if os.path.isabs(args.debug_dir) else os.path.join(_ROOT, args.debug_dir)
    session = new_session_dir(root)
    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f'nie wczytano {args.image}')
        if args.rotate:
            bgr = apply_rotate(bgr, args.rotate)
        process_frame(bgr, 'file_0001', session, xy_mode=args.xy_mode, panel_id='A',
                      no_color=args.no_color_detect)
    else:
        from release.camera_source import CameraConfig, CameraSource
        with CameraSource(CameraConfig(device_index=args.camera)) as cam:
            ok, bgr, fid = cam.read()
            if not ok or bgr is None:
                raise SystemExit('capture fail')
            if args.rotate:
                bgr = apply_rotate(bgr, args.rotate)
            process_frame(bgr, fid, session, xy_mode=args.xy_mode, panel_id='A', no_color=args.no_color_detect)
    print(f'Sesja: {session}')
    print('Analiza: python3 -m pipelines.analyze_live_debug --root live_debug --html')


if __name__ == '__main__':
    main()
