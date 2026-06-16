"""Analiza jednej klatki live — osobny wątek (YOLO + moduł B)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from module_panel.analyze import analyze_panel_image
from module_panel.report import predictions_to_report_lines
from module_panel.warp import warp_panel_rect
from module_geom.camera import resolve_intrinsics
from release.live_card_detect import detect_cards_live, warped_dets_to_predictions
from release.live_corners import detect_corners_live
from release.pose_runtime import PoseFrameOutput


def run_frame_analysis(
    ctx: Dict[str, Any],
    bgr: np.ndarray,
    fid: str,
    panel_id_live: str,
) -> Dict[str, Any]:
    """Ciężka analiza (YOLO, CXY) — do wykonania poza główną pętlą."""
    args = ctx['args']
    pose_rt = ctx.get('pose_rt')
    profile = ctx['profile']
    cam_calib = ctx.get('cam_calib')
    calib_path = ctx.get('calib_path')
    run_live_color = bool(ctx.get('run_live_color'))
    cxy_latch = ctx.get('cxy_latch')
    cxy_locked_announced = bool(ctx.get('cxy_locked_announced'))

    h, w = bgr.shape[:2]
    k, dist, _intr_meta = resolve_intrinsics(
        (h, w),
        profile=profile,
        zoom_ratio=float(args.zoom_ratio),
        calib_path=cam_calib,
    )
    det: List[Tuple[int, float, float, float, float]] = []
    warped_dets: List[dict] = []
    cand_rows: List[dict] = []
    module_a_po: Optional[PoseFrameOutput] = None
    corners_px: Optional[np.ndarray] = None
    corner_src = 'none'
    corner_meta: dict = {}

    a_yolo = pose_rt is not None and pose_rt.cfg.corner_source == 'yolo_pose'
    b_yolo = args.corner_mode == 'yolo_pose'
    shared_yolo = a_yolo and b_yolo

    if shared_yolo:
        corners_px, corner_src, corner_meta = detect_corners_live(
            bgr,
            k,
            dist,
            return_all_candidates=False,
            corner_mode='yolo_pose',
            use_tracker=not args.no_stabilize,
        )
        raw_cand = corner_meta.get('candidate_rows')
        if isinstance(raw_cand, list):
            cand_rows = raw_cand
        if corner_meta.get('tracker_held'):
            corner_src = f'{corner_src}+hold'

    if pose_rt is not None:
        module_a_po = pose_rt.process_bgr(
            bgr,
            fid,
            det=[],
            k=k,
            dist=dist,
            corners_px=corners_px if shared_yolo else None,
            corners_meta=corner_meta if shared_yolo else None,
            record=False,
        )

    if not shared_yolo:
        corners_px, corner_src, corner_meta = detect_corners_live(
            bgr,
            k,
            dist,
            return_all_candidates=False,
            corner_mode=args.corner_mode,
            use_tracker=not args.no_stabilize,
        )
        raw_cand = corner_meta.get('candidate_rows')
        if isinstance(raw_cand, list):
            cand_rows = raw_cand
        if corner_meta.get('tracker_held'):
            corner_src = f'{corner_src}+hold'

    if (
        corners_px is None
        and module_a_po is not None
        and module_a_po.ok
        and module_a_po.corners_px is not None
        and module_a_po.corners_px.shape == (4, 2)
    ):
        corners_px = module_a_po.corners_px.astype(np.float32).copy()
        corner_src = f'{module_a_po.meta.get("corner_source", "yolo_pose")}+module_a'
        corner_meta = dict(module_a_po.meta or {})
        corner_meta.setdefault('reproj_mean_px', float(module_a_po.reproj_mean_px))
        corner_meta.setdefault('method', 'yolo_pose')

    work_bgr = bgr
    gs_bgr: Optional[np.ndarray] = None
    black_th = None
    if corners_px is not None:
        from release.panel_black import calibrate_panel_black_from_corners

        black_th = calibrate_panel_black_from_corners(bgr, corners_px)
    if args.greenscreen:
        from release.greenscreen_panel import apply_workshop_greenscreen

        gs_bgr = apply_workshop_greenscreen(
            bgr, corners_px if corners_px is not None else None, calibrate_black=True,
        )
        work_bgr = gs_bgr

    h_mat = None
    grid_overlap_ratio = 0.0
    opencv_grid_x: Optional[List[float]] = None
    opencv_grid_y: Optional[List[float]] = None
    hough_grid_x: Optional[List[float]] = None
    hough_grid_y: Optional[List[float]] = None
    preds: List[dict] = []
    reliable = False
    reproj = 999.0
    xy_back = '-'
    angle = 0
    category = 'horizontal'
    pan_meta: dict = {'err': 'analysis_paused'}

    if corners_px is None:
        pan_meta = {'err': 'no_corners'}
        warped = work_bgr.copy()
    elif (
        not args.cxy_latch
        and (args.corners_only or (args.corner_mode == 'yolo_pose' and args.out_video))
    ):
        warped, h_mat = warp_panel_rect(work_bgr, corners_px)
        reproj = float(corner_meta.get('reproj_mean_px', 0.0))
        pan_meta = {
            'reproj_mean_px': reproj,
            'grid_xy_reliable': reproj < args.max_reproj_reliable,
            'xy_backend_selected': 'yolo_pose',
            'corner_source': corner_src,
        }
        reliable = bool(pan_meta['grid_xy_reliable'])
        xy_back = 'yolo_pose'
    else:
        warped, h_mat = warp_panel_rect(work_bgr, corners_px)
        if run_live_color:
            det, warped_dets = detect_cards_live(
                work_bgr, corners_px, h_mat, warped, black_thresholds=black_th,
            )
        if args.corner_mode == 'yolo_pose' and args.cxy_latch:
            from release.run_live_panel import _analyze_cxy_from_corners

            pan, preds, det, warped_dets, reproj, reliable, xy_back, angle, category, pan_meta = (
                _analyze_cxy_from_corners(
                    work_bgr,
                    corners_px,
                    corner_src,
                    k,
                    dist,
                    panel_id=panel_id_live,
                    xy_mode=args.xy_mode,
                    angle_source=args.angle_source,
                    calib_path=calib_path,
                    cam_calib=cam_calib,
                    max_reproj_px=float(args.max_reproj_reliable),
                    min_homography_inliers=int(args.min_homography_inliers),
                    no_color_detect=not run_live_color,
                )
            )
            warped = pan.warped_bgr
        else:
            pan = analyze_panel_image(
                work_bgr,
                det,
                k=k,
                dist=dist,
                xy_mode=args.xy_mode,
                angle_source=args.angle_source,
                angle_calibration_path=calib_path,
                camera_calib_path=cam_calib,
                allowed_orbit_steps=None,
                min_homography_inliers=int(args.min_homography_inliers),
                max_reproj_px=float(args.max_reproj_reliable),
                corners_px=corners_px,
                corner_source=corner_src,
            )
            pan_meta = pan.meta
            warped = pan.warped_bgr
            preds = list(pan.predictions)
            if warped_dets:
                live_preds_tmp = warped_dets_to_predictions(warped_dets)
                preds = live_preds_tmp
                pan_meta['xy_source'] = 'live_warp_grid_hsv'
                pan_meta['xy_backend_selected'] = 'live_warp_grid_hsv'
            reliable = bool(pan.meta.get('grid_xy_reliable', False))
            reproj = float(pan.meta.get('reproj_mean_px', 999.0))
            corner_src = str(pan.meta.get('corner_source', corner_src))
            if corner_meta.get('reproj_mean_px') is not None and args.corner_mode != 'yolo_pose':
                reproj = float(corner_meta.get('reproj_mean_px', reproj))
            xy_back = str(pan.meta.get('xy_backend_selected', pan.meta.get('xy_source', '?')))
            angle = int(pan.report_angle_deg)
            category = str(pan.panel_angle_category)

    hom_inliers = int(pan_meta.get('homography_inliers', 0))
    if corners_px is not None and pan_meta.get('err') != 'no_corners':
        from module_panel.grid_overlap import measure_panel_opencv_grid_overlap

        grid_overlap_ratio, g_ov = measure_panel_opencv_grid_overlap(
            warped,
            grid_lines_x=pan_meta.get('grid_lines_x'),
            grid_lines_y=pan_meta.get('grid_lines_y'),
            line_tol_frac=float(args.snapshot_grid_line_tol),
        )
        pan_meta['grid_overlap_ratio'] = float(grid_overlap_ratio)
        pan_meta['grid_overlap_method'] = str(g_ov.get('overlap_method', 'triple_grid'))
        pan_meta['triple_consensus'] = float(g_ov.get('triple_consensus', 0.0))
        pan_meta['spacing_min'] = float(g_ov.get('spacing_min', 0.0))
        pan_meta['warp_panel_coverage'] = float(g_ov.get('warp_panel_coverage', 0.0))
        pan_meta['grid_cell_iou'] = float(g_ov.get('cell_iou', 0.0))
        pan_meta['grid_line_match_ratio'] = float(g_ov.get('line_match_ratio', 0.0))
        pan_meta['hough_line_ok'] = bool(g_ov.get('hough_line_ok', False))
        if g_ov.get('hough_lines_x') and g_ov.get('hough_lines_y'):
            hough_grid_x = list(g_ov['hough_lines_x'])
            hough_grid_y = list(g_ov['hough_lines_y'])
            pan_meta['hough_lines_x'] = hough_grid_x
            pan_meta['hough_lines_y'] = hough_grid_y
        if g_ov.get('grid_lines_x') and g_ov.get('grid_lines_y'):
            opencv_grid_x = list(g_ov['grid_lines_x'])
            opencv_grid_y = list(g_ov['grid_lines_y'])
            pan_meta['grid_lines_x'] = opencv_grid_x
            pan_meta['grid_lines_y'] = opencv_grid_y

    live_preds = list(preds)
    live_lines = predictions_to_report_lines(panel_id_live, angle, live_preds)
    if args.require_reliable and not reliable:
        live_lines = []
        if not args.cxy_latch:
            live_preds = []

    latch_txt: Optional[str] = None
    latched_preds: Optional[List[dict]] = None
    latched_lines: Optional[List[str]] = None
    latch_events: List[str] = []
    if cxy_latch is not None:
        latch_meta = cxy_latch.update(
            frame_id=fid,
            reliable=reliable,
            reproj_mean_px=reproj,
            homography_inliers=hom_inliers,
            predictions=live_preds,
            report_lines=live_lines,
            meta=pan_meta,
        )
        pan_meta.update(latch_meta)
        snap = cxy_latch.snapshot
        if cxy_latch.locked and snap is not None:
            latched_preds = list(snap.predictions)
            latched_lines = list(snap.report_lines)
            latch_txt = (
                f'ZAMKNIĘTE @ {snap.frame_id}  reproj={snap.reproj_mean_px:.1f}px  '
                f'inl={snap.homography_inliers}'
            )
        elif snap is not None:
            latch_txt = (
                f'Oczekiwanie… best @ {snap.frame_id} '
                f'({int(latch_meta.get("cxy_stable_run", 0))}/{cxy_latch.min_stable_frames})'
            )
        if cxy_latch.locked and not cxy_locked_announced and snap is not None:
            latch_events.append('cxy_locked')
            for line in latched_lines or []:
                latch_events.append(f'latch_line:{line}')

    from release.panel_presence import resolve_panel_presence

    panel_present, panel_presence_reason, _presence_detail = resolve_panel_presence(
        corners_px,
        corner_meta,
        corner_src,
        image_shape=bgr.shape[:2],
        reliable_b=bool(reliable),
        reproj_b=float(reproj),
        max_reproj_b=float(args.max_reproj_reliable),
        analyze_err=str(pan_meta.get('err') or ''),
    )
    tracker_hold = bool(corner_meta.get('tracker_held')) or 'hold' in str(corner_src)

    return {
        'fid': fid,
        'panel_id_live': panel_id_live,
        'bgr': bgr,
        'work_bgr': work_bgr,
        'warped': warped,
        'module_a_po': module_a_po,
        'corners_px': corners_px,
        'corner_src': corner_src,
        'corner_meta': corner_meta,
        'det': det,
        'warped_dets': warped_dets,
        'cand_rows': cand_rows,
        'pan_meta': pan_meta,
        'live_preds': live_preds,
        'live_lines': live_lines,
        'latched_preds': latched_preds,
        'latched_lines': latched_lines,
        'latch_txt': latch_txt,
        'reliable': reliable,
        'reproj': reproj,
        'xy_back': xy_back,
        'angle': angle,
        'category': category,
        'hom_inliers': hom_inliers,
        'grid_overlap_ratio': grid_overlap_ratio,
        'opencv_grid_x': opencv_grid_x,
        'opencv_grid_y': opencv_grid_y,
        'hough_grid_x': hough_grid_x,
        'hough_grid_y': hough_grid_y,
        'panel_present': panel_present,
        'panel_presence_reason': panel_presence_reason,
        'tracker_hold': tracker_hold,
        'h_mat': h_mat,
        'black_th': black_th,
        'gs_bgr': gs_bgr,
        'latch_events': latch_events,
        'run_analysis': True,
    }
