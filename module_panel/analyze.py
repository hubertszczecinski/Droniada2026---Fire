from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
import pipeline_competition as pc
from module_panel.angle_from_pose import CATEGORY_BY_ANGLE, estimate_report_angle_and_category
from module_panel.grid import map_yolo_to_cells_corners_homography, map_yolo_to_cells_warp_grid
from module_panel.result import PanelAnalyzeResult
from module_panel.warp import warp_panel_rect
from module_panel.reliability import assess_grid_xy_reliable, probe_grid_homography_quality
from module_pose.api import canonicalize_corners_by_white_anchor, detect_corners_panel
from module_pose.pnp_panel import solve_panel_pose
_XY_MODES = frozenset({'grid_geom', 'grid_geom_white', 'warp_grid', 'geom_grid', 'line_grid'})

def _analyze_panel_from_warped_with_meta(warped_bgr: np.ndarray, homography_img_to_warp: np.ndarray, yolo_det: List[Tuple[int, float, float, float, float]], corners_image_px: np.ndarray, *, xy_mode: str='grid_geom_white', src_wh: Tuple[int, int]=(1024, 1024)) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if xy_mode not in _XY_MODES:
        raise ValueError(xy_mode)
    sw, sh = src_wh
    preds = map_yolo_to_cells_corners_homography(yolo_det, corners_image_px, sw, sh)
    if xy_mode == 'warp_grid':
        grid_preds, ok_grid, grid_meta = map_yolo_to_cells_warp_grid(yolo_det, warped_bgr, homography_img_to_warp, sw, sh)
        if ok_grid:
            return (grid_preds, {'xy_source': 'warp_grid_lines', **grid_meta})
        return (preds, {'xy_source': 'corner_homography_fallback', **grid_meta})
    if xy_mode == 'grid_geom_white':
        # Corners are already canonicalized by the white anchor before warping.
        # Re-applying the white-corner transform here flips the grid a second time.
        return (preds, {'xy_source': 'corner_homography_white_anchor'})
    return (preds, {'xy_source': 'corner_homography'})

def analyze_panel_from_warped(warped_bgr: np.ndarray, yolo_det: List[Tuple[int, float, float, float, float]], corners_image_px: np.ndarray, *, xy_mode: str='grid_geom_white', src_wh: Tuple[int, int]=(1024, 1024)) -> List[Dict[str, Any]]:
    h_mat = cv2.getPerspectiveTransform(corners_image_px.astype(np.float32), np.array([[0, 0], [pc.RECT_W - 1, 0], [pc.RECT_W - 1, pc.RECT_H - 1], [0, pc.RECT_H - 1]], dtype=np.float32))
    preds, _meta = _analyze_panel_from_warped_with_meta(warped_bgr, h_mat, yolo_det, corners_image_px, xy_mode=xy_mode, src_wh=src_wh)
    return preds

def detect_panel_corners_for_module_b(
    image_bgr: np.ndarray,
    yolo_det: List[Tuple[int, float, float, float, float]],
    *,
    prefer_geom_vp: bool = False,
    prefer_line_grid: bool = False,
    k: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    min_homography_inliers: int = 12,
) -> Tuple[Optional[np.ndarray], str]:
    if prefer_line_grid and k is not None:
        from module_geom.line_grid import detect_corners_line_grid
        c, lbl, _lg_meta = detect_corners_line_grid(
            image_bgr,
            k,
            dist if dist is not None else np.zeros((4, 1), np.float32),
            yolo_det=yolo_det,
            min_inliers=min_homography_inliers,
        )
        if c is not None:
            return (c, lbl)
        # Fallback only when lines completely fail.
    candidates: List[Tuple[str, Optional[np.ndarray]]] = []
    if prefer_geom_vp:
        from module_geom.pipeline import detect_corners_geom_vp
        g, geom_lbl, _gmeta = detect_corners_geom_vp(image_bgr)
        candidates.append((geom_lbl, g))
    if not prefer_line_grid:
        candidates.extend([
            ('img_panel', detect_corners_panel(image_bgr)),
            ('yolo_bbox', pc.detect_corners_yolo(yolo_det or [])),
        ])
    best: Optional[Tuple[np.ndarray, str, float]] = None
    for label, raw in candidates:
        if raw is None:
            continue
        if label == 'geom_vp':
            c = raw.astype(np.float32)
        else:
            c, _anc = canonicalize_corners_by_white_anchor(image_bgr, raw.astype(np.float32))
            c = c.astype(np.float32)
        score = 1e9
        if k is not None:
            ok_pnp, _rv, _tv, reproj = solve_panel_pose(c, k, dist)
            if ok_pnp:
                score = float(reproj)
        if best is None or score < best[2]:
            best = (c, label, score)
    if best is None:
        return (None, 'none')
    return (best[0], best[1])

def analyze_panel_image(
    image_bgr: np.ndarray,
    yolo_det: List[Tuple[int, float, float, float, float]],
    *,
    k: Optional[np.ndarray]=None,
    dist: Optional[np.ndarray]=None,
    xy_mode: str='grid_geom_white',
    angle_source: str='rmat_linear',
    json_report_angle_deg: Optional[int]=None,
    angle_calibration_path: Optional[str]=None,
    camera_calib_path: Optional[str]=None,
    pose_json: Optional[Dict[str, Any]]=None,
    allowed_orbit_steps: Optional[Set[int]] = None,
    min_homography_inliers: int = 12,
    max_reproj_px: float = 8.0,
) -> PanelAnalyzeResult:
    if xy_mode not in _XY_MODES:
        raise ValueError(xy_mode)
    h, w = image_bgr.shape[:2]
    if k is None:
        k = np.array([[1000.0, 0.0, w / 2.0], [0.0, 1000.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    if dist is None:
        dist = np.zeros((4, 1), dtype=np.float32)
    from module_geom.pipeline import analyze_cards_geom, prepare_image_geom
    img_work, k_work, dist_work, und_meta = prepare_image_geom(image_bgr, k, dist, calib_path=camera_calib_path)
    prefer_geom = xy_mode == 'geom_grid'
    prefer_lines = xy_mode == 'line_grid'
    corners_px, corner_source = detect_panel_corners_for_module_b(
        img_work,
        yolo_det,
        prefer_geom_vp=prefer_geom,
        prefer_line_grid=prefer_lines,
        k=k_work,
        dist=dist_work,
        min_homography_inliers=min_homography_inliers,
    )
    if corners_px is None:
        return PanelAnalyzeResult(predictions=[], warped_bgr=image_bgr.copy(), homography=np.eye(3, dtype=np.float32), report_angle_deg=0, panel_angle_category='horizontal', meta={'xy_mode': xy_mode, 'angle_source': angle_source, 'pnp_ok': False, 'corner_source': 'none', 'err': 'no_corners', **und_meta})
    ok_pnp, rvec, _tvec, reproj = solve_panel_pose(corners_px, k_work, dist_work)
    rmat = None
    if ok_pnp and rvec is not None:
        rmat, _ = cv2.Rodrigues(rvec)
    warped, h_mat = warp_panel_rect(img_work, corners_px)
    xy_meta: Dict[str, Any] = {}
    if xy_mode == 'line_grid':
        from module_geom.line_grid import analyze_cards_line_grid
        preds, lg_ok, xy_meta = analyze_cards_line_grid(
            img_work,
            yolo_det,
            corners_px,
            src_wh=(w, h),
            corner_label=corner_source,
            reproj_px=float(reproj),
            min_inliers=min_homography_inliers,
        )
        if not lg_ok:
            preds = []
            xy_meta['xy_source'] = 'line_grid_fail_no_homography'
    elif xy_mode == 'geom_grid':
        preds, geom_ok, xy_meta = analyze_cards_geom(img_work, yolo_det, corners_px, src_wh=(w, h))
        if not geom_ok:
            preds, xy_meta = _analyze_panel_from_warped_with_meta(warped, h_mat, yolo_det, corners_px, xy_mode='grid_geom_white', src_wh=(w, h))
            xy_meta['xy_source'] = 'geom_grid_fallback_white'
    else:
        preds, xy_meta = _analyze_panel_from_warped_with_meta(warped, h_mat, yolo_det, corners_px, xy_mode=xy_mode, src_wh=(w, h))
    meta: Dict[str, Any] = {
        'xy_mode': xy_mode,
        'angle_source': angle_source,
        'reproj_mean_px': reproj,
        'pnp_ok': bool(ok_pnp),
        'corner_source': corner_source,
        **und_meta,
    }
    meta.update(xy_meta)
    if 'homography_inliers' not in meta:
        meta.update(probe_grid_homography_quality(img_work, corners_px, min_inliers=min_homography_inliers))
    rel_v2, rel_detail = assess_grid_xy_reliable(
        pnp_ok=bool(ok_pnp),
        reproj_mean_px=float(reproj),
        meta={**meta, 'xy_mode': xy_mode},
        pose_json=pose_json,
        max_reproj_px=max_reproj_px,
        min_homography_inliers=min_homography_inliers,
        allowed_orbit_steps=allowed_orbit_steps,
    )
    meta.update(rel_detail)
    meta['grid_xy_reliable'] = rel_v2
    meta['grid_xy_reliable_legacy'] = bool(ok_pnp and reproj <= max_reproj_px)
    if angle_source == 'json' and json_report_angle_deg is not None:
        report_angle = int(json_report_angle_deg)
        cat = CATEGORY_BY_ANGLE.get(report_angle, 'horizontal')
    elif angle_source == 'geom':
        report_angle = pc.angle_from_geom(corners_px)
        cat = CATEGORY_BY_ANGLE.get(report_angle, 'horizontal')
    elif angle_source == 'pnp':
        report_angle = pc.angle_from_pnp(corners_px, image_bgr.shape)
        cat = CATEGORY_BY_ANGLE.get(report_angle, 'horizontal')
    elif rmat is not None:
        mode = 'rmat_theta' if angle_source == 'rmat_theta' else 'rmat_linear'
        report_angle, cat, am = estimate_report_angle_and_category(rmat, reproj_px=reproj, calibration_path=angle_calibration_path, mode=mode)
        meta.update(am)
    else:
        report_angle, cat = (0, 'horizontal')
    return PanelAnalyzeResult(predictions=preds, warped_bgr=warped, homography=h_mat, report_angle_deg=report_angle, panel_angle_category=cat, meta=meta)
