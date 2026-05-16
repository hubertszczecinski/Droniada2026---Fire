from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pipeline_competition as pc
from module_panel.angle_from_pose import CATEGORY_BY_ANGLE, estimate_report_angle_and_category
from module_panel.grid import map_yolo_to_cells_corners_homography
from module_panel.result import PanelAnalyzeResult
from module_panel.warp import warp_panel_rect
from module_pose.api import canonicalize_corners_by_white_anchor, detect_corners_img_robust
from module_pose.pnp_panel import solve_panel_pose
_XY_MODES = frozenset({'grid_geom', 'grid_geom_white'})

def analyze_panel_from_warped(warped_bgr: np.ndarray, yolo_det: List[Tuple[int, float, float, float, float]], corners_image_px: np.ndarray, *, xy_mode: str='grid_geom_white', src_wh: Tuple[int, int]=(1024, 1024)) -> List[Dict[str, Any]]:
    if xy_mode not in _XY_MODES:
        raise ValueError(xy_mode)
    sw, sh = src_wh
    preds = map_yolo_to_cells_corners_homography(yolo_det, corners_image_px, sw, sh)
    if xy_mode == 'grid_geom_white':
        t = pc.detect_white_corner_transform(warped_bgr)
        preds = pc.apply_transform_preds(preds, t)
    return preds

def detect_panel_corners_for_module_b(image_bgr: np.ndarray, yolo_det: List[Tuple[int, float, float, float, float]]) -> Tuple[Optional[np.ndarray], str]:
    attempts: List[Tuple[str, Optional[np.ndarray]]] = [('img_corners_robust', detect_corners_img_robust(image_bgr)), ('img_corners', pc.detect_corners_img(image_bgr)), ('yolo_bbox', pc.detect_corners_yolo(yolo_det or []))]
    for label, raw in attempts:
        if raw is None:
            continue
        c, _anc = canonicalize_corners_by_white_anchor(image_bgr, raw.astype(np.float32))
        return (c.astype(np.float32), label)
    return (None, 'none')

def analyze_panel_image(image_bgr: np.ndarray, yolo_det: List[Tuple[int, float, float, float, float]], *, k: Optional[np.ndarray]=None, dist: Optional[np.ndarray]=None, xy_mode: str='grid_geom_white', angle_source: str='rmat_linear', json_report_angle_deg: Optional[int]=None, angle_calibration_path: Optional[str]=None) -> PanelAnalyzeResult:
    if xy_mode not in _XY_MODES:
        raise ValueError(xy_mode)
    h, w = image_bgr.shape[:2]
    if k is None:
        k = np.array([[1000.0, 0.0, w / 2.0], [0.0, 1000.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    if dist is None:
        dist = np.zeros((4, 1), dtype=np.float32)
    corners_px, corner_source = detect_panel_corners_for_module_b(image_bgr, yolo_det)
    if corners_px is None:
        return PanelAnalyzeResult(predictions=[], warped_bgr=image_bgr.copy(), homography=np.eye(3, dtype=np.float32), report_angle_deg=0, panel_angle_category='horizontal', meta={'xy_mode': xy_mode, 'angle_source': angle_source, 'pnp_ok': False, 'corner_source': 'none', 'err': 'no_corners'})
    ok_pnp, rvec, _tvec, reproj = solve_panel_pose(corners_px, k, dist)
    rmat = None
    if ok_pnp and rvec is not None:
        rmat, _ = cv2.Rodrigues(rvec)
    warped, h_mat = warp_panel_rect(image_bgr, corners_px)
    preds = analyze_panel_from_warped(warped, yolo_det, corners_px, xy_mode=xy_mode, src_wh=(w, h))
    meta: Dict[str, Any] = {'xy_mode': xy_mode, 'angle_source': angle_source, 'reproj_mean_px': reproj, 'pnp_ok': bool(ok_pnp), 'corner_source': corner_source, 'grid_xy_reliable': bool(ok_pnp and reproj <= 8.0)}
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
