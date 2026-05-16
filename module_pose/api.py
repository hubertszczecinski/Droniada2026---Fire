import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pipeline_competition as pc
from module_pose.pnp_panel import PANEL_CELL_HEIGHT_M, PANEL_CELL_WIDTH_M, PANEL_GRID_COLS, PANEL_GRID_ROWS, PANEL_HEIGHT_M, PANEL_INTERNAL_LINES_PER_AXIS, PANEL_WIDTH_M, rotation_matrix_to_euler_deg, solve_panel_pose
from module_pose.refine_corners import refine_panel_corners_uniform_grid
from module_pose.types import PoseResult

def canonicalize_corners_by_white_anchor(image_bgr: np.ndarray, corners_tltrbrbl: np.ndarray) -> Tuple[np.ndarray, str]:
    if corners_tltrbrbl is None or corners_tltrbrbl.shape != (4, 2):
        return (corners_tltrbrbl, 'unknown')
    dst = np.array([[0, 0], [pc.RECT_W - 1, 0], [pc.RECT_W - 1, pc.RECT_H - 1], [0, pc.RECT_H - 1]], dtype=np.float32)
    h_mat = cv2.getPerspectiveTransform(corners_tltrbrbl.astype(np.float32), dst)
    warped = cv2.warpPerspective(image_bgr, h_mat, (pc.RECT_W, pc.RECT_H))
    t = pc.detect_white_corner_transform(warped)
    if t == 'id':
        a = 3
        anchor = 'bl'
    elif t == 'fx':
        a = 2
        anchor = 'br'
    elif t == 'fy':
        a = 0
        anchor = 'tl'
    else:
        a = 1
        anchor = 'tr'
    idx = [(a - i) % 4 for i in range(4)]
    out = corners_tltrbrbl[idx].astype(np.float32)
    return (out, anchor)

def detect_corners_img_robust(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(gray, 40, 130)
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    img_area = float(image_bgr.shape[0] * image_bgr.shape[1])
    best_quad = None
    best_area = 0.0
    best_rect = None
    best_rect_area = 0.0
    for c in cnts:
        a = float(cv2.contourArea(c))
        if a < 0.03 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and a > best_area:
            best_area = a
            best_quad = approx.reshape(-1, 2).astype(np.float32)
        rect = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
        if a > best_rect_area:
            best_rect_area = a
            best_rect = rect
    if best_quad is not None:
        return pc.order_points(best_quad)
    if best_rect is not None:
        return pc.order_points(best_rect)
    return None

def default_intrinsics(image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_shape[:2]
    k = np.array([[1000.0, 0.0, w / 2.0], [0.0, 1000.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    dist = np.zeros((4, 1), dtype=np.float32)
    return (k, dist)

def intrinsics_from_pose_json(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    intr = data.get('intrinsics') or {}
    fx = float(intr.get('fx', 1000.0))
    fy = float(intr.get('fy', fx))
    cx = float(intr.get('cx', intr.get('width', 1024) / 2.0))
    cy = float(intr.get('cy', intr.get('height', 1024) / 2.0))
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    dcoef = intr.get('dist_coeffs') or [0.0, 0.0, 0.0, 0.0, 0.0]
    dist = np.array(dcoef, dtype=np.float32).reshape(-1, 1)
    return (k, dist)

def load_pose_gt_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def enrich_pose_with_drone_gt(result: PoseResult, gt: Dict[str, Any]) -> None:
    cam = gt.get('camera') or {}
    if 'rotation_euler_xyz_rad' in cam:
        result.meta['ref_drone_camera_euler_xyz_deg'] = [math.degrees(float(x)) for x in cam['rotation_euler_xyz_rad']]
    if 'location_world' in cam:
        result.meta['ref_drone_camera_position_world_m'] = [float(x) for x in cam['location_world']]
    panel = gt.get('panel') or {}
    if 'panel_angle_category' in panel:
        result.meta['ref_panel_angle_category'] = panel['panel_angle_category']
    if 'panel_skew_report_deg' in panel:
        result.meta['ref_panel_skew_report_deg'] = int(panel['panel_skew_report_deg'])
    if 'orbit_azimuth_deg' in cam:
        result.meta['ref_orbit_azimuth_deg'] = float(cam['orbit_azimuth_deg'])

def pose_from_image(image_bgr: np.ndarray, yolo_det: Optional[List[Tuple[int, float, float, float, float]]]=None, k: Optional[np.ndarray]=None, dist: Optional[np.ndarray]=None, prefer_img_corners: bool=True, refine_corners_grid: bool=True) -> PoseResult:
    h, w = image_bgr.shape[:2]
    if k is None:
        k, dist = default_intrinsics((h, w))
    corners_img = pc.detect_corners_img(image_bgr) if prefer_img_corners else None
    corners_img_robust = detect_corners_img_robust(image_bgr) if prefer_img_corners else None
    corners_yolo = pc.detect_corners_yolo(yolo_det or []) if yolo_det else None
    if corners_img is None and corners_img_robust is None and (corners_yolo is None):
        return PoseResult(ok=False, method='none', confidence=0.0, meta={'reason': 'no_corners'})
    if corners_img is not None:
        corners_img = pc.order_points(corners_img)
    if corners_img_robust is not None:
        corners_img_robust = pc.order_points(corners_img_robust)
    if corners_yolo is not None:
        corners_yolo = pc.order_points(corners_yolo)
    candidates: List[Tuple[str, np.ndarray, str, Optional[float], str]] = []
    if corners_img is not None:
        c0, anc = canonicalize_corners_by_white_anchor(image_bgr, corners_img)
        candidates.append(('img_corners', c0, 'none', None, anc))
    if corners_img_robust is not None:
        c0, anc = canonicalize_corners_by_white_anchor(image_bgr, corners_img_robust)
        candidates.append(('img_corners_robust', c0, 'none', None, anc))
    if corners_yolo is not None:
        c0, anc = canonicalize_corners_by_white_anchor(image_bgr, corners_yolo)
        candidates.append(('yolo_bbox', c0, 'none', None, anc))
    if refine_corners_grid:
        if corners_img is not None:
            refined_img = refine_panel_corners_uniform_grid(image_bgr, corners_img)
            if refined_img is not None:
                ok_i, _, _, er_i = solve_panel_pose(corners_img, k, dist, refine_lm=True)
                ok_r, _, _, er_r = solve_panel_pose(refined_img, k, dist, refine_lm=True)
                if ok_i and ok_r and (er_r <= er_i + 0.75):
                    c1, anc = canonicalize_corners_by_white_anchor(image_bgr, refined_img)
                    candidates.append(('img_corners', c1, 'grid_9_internal_lines', er_i, anc))
        if corners_yolo is not None:
            refined_yolo = refine_panel_corners_uniform_grid(image_bgr, corners_yolo)
            if refined_yolo is not None:
                ok_i, _, _, er_i = solve_panel_pose(corners_yolo, k, dist, refine_lm=True)
                ok_r, _, _, er_r = solve_panel_pose(refined_yolo, k, dist, refine_lm=True)
                if ok_i and ok_r and (er_r <= er_i + 0.75):
                    c1, anc = canonicalize_corners_by_white_anchor(image_bgr, refined_yolo)
                    candidates.append(('yolo_bbox', c1, 'grid_9_internal_lines', er_i, anc))
        if corners_img_robust is not None:
            refined_robust = refine_panel_corners_uniform_grid(image_bgr, corners_img_robust)
            if refined_robust is not None:
                ok_i, _, _, er_i = solve_panel_pose(corners_img_robust, k, dist, refine_lm=True)
                ok_r, _, _, er_r = solve_panel_pose(refined_robust, k, dist, refine_lm=True)
                if ok_i and ok_r and (er_r <= er_i + 0.75):
                    c1, anc = canonicalize_corners_by_white_anchor(image_bgr, refined_robust)
                    candidates.append(('img_corners_robust', c1, 'grid_9_internal_lines', er_i, anc))
    best: Optional[Dict[str, Any]] = None
    candidate_reproj: Dict[str, float] = {}
    for idx, (base_method, cand_corners, refine_tag, before_er, anchor_tag) in enumerate(candidates):
        ok_c, rv_c, tv_c, er_c = solve_panel_pose(cand_corners, k, dist, refine_lm=True)
        key = f'{base_method}:{refine_tag}:{anchor_tag}:{idx}'
        candidate_reproj[key] = float(er_c) if ok_c else float('inf')
        if not ok_c or rv_c is None or tv_c is None:
            continue
        if best is None or er_c < best['reproj']:
            best = {'base_method': base_method, 'corners': cand_corners, 'refine_tag': refine_tag, 'before_er': before_er, 'anchor_tag': anchor_tag, 'rvec': rv_c, 'tvec': tv_c, 'reproj': er_c}
    if best is None:
        return PoseResult(ok=False, corners_px=corners_img if corners_img is not None else corners_img_robust if corners_img_robust is not None else corners_yolo, method='none', confidence=0.0, meta={'reason': 'pnp_failed_all_candidates', 'candidate_reproj_mean_px': candidate_reproj})
    assert best is not None
    corners_final = best['corners']
    method = best['base_method']
    rvec = best['rvec']
    tvec = best['tvec']
    reproj = float(best['reproj'])
    corner_meta: Dict[str, Any] = {'corner_refinement': best['refine_tag'], 'corner_anchor_corner': best['anchor_tag'], 'candidate_reproj_mean_px': candidate_reproj, 'selected_corner_candidate': f"{method}:{best['refine_tag']}:{best['anchor_tag']}"}
    if best['before_er'] is not None:
        corner_meta['reproj_mean_px_before_corner_refine'] = float(best['before_er'])
    rmat, _ = cv2.Rodrigues(rvec)
    euler = rotation_matrix_to_euler_deg(rmat)
    conf = float(max(0.0, 1.0 - reproj / 15.0))
    tv = np.asarray(tvec, dtype=np.float64).reshape(3)
    dist_m = float(np.linalg.norm(tv))
    return PoseResult(ok=True, rvec=rvec, tvec=tvec, corners_px=corners_final, euler_cam_deg=euler, confidence=conf, method=f'pnp_{method}', meta={'reproj_mean_px': reproj, 'distance_camera_to_panel_center_m': dist_m, 'panel_size_m': {'width': PANEL_WIDTH_M, 'height': PANEL_HEIGHT_M}, 'panel_grid': {'cols': PANEL_GRID_COLS, 'rows': PANEL_GRID_ROWS, 'internal_lines_per_axis': PANEL_INTERNAL_LINES_PER_AXIS, 'cell_width_m': PANEL_CELL_WIDTH_M, 'cell_height_m': PANEL_CELL_HEIGHT_M}, **corner_meta})

def pose_from_paths(image_path: str, yolo_path: Optional[str]=None, pose_gt_json_path: Optional[str]=None) -> PoseResult:
    img = cv2.imread(image_path)
    if img is None:
        return PoseResult(ok=False, method='none', meta={'reason': 'no_image'})
    det = pc.load_yolo(yolo_path) if yolo_path and os.path.isfile(yolo_path) else []
    k, dist = default_intrinsics(img.shape)
    gt: Optional[Dict[str, Any]] = None
    if pose_gt_json_path and os.path.isfile(pose_gt_json_path):
        gt = load_pose_gt_json(pose_gt_json_path)
        if gt and 'intrinsics' in gt:
            k, dist = intrinsics_from_pose_json(gt)
    res = pose_from_image(img, det, k=k, dist=dist)
    if gt is not None:
        enrich_pose_with_drone_gt(res, gt)
    return res
