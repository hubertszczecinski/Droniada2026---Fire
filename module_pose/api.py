import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pipeline_competition as pc
from module_pose.pnp_panel import PANEL_CELL_HEIGHT_M, PANEL_CELL_WIDTH_M, PANEL_GRID_COLS, PANEL_GRID_ROWS, PANEL_HEIGHT_M, PANEL_INTERNAL_LINES_PER_AXIS, PANEL_WIDTH_M, rotation_matrix_to_euler_deg, solve_panel_pose
from module_pose.grid_corners import detect_corners_white_grid
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

def _quad_aspect(quad: np.ndarray) -> float:
    side1 = float(np.linalg.norm(quad[0] - quad[1]))
    side2 = float(np.linalg.norm(quad[1] - quad[2]))
    return max(side1, side2) / max(1e-06, min(side1, side2))

def _quad_area_ratio(quad: np.ndarray, img_area: float) -> float:
    return float(cv2.contourArea(quad.astype(np.float32))) / max(1.0, img_area)

def _contour_to_quad(c: np.ndarray) -> np.ndarray:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        return pc.order_points(approx.reshape(-1, 2).astype(np.float32))
    return pc.order_points(cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32))

def _quads_from_binary_mask(mask: np.ndarray, img_area: float, *, min_area_frac: float, max_area_frac: float, min_aspect: float = 1.35, max_aspect: float = 3.2) -> List[np.ndarray]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[np.ndarray] = []
    for c in cnts:
        a = float(cv2.contourArea(c))
        if a < min_area_frac * img_area or a > max_area_frac * img_area:
            continue
        quad = _contour_to_quad(c)
        asp = _quad_aspect(quad)
        if min_aspect <= asp <= max_aspect:
            out.append(quad)
    return out

def _dark_masks(image_bgr: np.ndarray) -> List[np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    v, s, l = (hsv[:, :, 2], hsv[:, :, 1], lab[:, :, 0])
    dark = ((v <= 115) & (s <= 240)) | (l <= 115)
    m1 = (dark.astype(np.uint8) * 255)
    m1 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m1 = cv2.morphologyEx(m1, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    v_blur = cv2.GaussianBlur(v, (7, 7), 0)
    _, m2 = cv2.threshold(v_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, m3 = cv2.threshold(blur, 85, 255, cv2.THRESH_BINARY_INV)
    m3 = cv2.morphologyEx(m3, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return [m1, m2, m3]

def _black_panel_interior_score(image_bgr: np.ndarray, corners: np.ndarray) -> float:
    dst = np.array([[0, 0], [pc.RECT_W - 1, 0], [pc.RECT_W - 1, pc.RECT_H - 1], [0, pc.RECT_H - 1]], dtype=np.float32)
    h_mat = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(image_bgr, h_mat, (pc.RECT_W, pc.RECT_H))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    mx, my = (int(0.12 * pc.RECT_W), int(0.15 * pc.RECT_H))
    roi = gray[my:pc.RECT_H - my, mx:pc.RECT_W - mx]
    if roi.size == 0:
        return 0.0
    dark_frac = float(np.mean(roi < 95))
    white_t = pc.detect_white_corner_transform(warped)
    white_bonus = 0.12 if white_t in ('id', 'fx', 'fy') else 0.0
    return dark_frac + white_bonus

def _warp_panel_candidate(image_bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    dst = np.array([[0, 0], [pc.RECT_W - 1, 0], [pc.RECT_W - 1, pc.RECT_H - 1], [0, pc.RECT_H - 1]], dtype=np.float32)
    h_mat = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(image_bgr, h_mat, (pc.RECT_W, pc.RECT_H))

def _white_marker_score(warped_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    cw = pc.RECT_W // 10
    ch = pc.RECT_H // 10
    pad_x = max(2, int(cw * 0.18))
    pad_y = max(2, int(ch * 0.18))

    def score_patch(x0: int, y0: int) -> float:
        patch_hsv = hsv[y0 + pad_y:y0 + ch - pad_y, x0 + pad_x:x0 + cw - pad_x]
        patch_gray = gray[y0 + pad_y:y0 + ch - pad_y, x0 + pad_x:x0 + cw - pad_x]
        if patch_hsv.size == 0 or patch_gray.size == 0:
            return 0.0
        bright = patch_gray.astype(np.float32) / 255.0
        low_sat = 1.0 - patch_hsv[:, :, 1].astype(np.float32) / 255.0
        return float(0.7 * np.percentile(bright, 75) + 0.3 * np.percentile(low_sat, 75))

    scores = [
        score_patch(0, 0),
        score_patch(pc.RECT_W - cw, 0),
        score_patch(pc.RECT_W - cw, pc.RECT_H - ch),
        score_patch(0, pc.RECT_H - ch),
    ]
    best = max(scores)
    second = sorted(scores)[-2]
    return float(best + max(0.0, best - second))

def _white_marker_image_score(image_bgr: np.ndarray, quad: np.ndarray) -> float:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = (((gray > 150) & (hsv[:, :, 1] < 130)).astype(np.uint8) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    q = quad.astype(np.float32)
    diag = max(1.0, float(np.linalg.norm(q[0] - q[2])))
    best = 0.0
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < 350.0:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w <= 6 or h <= 6:
            continue
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > 4.0:
            continue
        cx = float(x + 0.5 * w)
        cy = float(y + 0.5 * h)
        if cv2.pointPolygonTest(q, (cx, cy), False) < 0:
            continue
        corner_dist = min(float(np.linalg.norm(np.array([cx, cy], dtype=np.float32) - p)) for p in q) / diag
        if corner_dist > 0.22:
            continue
        fill = area / float(max(1, w * h))
        size_score = min(1.0, area / 4500.0)
        best = max(best, float((1.0 - corner_dist / 0.22) * 0.55 + fill * 0.25 + size_score * 0.2))
    return best

def _grid_structure_score(warped_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    lines = cv2.createLineSegmentDetector().detect(eq)[0]
    if lines is None:
        return 0.0
    vert = 0
    hori = 0
    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx, dy = (abs(x2 - x1), abs(y2 - y1))
        if dy > dx * 1.3:
            vert += 1
        elif dx > dy * 1.3:
            hori += 1
    return float(min(1.0, (vert + hori) / 18.0))

def score_panel_quad(image_bgr: np.ndarray, quad: np.ndarray, k: np.ndarray, dist: np.ndarray) -> float:
    h, w = image_bgr.shape[:2]
    img_area = float(h * w)
    area_ratio = _quad_area_ratio(quad, img_area)
    interior = _black_panel_interior_score(image_bgr, quad)
    c_can, _anc = canonicalize_corners_by_white_anchor(image_bgr, quad)
    ok_pnp, _rv, _tv, reproj = solve_panel_pose(c_can, k, dist, refine_lm=False)
    if not ok_pnp:
        return -1e6
    score = 220.0 - float(reproj)
    if 0.12 <= area_ratio <= 0.70:
        score += 60.0
    elif area_ratio > 0.78:
        score -= 260.0
    elif area_ratio < 0.05:
        score -= 90.0
    score += 55.0 * interior
    warped = _warp_panel_candidate(image_bgr, c_can)
    score += 45.0 * _grid_structure_score(warped)
    score += 70.0 * _white_marker_score(warped)
    marker_img = _white_marker_image_score(image_bgr, quad)
    score += 180.0 * marker_img
    if marker_img == 0.0:
        score -= 120.0
    if pc.detect_white_corner_transform(warped) in ('id', 'fx', 'fy'):
        score += 12.0
    return score

def strip_camera_overlays(image_bgr: np.ndarray, *, top_frac: float = 0.07, bottom_frac: float = 0.0) -> Tuple[np.ndarray, float, float]:
    """Usuń typowe paski UI kamery (timer u góry, OSD na dole)."""
    h, w = image_bgr.shape[:2]
    y0 = int(top_frac * h)
    y1 = int((1.0 - bottom_frac) * h)
    if y1 <= y0 + 48:
        return (image_bgr, 0.0, 0.0)
    return (image_bgr[y0:y1, :].copy(), 0.0, float(y0))

def _rough_quad_from_white_hlines(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Białe poziome linie siatki 10x10 — typowy widok terenowy panelu."""
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    dark_panel_context = ((gray < 105) | ((v < 155) & (s > 50))).astype(np.uint8) * 255
    dark_panel_context = cv2.dilate(dark_panel_context, np.ones((31, 31), np.uint8), iterations=1)
    bright = ((gray > 118) & (dark_panel_context > 0)).astype(np.uint8) * 255
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 5), 1))
    hlines = cv2.morphologyEx(bright, cv2.MORPH_OPEN, hk)
    row_hit = (hlines > 0).mean(axis=1)
    ys = np.where(row_hit > 0.025)[0]
    if ys.size < 5:
        return None
    y_groups = np.split(ys, np.where(np.diff(ys) > 3)[0] + 1)
    y_centers = np.array([float(g.mean()) for g in y_groups if g.size > 0], dtype=np.float32)
    if y_centers.size >= 4:
        y_step = float(np.median(np.diff(np.sort(y_centers))))
    else:
        y_step = 0.0
    y0 = int(max(0.0, float(ys[0]) - 0.35 * y_step))
    y1 = int(min(float(h - 1), float(ys[-1]) + 0.35 * y_step))
    band = hlines[y0:y1 + 1, :]
    col_hit = (band > 0).mean(axis=0)
    xs = np.where(col_hit > 0.02)[0]
    if xs.size < 5:
        return None
    x_groups = np.split(xs, np.where(np.diff(xs) > 3)[0] + 1)
    x_centers = np.array([float(g.mean()) for g in x_groups if g.size > 0], dtype=np.float32)
    if x_centers.size >= 4:
        x_step = float(np.median(np.diff(np.sort(x_centers))))
    else:
        x_step = 0.0
    x0 = int(max(0.0, float(xs[0]) - 0.35 * x_step))
    x1 = int(min(float(w - 1), float(xs[-1]) + 0.35 * x_step))
    quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    asp = _quad_aspect(quad)
    if asp < 1.25 or asp > 3.6:
        return None
    return pc.order_points(quad)

def _largest_dark_quad(image_bgr: np.ndarray, img_area: float) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_q = None
    best_a = 0.0
    for c in cnts:
        a = float(cv2.contourArea(c))
        if a < 0.03 * img_area or a > 0.72 * img_area:
            continue
        q = _contour_to_quad(c)
        asp = _quad_aspect(q)
        if asp < 1.2 or asp > 3.8:
            continue
        if a > best_a:
            best_a = a
            best_q = q
    return best_q

def _panel_color_quad(image_bgr: np.ndarray, img_area: float) -> Optional[np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # W terenie panel jest ciemny, ale też wyraźnie bardziej nasycony
    # (zielono-niebieski) niż jasna ściana/tło.
    mask = (((h >= 75) & (h <= 115) & (s > 75) & (v < 150)).astype(np.uint8) * 255)
    mask[:max(1, int(0.03 * mask.shape[0])), :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_q = None
    best_a = 0.0
    for c in cnts:
        a = float(cv2.contourArea(c))
        if a < 0.08 * img_area or a > 0.78 * img_area:
            continue
        q = _contour_to_quad(c)
        asp = _quad_aspect(q)
        if asp < 1.35 or asp > 3.2:
            continue
        if a > best_a:
            best_a = a
            best_q = q
    return best_q

def _shift_quad(quad: np.ndarray, dx: float, dy: float) -> np.ndarray:
    q = quad.copy()
    q[:, 0] += dx
    q[:, 1] += dy
    return q

def gather_panel_quad_candidates(image_bgr: np.ndarray) -> List[np.ndarray]:
    work, ox, oy = strip_camera_overlays(image_bgr)
    h, w = work.shape[:2]
    img_area = float(h * w)
    out: List[np.ndarray] = []
    qc = _panel_color_quad(work, img_area)
    if qc is not None:
        out.append(_shift_quad(qc, ox, oy))
    qh = _rough_quad_from_white_hlines(work)
    if qh is not None:
        out.append(_shift_quad(qh, ox, oy))
    qd = _largest_dark_quad(work, img_area)
    if qd is not None:
        out.append(_shift_quad(qd, ox, oy))
    mx, my = int(0.12 * w), int(0.10 * h)
    roi = work[my:h - my, mx:w - mx]
    roi_area = float(roi.shape[0] * roi.shape[1])
    for mask in _dark_masks(roi):
        for q in _quads_from_binary_mask(mask, roi_area, min_area_frac=0.06, max_area_frac=0.88):
            out.append(_shift_quad(q, ox + float(mx), oy + float(my)))
    for mask in _dark_masks(work):
        for q in _quads_from_binary_mask(mask, img_area, min_area_frac=0.03, max_area_frac=0.72):
            out.append(_shift_quad(q, ox, oy))
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(gray, 40, 130)
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    for q in _quads_from_binary_mask(edge, img_area, min_area_frac=0.04, max_area_frac=0.50):
        out.append(_shift_quad(q, ox, oy))
    c_old = pc.detect_corners_img(work)
    if c_old is not None:
        out.append(_shift_quad(pc.order_points(c_old), ox, oy))
    c_grid = detect_corners_white_grid(work)
    if c_grid is not None:
        out.append(_shift_quad(c_grid, ox, oy))
    dedup: List[np.ndarray] = []
    for q in out:
        if not any(float(np.max(np.abs(q - d))) < 8.0 for d in dedup):
            dedup.append(q)
    extra: List[np.ndarray] = []
    for q in dedup:
        refined = refine_panel_corners_uniform_grid(image_bgr, q)
        if refined is not None and not any(float(np.max(np.abs(refined - d))) < 8.0 for d in dedup + extra):
            extra.append(refined)
    dedup.extend(extra)
    return dedup

def detect_corners_black_panel(image_bgr: np.ndarray, k: Optional[np.ndarray]=None, dist: Optional[np.ndarray]=None) -> Optional[np.ndarray]:
    h, w = image_bgr.shape[:2]
    if k is None:
        k, dist = default_intrinsics((h, w))
    img_area = float(h * w)
    quads: List[np.ndarray] = []
    for mask in _dark_masks(image_bgr)[:2]:
        quads.extend(_quads_from_binary_mask(mask, img_area, min_area_frac=0.04, max_area_frac=0.42))
    if not quads:
        return None
    best = max(quads, key=lambda q: score_panel_quad(image_bgr, q, k, dist))
    if score_panel_quad(image_bgr, best, k, dist) < 30.0:
        return None
    return best

def detect_corners_panel(image_bgr: np.ndarray, k: Optional[np.ndarray]=None, dist: Optional[np.ndarray]=None) -> Optional[np.ndarray]:
    h, w = image_bgr.shape[:2]
    if k is None:
        k, dist = default_intrinsics((h, w))
    work, ox, oy = strip_camera_overlays(image_bgr)
    color_quad = _panel_color_quad(work, float(work.shape[0] * work.shape[1]))
    if color_quad is not None:
        return _shift_quad(color_quad, ox, oy)
    quads = gather_panel_quad_candidates(image_bgr)
    if not quads:
        return None
    img_area = float(h * w)
    ranked = sorted(quads, key=lambda q: score_panel_quad(image_bgr, q, k, dist), reverse=True)
    for q in ranked:
        c_can, _ = canonicalize_corners_by_white_anchor(image_bgr, q)
        ok_pnp, _rv, _tv, _reproj = solve_panel_pose(c_can, k, dist, refine_lm=False)
        if ok_pnp:
            return q
    return ranked[0]

def detect_corners_img_robust(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    return detect_corners_panel(image_bgr)

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

def _append_corner_candidates(
    image_bgr: np.ndarray,
    corners: np.ndarray,
    base_method: str,
    k: np.ndarray,
    dist: np.ndarray,
    candidates: List[Tuple[str, np.ndarray, str, Optional[float], str]],
    *,
    refine_corners_grid: bool,
) -> None:
    corners = pc.order_points(corners.astype(np.float32))
    c0, anc = canonicalize_corners_by_white_anchor(image_bgr, corners)
    candidates.append((base_method, c0, 'none', None, anc))
    if not refine_corners_grid:
        return
    refined = refine_panel_corners_uniform_grid(image_bgr, corners)
    if refined is None:
        return
    ok_i, _, _, er_i = solve_panel_pose(corners, k, dist, refine_lm=True)
    ok_r, _, _, er_r = solve_panel_pose(refined, k, dist, refine_lm=True)
    if ok_i and ok_r and (er_r <= er_i + 0.75):
        c1, anc = canonicalize_corners_by_white_anchor(image_bgr, refined)
        candidates.append((base_method, c1, 'grid_9_internal_lines', er_i, anc))


def _pose_result_from_candidates(
    image_bgr: np.ndarray,
    candidates: List[Tuple[str, np.ndarray, str, Optional[float], str]],
    k: np.ndarray,
    dist: np.ndarray,
    *,
    fallback_corners: Optional[np.ndarray] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> PoseResult:
    h, w = image_bgr.shape[:2]
    best: Optional[Dict[str, Any]] = None
    candidate_reproj: Dict[str, float] = {}
    for idx, (base_method, cand_corners, refine_tag, before_er, anchor_tag) in enumerate(candidates):
        ok_c, rv_c, tv_c, er_c = solve_panel_pose(cand_corners, k, dist, refine_lm=True)
        key = f'{base_method}:{refine_tag}:{anchor_tag}:{idx}'
        candidate_reproj[key] = float(er_c) if ok_c else float('inf')
        if not ok_c or rv_c is None or tv_c is None:
            continue
        if best is None or er_c < best['reproj']:
            best = {
                'base_method': base_method,
                'corners': cand_corners,
                'refine_tag': refine_tag,
                'before_er': before_er,
                'anchor_tag': anchor_tag,
                'rvec': rv_c,
                'tvec': tv_c,
                'reproj': er_c,
            }
    if best is None:
        meta: Dict[str, Any] = {
            'reason': 'pnp_failed_all_candidates',
            'candidate_reproj_mean_px': candidate_reproj,
        }
        if extra_meta:
            meta.update(extra_meta)
        return PoseResult(
            ok=False,
            corners_px=fallback_corners,
            method='none',
            confidence=0.0,
            meta=meta,
        )
    corners_final = best['corners']
    method = best['base_method']
    rvec = best['rvec']
    tvec = best['tvec']
    reproj = float(best['reproj'])
    corner_meta: Dict[str, Any] = {
        'corner_refinement': best['refine_tag'],
        'corner_anchor_corner': best['anchor_tag'],
        'candidate_reproj_mean_px': candidate_reproj,
        'selected_corner_candidate': f"{method}:{best['refine_tag']}:{best['anchor_tag']}",
    }
    if best['before_er'] is not None:
        corner_meta['reproj_mean_px_before_corner_refine'] = float(best['before_er'])
    if extra_meta:
        corner_meta.update(extra_meta)
    rmat, _ = cv2.Rodrigues(rvec)
    euler = rotation_matrix_to_euler_deg(rmat)
    from module_pose.panel_stand import estimate_panel_stand
    report_angle, panel_cat, stand_conf, stand_meta = estimate_panel_stand(
        rmat,
        reproj,
        corners_final,
        (h, w),
    )
    conf = float(max(0.0, min(1.0, 0.55 * stand_conf + 0.45 * max(0.0, 1.0 - reproj / 15.0))))
    tv = np.asarray(tvec, dtype=np.float64).reshape(3)
    dist_m = float(np.linalg.norm(tv))
    corner_meta.update(stand_meta)
    return PoseResult(
        ok=True,
        rvec=rvec,
        tvec=tvec,
        corners_px=corners_final,
        euler_cam_deg=euler,
        report_angle_deg=int(report_angle),
        panel_angle_category=str(panel_cat),
        stand_confidence=float(stand_conf),
        confidence=conf,
        method=f'pnp_{method}',
        meta={
            'reproj_mean_px': reproj,
            'distance_camera_to_panel_center_m': dist_m,
            'panel_size_m': {'width': PANEL_WIDTH_M, 'height': PANEL_HEIGHT_M},
            'panel_grid': {
                'cols': PANEL_GRID_COLS,
                'rows': PANEL_GRID_ROWS,
                'internal_lines_per_axis': PANEL_INTERNAL_LINES_PER_AXIS,
                'cell_width_m': PANEL_CELL_WIDTH_M,
                'cell_height_m': PANEL_CELL_HEIGHT_M,
            },
            **corner_meta,
        },
    )


def pose_from_corners(
    image_bgr: np.ndarray,
    corners_tltrbrbl: np.ndarray,
    k: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    *,
    base_method: str = 'yolo_pose',
    refine_corners_grid: bool = True,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> PoseResult:
    """PnP + ustawienie panelu z gotowych 4 rogów (np. współdzielonych z modułem B)."""
    h, w = image_bgr.shape[:2]
    if k is None:
        k, dist = default_intrinsics((h, w))
    if corners_tltrbrbl is None or corners_tltrbrbl.shape != (4, 2):
        return PoseResult(ok=False, method='none', confidence=0.0, meta={'reason': 'bad_corners'})
    candidates: List[Tuple[str, np.ndarray, str, Optional[float], str]] = []
    _append_corner_candidates(
        image_bgr,
        corners_tltrbrbl,
        base_method,
        k,
        dist,
        candidates,
        refine_corners_grid=refine_corners_grid,
    )
    meta = dict(extra_meta or {})
    meta.setdefault('corner_source', base_method)
    return _pose_result_from_candidates(
        image_bgr,
        candidates,
        k,
        dist,
        fallback_corners=pc.order_points(corners_tltrbrbl.astype(np.float32)),
        extra_meta=meta,
    )


def pose_from_yolo_pose(
    image_bgr: np.ndarray,
    k: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    *,
    refine_corners_grid: bool = True,
    use_tracker: bool = True,
) -> PoseResult:
    """Moduł A: YOLO-Pose → te same rogi co moduł B (bias, scoring) → PnP."""
    from module_pose.yolo_pose_bridge import acquire_corners_for_pose

    h, w = image_bgr.shape[:2]
    if k is None:
        k, dist = default_intrinsics((h, w))
    corners, label, ymeta = acquire_corners_for_pose(
        image_bgr, k, dist, use_tracker=use_tracker,
    )
    meta = dict(ymeta)
    meta['corner_source'] = 'yolo_pose'
    meta['corner_label'] = str(label)
    if corners is None:
        meta['reason'] = meta.get('fail', 'yolo_pose_no_corners')
        return PoseResult(ok=False, method='yolo_pose', confidence=0.0, meta=meta)
    res = pose_from_corners(
        image_bgr,
        corners,
        k,
        dist,
        base_method='yolo_pose',
        refine_corners_grid=refine_corners_grid,
        extra_meta=meta,
    )
    if res.meta is None:
        res.meta = meta
    else:
        res.meta.update(meta)
    return res


def pose_from_image(image_bgr: np.ndarray, yolo_det: Optional[List[Tuple[int, float, float, float, float]]]=None, k: Optional[np.ndarray]=None, dist: Optional[np.ndarray]=None, prefer_img_corners: bool=True, refine_corners_grid: bool=True) -> PoseResult:
    h, w = image_bgr.shape[:2]
    if k is None:
        k, dist = default_intrinsics((h, w))
    corners_panel = detect_corners_panel(image_bgr) if prefer_img_corners else None
    corners_yolo = pc.detect_corners_yolo(yolo_det or []) if yolo_det else None
    if corners_panel is None and corners_yolo is None:
        return PoseResult(ok=False, method='none', confidence=0.0, meta={'reason': 'no_corners', 'corner_source': 'cv'})
    candidates: List[Tuple[str, np.ndarray, str, Optional[float], str]] = []
    if corners_panel is not None:
        _append_corner_candidates(
            image_bgr, corners_panel, 'img_panel', k, dist, candidates,
            refine_corners_grid=refine_corners_grid,
        )
    if corners_yolo is not None:
        _append_corner_candidates(
            image_bgr, corners_yolo, 'yolo_bbox', k, dist, candidates,
            refine_corners_grid=refine_corners_grid,
        )
    fb = corners_panel if corners_panel is not None else corners_yolo
    return _pose_result_from_candidates(
        image_bgr,
        candidates,
        k,
        dist,
        fallback_corners=fb,
        extra_meta={'corner_source': 'cv'},
    )

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
