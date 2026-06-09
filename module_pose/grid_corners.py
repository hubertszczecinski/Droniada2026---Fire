"""Detect panel outer corners from 9x9 white grid lines (2m x 1m competition panel)."""
from __future__ import annotations
from typing import List, Optional, Tuple
import cv2
import numpy as np
import pipeline_competition as pc
from module_pose.pnp_panel import PANEL_INTERNAL_LINES_PER_AXIS
from module_pose.refine_corners import _bounds_from_internal_lines

def _line_positions_from_lsd(mask: np.ndarray, min_len_frac: float = 0.12) -> Tuple[List[float], List[float]]:
    h, w = mask.shape[:2]
    min_len = min_len_frac * float(min(h, w))
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(mask)[0]
    vert: List[float] = []
    hori: List[float] = []
    if lines is None:
        return (vert, hori)
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        dx, dy = (abs(x2 - x1), abs(y2 - y1))
        length = float(np.hypot(dx, dy))
        if length < min_len:
            continue
        if dy > dx * 1.25:
            vert.append(float((x1 + x2) * 0.5))
        elif dx > dy * 1.25:
            hori.append(float((y1 + y2) * 0.5))
    return (vert, hori)

def _quad_from_grid_lines(vert: List[float], hori: List[float], w: int, h: int) -> Optional[np.ndarray]:
    n_in = PANEL_INTERNAL_LINES_PER_AXIS
    if len(vert) < 4 or len(hori) < 4:
        return None
    x_lines = pc.snap_lines_to_grid(vert, expected=n_in, low=0.0, high=float(w - 1))
    y_lines = pc.snap_lines_to_grid(hori, expected=n_in, low=0.0, high=float(h - 1))
    xb = _bounds_from_internal_lines(np.array(x_lines, dtype=np.float64), float(w - 1), n_in)
    yb = _bounds_from_internal_lines(np.array(y_lines, dtype=np.float64), float(h - 1), n_in)
    if xb is None or yb is None:
        return None
    left, right = xb
    top, bottom = yb
    quad = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
    area = float(cv2.contourArea(quad.reshape(1, 4, 2)))
    img_area = float(h * w)
    if area < 0.04 * img_area or area > 0.55 * img_area:
        return None
    asp = max(right - left, bottom - top) / max(1e-06, min(right - left, bottom - top))
    if asp < 1.3 or asp > 3.4:
        return None
    return pc.order_points(quad)

def _white_grid_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    v, s = (hsv[:, :, 2], hsv[:, :, 1])
    bright = ((v > 125) & (s < 150)) | (gray > 160)
    m = (bright.astype(np.uint8) * 255)
    h, w = m.shape[:2]
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(7, h // 50)))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(7, w // 50), 1))
    vert = cv2.morphologyEx(m, cv2.MORPH_OPEN, vk)
    horiz = cv2.morphologyEx(m, cv2.MORPH_OPEN, hk)
    comb = cv2.bitwise_or(vert, horiz)
    comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return comb

def _quad_from_warped_grid(image_bgr: np.ndarray, rough: np.ndarray) -> Optional[np.ndarray]:
    rw, rh = (pc.RECT_W, pc.RECT_H)
    dst = np.array([[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]], dtype=np.float32)
    try:
        h_i2r = cv2.getPerspectiveTransform(rough.astype(np.float32), dst)
    except cv2.error:
        return None
    warped = cv2.warpPerspective(image_bgr, h_i2r, (rw, rh))
    mask = _white_grid_mask(warped)
    vert, hori = _line_positions_from_lsd(mask, min_len_frac=0.06)
    quad_r = _quad_from_grid_lines(vert, hori, rw, rh)
    if quad_r is None:
        return None
    try:
        h_r2i = np.linalg.inv(h_i2r)
    except np.linalg.LinAlgError:
        return None
    out = cv2.perspectiveTransform(quad_r.reshape(1, 4, 2), h_r2i.astype(np.float64)).reshape(4, 2)
    if not np.isfinite(out).all():
        return None
    return pc.order_points(out.astype(np.float32))

def detect_corners_white_grid(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Find outer panel quad from white 9x9 grid lines (PV / EL style)."""
    h, w = image_bgr.shape[:2]
    mask = _white_grid_mask(image_bgr)
    vert, hori = _line_positions_from_lsd(mask, min_len_frac=0.10)
    quad = _quad_from_grid_lines(vert, hori, w, h)
    if quad is not None:
        return quad
    edges = cv2.Canny(mask, 50, 150)
    vert2, hori2 = _line_positions_from_lsd(edges, min_len_frac=0.08)
    quad = _quad_from_grid_lines(vert2, hori2, w, h)
    if quad is not None:
        return quad
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = float(h * w)
    rough_cands: List[np.ndarray] = []
    for c in cnts:
        a = float(cv2.contourArea(c))
        if a < 0.04 * img_area or a > 0.50 * img_area:
            continue
        rough_cands.append(pc.order_points(cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)))
    for rough in sorted(rough_cands, key=lambda q: cv2.contourArea(q.reshape(1, 4, 2)), reverse=True):
        q2 = _quad_from_warped_grid(image_bgr, rough)
        if q2 is not None:
            return q2
    return None
