from typing import Optional
import cv2
import numpy as np
import pipeline_competition as pc
from module_pose.pnp_panel import PANEL_INTERNAL_LINES_PER_AXIS

def _dst_rect_pts(rect_w: int, rect_h: int) -> np.ndarray:
    return np.array([[0, 0], [rect_w - 1, 0], [rect_w - 1, rect_h - 1], [0, rect_h - 1]], dtype=np.float32)

def _bounds_from_internal_lines(line_positions: np.ndarray, span_px: float, n_internal: int) -> Optional[tuple]:
    xs = np.sort(np.asarray(line_positions, dtype=np.float64).reshape(-1))
    if xs.size < 2:
        return None
    step = float(np.median(np.diff(xs)))
    if not np.isfinite(step) or step < 1.0:
        step = span_px / float(n_internal + 1)
    left = float(xs[0] - 0.5 * step)
    right = float(xs[-1] + 0.5 * step)
    left = float(np.clip(left, 0.0, span_px))
    right = float(np.clip(right, 0.0, span_px))
    if right <= left + 8.0:
        return None
    return (left, right)

def refine_panel_corners_uniform_grid(image_bgr: np.ndarray, corners_tltrbrbl: np.ndarray, rect_w: int=None, rect_h: int=None) -> Optional[np.ndarray]:
    if corners_tltrbrbl is None or corners_tltrbrbl.shape != (4, 2):
        return None
    rw = int(rect_w or pc.RECT_W)
    rh = int(rect_h or pc.RECT_H)
    if rw < 32 or rh < 32:
        return None
    n_in = PANEL_INTERNAL_LINES_PER_AXIS
    dst = _dst_rect_pts(rw, rh)
    try:
        h_i2r = cv2.getPerspectiveTransform(corners_tltrbrbl.astype(np.float32), dst)
    except cv2.error:
        return None
    if not np.isfinite(h_i2r).all():
        return None
    warped = cv2.warpPerspective(image_bgr, h_i2r, (rw, rh))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(eq)[0]
    vert: list = []
    hori: list = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy > dx * 1.3:
                vert.append((x1 + x2) * 0.5)
            elif dx > dy * 1.3:
                hori.append((y1 + y2) * 0.5)
    x_lines = pc.snap_lines_to_grid(vert, expected=n_in, low=0.0, high=float(rw - 1))
    y_lines = pc.snap_lines_to_grid(hori, expected=n_in, low=0.0, high=float(rh - 1))
    xb = _bounds_from_internal_lines(np.array(x_lines, dtype=np.float64), float(rw - 1), n_in)
    yb = _bounds_from_internal_lines(np.array(y_lines, dtype=np.float64), float(rh - 1), n_in)
    if xb is None or yb is None:
        return None
    left, right = xb
    top, bottom = yb
    tl = np.array([left, top], dtype=np.float32)
    tr = np.array([right, top], dtype=np.float32)
    br = np.array([right, bottom], dtype=np.float32)
    bl = np.array([left, bottom], dtype=np.float32)
    rect_pts = np.stack([tl, tr, br, bl], axis=0)
    area = float(cv2.contourArea(rect_pts.reshape(1, 4, 2)))
    if area < 0.15 * float(rw * rh):
        return None
    try:
        h_r2i = np.linalg.inv(h_i2r)
    except np.linalg.LinAlgError:
        return None
    refined = cv2.perspectiveTransform(rect_pts.reshape(1, 4, 2), h_r2i.astype(np.float64)).reshape(4, 2)
    if not np.isfinite(refined).all():
        return None
    h_img, w_img = image_bgr.shape[:2]
    refined[:, 0] = np.clip(refined[:, 0], 0.0, float(w_img - 1))
    refined[:, 1] = np.clip(refined[:, 1], 0.0, float(h_img - 1))
    return refined.astype(np.float32)
