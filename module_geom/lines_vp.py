"""Panel corners from LSD line segments + vanishing points + line intersections."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pipeline_competition as pc

Line2D = Tuple[float, float, float]  # nx, ny, c  with nx*x + ny*y + c = 0


def _segment_to_line(x1: float, y1: float, x2: float, y2: float) -> Optional[Line2D]:
    dx, dy = (x2 - x1, y2 - y1)
    length = float(np.hypot(dx, dy))
    if length < 1e-3:
        return None
    nx, ny = (-dy / length, dx / length)
    c = -(nx * x1 + ny * y1)
    return (float(nx), float(ny), float(c))


def _intersect_lines(l1: Line2D, l2: Line2D) -> Optional[Tuple[float, float]]:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return (float(x), float(y))


def _line_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)


def _classify_segment(x1: float, y1: float, x2: float, y2: float) -> Optional[str]:
    ang = _line_angle_deg(x1, y1, x2, y2)
    # Grid lines on tilted panel: two dominant orientations in the image.
    if ang < 32.0 or ang > 148.0:
        return 'h'
    if 58.0 < ang < 122.0:
        return 'v'
    return None


def _lsd_segments(mask_or_gray: np.ndarray, min_len_frac: float = 0.08) -> List[Tuple[float, float, float, float, str]]:
    h, w = mask_or_gray.shape[:2]
    min_len = min_len_frac * float(min(h, w))
    if mask_or_gray.ndim == 3:
        gray = cv2.cvtColor(mask_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask_or_gray
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    out: List[Tuple[float, float, float, float, str]] = []
    if lines is None:
        return out
    for seg in lines:
        x1, y1, x2, y2 = (float(seg[0][0]), float(seg[0][1]), float(seg[0][2]), float(seg[0][3]))
        if float(np.hypot(x2 - x1, y2 - y1)) < min_len:
            continue
        cls = _classify_segment(x1, y1, x2, y2)
        if cls is not None:
            out.append((x1, y1, x2, y2, cls))
    return out


def _vp_ransac(lines: List[Line2D], w: int, h: int, *, iters: int = 120, thresh: float = 2.5) -> Optional[Tuple[float, float]]:
    if len(lines) < 2:
        return None
    best_inliers: List[int] = []
    best_vp: Optional[Tuple[float, float]] = None
    rng = np.random.default_rng(42)
    arr = list(lines)
    n = len(arr)
    for _ in range(iters):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        vp = _intersect_lines(arr[int(i)], arr[int(j)])
        if vp is None:
            continue
        vx, vy = vp
        inliers = []
        for k, (nx, ny, c) in enumerate(arr):
            d = abs(nx * vx + ny * vy + c)
            if d < thresh:
                inliers.append(k)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_vp = vp
    if best_vp is None or len(best_inliers) < 2:
        return None
    # Refine VP: least-squares intersection of inlier lines.
    pts = []
    for k in best_inliers:
        for j in best_inliers:
            if j <= k:
                continue
            p = _intersect_lines(arr[k], arr[j])
            if p is not None:
                pts.append(p)
    if not pts:
        return best_vp
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (float(np.median(xs)), float(np.median(ys)))


def _line_offset_from_vp(line: Line2D, vp: Tuple[float, float]) -> float:
    nx, ny, c = line
    vx, vy = vp
    # Signed distance along normal between VP and line.
    return float(c + nx * vx + ny * vy)


def _pick_border_lines(lines: List[Line2D], vp: Optional[Tuple[float, float]]) -> Tuple[Optional[Line2D], Optional[Line2D]]:
    if len(lines) < 2:
        return None, None
    if vp is None:
        sorted_by_c = sorted(lines, key=lambda ln: ln[2])
        return sorted_by_c[0], sorted_by_c[-1]
    offsets = [(_line_offset_from_vp(ln, vp), ln) for ln in lines]
    offsets.sort(key=lambda t: t[0])
    return offsets[0][1], offsets[-1][1]


def _white_grid_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    bright = ((v > 115) & (s < 110)) | (gray > 165)
    m = bright.astype(np.uint8) * 255
    h, w = m.shape[:2]
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(9, h // 40)))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 40), 1))
    vert = cv2.morphologyEx(m, cv2.MORPH_OPEN, vk)
    hori = cv2.morphologyEx(m, cv2.MORPH_OPEN, hk)
    comb = cv2.bitwise_or(vert, hori)
    return cv2.morphologyEx(comb, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def detect_corners_geom_vp(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, float | str | int]]:
    """4 panel corners as intersections of extreme grid lines (LSD + VP)."""
    h, w = image_bgr.shape[:2]
    meta: Dict[str, float | str | int] = {'method': 'geom_vp_lsd'}
    mask = _white_grid_mask(image_bgr)
    segs = _lsd_segments(mask, min_len_frac=0.07)
    if len(segs) < 8:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        segs = _lsd_segments(clahe.apply(gray), min_len_frac=0.09)
    h_lines: List[Line2D] = []
    v_lines: List[Line2D] = []
    for x1, y1, x2, y2, cls in segs:
        ln = _segment_to_line(x1, y1, x2, y2)
        if ln is None:
            continue
        if cls == 'h':
            h_lines.append(ln)
        else:
            v_lines.append(ln)
    meta['h_segments'] = len(h_lines)
    meta['v_segments'] = len(v_lines)
    if len(h_lines) < 2 or len(v_lines) < 2:
        meta['fail'] = 'insufficient_lines'
        return None, meta
    vp_h = _vp_ransac(h_lines, w, h)
    vp_v = _vp_ransac(v_lines, w, h)
    top, bottom = _pick_border_lines(h_lines, vp_h)
    left, right = _pick_border_lines(v_lines, vp_v)
    if top is None or bottom is None or left is None or right is None:
        meta['fail'] = 'border_lines'
        return None, meta
    corners = []
    for hl in (top, bottom):
        for vl in (left, right):
            p = _intersect_lines(hl, vl)
            if p is None:
                meta['fail'] = 'corner_intersection'
                return None, meta
            corners.append(p)
    # Order: top-left, top-right, bottom-right, bottom-left from line pairing.
    tl = _intersect_lines(top, left)
    tr = _intersect_lines(top, right)
    br = _intersect_lines(bottom, right)
    bl = _intersect_lines(bottom, left)
    if any(p is None for p in (tl, tr, br, bl)):
        meta['fail'] = 'ordered_corners'
        return None, meta
    quad = pc.order_points(np.array([tl, tr, br, bl], dtype=np.float32))
    # Sanity checks.
    area = float(cv2.contourArea(quad.reshape(1, 4, 2)))
    img_area = float(h * w)
    if area < 0.03 * img_area or area > 0.85 * img_area:
        meta['fail'] = 'area_ratio'
        return None, meta
    asp = max(np.ptp(quad[:, 0]), np.ptp(quad[:, 1])) / max(1.0, min(np.ptp(quad[:, 0]), np.ptp(quad[:, 1])))
    if asp < 1.2 or asp > 4.0:
        meta['fail'] = 'aspect'
        return None, meta
    margin = 0.35 * max(h, w)
    for x, y in quad:
        if x < -margin or y < -margin or x > w + margin or y > h + margin:
            meta['fail'] = 'out_of_frame'
            return None, meta
    meta['ok'] = 1
    meta['vp_h_x'] = float(vp_h[0]) if vp_h else 0.0
    meta['vp_v_x'] = float(vp_v[0]) if vp_v else 0.0
    return quad.astype(np.float32), meta
