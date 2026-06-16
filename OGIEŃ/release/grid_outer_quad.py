"""
Zewnętrzna siatka panelu (LSD/Hough) — główne źródło geometrii live.

HSV/alignment tylko jako fallback (panel w kadrze), nie jako kotwica krawędzi.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pipeline_competition as pc

from module_geom.lines_vp import _classify_segment, _white_grid_mask
from module_pose.api import strip_camera_overlays


@dataclass
class OuterGridBounds:
    """Skrajne białe linie siatki w układzie pełnego kadru (po strip OSD)."""
    top: Tuple[float, float, float]
    bottom: Tuple[float, float, float]
    left: Tuple[float, float, float]
    right: Tuple[float, float, float]
    x_left_line: float
    x_right_line: float
    y_top_line: float
    y_bottom_line: float
    ox: float
    oy: float
    detect_scale: float
    n_v: int
    n_h: int


def _segment_to_line(x1: float, y1: float, x2: float, y2: float) -> Optional[Tuple[float, float, float]]:
    dx, dy = (x2 - x1, y2 - y1)
    length = float(np.hypot(dx, dy))
    if length < 1e-3:
        return None
    nx, ny = (-dy / length, dx / length)
    c = -(nx * x1 + ny * y1)
    return (float(nx), float(ny), float(c))


def _intersect_lines(
    l1: Tuple[float, float, float],
    l2: Tuple[float, float, float],
) -> Optional[Tuple[float, float]]:
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


def _lsd_hv_segments(
    gray: np.ndarray,
    *,
    min_len_frac: float = 0.04,
) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float]]]:
    h_img, w_img = gray.shape[:2]
    min_len = min_len_frac * float(min(h_img, w_img))
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    h_segs: List[Tuple[float, float, float, float]] = []
    v_segs: List[Tuple[float, float, float, float]] = []
    if lines is None:
        return h_segs, v_segs
    for seg in lines:
        x1, y1, x2, y2 = (
            float(seg[0][0]),
            float(seg[0][1]),
            float(seg[0][2]),
            float(seg[0][3]),
        )
        if float(np.hypot(x2 - x1, y2 - y1)) < min_len:
            continue
        cls = _classify_segment(x1, y1, x2, y2)
        if cls == 'h':
            h_segs.append((x1, y1, x2, y2))
        elif cls == 'v':
            v_segs.append((x1, y1, x2, y2))
    return h_segs, v_segs


def _hough_hv_segments(
    gray: np.ndarray,
    *,
    min_len_frac: float = 0.12,
) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float]]]:
    """Fallback gdy LSD zwraca mało segmentów."""
    h_img, w_img = gray.shape[:2]
    min_len = int(min_len_frac * float(min(h_img, w_img)))
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=80,
        minLineLength=max(40, min_len),
        maxLineGap=18,
    )
    h_segs: List[Tuple[float, float, float, float]] = []
    v_segs: List[Tuple[float, float, float, float]] = []
    if lines is None:
        return h_segs, v_segs
    for ln in lines:
        x1, y1, x2, y2 = (float(ln[0][0]), float(ln[0][1]), float(ln[0][2]), float(ln[0][3]))
        cls = _classify_segment(x1, y1, x2, y2)
        if cls == 'h':
            h_segs.append((x1, y1, x2, y2))
        elif cls == 'v':
            v_segs.append((x1, y1, x2, y2))
    return h_segs, v_segs


def _line_x_at_y(line: Tuple[float, float, float], y: float) -> float:
    nx, ny, c = line
    if abs(ny) < 1e-6:
        return float('nan')
    return float(-(ny * y + c) / nx)


def _line_y_at_x(line: Tuple[float, float, float], x: float) -> float:
    nx, ny, c = line
    if abs(nx) < 1e-6:
        return float('nan')
    return float(-(nx * x + c) / ny)


def _best_line_near_coordinate(
    segs: List[Tuple[float, float, float, float]],
    target: float,
    *,
    coord: str,
    tol_frac: float = 0.04,
    img_size: int,
) -> Optional[Tuple[float, float, float]]:
    """Wybierz najdłuższy segment blisko skrajnej współrzędnej (min/max X lub Y)."""
    if not segs:
        return None
    tol = tol_frac * float(img_size)
    scored: List[Tuple[float, Tuple[float, float, float]]] = []
    for seg in segs:
        ln = _segment_to_line(seg[0], seg[1], seg[2], seg[3])
        if ln is None:
            continue
        if coord == 'x':
            cx = 0.5 * (seg[0] + seg[2])
            if abs(cx - target) > tol:
                continue
        else:
            cy = 0.5 * (seg[1] + seg[3])
            if abs(cy - target) > tol:
                continue
        length = float(np.hypot(seg[2] - seg[0], seg[3] - seg[1]))
        dist = abs((seg[0] + seg[2]) * 0.5 - target) if coord == 'x' else abs((seg[1] + seg[3]) * 0.5 - target)
        scored.append((length - 0.15 * dist, ln))
    if not scored:
        for seg in segs:
            ln = _segment_to_line(seg[0], seg[1], seg[2], seg[3])
            if ln is not None:
                length = float(np.hypot(seg[2] - seg[0], seg[3] - seg[1]))
                cx = 0.5 * (seg[0] + seg[2])
                cy = 0.5 * (seg[1] + seg[3])
                dist = abs(cx - target) if coord == 'x' else abs(cy - target)
                scored.append((length - 0.25 * dist, ln))
    if not scored:
        return None
    return max(scored, key=lambda t: t[0])[1]


def _vertical_line_at_x(x: float) -> Tuple[float, float, float]:
    return (1.0, 0.0, -float(x))


def _horizontal_line_at_y(y: float) -> Tuple[float, float, float]:
    return (0.0, 1.0, -float(y))


def pick_absolute_outer_lines(
    h_segs: List[Tuple[float, float, float, float]],
    v_segs: List[Tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
) -> Tuple[Optional[Tuple], Optional[Tuple], Optional[Tuple], Optional[Tuple], float, float, float, float]:
    """
    Skrajne linie: dla pionowych min(x1,x2) / max(x1,x2) (nie środek — unika wewnętrznej siatki).
    Preferuj długie segmenty (≥22% wysokości kadru).
    """
    min_v_len = 0.20 * float(img_h)
    min_h_len = 0.14 * float(img_w)

    v_long = [
        s for s in v_segs
        if float(np.hypot(s[2] - s[0], s[3] - s[1])) >= min_v_len
    ]
    h_long = [
        s for s in h_segs
        if float(np.hypot(s[2] - s[0], s[3] - s[1])) >= min_h_len
    ]
    if len(v_long) < 2:
        v_long = [
            s for s in v_segs
            if float(np.hypot(s[2] - s[0], s[3] - s[1])) >= 0.10 * float(min(img_h, img_w))
        ]
    if len(h_long) < 2:
        h_long = [
            s for s in h_segs
            if float(np.hypot(s[2] - s[0], s[3] - s[1])) >= 0.08 * float(min(img_h, img_w))
        ]
    if len(v_long) < 2 or len(h_long) < 2:
        return None, None, None, None, 0.0, 0.0, 0.0, 0.0

    v_xmins = np.array([min(s[0], s[2]) for s in v_long], dtype=np.float32)
    v_xmaxs = np.array([max(s[0], s[2]) for s in v_long], dtype=np.float32)
    h_ymins = np.array([min(s[1], s[3]) for s in h_long], dtype=np.float32)
    h_ymaxs = np.array([max(s[1], s[3]) for s in h_long], dtype=np.float32)

    # Skrajne linie siatki = min/max (nie percentyl — unika wewnętrznych kratek).
    x_left_line = float(np.min(v_xmins))
    x_right_line = float(np.max(v_xmaxs))
    y_top_line = float(np.min(h_ymins))
    y_bottom_line = float(np.max(h_ymaxs))

    left = _best_line_near_coordinate(v_long, x_left_line, coord='x', img_size=img_w, tol_frac=0.08)
    right = _best_line_near_coordinate(v_long, x_right_line, coord='x', img_size=img_w, tol_frac=0.08)
    top = _best_line_near_coordinate(h_long, y_top_line, coord='y', img_size=img_h, tol_frac=0.08)
    bottom = _best_line_near_coordinate(h_long, y_bottom_line, coord='y', img_size=img_h, tol_frac=0.08)

    if left is None:
        left = _vertical_line_at_x(x_left_line)
    if right is None:
        right = _vertical_line_at_x(x_right_line)
    if top is None:
        top = _horizontal_line_at_y(y_top_line)
    if bottom is None:
        bottom = _horizontal_line_at_y(y_bottom_line)

    return top, bottom, left, right, x_left_line, x_right_line, y_top_line, y_bottom_line


def pick_outer_lines_snapped_grid(
    h_segs: List[Tuple[float, float, float, float]],
    v_segs: List[Tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
) -> Optional[Tuple[Optional[Tuple], Optional[Tuple], Optional[Tuple], Optional[Tuple], float, float, float, float]]:
    """
    9×9 siatka: snap pozycji linii → zewnętrzna obwódka (pół komórki na zewnątrz).
    Skalowalne — krok siatki z mediany odstępów, nie stałe px.
    """
    from module_pose.pnp_panel import PANEL_INTERNAL_LINES_PER_AXIS
    from module_pose.refine_corners import _bounds_from_internal_lines

    n_in = int(PANEL_INTERNAL_LINES_PER_AXIS)
    min_side = float(min(img_w, img_h))

    vert: List[float] = []
    for s in v_segs:
        if float(np.hypot(s[2] - s[0], s[3] - s[1])) >= 0.12 * img_h:
            vert.append(0.5 * (s[0] + s[2]))
    hori: List[float] = []
    for s in h_segs:
        if float(np.hypot(s[2] - s[0], s[3] - s[1])) >= 0.10 * img_w:
            hori.append(0.5 * (s[1] + s[3]))

    if len(vert) < 4 or len(hori) < 4:
        return None

    x_lines = pc.snap_lines_to_grid(vert, expected=n_in, low=0.0, high=float(img_w - 1))
    y_lines = pc.snap_lines_to_grid(hori, expected=n_in, low=0.0, high=float(img_h - 1))
    if len(x_lines) < 4 or len(y_lines) < 4:
        return None

    xb = _bounds_from_internal_lines(np.array(x_lines, dtype=np.float64), float(img_w - 1), n_in)
    yb = _bounds_from_internal_lines(np.array(y_lines, dtype=np.float64), float(img_h - 1), n_in)
    if xb is None or yb is None:
        return None

    xl, xr = xb
    yt, yb = yb
    left = _vertical_line_at_x(xl)
    right = _vertical_line_at_x(xr)
    top = _horizontal_line_at_y(yt)
    bottom = _horizontal_line_at_y(yb)

    span_w = max(1.0, xr - xl)
    span_h = max(1.0, yb - yt)
    if span_w < 0.22 * img_w or span_h < 0.14 * img_h:
        return None
    if span_w > 0.96 * img_w and span_h > 0.96 * img_h:
        return None
    return top, bottom, left, right, xl, xr, yt, yb


def _dark_panel_extent_full(
    image_bgr: np.ndarray,
) -> Optional[Tuple[float, float, float, float]]:
    """Czarna ramka panelu 2:1 — tylko sensowny zasięg (nie cały kadr warsztatu)."""
    h, w = image_bgr.shape[:2]

    try:
        from module_pose.api import detect_corners_black_panel

        quad = detect_corners_black_panel(image_bgr)
        if quad is not None and quad.shape == (4, 2):
            q = quad.astype(np.float32)
            span_w = float(q[:, 0].max() - q[:, 0].min())
            span_h = float(q[:, 1].max() - q[:, 1].min())
            if 0.40 * w <= span_w <= 0.86 * w and 0.22 * h <= span_h <= 0.92 * h:
                return (
                    float(q[:, 0].min()),
                    float(q[:, 0].max()),
                    float(q[:, 1].min()),
                    float(q[:, 1].max()),
                )
    except Exception:
        pass
    return None


def _merge_bounds_with_dark_panel(
    image_bgr: np.ndarray,
    bounds: OuterGridBounds,
    *,
    min_span_frac: float = 0.50,
) -> OuterGridBounds:
    """Jeśli LSD wąski — poszerz X/Y do ciemnej ramki fizycznego panelu."""
    h, w = image_bgr.shape[:2]
    span_w = bounds.x_right_line - bounds.x_left_line
    if span_w >= min_span_frac * w:
        return bounds

    ext = _dark_panel_extent_full(image_bgr)
    if ext is None:
        return bounds

    px0, px1, py0, py1 = ext
    dark_w = px1 - px0
    if dark_w < 0.38 * w or dark_w > 0.86 * w:
        return bounds

    xl = min(bounds.x_left_line, px0)
    xr = max(bounds.x_right_line, px1)
    yt = min(bounds.y_top_line, py0)
    yb = max(bounds.y_bottom_line, py1)

    return OuterGridBounds(
        top=_horizontal_line_at_y(yt),
        bottom=_horizontal_line_at_y(yb),
        left=_vertical_line_at_x(xl),
        right=_vertical_line_at_x(xr),
        x_left_line=xl,
        x_right_line=xr,
        y_top_line=yt,
        y_bottom_line=yb,
        ox=bounds.ox,
        oy=bounds.oy,
        detect_scale=bounds.detect_scale,
        n_v=bounds.n_v,
        n_h=bounds.n_h,
    )


def _prepare_work_image(
    image_bgr: np.ndarray,
    *,
    max_width: int = 1280,
    top_frac: float = 0.07,
    bottom_frac: float = 0.10,
) -> Tuple[np.ndarray, float, float, float]:
    work, ox, oy = strip_camera_overlays(image_bgr, top_frac=top_frac, bottom_frac=bottom_frac)
    h, w = work.shape[:2]
    scale = 1.0
    if w > max_width:
        scale = max_width / float(w)
        work = cv2.resize(
            work,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return work, scale, float(ox), float(oy)


def _map_bounds_to_full(
    bounds: OuterGridBounds,
    full_shape: Tuple[int, int],
) -> OuterGridBounds:
    """Przeskaluj bounds z obrazu roboczego na pełny kadr."""
    inv = 1.0 / max(bounds.detect_scale, 1e-6)
    ox, oy = bounds.ox, bounds.oy

    def map_line(ln: Tuple[float, float, float]) -> Tuple[float, float, float]:
        nx, ny, c = ln
        c_full = c * inv - nx * ox - ny * oy
        return (nx, ny, c_full)

    return OuterGridBounds(
        top=map_line(bounds.top),
        bottom=map_line(bounds.bottom),
        left=map_line(bounds.left),
        right=map_line(bounds.right),
        x_left_line=(bounds.x_left_line * inv + bounds.ox),
        x_right_line=(bounds.x_right_line * inv + bounds.ox),
        y_top_line=(bounds.y_top_line * inv + bounds.oy),
        y_bottom_line=(bounds.y_bottom_line * inv + bounds.oy),
        ox=bounds.ox,
        oy=bounds.oy,
        detect_scale=bounds.detect_scale,
        n_v=bounds.n_v,
        n_h=bounds.n_h,
    )


def _segment_border_distance(
    seg: Tuple[float, float, float, float],
    cw: int,
    ch: int,
) -> float:
    """Odległość segmentu od najbliższej krawędzi cropu (0 = na brzegu)."""
    x1, y1, x2, y2 = seg
    return float(
        min(
            x1, x2, cw - 1 - max(x1, x2),
            y1, y2, ch - 1 - max(y1, y2),
        )
    )


def _filter_outer_rim_segments(
    segs: List[Tuple[float, float, float, float]],
    cw: int,
    ch: int,
    *,
    rim_frac: float = 0.14,
) -> List[Tuple[float, float, float, float]]:
    """Tylko segmenty przy krawędzi cropu — odrzuca wewnętrzną siatkę 10×10."""
    lim = float(rim_frac) * float(min(cw, ch))
    out = [s for s in segs if _segment_border_distance(s, cw, ch) <= lim]
    return out if len(out) >= 2 else segs


def _expand_quad_to_blue_inset(
    quad: np.ndarray,
    blue_roi: Tuple[int, int, int, int],
    *,
    inset_frac: float = 0.028,
    pull_frac: float = 0.92,
) -> np.ndarray:
    """
    Dociągnij rogi żółtego do zewnętrznej obwódki niebieskiego ROI (nie wewnętrznych linii).
    Częściowe przesunięcie (pull_frac) zachowuje trapez perspektywiczny.
    """
    bx0, by0, bx1, by1 = (float(blue_roi[0]), float(blue_roi[1]), float(blue_roi[2]), float(blue_roi[3]))
    bw = max(1.0, bx1 - bx0)
    bh = max(1.0, by1 - by0)
    ins = float(inset_frac)
    tx0 = bx0 + ins * bw
    ty0 = by0 + ins * bh
    tx1 = bx1 - ins * bw
    ty1 = by1 - ins * bh
    pull = float(np.clip(pull_frac, 0.5, 1.0))

    q = pc.order_points(quad.astype(np.float32).copy())

    def _pull_val(cur: float, target: float, inward: bool) -> float:
        if not inward:
            return cur
        return float(cur + pull * (target - cur))

    q[0, 0] = _pull_val(float(q[0, 0]), tx0, float(q[0, 0]) > tx0)
    q[0, 1] = _pull_val(float(q[0, 1]), ty0, float(q[0, 1]) > ty0)
    q[1, 0] = _pull_val(float(q[1, 0]), tx1, float(q[1, 0]) < tx1)
    q[1, 1] = _pull_val(float(q[1, 1]), ty0, float(q[1, 1]) > ty0)
    q[2, 0] = _pull_val(float(q[2, 0]), tx1, float(q[2, 0]) < tx1)
    q[2, 1] = _pull_val(float(q[2, 1]), ty1, float(q[2, 1]) < ty1)
    q[3, 0] = _pull_val(float(q[3, 0]), tx0, float(q[3, 0]) > tx0)
    q[3, 1] = _pull_val(float(q[3, 1]), ty1, float(q[3, 1]) < ty1)
    return pc.order_points(q)


def _bright_line_mask(crop_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Białe linie siatki: pełna maska + morfologia pozioma / pionowa."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    dark = (gray < 115) | ((v < 150) & (s > 40))
    bright = (gray > 118) & dark
    base = (bright.astype(np.uint8)) * 255
    base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    ch, cw = base.shape[:2]
    kw = max(21, int(round(0.07 * cw)))
    kh = max(21, int(round(0.07 * ch)))
    h_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    v_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    h_mask = cv2.morphologyEx(base, cv2.MORPH_OPEN, h_ker)
    v_mask = cv2.morphologyEx(base, cv2.MORPH_OPEN, v_ker)
    return base, h_mask, v_mask


def _max_run_frac(line_1d: np.ndarray) -> float:
    best = cur = 0
    for v in line_1d:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return float(best) / max(1, len(line_1d))


def _column_is_grid_line(
    col_mask: np.ndarray,
    *,
    min_vert_span_frac: float = 0.14,
    max_blob_frac: float = 0.55,
) -> bool:
    """Odrzuć biały prostokąt (kartę): za dużo jasnych pikseli w kolumnie."""
    h = col_mask.shape[0]
    if h < 8:
        return False
    vert_span = _max_run_frac(col_mask > 0)
    fill = float((col_mask > 0).mean())
    if fill > max_blob_frac:
        return False
    return vert_span >= min_vert_span_frac


def _scan_x_near_anchor(
    line_mask: np.ndarray,
    anchor_x: float,
    *,
    from_left: bool = True,
    margin_frac: float = 0.02,
    hit_frac: float = 0.028,
    min_span_frac: float = 0.06,
    max_delta_frac: float = 0.22,
) -> Optional[float]:
    """Szukaj pionowej linii siatki blisko anchor_x (np. BL przy TL, nie biała karta)."""
    h, w = line_mask.shape[:2]
    margin = max(2, int(margin_frac * w))
    max_delta = max(40.0, float(max_delta_frac) * w)
    xs = range(margin, w - margin) if from_left else range(w - margin - 1, margin - 1, -1)
    hits: List[float] = []
    for x in xs:
        if (line_mask[:, int(x)] > 0).mean() < hit_frac:
            continue
        if _max_run_frac(line_mask[:, int(x)] > 0) < min_span_frac:
            continue
        if not _column_is_grid_line(line_mask[:, int(x)]):
            continue
        if abs(float(x) - float(anchor_x)) > max_delta:
            continue
        hits.append(float(x))
    if not hits:
        return None
    return min(hits, key=lambda xv: abs(xv - float(anchor_x)))


def _scan_x_from_side(
    line_mask: np.ndarray,
    *,
    from_left: bool,
    margin_frac: float = 0.02,
    hit_frac: float = 0.035,
    min_span_frac: float = 0.22,
    require_grid_line: bool = False,
) -> Optional[float]:
    h, w = line_mask.shape[:2]
    margin = max(2, int(margin_frac * w))
    col_hit = (line_mask > 0).mean(axis=0)
    xs = range(margin, w - margin) if from_left else range(w - margin - 1, margin - 1, -1)
    for x in xs:
        if col_hit[x] < hit_frac:
            continue
        if _max_run_frac(line_mask[:, x] > 0) < min_span_frac:
            continue
        if require_grid_line and not _column_is_grid_line(line_mask[:, x]):
            continue
        return float(x)
    return None


def _scan_y_from_side(
    line_mask: np.ndarray,
    *,
    from_top: bool,
    margin_frac: float = 0.02,
    hit_frac: float = 0.035,
    min_span_frac: float = 0.012,
) -> Optional[float]:
    h, w = line_mask.shape[:2]
    margin = max(2, int(margin_frac * h))
    row_hit = (line_mask > 0).mean(axis=1)
    ys = range(margin, h - margin) if from_top else range(h - margin - 1, margin - 1, -1)
    for y in ys:
        if row_hit[y] >= hit_frac and _max_run_frac(line_mask[y, :] > 0) >= min_span_frac:
            return float(y)
    return None


def detect_yellow_quad_border_scan(
    image_bgr: np.ndarray,
    blue_roi: Optional[Tuple[int, int, int, int]],
    *,
    inset_px: float = 4.0,
    search_roi: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[np.ndarray]:
    """
    Zewnętrzne rogi siatki: skan od brzegu obszaru (domyślnie poszerzony względem niebieskiego).
    Niebieski ROI nie przycina detekcji — tylko luźna referencja na końcu.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    fh, fw = image_bgr.shape[:2]
    if search_roi is not None:
        sx0, sy0, sx1, sy1 = (int(search_roi[0]), int(search_roi[1]), int(search_roi[2]), int(search_roi[3]))
    elif blue_roi is not None:
        from release.safe_blue_roi import expand_roi

        sx0, sy0, sx1, sy1 = expand_roi(blue_roi, (fh, fw), margin_frac=0.12, pad_px=32)
    else:
        sx0, sy0, sx1, sy1 = 0, 0, fw - 1, fh - 1
    bx0, by0, bx1, by1 = sx0, sy0, sx1, sy1
    if bx1 <= bx0 + 40 or by1 <= by0 + 40:
        return None

    crop = image_bgr[by0:by1, bx0:bx1].copy()
    ch, cw = crop.shape[:2]
    _base, h_mask, v_mask = _bright_line_mask(crop)

    scan_h = _base
    y_top = _scan_y_from_side(
        scan_h, from_top=True, margin_frac=0.008, hit_frac=0.042, min_span_frac=0.012,
    )
    y_bot = _scan_y_from_side(
        scan_h, from_top=False, margin_frac=0.008, hit_frac=0.042, min_span_frac=0.012,
    )
    if y_top is None or y_bot is None:
        return None

    band = max(14.0, 0.08 * (y_bot - y_top))
    top_strip = _base[int(y_top) : int(min(ch, y_top + band)), :]
    bot_strip = _base[int(max(0, y_bot - band)) : int(y_bot), :]

    x_tl = _scan_x_from_side(top_strip, from_left=True, margin_frac=0.01, hit_frac=0.028, min_span_frac=0.06)
    x_tr = _scan_x_from_side(top_strip, from_left=False, margin_frac=0.01, hit_frac=0.028, min_span_frac=0.06)
    x_br = _scan_x_from_side(bot_strip, from_left=False, margin_frac=0.01, hit_frac=0.028, min_span_frac=0.06)

    if x_tl is None or x_tr is None or x_br is None:
        return None

    x_bl = None
    if x_tl is not None:
        x_bl = _scan_x_near_anchor(
            bot_strip, float(x_tl), from_left=True, hit_frac=0.024, max_delta_frac=0.09,
        )
    if x_bl is None and x_tl is not None:
        x_bl = float(x_tl)

    if x_bl is None:
        return None

    if x_tl is not None and float(x_bl) < float(x_tl) - 18.0:
        x_bl = float(x_tl)
    if x_tl is not None and x_tr is not None:
        top_w = max(40.0, float(x_tr) - float(x_tl))
        bl_max = float(x_tl) + 0.14 * top_w
        if float(x_bl) > bl_max:
            x_bl = bl_max

    ins = float(inset_px)
    tl = (float(x_tl) + ins + bx0, float(y_top) + ins + by0)
    tr = (float(x_tr) - ins + bx0, float(y_top) + ins + by0)
    br = (float(x_br) - ins + bx0, float(y_bot) - ins + by0)
    bl = (float(x_bl) + ins + bx0, float(y_bot) - ins + by0)

    quad = pc.order_points(np.array([tl, tr, br, bl], dtype=np.float32))
    quad[:, 0] = np.clip(quad[:, 0], 2.0, float(fw - 1))
    quad[:, 1] = np.clip(quad[:, 1], 2.0, float(fh - 1))

    if blue_roi is not None:
        quad = _expand_quad_to_blue_inset(quad, blue_roi, inset_frac=0.012, pull_frac=0.38)

    search_area = float((bx1 - bx0) * (by1 - by0))
    area = float(cv2.contourArea(quad.reshape(1, 4, 2).astype(np.float32)))
    if area < 0.48 * search_area or area > 0.96 * search_area:
        return None
    top_w = float(quad[1, 0] - quad[0, 0])
    bot_w = float(quad[2, 0] - quad[3, 0])
    if top_w < 0.32 * (bx1 - bx0) or bot_w < 0.32 * (bx1 - bx0):
        return None
    return quad


def detect_yellow_quad_scalable(
    image_bgr: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], str]:
    """
    Żółty trapez + niebieski ROI wyłącznie z geometrii siatki (LSD, ułamki kadru).
    Bez kalibracji pikselowej i bez zależności żółty→crop niebieskiego.
    """
    quad, roi, src, _bounds = detect_panel_quad_and_roi(image_bgr)
    return quad, roi, src


def _ring_masked_gray(
    crop_bgr: np.ndarray,
    *,
    inner_clear_frac: float = 0.88,
) -> np.ndarray:
    """Aktywny pierścień ~6% z każdej strony — bez wewnętrznych kratek."""
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mx = int(round(0.5 * float(inner_clear_frac) * w))
    my = int(round(0.5 * float(inner_clear_frac) * h))
    mx = max(4, min(mx, w // 2 - 2))
    my = max(4, min(my, h // 2 - 2))
    mask[my : h - my, mx : w - mx] = 0
    out = enhanced.copy()
    out[mask == 0] = 0
    return out


def detect_yellow_quad_ring_lsd(
    image_bgr: np.ndarray,
    blue_roi: Tuple[int, int, int, int],
    *,
    inner_clear_frac: float = 0.88,
    min_seg_len_frac: float = 0.08,
    rim_frac: float = 0.14,
    blue_inset_frac: float = 0.028,
) -> Optional[np.ndarray]:
    """
    Żółty trapez: preferuj skan od brzegu ROI; fallback LSD w pierścieniu.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    scan_q = detect_yellow_quad_border_scan(image_bgr, blue_roi, inset_px=5.0)
    if scan_q is not None:
        return scan_q
    bx0, by0, bx1, by1 = (int(blue_roi[0]), int(blue_roi[1]), int(blue_roi[2]), int(blue_roi[3]))
    if bx1 <= bx0 + 40 or by1 <= by0 + 40:
        return None

    crop = image_bgr[by0:by1, bx0:bx1].copy()
    ch, cw = crop.shape[:2]
    ring_gray = _ring_masked_gray(crop, inner_clear_frac=inner_clear_frac)

    h_segs, v_segs = _lsd_hv_segments(ring_gray, min_len_frac=min_seg_len_frac)
    if len(v_segs) < 2 or len(h_segs) < 2:
        hv, vv = _hough_hv_segments(ring_gray, min_len_frac=0.14)
        h_segs = h_segs + hv
        v_segs = v_segs + vv
    if len(v_segs) < 2 or len(h_segs) < 2:
        return None

    v_segs = _filter_outer_rim_segments(v_segs, cw, ch, rim_frac=rim_frac)
    h_segs = _filter_outer_rim_segments(h_segs, cw, ch, rim_frac=rim_frac)

    rim_lim = float(rim_frac) * float(min(cw, ch))
    top_band = [s for s in h_segs if _segment_border_distance(s, cw, ch) <= rim_lim and min(s[1], s[3]) < 0.22 * ch]
    bot_band = [s for s in h_segs if _segment_border_distance(s, cw, ch) <= rim_lim and max(s[1], s[3]) > 0.78 * ch]
    if top_band:
        y_top = float(min(min(s[1], s[3]) for s in top_band))
    else:
        y_top = float(np.percentile([min(s[1], s[3]) for s in h_segs], 3))
    if bot_band:
        y_bottom = float(max(max(s[1], s[3]) for s in bot_band))
    else:
        y_bottom = float(np.percentile([max(s[1], s[3]) for s in h_segs], 97))
    span_h = max(1.0, y_bottom - y_top)
    band = max(14.0, 0.10 * span_h)

    def _v_x_extents(y_a: float, y_b: float) -> Tuple[float, float]:
        ya, yb = float(y_a), float(y_b)
        xs_lo: List[float] = []
        xs_hi: List[float] = []
        for s in v_segs:
            my = 0.5 * (s[1] + s[3])
            if my < ya or my > yb:
                continue
            xs_lo.append(min(s[0], s[2]))
            xs_hi.append(max(s[0], s[2]))
        if len(xs_lo) < 2:
            return float('nan'), float('nan')
        return float(np.percentile(xs_lo, 3)), float(np.percentile(xs_hi, 97))

    x_tl, x_tr = _v_x_extents(y_top, y_top + band)
    x_bl, x_br = _v_x_extents(y_bottom - band, y_bottom)
    if not all(np.isfinite(v) for v in (x_tl, x_tr, x_bl, x_br)):
        v_xmins = [min(s[0], s[2]) for s in v_segs]
        v_xmaxs = [max(s[0], s[2]) for s in v_segs]
        x_tl = x_bl = float(np.percentile(v_xmins, 3))
        x_tr = x_br = float(np.percentile(v_xmaxs, 97))

    span_w = max(1.0, max(x_tr, x_br) - min(x_tl, x_bl))
    if span_w < 0.35 * cw or span_h < 0.35 * ch:
        return None
    if span_w > 0.98 * cw and span_h > 0.98 * ch:
        return None

    top_ln = _horizontal_line_at_y(y_top)
    bot_ln = _horizontal_line_at_y(y_bottom)
    left_top_ln = _vertical_line_at_x(x_tl)
    right_top_ln = _vertical_line_at_x(x_tr)
    left_bot_ln = _vertical_line_at_x(x_bl)
    right_bot_ln = _vertical_line_at_x(x_br)

    tl = _intersect_lines(top_ln, left_top_ln)
    tr = _intersect_lines(top_ln, right_top_ln)
    br = _intersect_lines(bot_ln, right_bot_ln)
    bl = _intersect_lines(bot_ln, left_bot_ln)
    if any(p is None for p in (tl, tr, br, bl)):
        return None

    quad = pc.order_points(np.array([tl, tr, br, bl], dtype=np.float32))
    quad[:, 0] += float(bx0)
    quad[:, 1] += float(by0)

    fh, fw = image_bgr.shape[:2]
    quad[:, 0] = np.clip(quad[:, 0], float(bx0), float(fw - 1))
    quad[:, 1] = np.clip(quad[:, 1], float(by0), float(fh - 1))

    quad = _expand_quad_to_blue_inset(quad, blue_roi, inset_frac=blue_inset_frac)

    area = float(cv2.contourArea(quad.reshape(1, 4, 2).astype(np.float32)))
    roi_area = float((bx1 - bx0) * (by1 - by0))
    if area < 0.62 * roi_area or area > 0.96 * roi_area:
        return None
    return quad


def detect_outer_grid_bounds(
    image_bgr: np.ndarray,
    *,
    max_width: int = 1280,
) -> Optional[OuterGridBounds]:
    """LSD (+ Hough) na lekko zmniejszonym kadrze → skrajne linie siatki."""
    work, scale, ox, oy = _prepare_work_image(image_bgr, max_width=max_width)
    h, w = work.shape[:2]

    mask = _white_grid_mask(work)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    h_segs, v_segs = _lsd_hv_segments(enhanced, min_len_frac=0.035)
    if len(v_segs) < 4:
        hv, vv = _lsd_hv_segments(mask, min_len_frac=0.03)
        h_segs = h_segs + hv
        v_segs = v_segs + vv
    if len(v_segs) < 4:
        hv, vv = _hough_hv_segments(enhanced, min_len_frac=0.10)
        h_segs = h_segs + hv
        v_segs = v_segs + vv

    picked = pick_outer_lines_snapped_grid(h_segs, v_segs, w, h)
    if picked is None:
        picked = pick_absolute_outer_lines(h_segs, v_segs, w, h)
    if picked is None:
        return None
    top, bottom, left, right, xl, xr, yt, yb = picked
    if top is None or bottom is None or left is None or right is None:
        return None

    span_w = max(1.0, xr - xl)
    span_h = max(1.0, yb - yt)
    if span_w < 0.14 * w or span_h < 0.10 * h:
        return None
    if span_w > 0.98 * w and span_h > 0.98 * h:
        return None

    b = OuterGridBounds(
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        x_left_line=xl,
        x_right_line=xr,
        y_top_line=yt,
        y_bottom_line=yb,
        ox=ox,
        oy=oy,
        detect_scale=scale,
        n_v=len(v_segs),
        n_h=len(h_segs),
    )
    b_full = _map_bounds_to_full(b, image_bgr.shape[:2])
    return _merge_bounds_with_dark_panel(image_bgr, b_full)


def quad_from_outer_grid_bounds(
    bounds: OuterGridBounds,
    image_shape: Tuple[int, int],
    *,
    cell_margin_frac: float = 0.008,
) -> Optional[np.ndarray]:
    """
    Trapez z przecięć skrajnych linii; lekki inset do wnętrza panelu (ułamek komórki).
    """
    h, w = image_shape[:2]
    tl = _intersect_lines(bounds.top, bounds.left)
    tr = _intersect_lines(bounds.top, bounds.right)
    br = _intersect_lines(bounds.bottom, bounds.right)
    bl = _intersect_lines(bounds.bottom, bounds.left)
    if any(p is None for p in (tl, tr, br, bl)):
        return None

    quad = pc.order_points(np.array([tl, tr, br, bl], dtype=np.float32))
    span_w = max(1.0, float(bounds.x_right_line - bounds.x_left_line))
    span_h = max(1.0, float(bounds.y_bottom_line - bounds.y_top_line))
    mx = cell_margin_frac * span_w
    my = cell_margin_frac * span_h

    out = quad.copy()
    # Lekki inset do wnętrza (bliżej żółtej obwódki siatki, nie na zewnątrz kadru)
    out[0, 0] += mx
    out[0, 1] += my
    out[1, 0] -= mx
    out[1, 1] += my
    out[2, 0] -= mx
    out[2, 1] -= my
    out[3, 0] += mx
    out[3, 1] -= my

    out[:, 0] = np.clip(out[:, 0], 0.0, float(w - 1))
    out[:, 1] = np.clip(out[:, 1], 0.0, float(h - 1))
    out = pc.order_points(out)

    area = float(cv2.contourArea(out.reshape(1, 4, 2).astype(np.float32)))
    if area < 0.03 * h * w or area > 0.88 * h * w:
        return None
    q = pc.order_points(out)
    top_w = float(np.linalg.norm(q[1] - q[0]))
    bot_w = float(np.linalg.norm(q[2] - q[3]))
    left_h = float(np.linalg.norm(q[3] - q[0]))
    right_h = float(np.linalg.norm(q[2] - q[1]))
    est_w = 0.5 * (top_w + bot_w)
    est_h = 0.5 * (left_h + right_h)
    asp = max(est_w, est_h) / max(1.0, min(est_w, est_h))
    if not (0.85 <= asp <= 4.5):
        return None
    return out


def roi_bbox_from_quad(
    quad: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    margin_frac: float = 0.04,
    margin_top_frac: float = 0.08,
    margin_bottom_frac: float = 0.08,
    margin_right_frac: float = 0.012,
    margin_right_max_px: float = 22.0,
    expand_top_frac: float = 0.05,
    expand_bottom_frac: float = 0.05,
    expand_right_frac: float = 0.0,
) -> Tuple[int, int, int, int]:
    """
    Niebieski ROI = bbox(żółty trapez) + margines.

    Prawo: mały stały pad (px), nie % szerokości — unika „zjadania” tła lub ucinania panelu.
    """
    h, w = image_shape[:2]
    q = pc.order_points(quad.astype(np.float32))
    span_h = max(1.0, float(np.max(q[:, 1]) - np.min(q[:, 1])))
    span_w = max(1.0, float(np.max(q[:, 0]) - np.min(q[:, 0])))
    if expand_top_frac > 0.0 and span_h > 8.0:
        lift = float(expand_top_frac) * span_h
        top_idx = np.argsort(q[:, 1])[:2]
        q = q.copy()
        q[top_idx, 1] = np.maximum(0.0, q[top_idx, 1] - lift)
    if expand_bottom_frac > 0.0 and span_h > 8.0:
        drop = float(expand_bottom_frac) * span_h
        bot_idx = np.argsort(q[:, 1])[-2:]
        q = q.copy()
        q[bot_idx, 1] = np.minimum(float(h - 1), q[bot_idx, 1] + drop)
    if expand_right_frac > 0.0 and span_w > 8.0:
        extend = float(expand_right_frac) * span_w
        right_idx = np.argsort(q[:, 0])[-2:]
        q = q.copy()
        q[right_idx, 0] = np.minimum(float(w - 1), q[right_idx, 0] + extend)
    x0, y0 = float(q[:, 0].min()), float(q[:, 1].min())
    x1, y1 = float(q[:, 0].max()), float(q[:, 1].max())
    span_w = max(1.0, x1 - x0)
    span_h = max(1.0, y1 - y0)
    quad_x1 = x1
    x0 -= margin_frac * span_w
    right_pad = min(float(margin_right_frac) * span_w, float(margin_right_max_px))
    x1 = quad_x1 + right_pad
    y0 -= float(margin_top_frac) * span_h
    y1 += float(margin_bottom_frac) * span_h
    bx0 = int(np.floor(np.clip(x0, 0, w - 2)))
    by0 = int(np.floor(np.clip(y0, 0, h - 2)))
    bx1 = int(np.ceil(np.clip(x1, bx0 + 2, w - 1)))
    by1 = int(np.ceil(np.clip(y1, by0 + 2, h - 1)))
    return bx0, by0, bx1, by1


def quad_width_frac(quad: np.ndarray, image_shape: Tuple[int, int]) -> float:
    h, w = image_shape[:2]
    q = quad.astype(np.float32)
    return float(q[:, 0].max() - q[:, 0].min()) / max(1.0, float(w))


def detect_panel_quad_and_roi(
    image_bgr: np.ndarray,
    *,
    roi_margin_frac: float = 0.04,
    min_width_frac: float = 0.42,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], str, Optional[OuterGridBounds]]:
    """
    Główna ścieżka live: linie → trapez → ROI z quada (ten sam czworokąt).
    """
    h, w = image_bgr.shape[:2]
    bounds = detect_outer_grid_bounds(image_bgr)
    quad: Optional[np.ndarray] = None
    src = 'none'

    if bounds is not None:
        quad = quad_from_outer_grid_bounds(bounds, (h, w))
        if quad is not None:
            from module_pose.refine_corners import refine_panel_corners_uniform_grid

            refined = refine_panel_corners_uniform_grid(image_bgr, quad)
            if refined is not None:
                quad = refined
            src = 'grid_outer_lsd'

    if quad is None or quad_width_frac(quad, (h, w)) < min_width_frac:
        ext = _dark_panel_extent_full(image_bgr)
        if ext is not None and 0.40 * w <= (ext[1] - ext[0]) <= 0.86 * w:
            px0, px1, py0, py1 = ext
            dark_bounds = OuterGridBounds(
                top=_horizontal_line_at_y(py0),
                bottom=_horizontal_line_at_y(py1),
                left=_vertical_line_at_x(px0),
                right=_vertical_line_at_x(px1),
                x_left_line=px0,
                x_right_line=px1,
                y_top_line=py0,
                y_bottom_line=py1,
                ox=0.0,
                oy=0.0,
                detect_scale=1.0,
                n_v=0,
                n_h=0,
            )
            q_dark = quad_from_outer_grid_bounds(dark_bounds, (h, w), cell_margin_frac=0.02)
            if q_dark is not None and (
                quad is None or quad_width_frac(q_dark, (h, w)) > quad_width_frac(quad, (h, w))
            ):
                quad = q_dark
                bounds = dark_bounds
                src = 'dark_panel_frame'

    if quad is None:
        return None, None, 'grid_lines_fail', bounds

    roi = roi_bbox_from_quad(quad, (h, w), margin_frac=roi_margin_frac)
    return quad, roi, src, bounds


def external_coverage_penalty(
    quad: np.ndarray,
    bounds: OuterGridBounds,
    *,
    tol_px: float = 12.0,
) -> Tuple[float, float, float, float]:
    """
    Kara za ucinanie kolumn/wierszy względem skrajnych linii.
    Zwraca (left_gap, right_gap, top_gap, bottom_gap) w px (tylko dodatnie = źle).
    """
    q = quad.astype(np.float32)
    qx0, qy0 = float(q[:, 0].min()), float(q[:, 1].min())
    qx1, qy1 = float(q[:, 0].max()), float(q[:, 1].max())
    cy = 0.5 * (qy0 + qy1)
    cx = 0.5 * (qx0 + qx1)

    x_left_ref = _line_x_at_y(bounds.left, cy)
    if not np.isfinite(x_left_ref):
        x_left_ref = bounds.x_left_line
    x_right_ref = _line_x_at_y(bounds.right, cy)
    if not np.isfinite(x_right_ref):
        x_right_ref = bounds.x_right_line
    y_top_ref = _line_y_at_x(bounds.top, cx)
    if not np.isfinite(y_top_ref):
        y_top_ref = bounds.y_top_line
    y_bot_ref = _line_y_at_x(bounds.bottom, cx)
    if not np.isfinite(y_bot_ref):
        y_bot_ref = bounds.y_bottom_line

    left_gap = max(0.0, qx0 - x_left_ref - tol_px)
    right_gap = max(0.0, x_right_ref - qx1 - tol_px)
    top_gap = max(0.0, qy0 - y_top_ref - tol_px)
    bottom_gap = max(0.0, y_bot_ref - qy1 - tol_px)
    return left_gap, right_gap, top_gap, bottom_gap


def external_penalty_score(
    quad: np.ndarray,
    bounds: Optional[OuterGridBounds],
    *,
    tol_px: float = 12.0,
) -> float:
    """Dodatek do rank score — duża kara za lewy brzeg wewnątrz siatki."""
    if bounds is None:
        return 0.0
    lg, rg, tg, bg = external_coverage_penalty(quad, bounds, tol_px=tol_px)
    # Lewy brzeg najważniejszy (sesje live 20260520)
    return lg * 4.5 + rg * 2.2 + tg * 1.8 + bg * 1.8
