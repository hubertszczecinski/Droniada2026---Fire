"""Rogowanie panelu pod kamerę (Continuity / duże rozdzielczości)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pipeline_competition as pc

from module_pose.api import (
    _black_panel_interior_score,
    _grid_structure_score,
    _warp_panel_candidate,
    canonicalize_corners_by_white_anchor,
    default_intrinsics,
    detect_corners_black_panel,
    detect_corners_panel,
    strip_camera_overlays,
)
from module_pose.pnp_panel import solve_panel_pose
from release.alignment_pipelines import (
    _expand_quad,
    pipeline_dark_blob,
    pipeline_hsv_panel,
    pipeline_trapezoid,
)

_MAX_LIVE_REPROJ_PX = 22.0
_MAX_LIVE_REPROJ_FALLBACK_PX = 55.0


def _segment_to_line(x1: float, y1: float, x2: float, y2: float) -> Optional[Tuple[float, float, float]]:
    dx, dy = (x2 - x1, y2 - y1)
    length = float(np.hypot(dx, dy))
    if length < 1e-3:
        return None
    nx, ny = (-dy / length, dx / length)
    c = -(nx * x1 + ny * y1)
    return (float(nx), float(ny), float(c))


def _intersect_lines(l1: Tuple[float, float, float], l2: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
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


def _lsd_hv_segments(gray: np.ndarray, min_len_frac: float = 0.06) -> Tuple[List[Tuple], List[Tuple]]:
    from module_geom.lines_vp import _classify_segment

    h_img, w_img = gray.shape[:2]
    min_len = min_len_frac * float(min(h_img, w_img))
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    h_segs: List[Tuple] = []
    v_segs: List[Tuple] = []
    if lines is None:
        return h_segs, v_segs
    for seg in lines:
        x1, y1, x2, y2 = (float(seg[0][0]), float(seg[0][1]), float(seg[0][2]), float(seg[0][3]))
        if float(np.hypot(x2 - x1, y2 - y1)) < min_len:
            continue
        cls = _classify_segment(x1, y1, x2, y2)
        if cls == 'h':
            h_segs.append((x1, y1, x2, y2))
        elif cls == 'v':
            v_segs.append((x1, y1, x2, y2))
    return h_segs, v_segs


def _pick_extreme_border_lines(
    h_segs: List[Tuple[float, float, float, float]],
    v_segs: List[Tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
) -> Tuple[Optional[Tuple], Optional[Tuple], Optional[Tuple], Optional[Tuple]]:
    """Zewnętrzne linie siatki: długie segmenty + percentyle (bez VP)."""
    if len(h_segs) < 2 or len(v_segs) < 2:
        return None, None, None, None

    def to_line(seg: Tuple[float, float, float, float]) -> Optional[Tuple[float, float, float]]:
        return _segment_to_line(seg[0], seg[1], seg[2], seg[3])

    # Na kadrze z perspektywą „poziome” linie siatki mają krótki zasięg w X — filtruj po długości segmentu.
    min_len = 0.14 * float(min(img_h, img_w))

    h_with_y = []
    for seg in h_segs:
        if float(np.hypot(seg[2] - seg[0], seg[3] - seg[1])) < min_len:
            continue
        ln = to_line(seg)
        if ln is not None:
            h_with_y.append(((seg[1] + seg[3]) * 0.5, ln))
    v_with_x = []
    for seg in v_segs:
        if float(np.hypot(seg[2] - seg[0], seg[3] - seg[1])) < min_len:
            continue
        ln = to_line(seg)
        if ln is not None:
            v_with_x.append(((seg[0] + seg[2]) * 0.5, ln))

    if len(h_with_y) < 2 or len(v_with_x) < 2:
        return None, None, None, None

    ys = np.array([t[0] for t in h_with_y], dtype=np.float32)
    xs = np.array([t[0] for t in v_with_x], dtype=np.float32)
    y_lo, y_hi = float(np.percentile(ys, 8)), float(np.percentile(ys, 92))
    x_lo, x_hi = float(np.percentile(xs, 8)), float(np.percentile(xs, 92))

    top = min((t for t in h_with_y if t[0] <= y_lo + 1.0), key=lambda t: t[0], default=h_with_y[0])[1]
    bottom = max((t for t in h_with_y if t[0] >= y_hi - 1.0), key=lambda t: t[0], default=h_with_y[-1])[1]
    left = min((t for t in v_with_x if t[0] <= x_lo + 1.0), key=lambda t: t[0], default=v_with_x[0])[1]
    right = max((t for t in v_with_x if t[0] >= x_hi - 1.0), key=lambda t: t[0], default=v_with_x[-1])[1]
    return top, bottom, left, right


def _quad_from_lsd_outer_grid(work_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Trapezoid z zewnętrznych linii siatki (LSD) na obrazie roboczym."""
    from module_geom.lines_vp import _white_grid_mask

    h, w = work_bgr.shape[:2]
    mask = _white_grid_mask(work_bgr)
    gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    h_segs, v_segs = _lsd_hv_segments(clahe.apply(gray), min_len_frac=0.05)
    if len(h_segs) < 6:
        h_segs, v_segs = _lsd_hv_segments(mask, min_len_frac=0.04)
    top, bottom, left, right = _pick_extreme_border_lines(h_segs, v_segs, w, h)
    if top is None or bottom is None or left is None or right is None:
        return None
    tl = _intersect_lines(top, left)
    tr = _intersect_lines(top, right)
    br = _intersect_lines(bottom, right)
    bl = _intersect_lines(bottom, left)
    if any(p is None for p in (tl, tr, br, bl)):
        return None
    quad = pc.order_points(np.array([tl, tr, br, bl], dtype=np.float32))
    area = float(cv2.contourArea(quad.reshape(1, 4, 2)))
    if area < 0.04 * h * w or area > 0.62 * h * w:
        return None
    quad[:, 0] = np.clip(quad[:, 0], 0.0, float(w - 1))
    quad[:, 1] = np.clip(quad[:, 1], 0.0, float(h - 1))
    return quad


def _grid_trapezoid_from_white_profile(work_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Trapezoid z chmury białych linii w ROI profilu (bez maski HSV panelu)."""
    from module_pose.api import _quad_area_ratio, _quad_aspect, _rough_quad_from_white_hlines

    profile = _rough_quad_from_white_hlines(work_bgr)
    if profile is None:
        return None
    h, w = work_bgr.shape[:2]
    x0, y0, x1, y1 = (
        float(profile[:, 0].min()),
        float(profile[:, 1].min()),
        float(profile[:, 0].max()),
        float(profile[:, 1].max()),
    )
    pad_x, pad_y = 0.08 * (x1 - x0), 0.05 * (y1 - y0)
    rx0 = int(max(0.0, x0 - pad_x))
    rx1 = int(min(float(w - 1), x1 + pad_x))
    ry0 = int(max(0.0, y0 - pad_y))
    ry1 = int(min(float(h - 1), y1 + pad_y))
    roi = work_bgr[ry0:ry1 + 1, rx0:rx1 + 1]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = (((gray > 105) & (hsv[:, :, 1] < 125)) | (gray > 155)).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white = cv2.dilate(white, np.ones((5, 5), np.uint8), iterations=1)
    pts = cv2.findNonZero(white)
    if pts is None or len(pts) < 80:
        return None
    pts2 = pts.reshape(-1, 2).astype(np.float32)
    qx0, qx1 = np.percentile(pts2[:, 0], [4.0, 96.0])
    qy0, qy1 = np.percentile(pts2[:, 1], [4.0, 96.0])
    pts2 = pts2[
        (pts2[:, 0] >= qx0)
        & (pts2[:, 0] <= qx1)
        & (pts2[:, 1] >= qy0)
        & (pts2[:, 1] <= qy1)
    ]
    if pts2.shape[0] < 50:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(pts2.reshape(-1, 1, 2))).astype(np.float32)
    box[:, 0] += float(rx0)
    box[:, 1] += float(ry0)
    box = pc.order_points(box)
    asp = _quad_aspect(box)
    area_ratio = _quad_area_ratio(box, float(h * w))
    if not (1.2 <= asp <= 3.6 and 0.05 <= area_ratio <= 0.62):
        return None
    return box


_LIVE_MAX_DETECT_WIDTH = 1280
# white_grid / gather_quad na live dają profil na cały kadr — wyłączone (sesja 20260520_163052)
_LIVE_BLOCKED_LABELS = frozenset({'white_grid', 'gather_quad', 'img_panel'})
_PREFERRED_LABELS = (
    'lsd_outer_grid',
    'white_trap',
    'trapezoid',
    'hsv_panel',
    'img_panel',
    'black_panel',
    'dark_blob',
)


def _prepare_detection_image(
    image_bgr: np.ndarray,
    *,
    max_width: int = _LIVE_MAX_DETECT_WIDTH,
    top_frac: float = 0.07,
    bottom_frac: float = 0.10,
) -> Tuple[np.ndarray, float, float, float]:
    """Obetnij OSD kamery i przeskaluj do max_width (detekcja jak na dataset ~1k)."""
    work, ox, oy = strip_camera_overlays(image_bgr, top_frac=top_frac, bottom_frac=bottom_frac)
    h, w = work.shape[:2]
    scale = 1.0
    if w > max_width:
        scale = max_width / float(w)
        work = cv2.resize(work, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return work, scale, ox, oy


def _quad_to_full_image(
    quad: np.ndarray,
    *,
    scale: float,
    ox: float,
    oy: float,
) -> np.ndarray:
    q = quad.astype(np.float32).copy()
    if scale > 1e-6 and abs(scale - 1.0) > 1e-4:
        q /= scale
    q[:, 0] += ox
    q[:, 1] += oy
    return q


def _is_axis_aligned_profile_box(quad: np.ndarray, img_w: int, img_h: int) -> bool:
    """Profil H/V na całą szerokość — typowy błąd white_grid na Continuity."""
    """Odrzuć prostokąt H/V na ~całą szerokość kadru (typowy white_grid na Continuity)."""
    x0, y0 = float(np.min(quad[:, 0])), float(np.min(quad[:, 1]))
    x1, y1 = float(np.max(quad[:, 0])), float(np.max(quad[:, 1]))
    w_span = x1 - x0
    h_span = y1 - y0
    if w_span < 0.55 * img_w:
        return False
    # prawie równoległy do osi obrazu
    edges = [quad[i] - quad[(i + 1) % 4] for i in range(4)]
    horiz = sum(1 for e in edges if abs(float(e[1])) < 0.08 * max(abs(float(e[0])), 1.0))
    vert = sum(1 for e in edges if abs(float(e[0])) < 0.08 * max(abs(float(e[1])), 1.0))
    return horiz >= 2 and vert >= 2


def _quad_sane(quad: np.ndarray, img_w: int, img_h: int) -> bool:
    if quad is None or quad.shape != (4, 2):
        return False
    img_area = float(img_w * img_h)
    area = float(cv2.contourArea(quad.reshape(1, 4, 2).astype(np.float32)))
    ratio = area / max(1.0, img_area)
    if ratio < 0.04 or ratio > 0.65:
        return False
    if _is_axis_aligned_profile_box(quad, img_w, img_h):
        return False  # caller may override for white_hlines_profile
    w_span = float(np.ptp(quad[:, 0]))
    h_span = float(np.ptp(quad[:, 1]))
    asp = max(w_span, h_span) / max(1.0, min(w_span, h_span))
    return 1.12 <= asp <= 4.5


def _evaluate_panel_roi_candidate(
    image_bgr: np.ndarray,
    raw_quad: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    *,
    label: str,
    preserve_geometry: bool = False,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Ocena rogów z niebieskiego ROI / alignment — bez odrzucania dużego, osiowego panelu."""
    h, w = image_bgr.shape[:2]
    img_area = float(h * w)
    q_raw = pc.order_points(raw_quad.astype(np.float32))
    area = float(cv2.contourArea(q_raw.reshape(1, 4, 2).astype(np.float32)))
    ratio = area / max(1.0, img_area)
    if ratio < 0.03 or ratio > 0.88:
        return None, {'reject': 1.0, 'reject_reason': 'area_ratio'}
    c_can, _anc = canonicalize_corners_by_white_anchor(image_bgr, q_raw)
    ok_pnp, _rv, _tv, reproj = solve_panel_pose(c_can, k, dist, refine_lm=True)
    reproj_v = float(reproj) if ok_pnp else 999.0
    interior = _black_panel_interior_score(image_bgr, c_can)
    grid_sc = 0.0
    try:
        warped = _warp_panel_candidate(image_bgr, c_can)
        grid_sc = float(_grid_structure_score(warped))
    except cv2.error:
        pass
    # Zawsze zwracaj wersję z kotwicą białego (1,1) — inaczej pionowy panel
    # bywa lustrzany na warpie mimo poprawnej geometrii LSD.
    _ = preserve_geometry
    return c_can.astype(np.float32), {
        'reproj_mean_px': reproj_v,
        'panel_interior_score': interior,
        'grid_structure_score': grid_sc,
        'pnp_ok': float(ok_pnp),
        'area_ratio': ratio,
    }


def _evaluate_candidate(
    image_bgr: np.ndarray,
    raw_quad: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    *,
    label: str,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    h, w = image_bgr.shape[:2]
    sane = _quad_sane(raw_quad, w, h)
    if not sane and label != 'white_hlines_profile':
        return None, {'reject': 1.0}
    if label == 'white_hlines_profile' and _is_axis_aligned_profile_box(raw_quad, w, h):
        # profil boczny jest OK jeśli PnP potwierdzi niski reproj
        pass
    elif not sane:
        return None, {'reject': 1.0}
    c_can, _anc = canonicalize_corners_by_white_anchor(image_bgr, raw_quad.astype(np.float32))
    ok_pnp, _rv, _tv, reproj = solve_panel_pose(c_can, k, dist, refine_lm=True)
    reproj_v = float(reproj) if ok_pnp else 999.0
    interior = _black_panel_interior_score(image_bgr, c_can)
    grid_sc = 0.0
    try:
        warped = _warp_panel_candidate(image_bgr, c_can)
        grid_sc = float(_grid_structure_score(warped))
    except cv2.error:
        pass
    return c_can.astype(np.float32), {
        'reproj_mean_px': reproj_v,
        'panel_interior_score': interior,
        'grid_structure_score': grid_sc,
        'pnp_ok': float(ok_pnp),
    }


def _collect_raw_candidates(
    image_bgr: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
) -> List[Tuple[str, np.ndarray]]:
    work, scale, ox, oy = _prepare_detection_image(image_bgr)
    wh, ww = work.shape[:2]
    wk, wdist = default_intrinsics((wh, ww))

    out: List[Tuple[str, np.ndarray]] = []

    def add(label: str, quad_work: Optional[np.ndarray]) -> None:
        if quad_work is None:
            return
        full = _quad_to_full_image(quad_work, scale=scale, ox=ox, oy=oy)
        out.append((label, pc.order_points(full)))

    black = detect_corners_black_panel(work, wk, wdist)
    add('black_panel', black)

    # Alignment na obrazie roboczym (~1280 px) — na pełnym 1920×1080 HSV/trapezoid zwykle pada.
    for name, fn in (
        ('hsv_panel', pipeline_hsv_panel),
        ('trapezoid', pipeline_trapezoid),
        ('dark_blob', pipeline_dark_blob),
    ):
        res = fn(work)
        if res.ok and res.quad is not None:
            add(name, res.quad.astype(np.float32))

    lsd_q = _quad_from_lsd_outer_grid(work)
    add('lsd_outer_grid', lsd_q)
    add('white_trap', _grid_trapezoid_from_white_profile(work))

    from module_pose.api import _rough_quad_from_white_hlines

    wh = _rough_quad_from_white_hlines(work)
    add('white_hlines_profile', wh)

    return out


def _pick_best(
    rows: List[Dict[str, Any]],
    *,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Dict[str, Any]]:
    """Min reproj; przy ROI penalizuj ciasny żółty box (tylko środek siatki)."""
    if not rows:
        return None
    viable = [r for r in rows if float(r.get('reproj_mean_px', 999)) <= _MAX_LIVE_REPROJ_PX]
    if not viable:
        viable = [r for r in rows if float(r.get('reproj_mean_px', 999)) <= _MAX_LIVE_REPROJ_FALLBACK_PX]
    if not viable:
        return None

    from release.panel_roi import quad_coverage_vs_roi

    def sort_key(r: Dict[str, Any]) -> Tuple[float, float, float]:
        reproj = float(r['reproj_mean_px'])
        pref = 0.0 if r['label'] in _PREFERRED_LABELS or str(r['label']).startswith(
            ('lsd_outer', 'white_trap', 'trapezoid', 'hsv_panel', 'img_panel', 'black_panel'),
        ) else 6.0
        lbl = str(r['label'])
        if lbl == 'white_hlines_profile' and reproj > 32.0:
            pref += 10.0
        if 'warp' in lbl and roi is not None:
            cov = float(r.get('roi_coverage', 0.0))
            if cov < 0.62:
                pref += 14.0
        if 'roi_bbox' in lbl or 'align_quad' in lbl:
            pref -= 6.0
        elif 'lsd_outer' in lbl or 'expand_roi' in lbl or 'outer' in lbl:
            pref -= 4.0
        tight_pen = 0.0
        if roi is not None and 'corners' in r:
            cov = float(r.get('roi_coverage', quad_coverage_vs_roi(r['corners'], roi)))
            r['roi_coverage'] = cov
            if cov < 0.80:
                tight_pen = (0.80 - cov) * 55.0
        grid = -float(r.get('grid_structure_score', 0))
        return (reproj + pref + tight_pen, pref, grid)

    viable.sort(key=sort_key)
    return viable[0]


def _try_add_corner_row(
    rows: List[Dict[str, Any]],
    seen: List[np.ndarray],
    image_bgr: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    quad_full: np.ndarray,
    label: str,
    *,
    roi: Optional[Tuple[int, int, int, int]] = None,
    roi_src: str = '',
) -> None:
    if any(float(np.max(np.abs(quad_full - s))) < 10.0 for s in seen):
        return
    c_can, detail = _evaluate_candidate(image_bgr, quad_full, k, dist, label=label)
    if c_can is None:
        return
    seen.append(c_can)
    row: Dict[str, Any] = {
        'label': label,
        'corners': c_can,
        'rank_score': float(detail['reproj_mean_px']),
        'chosen': False,
        'roi_source': roi_src,
        **detail,
    }
    if roi is not None:
        from release.panel_roi import quad_coverage_vs_roi
        row['roi_coverage'] = float(quad_coverage_vs_roi(c_can, roi))
    rows.append(row)


def _append_outer_corner_variants(
    rows: List[Dict[str, Any]],
    seen: List[np.ndarray],
    image_bgr: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    base_corners: np.ndarray,
    base_label: str,
    *,
    roi: Optional[Tuple[int, int, int, int]] = None,
    roi_src: str = '',
) -> None:
    """Rozszerz czworokąt od środka i do pokrycia ROI — żółty ≈ niebieski."""
    from release.panel_roi import expand_quad_from_center, expand_quad_to_roi_coverage

    h, w = image_bgr.shape[:2]
    for sc, tag in ((1.06, 'x106'), (1.10, 'x110'), (1.14, 'x114')):
        exp = expand_quad_from_center(base_corners, (h, w), scale=sc)
        _try_add_corner_row(
            rows, seen, image_bgr, k, dist, exp, f'{base_label}_{tag}',
            roi=roi, roi_src=roi_src,
        )
    if roi is not None:
        wide = expand_quad_to_roi_coverage(base_corners, roi, (h, w), target_coverage=0.84)
        _try_add_corner_row(
            rows, seen, image_bgr, k, dist, wide, f'{base_label}_expand_roi',
            roi=roi, roi_src=roi_src,
        )


def probe_all_corner_candidates(
    image_bgr: np.ndarray,
    k: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    h, w = image_bgr.shape[:2]
    if k is None or dist is None:
        k, dist = default_intrinsics((h, w))

    rows: List[Dict[str, Any]] = []
    seen: List[np.ndarray] = []

    for label, raw in _collect_raw_candidates(image_bgr, k, dist):
        if label in _LIVE_BLOCKED_LABELS:
            continue
        if any(float(np.max(np.abs(raw - s))) < 15.0 for s in seen):
            continue
        c_can, detail = _evaluate_candidate(image_bgr, raw, k, dist, label=label)
        if c_can is None:
            continue
        seen.append(c_can)
        rows.append({
            'label': label,
            'corners': c_can,
            'rank_score': float(detail['reproj_mean_px']),
            'chosen': False,
            **detail,
        })

    from module_pose.refine_corners import refine_panel_corners_uniform_grid

    extra: List[Dict[str, Any]] = []
    for row in rows:
        if float(row['reproj_mean_px']) > 40.0:
            continue
        raw = np.asarray(row['corners'], dtype=np.float32)
        refined = refine_panel_corners_uniform_grid(image_bgr, raw)
        if refined is not None:
            c_can, detail = _evaluate_candidate(
                image_bgr, refined, k, dist, label=f"{row['label']}_refined",
            )
            if c_can is not None and float(detail.get('reproj_mean_px', 999)) < float(row['reproj_mean_px']):
                extra.append({
                    'label': f"{row['label']}_refined",
                    'corners': c_can,
                    'rank_score': float(detail['reproj_mean_px']),
                    'chosen': False,
                    **detail,
                })
        if float(row['reproj_mean_px']) > 14.0:
            continue
        if not any(row['label'].startswith(p) for p in ('trapezoid', 'hsv_panel', 'img_panel', 'black_panel', 'lsd_outer', 'white_trap')):
            continue
        for sx, sy, tag in ((1.04, 1.06, 'expand_s'), (1.08, 1.10, 'expand_m')):
            exp = _expand_quad(raw, image_bgr.shape[:2], sx=sx, sy=sy)
            c_can, detail = _evaluate_candidate(
                image_bgr, exp, k, dist, label=f"{row['label']}_{tag}",
            )
            if c_can is not None:
                extra.append({
                    'label': f"{row['label']}_{tag}",
                    'corners': c_can,
                    'rank_score': float(detail['reproj_mean_px']),
                    'chosen': False,
                    **detail,
                })
    rows.extend(extra)

    best = _pick_best(rows)
    if best is not None:
        best['chosen'] = True
    rows.sort(key=lambda r: (0.0 if r.get('chosen') else 1.0, float(r['reproj_mean_px'])))
    return rows


def _finalize_rows_with_roi(
    rows: List[Dict[str, Any]],
    roi: Optional[Tuple[int, int, int, int]],
) -> None:
    best = _pick_best(rows, roi=roi)
    for r in rows:
        r['chosen'] = False
    if best is not None:
        best['chosen'] = True
    rows.sort(key=lambda r: (0.0 if r.get('chosen') else 1.0, float(r['reproj_mean_px'])))


def _quad_area_px(quad: np.ndarray) -> float:
    q = quad.astype(np.float32).reshape(-1, 1, 2)
    return float(abs(cv2.contourArea(q)))


def _quad_min_internal_angle_deg(quad: np.ndarray) -> float:
    q = pc.order_points(quad.astype(np.float32))
    angles: List[float] = []
    for i in range(4):
        p0 = q[(i - 1) % 4]
        p1 = q[i]
        p2 = q[(i + 1) % 4]
        v1 = p0 - p1
        v2 = p2 - p1
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-3 or n2 < 1e-3:
            continue
        cos_a = float(np.dot(v1, v2) / (n1 * n2))
        cos_a = float(np.clip(cos_a, -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cos_a))))
    return min(angles) if angles else 90.0


def _tracker_geometry_ok(
    incoming: np.ndarray,
    last: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    reproj_px: float = 999.0,
    good_reproj: float = 28.0,
) -> Tuple[bool, str]:
    """Reject tracker update when quad jumps or collapses (bad hold / inner grid lock)."""
    h, w = image_shape[:2]
    diag = float(np.hypot(w, h))
    area_new = _quad_area_px(incoming)
    area_old = _quad_area_px(last)
    trust_incoming = reproj_px <= good_reproj
    if area_old > 1.0:
        ratio = area_new / area_old
        lo = 0.52 if trust_incoming else 0.68
        hi = 1.55 if trust_incoming else 1.38
        if ratio < lo or ratio > hi:
            return False, 'area_jump'
    min_ang = min(_quad_min_internal_angle_deg(incoming), _quad_min_internal_angle_deg(last))
    if min_ang < 38.0 if trust_incoming else 42.0:
        return False, 'acute_angle'
    delta = np.abs(incoming.astype(np.float32) - last.astype(np.float32))
    jump_scale = 1.35 if trust_incoming else 1.0
    if float(np.max(delta[:, 0])) > 72.0 * jump_scale:
        return False, 'corner_jump_x'
    if float(np.max(delta[:, 1])) > 62.0 * jump_scale:
        return False, 'corner_jump_y'
    shift = float(np.max(np.linalg.norm(delta, axis=1)))
    if shift > (0.14 if trust_incoming else 0.11) * diag:
        return False, 'corner_jump'
    return True, 'ok'


def _expand_if_inner_shrink(
    incoming: np.ndarray,
    last: Optional[np.ndarray],
    image_shape: Tuple[int, int],
    *,
    reproj_px: float,
    good_reproj: float,
) -> np.ndarray:
    """YOLO czasem łapie wewnętrzną siatkę — lekko rozpychaj quad na zewnątrz."""
    if last is None:
        return incoming
    area_new = _quad_area_px(incoming)
    area_old = _quad_area_px(last)
    if area_old <= 1.0 or area_new <= 1.0:
        return incoming
    ratio = area_new / area_old
    if ratio >= 0.82 or reproj_px > good_reproj * 1.35:
        return incoming
    from release.panel_roi import expand_quad_from_center

    scale = float(np.clip(1.04 / max(ratio, 0.50), 1.05, 1.20))
    return expand_quad_from_center(incoming, image_shape[:2], scale=scale)


class LiveCornerTracker:
    """EMA rogów + hold; dobry reproj → szybsze skoki; słaby → trzymaj ostatni dobry."""

    def __init__(
        self,
        *,
        alpha: float = 0.28,
        hold_frames: int = 14,
        good_reproj_px: float = 28.0,
    ) -> None:
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self.hold_frames = max(0, int(hold_frames))
        self.good_reproj = float(good_reproj_px)
        self._last_good: Optional[np.ndarray] = None
        self._last_roi: Optional[Tuple[int, int, int, int]] = None
        self._last_blue_roi: Optional[Tuple[int, int, int, int]] = None  # legacy; nie używane do ścinania rogów
        self._misses = 0
        self._last_area: float = 0.0
        self._last_reproj: float = 999.0

    def _blend_alpha(self, reproj_px: float, *, incoming_better: bool = False) -> float:
        if incoming_better and reproj_px <= self.good_reproj:
            return min(0.72, self.alpha * 5.0)
        if reproj_px <= 12.0:
            return min(0.58, self.alpha * 4.0)
        if reproj_px <= self.good_reproj:
            return min(0.42, self.alpha * 3.0)
        if reproj_px <= self.good_reproj * 1.6:
            return min(0.28, self.alpha * 2.0)
        return self.alpha

    def _commit(
        self,
        corners: np.ndarray,
        reproj_px: float,
        image_shape: Optional[Tuple[int, int]],
    ) -> np.ndarray:
        from release.safe_blue_roi import clip_quad_to_image

        self._last_good = corners.astype(np.float32)
        if image_shape is not None:
            self._last_good = clip_quad_to_image(self._last_good, image_shape)
        self._last_area = _quad_area_px(self._last_good)
        self._last_reproj = float(reproj_px)
        if image_shape is not None:
            self._last_roi = self._roi_from_corners(self._last_good, image_shape)
        return self._last_good.copy()

    def reset(self) -> None:
        self._last_good = None
        self._last_roi = None
        self._last_blue_roi = None
        self._misses = 0
        self._last_area = 0.0
        self._last_reproj = 999.0
        from release.safe_blue_roi import reset_blue_roi_stabilizer

        reset_blue_roi_stabilizer()

    def set_blue_roi(self, roi: Optional[Tuple[int, int, int, int]]) -> None:
        """No-op — niebieski ROI usunięty z live."""
        pass

    def _roi_from_corners(
        self,
        corners: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        h, w = image_shape[:2]
        pad = 22
        x0 = int(np.floor(np.clip(float(np.min(corners[:, 0])) - pad, 0, w - 2)))
        y0 = int(np.floor(np.clip(float(np.min(corners[:, 1])) - pad, 0, h - 2)))
        x1 = int(np.ceil(np.clip(float(np.max(corners[:, 0])) + pad, x0 + 2, w - 1)))
        y1 = int(np.ceil(np.clip(float(np.max(corners[:, 1])) + pad, y0 + 2, h - 1)))
        return x0, y0, x1, y1

    def apply(
        self,
        corners: Optional[np.ndarray],
        reproj_px: float,
        *,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[np.ndarray], bool]:
        if corners is None or corners.shape != (4, 2):
            self._misses += 1
            if self._last_good is not None and self._misses <= self.hold_frames:
                return self._last_good.copy(), True
            return None, False

        if reproj_px > _MAX_LIVE_REPROJ_FALLBACK_PX:
            self._misses += 1
            if self._last_good is not None and self._misses <= self.hold_frames:
                return self._last_good.copy(), True
            incoming = corners.astype(np.float32)
            self._last_good = incoming.copy()
            if image_shape is not None:
                self._last_roi = self._roi_from_corners(incoming, image_shape)
            return incoming, False

        self._misses = 0
        incoming = corners.astype(np.float32)
        if image_shape is not None:
            incoming = _expand_if_inner_shrink(
                incoming,
                self._last_good,
                image_shape,
                reproj_px=reproj_px,
                good_reproj=self.good_reproj,
            )
        incoming_better = reproj_px + 2.0 < self._last_reproj
        if image_shape is not None and self._last_good is not None:
            ok_geom, _reason = _tracker_geometry_ok(
                incoming,
                self._last_good,
                image_shape,
                reproj_px=reproj_px,
                good_reproj=self.good_reproj,
            )
            if not ok_geom:
                if incoming_better and reproj_px <= self.good_reproj * 1.25:
                    return self._commit(incoming, reproj_px, image_shape), False
                soft_a = self._blend_alpha(reproj_px) * 0.45
                blended = (1.0 - soft_a) * self._last_good + soft_a * incoming
                return self._commit(blended.astype(np.float32), reproj_px, image_shape), False

        if self._last_good is None or reproj_px > self.good_reproj * 1.8:
            return self._commit(incoming, reproj_px, image_shape), False

        if reproj_px <= self.good_reproj or incoming_better:
            a = self._blend_alpha(reproj_px, incoming_better=incoming_better)
            if incoming_better and reproj_px <= self.good_reproj * 0.85:
                a = max(a, 0.55)
            blended = (1.0 - a) * self._last_good + a * incoming
            return self._commit(blended.astype(np.float32), reproj_px, image_shape), False

        if self._last_good is not None:
            if image_shape is not None:
                ok_geom, _ = _tracker_geometry_ok(
                    incoming,
                    self._last_good,
                    image_shape,
                    reproj_px=reproj_px,
                    good_reproj=self.good_reproj,
                )
                if not ok_geom and reproj_px <= self.good_reproj * 1.4:
                    a = self._blend_alpha(reproj_px)
                    blended = (1.0 - a) * self._last_good + a * incoming
                    return self._commit(blended.astype(np.float32), reproj_px, image_shape), False
                if not ok_geom:
                    self.reset()
                    return self._commit(incoming, reproj_px, image_shape), False
            a = self._blend_alpha(reproj_px)
            blended = (1.0 - a) * self._last_good + a * incoming
            return self._commit(blended.astype(np.float32), reproj_px, image_shape), False
        return self._commit(incoming, reproj_px, image_shape), False

    def panel_roi(
        self,
        image_shape: Tuple[int, int],
    ) -> Optional[Tuple[int, int, int, int]]:
        if self._last_good is not None:
            return self._roi_from_corners(self._last_good, image_shape)
        return self._last_roi


_TRACKER = LiveCornerTracker()

# Co N-tą klatkę odpalać pełny align_hybrid (reszta: tracker_fast jeśli stabilny).
_ALIGN_FULL_PROBE_EVERY = 2
_align_probe_frame_idx = 0

_LIVE_CORNER_MODES = frozenset({
    'live', 'line_grid', 'auto', 'hybrid', 'roi_line_grid', 'roi_hybrid',
    'align_hybrid', 'outer_corners', 'yolo_pose',
})

# Domyślny: pipeline alignment (HSV + grid + dark + scored) jak run_alignment --pipeline hybrid
DEFAULT_LIVE_CORNER_MODE = 'align_hybrid'

_MIN_PANEL_WIDTH_FRAC = 0.40

_ALIGN_PRESERVE_GEOMETRY = frozenset({
    'morph_blob', 'hsv_panel', 'white_grid', 'trapezoid', 'current_scored', 'hybrid',
    'outer_grid', 'outer_grid_ring', 'grid_outer_lsd', 'border_scan',
})

# Progi reproj na żywo (bez GT intrinsics)
LIVE_MAX_REPROJ_RELIABLE_PX = 18.0
LIVE_MAX_REPROJ_DISPLAY_PX = 48.0


def _quad_width_frac_live(quad: np.ndarray, img_w: int) -> float:
    q = quad.astype(np.float32)
    return float(q[:, 0].max() - q[:, 0].min()) / max(1.0, float(img_w))


def _try_add_align_corner_row(
    rows: List[Dict[str, Any]],
    seen: List[np.ndarray],
    image_bgr: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    quad_full: np.ndarray,
    label: str,
    *,
    align_name: str = '',
    align_conf: float = 0.0,
    align_basis: str = '',
) -> None:
    if any(float(np.max(np.abs(quad_full - s))) < 10.0 for s in seen):
        return
    preserve = align_name in _ALIGN_PRESERVE_GEOMETRY or label.startswith('align_')
    c_can, detail = _evaluate_panel_roi_candidate(
        image_bgr, quad_full, k, dist, label=label, preserve_geometry=preserve,
    )
    if c_can is None:
        return
    h, w = image_bgr.shape[:2]
    min_w = 0.28 if align_name == 'outer_grid' else _MIN_PANEL_WIDTH_FRAC
    if _quad_width_frac_live(c_can, w) < min_w and 'dark' not in label:
        return
    seen.append(c_can)
    rows.append({
        'label': label,
        'corners': c_can,
        'rank_score': float(detail['reproj_mean_px']),
        'chosen': False,
        'align_name': align_name or label,
        'align_confidence': float(align_conf),
        'align_basis': str(align_basis),
        **detail,
    })


def _ref_outer_width_baseline(rows: List[Dict[str, Any]], img_w: int) -> Optional[float]:
    """Baseline „idealnie wycina”: max szerokość z HSV/morph/dark (bez white_hlines)."""
    best = 0.0
    for r in rows:
        name = str(r.get('align_name', r.get('label', ''))).lower()
        if 'white_hlines' in name or 'white_grid' in name:
            continue
        c = r.get('corners')
        if c is None:
            continue
        wf = _quad_width_frac_live(np.asarray(c, dtype=np.float32), img_w)
        best = max(best, wf)
    return best if best > 0.42 else None


def _baseline_method_tier(lbl: str, align: str) -> int:
    """Wyżej = preferowany przy podobnym rank_score (line_grid > border_scan > grid_outer)."""
    name = f'{lbl} {align}'.lower()
    if 'lg_img_panel' in name or align == 'img_panel':
        return 4
    if 'warp' in name and 'img_panel' in name:
        return 3
    if 'border_scan' in name:
        return 2
    if 'grid_outer' in name or 'outer_grid' in name or align == 'grid_outer_lsd':
        return 0
    if 'white_hlines' in name:
        return 0
    return 1


def _score_yellow_baseline_row(
    row: Dict[str, Any],
    img_w: int,
    *,
    ref_w_frac: Optional[float] = None,
    blue_roi: Optional[Tuple[int, int, int, int]] = None,
    img_h: int = 0,
) -> float:
    """Wybór kandydata: struktura siatki + perspektywa + ułamek kadru (bez wiązania z niebieskim ROI)."""
    reproj = float(row.get('reproj_mean_px', 999.0))
    lbl = str(row.get('label', ''))
    align = str(row.get('align_name', lbl))
    corners = row.get('corners')
    c_arr = np.asarray(corners, dtype=np.float32) if corners is not None else None
    grid_s = float(row.get('grid_structure_score', 0.0))
    interior_s = float(row.get('panel_interior_score', 0.0))

    # Wysoka struktura siatki → słabsza kara PnP (lg_img_panel często ma duży reproj przy dobrych rogach).
    reproj_w = 0.15 * max(0.10, 1.0 - 0.82 * min(1.0, grid_s))
    score = reproj * reproj_w
    score -= min(200.0, grid_s * 24.0)
    score -= min(95.0, interior_s * 35.0)

    if 'lg_img_panel' in lbl or 'img_panel' in lbl or align == 'img_panel':
        score -= 150.0
    if 'warp2_img_panel' in lbl or ('warp' in lbl and 'img_panel' in lbl):
        score -= 125.0
    if 'border_scan' in lbl or align == 'border_scan':
        score -= 55.0
    if 'grid_outer' in lbl or align in ('grid_outer_lsd', 'outer_grid'):
        score += 85.0
    if 'white_hlines' in lbl or align == 'white_hlines':
        score += 200.0
    elif 'morph' in lbl or 'dark_blob' in lbl or align in ('morph_blob', 'dark_blob'):
        if grid_s < 0.30:
            score += 160.0
        elif grid_s >= 0.55:
            score -= 45.0

    if c_arr is not None:
        if _is_near_axis_aligned_box(c_arr, tol_deg=4.0):
            score += 220.0
        if not _quad_has_perspective_skew(c_arr):
            score += 110.0
        w_frac = _quad_width_frac_live(c_arr, img_w)
        if img_h > 0:
            area_frac = float(cv2.contourArea(c_arr.reshape(1, 4, 2).astype(np.float32))) / float(img_w * img_h)
            if area_frac < 0.10 or area_frac > 0.62:
                score += 280.0
            elif 0.14 <= area_frac <= 0.48:
                score -= 35.0
        if w_frac > 0.90:
            score += 400.0
        elif w_frac < 0.38:
            score += 200.0
        elif 0.48 <= w_frac <= 0.88:
            score -= 25.0

        if blue_roi is not None and img_h > 0:
            bx0, by0, bx1, by1 = map(float, blue_roi)
            roi_area = max(1.0, (bx1 - bx0) * (by1 - by0))
            quad_area = float(cv2.contourArea(c_arr.reshape(1, 4, 2).astype(np.float32)))
            area_ratio = quad_area / roi_area
            if area_ratio > 0.98:
                score += (area_ratio - 0.98) * 120.0
            elif 0.72 <= area_ratio <= 0.95:
                score -= 40.0

    return score


def _corner_consensus_penalty(
    row: Dict[str, Any],
    peers: List[Dict[str, Any]],
) -> float:
    """Kara gdy rogi mocno odbiegają od mediany innych sensownych kandydatów (np. dark_blob vs line_grid)."""
    c = row.get('corners')
    if c is None or len(peers) < 2:
        return 0.0
    q = np.asarray(c, dtype=np.float32)
    others: List[np.ndarray] = []
    for p in peers:
        if p is row:
            continue
        pc = p.get('corners')
        if pc is None:
            continue
        if float(p.get('grid_structure_score', 0.0)) < 0.35:
            continue
        others.append(np.asarray(pc, dtype=np.float32))
    if len(others) < 2:
        return 0.0
    med = np.median(np.stack(others, axis=0), axis=0)
    dist = float(np.mean(np.linalg.norm(q - med, axis=1)))
    if dist < 55.0:
        return 0.0
    return min(320.0, (dist - 55.0) * 2.2)


def _left_anchor_hsv_x(image_bgr: np.ndarray) -> Optional[float]:
    """Lewy brzeg czarnej ramki z profilu HSV (nie x=0 z linii siatki)."""
    from release.alignment_pipelines import _hsv_profile_quad, _strip_alignment_overlays

    work, ox, oy = _strip_alignment_overlays(image_bgr)
    q = _hsv_profile_quad(work)
    if q is None:
        return None
    q = q.astype(np.float32).copy()
    q[:, 0] += float(ox)
    q[:, 1] += float(oy)
    return float(np.min(q[:, 0]))


def _left_anchor_baseline_x(
    rows: List[Dict[str, Any]],
    image_bgr: np.ndarray,
) -> Optional[float]:
    """Lewy anchor: HSV ramka; nie min() z dark/morph/hlines (często 0 = krawędź kadru)."""
    hsv_x = _left_anchor_hsv_x(image_bgr)
    if hsv_x is not None and hsv_x > 35.0:
        return hsv_x
    xs: List[float] = []
    for r in rows:
        name = str(r.get('align_name', r.get('label', ''))).lower()
        if 'white_hlines' in name or 'morph' in name or 'dark' in name:
            continue
        if not any(k in name for k in ('hsv_panel', 'hsv', 'black_panel', 'trapezoid', 'white_grid')):
            continue
        c = r.get('corners')
        if c is None:
            continue
        xl = float(np.min(np.asarray(c, dtype=np.float32)[:, 0]))
        if xl > 28.0:
            xs.append(xl)
    if xs:
        xs.sort()
        return float(xs[len(xs) // 2])
    return hsv_x


def _align_quad_left_to_outer_edge(
    quad: np.ndarray,
    x_outer_left: float,
    image_shape: Tuple[int, int],
    *,
    margin_px: float = 12.0,
) -> np.ndarray:
    """
    Lewa krawędź utknęła na x≈0 (kadr) — przesuń w prawo do zewnętrznej ramki panelu (HSV).
    """
    h, w = image_shape[:2]
    q = pc.order_points(quad.astype(np.float32))
    x_floor = float(np.clip(x_outer_left - margin_px, 0.0, float(w - 1)))
    cur_left = float(min(q[0, 0], q[3, 0]))
    if cur_left >= x_floor - 4.0:
        return q
    q = q.copy()
    if float(q[0, 0]) < x_floor - 2.0:
        q[0, 0] = x_floor
    if float(q[3, 0]) < x_floor - 2.0:
        q[3, 0] = x_floor
    q[:, 0] = np.clip(q[:, 0], 0.0, float(w - 1))
    q[:, 1] = np.clip(q[:, 1], 0.0, float(h - 1))
    return pc.order_points(q)


def _right_anchor_hsv_x(image_bgr: np.ndarray) -> Optional[float]:
    """Prawy brzeg czarnej ramki z profilu HSV."""
    from release.alignment_pipelines import _hsv_profile_quad, _strip_alignment_overlays

    work, ox, oy = _strip_alignment_overlays(image_bgr)
    q = _hsv_profile_quad(work)
    if q is None:
        return None
    q = q.astype(np.float32).copy()
    q[:, 0] += float(ox)
    q[:, 1] += float(oy)
    return float(np.max(q[:, 0]))


def _right_anchor_baseline_x(
    rows: List[Dict[str, Any]],
    image_bgr: np.ndarray,
) -> Optional[float]:
    """Prawy anchor: HSV; white_hlines bywa ~150–250 px za wąski."""
    hsv_x = _right_anchor_hsv_x(image_bgr)
    if hsv_x is not None and hsv_x > 200.0:
        return hsv_x
    xs: List[float] = []
    for r in rows:
        name = str(r.get('align_name', r.get('label', ''))).lower()
        if 'white_hlines' in name or 'morph' in name:
            continue
        if not any(k in name for k in ('hsv_panel', 'hsv', 'black_panel', 'trapezoid', 'white_grid')):
            continue
        c = r.get('corners')
        if c is None:
            continue
        xr = float(np.max(np.asarray(c, dtype=np.float32)[:, 0]))
        xs.append(xr)
    if xs and hsv_x is not None:
        return float(max(max(xs), hsv_x * 0.98))
    return max(xs) if xs else hsv_x


def _align_quad_right_to_outer_edge(
    quad: np.ndarray,
    x_outer_right: float,
    image_shape: Tuple[int, int],
    *,
    margin_px: float = 14.0,
) -> np.ndarray:
    """Prawa krawędź za wąska (white_hlines) — domknij do zewnętrznej ramki HSV."""
    h, w = image_shape[:2]
    q = pc.order_points(quad.astype(np.float32))
    x_cap = float(np.clip(x_outer_right - margin_px, 0.0, float(w - 1)))
    cur_right = float(max(q[1, 0], q[2, 0]))
    if cur_right >= x_cap - 4.0:
        return q
    q = q.copy()
    if float(q[1, 0]) < x_cap - 2.0:
        q[1, 0] = x_cap
    if float(q[2, 0]) < x_cap - 2.0:
        q[2, 0] = x_cap
    q[:, 0] = np.clip(q[:, 0], 0.0, float(w - 1))
    q[:, 1] = np.clip(q[:, 1], 0.0, float(h - 1))
    return pc.order_points(q)


def _trapezoid_from_grid_bands(
    image_bgr: np.ndarray,
    seed_box: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Trapez z perspektywy: lewy/prawy brzeg panelu osobno u góry i u dołu (tylko w bbox żółtego).
    """
    h, w = image_bgr.shape[:2]
    q = pc.order_points(seed_box.astype(np.float32))
    y0, y1 = float(q[0, 1]), float(q[2, 1])
    x_pad = max(8.0, 0.02 * float(q[1, 0] - q[0, 0]))
    x_lo = int(np.clip(float(q[0, 0]) - x_pad, 0, w - 2))
    x_hi = int(np.clip(float(q[1, 0]) + x_pad, x_lo + 2, w - 1))
    span_y = max(20.0, y1 - y0)
    band = max(10.0, min(0.10 * span_y, 0.14 * span_y))

    def _x_extents(y_a: float, y_b: float) -> Optional[Tuple[float, float]]:
        ya = int(np.clip(y_a, 0, h - 1))
        yb = int(np.clip(y_b, 0, h - 1))
        if yb <= ya:
            return None
        strip = image_bgr[ya:yb, x_lo : x_hi + 1]
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gx = np.abs(gx)
        col = gx.mean(axis=0)
        if col.size < 8:
            return None
        peak = float(np.percentile(col, 92))
        thr = max(peak * 0.35, float(np.median(col)) * 1.8)
        xs = np.where(col >= thr)[0]
        if xs.size < 6:
            hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
            v, s = hsv[:, :, 2], hsv[:, :, 1]
            dark = ((gray < 108) | ((v < 150) & (s > 45))).astype(np.uint8) * 255
            bright = ((gray > 115) & (dark > 0)).astype(np.uint8) * 255
            col_hit = (bright > 0).mean(axis=0)
            xs = np.where(col_hit > 0.04)[0]
            if xs.size < 6:
                return None
        return float(xs[0] + x_lo), float(xs[-1] + x_lo)

    top = _x_extents(y0, y0 + band)
    bot = _x_extents(y1 - band, y1)
    if top is None or bot is None:
        return None
    x_tl, x_tr = top
    x_bl, x_br = bot
    w_top = x_tr - x_tl
    w_bot = x_br - x_bl
    if w_top < 80.0 or w_bot < 80.0:
        return None
    # Góra panelu w perspektywie jest węższa niż dół (widok z góry).
    if w_top > w_bot * 1.08:
        shrink = 0.5 * (w_top - w_bot * 0.98)
        x_tl += shrink * 0.5
        x_tr -= shrink * 0.5
    trap = np.array(
        [[x_tl, y0], [x_tr, y0], [x_br, y1], [x_bl, y1]],
        dtype=np.float32,
    )
    trap[:, 0] = np.clip(trap[:, 0], 0.0, float(w - 1))
    trap[:, 1] = np.clip(trap[:, 1], 0.0, float(h - 1))
    out = pc.order_points(trap)
    top_w = float(out[1, 0] - out[0, 0])
    bot_w = float(out[2, 0] - out[3, 0])
    if top_w < 80.0 or bot_w < 80.0:
        return None
    left_delta = abs(float(out[0, 0] - out[3, 0]))
    right_delta = abs(float(out[1, 0] - out[2, 0]))
    # Perspektywa = skośne boki (różne x u góry i dołu), nie tylko różna szerokość całej krawędzi.
    if left_delta < 10.0 and right_delta < 10.0:
        if abs(top_w - bot_w) < max(14.0, 0.008 * max(top_w, bot_w)):
            return None
    return out


def _perspective_quad_from_panel_contour(
    image_bgr: np.ndarray,
    seed_box: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Z osiowego bboxu wyciągnij 4-punktowy trapez z konturu ciemnego panelu (Canny + approx).
    """
    h, w = image_bgr.shape[:2]
    q = pc.order_points(seed_box.astype(np.float32))
    x0 = int(max(0, np.floor(float(q[:, 0].min()) - 8)))
    y0 = int(max(0, np.floor(float(q[:, 1].min()) - 8)))
    x1 = int(min(w - 1, np.ceil(float(q[:, 0].max()) + 8)))
    y1 = int(min(h - 1, np.ceil(float(q[:, 1].max()) + 8)))
    if x1 <= x0 + 40 or y1 <= y0 + 40:
        return None
    crop = image_bgr[y0 : y1 + 1, x0 : x1 + 1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if float(cv2.contourArea(cnt)) < 0.08 * float(crop.shape[0] * crop.shape[1]):
        return None
    peri = float(cv2.arcLength(cnt, True))
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True).reshape(-1, 2)
    if approx.shape[0] < 4:
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True).reshape(-1, 2)
    if approx.shape[0] < 4:
        rect = cv2.minAreaRect(cnt)
        approx = cv2.boxPoints(rect)
    if approx.shape[0] != 4:
        return None
    out = approx.astype(np.float32)
    out[:, 0] += float(x0)
    out[:, 1] += float(y0)
    out = pc.order_points(out)
    if _is_near_axis_aligned_box(out, tol_deg=4.0):
        return None
    span = float(np.max(out[:, 0]) - np.min(out[:, 0]))
    if span < 0.45 * (x1 - x0):
        return None
    return out


def _yellow_quad_scalable(
    image_bgr: np.ndarray,
    blue_roi: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[np.ndarray]:
    """Żółty: LSD/Hough skrajnych linii siatki na kadrze (blue_roi tylko opcjonalny fallback)."""
    from release.grid_outer_quad import (
        detect_yellow_quad_border_scan,
        detect_yellow_quad_ring_lsd,
        detect_yellow_quad_scalable,
    )

    quad, _roi, _src = detect_yellow_quad_scalable(image_bgr)
    if quad is not None:
        return quad
    if blue_roi is not None:
        return detect_yellow_quad_ring_lsd(image_bgr, blue_roi)
    return detect_yellow_quad_border_scan(image_bgr, None)


def _quad_has_perspective_skew(quad: np.ndarray) -> bool:
    """Czy rogi tworzą trapez (różne x u góry/dół), nie osiowy prostokąt."""
    q = pc.order_points(quad.astype(np.float32))
    left_delta = abs(float(q[0, 0] - q[3, 0]))
    right_delta = abs(float(q[1, 0] - q[2, 0]))
    top_w = float(q[1, 0] - q[0, 0])
    bot_w = float(q[2, 0] - q[3, 0])
    if left_delta > 12.0 or right_delta > 12.0:
        return True
    return abs(top_w - bot_w) >= max(14.0, 0.008 * max(top_w, bot_w))


def _upgrade_baseline_to_perspective_trapezoid(
    image_bgr: np.ndarray,
    corners: np.ndarray,
    label: str,
    rows: List[Dict[str, Any]],
    k: np.ndarray,
    dist: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """
    white_hlines daje osiowy prostokąt; zamień na trapez (line_grid / trapezoid / morph)
    gdy ma sensowny reproj — wtedy żółte linie znowu „łączą 4 rogi” w perspektywie.
    """
    if not _is_near_axis_aligned_box(corners, tol_deg=5.5):
        return corners, label

    best: Optional[np.ndarray] = None
    best_reproj = 999.0
    best_lbl = ''

    for r in rows:
        name = str(r.get('label', ''))
        if 'white_hlines' in name:
            continue
        c = r.get('corners')
        if c is None:
            continue
        c_arr = np.asarray(c, dtype=np.float32)
        if not _quad_has_perspective_skew(c_arr):
            continue
        reproj = float(r.get('reproj_mean_px', 999.0))
        if reproj < best_reproj and reproj <= _MAX_LIVE_REPROJ_FALLBACK_PX:
            best_reproj = reproj
            best = c_arr.copy()
            best_lbl = name

    if best is None:
        from release.alignment_pipelines import pipeline_trapezoid

        res = pipeline_trapezoid(image_bgr)
        if res.ok and res.quad is not None:
            c_can, detail = _evaluate_panel_roi_candidate(
                image_bgr,
                res.quad.astype(np.float32),
                k,
                dist,
                label='align_trapezoid',
                preserve_geometry=True,
            )
            if c_can is not None and _quad_has_perspective_skew(c_can):
                reproj = float(detail.get('reproj_mean_px', 999.0))
                if reproj <= _MAX_LIVE_REPROJ_FALLBACK_PX:
                    best = c_can
                    best_reproj = reproj
                    best_lbl = 'align_trapezoid'

    if best is None:
        trap = _trapezoid_from_grid_bands(image_bgr, corners)
        if trap is None:
            trap = _perspective_quad_from_panel_contour(image_bgr, corners)
        if trap is not None:
            c_can, detail = _evaluate_panel_roi_candidate(
                image_bgr, trap, k, dist, label='grid_bands_trap', preserve_geometry=True,
            )
            if c_can is not None and _quad_has_perspective_skew(c_can):
                reproj = float(detail.get('reproj_mean_px', 999.0))
                grid_s = float(detail.get('grid_structure_score', 0.0))
                interior_s = float(detail.get('panel_interior_score', 0.0))
                persp_ok = (
                    reproj <= _MAX_LIVE_REPROJ_FALLBACK_PX
                    or (
                        reproj <= 220.0
                        and grid_s >= 0.12
                        and interior_s >= 0.08
                    )
                )
                if persp_ok:
                    best = c_can
                    best_lbl = 'grid_bands_trap'

    if best is None:
        return corners, label

    suffix = '+persp_trap' if '+persp_trap' not in label else ''
    return best.astype(np.float32), f'{label}{suffix}'


def _pick_yellow_baseline_corners(
    rows: List[Dict[str, Any]],
    image_bgr: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    *,
    blue_roi: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[Optional[np.ndarray], str, bool, Optional[float]]:
    """Wybór żółtego trapezu; niebieski ROI rozszerzany później (nie ścina rogów)."""
    from release.safe_blue_roi import clip_quad_to_image

    h, w = image_bgr.shape[:2]
    ref_w_frac = _ref_outer_width_baseline(rows, w)

    for r in rows:
        r['rank_score'] = _score_yellow_baseline_row(
            r, w, ref_w_frac=ref_w_frac, blue_roi=blue_roi, img_h=h,
        )

    ordered = sorted(rows, key=lambda r: float(r.get('rank_score', 999.0)))
    if len(ordered) >= 2:
        best_sc = float(ordered[0].get('rank_score', 999.0))
        pool = [
            r for r in ordered
            if r.get('corners') is not None
            and (
                float(r.get('rank_score', 999.0)) <= best_sc + 200.0
                or (
                    float(r.get('grid_structure_score', 0.0)) >= 0.45
                    and float(r.get('reproj_mean_px', 999.0)) <= 180.0
                )
            )
        ]
        if len(pool) >= 2:
            for r in pool:
                r['_consensus_penalty'] = _corner_consensus_penalty(r, pool)

            def _pool_pick_key(r: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
                gs = float(r.get('grid_structure_score', 0.0))
                rp = float(r.get('reproj_mean_px', 999.0))
                tier = float(_baseline_method_tier(str(r.get('label', '')), str(r.get('align_name', ''))))
                cons = float(r.get('_consensus_penalty', 0.0))
                return (tier, gs, -cons, -min(rp, 140.0), -float(r.get('rank_score', 999.0)))

            pool.sort(key=_pool_pick_key, reverse=True)
            if pool[0] is not ordered[0]:
                ordered = [pool[0]] + [r for r in ordered if r is not pool[0]]

    best: Optional[Dict[str, Any]] = None
    corners: Optional[np.ndarray] = None
    label = 'none'

    for cand in ordered:
        raw = cand.get('corners')
        if raw is None:
            continue
        corners = clip_quad_to_image(np.asarray(raw, dtype=np.float32), (h, w))
        best = cand
        label = str(cand.get('label', 'yellow'))
        break

    for r in rows:
        r['chosen'] = r is best

    if best is None or corners is None:
        rows.sort(key=lambda r: float(r.get('rank_score', 999.0)))
        return None, 'none', False, None

    best['corners'] = corners
    best['label'] = label
    rows.sort(key=lambda r: (0.0 if r.get('chosen') else 1.0, float(r.get('rank_score', 999))))
    return corners, label, False, None


def probe_yellow_corners_baseline(
    image_bgr: np.ndarray,
    k: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    *,
    fast: bool = True,
) -> Tuple[List[Dict[str, Any]], str]:
    """Zewnętrzne rogi siatki — delegacja do release.outer_corners (bez niebieskiego ROI)."""
    from release.outer_corners import detect_outer_corners

    corners, label, rows, _meta = detect_outer_corners(
        image_bgr, k, dist, fast=fast, refine=False,
    )
    if corners is None:
        return rows, 'none'
    return rows, label


def _ref_outer_width_frac(rows: List[Dict[str, Any]], img_w: int) -> Optional[float]:
    """Typowa szerokość panelu — bez kandydatów na cały kadr (dark/morph >90%)."""
    wfs: List[float] = []
    for r in rows:
        name = str(r.get('align_name', r.get('label', ''))).lower()
        if 'white_hlines' in name or 'white_grid' in name or 'morph' in name:
            continue
        if not any(k in name for k in _EXTERIOR_ANCHOR_KEYS):
            continue
        c = r.get('corners')
        if c is None:
            continue
        wf = _quad_width_frac_live(np.asarray(c, dtype=np.float32), img_w)
        if wf > 0.90:
            continue
        wfs.append(wf)
    if not wfs:
        return None
    wfs.sort()
    return float(wfs[len(wfs) // 2])


def _score_align_candidate_row(row: Dict[str, Any], img_w: int, *, ref_w_frac: Optional[float] = None) -> float:
    reproj = float(row.get('reproj_mean_px', 999.0))
    lbl = str(row.get('label', ''))
    align = str(row.get('align_name', lbl))
    corners = row.get('corners')
    w_frac = 1.0
    if corners is not None:
        w_frac = _quad_width_frac_live(np.asarray(corners, dtype=np.float32), img_w)

    score = reproj
    if w_frac > 0.88:
        score += 650.0
    elif w_frac > 0.78:
        score += 220.0
    elif w_frac < 0.38:
        score += 700.0
    elif w_frac < 0.50:
        score += 140.0
    elif w_frac >= 0.58:
        score -= 12.0

    if 'hsv' in align or 'hsv' in lbl:
        score -= 28.0
    if 'trapezoid' in align or 'trapezoid' in lbl:
        score -= 18.0
    if 'scored' in align or 'scored' in lbl:
        score -= 14.0
    if align == 'hybrid' or 'align_hybrid' in lbl:
        score -= 20.0
    if 'white_grid' in align or 'white_grid' in lbl:
        score -= 52.0
    if 'white_hlines' in lbl or align == 'white_hlines':
        score -= 18.0
        if ref_w_frac is not None and w_frac < ref_w_frac * 0.85:
            score += 55.0
    if 'dark' in align or 'dark' in lbl:
        score += 90.0
        if w_frac > 0.86:
            score += 320.0
    if 'img_panel' in lbl or lbl.startswith('align_lg_'):
        score -= 32.0
    if 'morph_blob' in align or 'morph' in lbl:
        score += 140.0

    score -= float(row.get('align_confidence', 0.0)) * 12.0
    score -= min(14.0, float(row.get('grid_structure_score', 0.0)) * 10.0)
    score += max(0.0, 0.55 - float(row.get('panel_interior_score', 0.0))) * 40.0

    ref_left = row.get('_ref_left_x')
    if ref_left is not None and corners is not None:
        inset = max(0.0, float(np.min(np.asarray(corners)[:, 0])) - float(ref_left) - 35.0)
        score += inset * 3.2

    ref_right = row.get('_ref_right_x')
    if ref_right is not None and corners is not None:
        right_x = float(np.max(np.asarray(corners)[:, 0]))
        outset = max(0.0, float(ref_right) - right_x - 35.0)
        score += outset * 3.2
        overhang = max(0.0, right_x - float(ref_right) - 28.0)
        score += overhang * 5.5

    if ref_w_frac is not None and w_frac > ref_w_frac * 1.04:
        score += (w_frac - ref_w_frac * 1.04) * 720.0

    return score


_LEFT_ANCHOR_KEYS = ('hsv_panel', 'hsv', 'black_panel', 'trapezoid', 'dark_blob', 'morph_blob', 'morph')
# morph bywa szerszy od tła — nie używaj do prawej krawędzi.
_EXTERIOR_ANCHOR_KEYS = ('hsv_panel', 'hsv', 'black_panel', 'trapezoid', 'dark_blob', 'white_grid')


def _right_anchor_x_from_rows(
    rows: List[Dict[str, Any]],
    img_w: int,
) -> Optional[float]:
    """Prawy brzeg — mediana stabilnych detektorów (bez morph i „pełnego kadru”)."""
    xs: List[float] = []
    x_hlines: Optional[float] = None
    for r in rows:
        name = str(r.get('align_name', r.get('label', ''))).lower()
        if 'morph' in name:
            continue
        c = r.get('corners')
        if c is None:
            continue
        wf = _quad_width_frac_live(np.asarray(c, dtype=np.float32), img_w)
        if wf > 0.90:
            continue
        xr = float(np.max(np.asarray(c, dtype=np.float32)[:, 0]))
        if 'white_hlines' in name:
            x_hlines = xr if x_hlines is None else max(x_hlines, xr)
            continue
        if not any(k in name for k in _EXTERIOR_ANCHOR_KEYS):
            continue
        xs.append(xr)
    if not xs and x_hlines is None:
        return None
    anchor = float(np.median(xs)) if xs else float(x_hlines)
    if x_hlines is not None and xs:
        # Zewnętrzna ramka, ale nie szersza niż linie siatki + ~4% kadru.
        cap = float(x_hlines) + 0.042 * float(img_w)
        anchor = float(min(max(anchor, x_hlines + 18.0), cap))
    return anchor


def _left_anchor_x_from_rows(rows: List[Dict[str, Any]]) -> Optional[float]:
    """Lewy brzeg panelu z HSV / czarnej ramki / trapezu (nie white_hlines)."""
    xs: List[float] = []
    for r in rows:
        name = str(r.get('align_name', r.get('label', ''))).lower()
        if not any(k in name for k in _LEFT_ANCHOR_KEYS):
            continue
        c = r.get('corners')
        if c is None:
            continue
        xs.append(float(np.min(np.asarray(c, dtype=np.float32)[:, 0])))
    return min(xs) if xs else None


def _fuse_quad_left_edge(
    quad: np.ndarray,
    x_left: float,
    image_shape: Tuple[int, int],
    *,
    margin_px: float = 10.0,
) -> np.ndarray:
    """Rozszerz TL/BL do zewnętrznego lewego brzegu (HSV), zachowując prawą krawędź."""
    h, w = image_shape[:2]
    q = pc.order_points(quad.astype(np.float32))
    x_target = float(np.clip(x_left - margin_px, 0.0, float(w - 1)))
    cur_left = float(min(q[0, 0], q[3, 0]))
    if cur_left <= x_target + 3.0:
        return q
    q[0, 0] = x_target
    q[3, 0] = x_target
    q[:, 0] = np.clip(q[:, 0], 0.0, float(w - 1))
    q[:, 1] = np.clip(q[:, 1], 0.0, float(h - 1))
    return pc.order_points(q)


def _top_anchor_y_from_rows(rows: List[Dict[str, Any]]) -> Optional[float]:
    """Górny brzeg panelu (bez white_hlines / morph)."""
    ys: List[float] = []
    for r in rows:
        name = str(r.get('align_name', r.get('label', ''))).lower()
        if 'white_hlines' in name or 'morph' in name:
            continue
        if not any(k in name for k in _EXTERIOR_ANCHOR_KEYS):
            continue
        c = r.get('corners')
        if c is None:
            continue
        ys.append(float(np.min(np.asarray(c, dtype=np.float32)[:, 1])))
    return min(ys) if ys else None


def _bottom_anchor_y_from_rows(rows: List[Dict[str, Any]]) -> Optional[float]:
    """Dolny brzeg panelu — zewnętrzna ramka + white_hlines (ostatni wiersz siatki)."""
    ys: List[float] = []
    for r in rows:
        name = str(r.get('align_name', r.get('label', ''))).lower()
        if 'morph' in name:
            continue
        if 'white_hlines' in name:
            c = r.get('corners')
            if c is not None:
                ys.append(float(np.max(np.asarray(c, dtype=np.float32)[:, 1])))
            continue
        if not any(k in name for k in _EXTERIOR_ANCHOR_KEYS):
            continue
        c = r.get('corners')
        if c is None:
            continue
        ys.append(float(np.max(np.asarray(c, dtype=np.float32)[:, 1])))
    return max(ys) if ys else None


def _fuse_quad_top_edge(
    quad: np.ndarray,
    y_top: float,
    image_shape: Tuple[int, int],
    *,
    margin_px: float = 8.0,
) -> np.ndarray:
    """Rozszerz TL/TR do zewnętrznej górnej krawędzi."""
    h, w = image_shape[:2]
    q = pc.order_points(quad.astype(np.float32))
    y_target = float(np.clip(y_top - margin_px, 0.0, float(h - 1)))
    cur_top = float(min(q[0, 1], q[1, 1]))
    if cur_top <= y_target + 3.0:
        return q
    q[0, 1] = y_target
    q[1, 1] = y_target
    q[:, 1] = np.clip(q[:, 1], 0.0, float(h - 1))
    return pc.order_points(q)


def _fuse_quad_right_edge(
    quad: np.ndarray,
    x_right: float,
    image_shape: Tuple[int, int],
    *,
    margin_px: float = 6.0,
    max_push_frac: float = 0.03,
) -> np.ndarray:
    """Domknij lukę w prawo — częściowo (bez wchodzenia w tło za panelem)."""
    h, w = image_shape[:2]
    q = pc.order_points(quad.astype(np.float32))
    cur_right = float(max(q[1, 0], q[2, 0]))
    gap = float(x_right) - cur_right
    if gap < 50.0:
        return q
    span_w = max(1.0, float(np.max(q[:, 0]) - np.min(q[:, 0])))
    push = min(max(0.0, gap - margin_px), span_w * max_push_frac)
    x_target = float(np.clip(cur_right + push, 0.0, float(w - 1)))
    if cur_right >= x_target - 2.0:
        return q
    q = q.copy()
    q[1, 0] = x_target
    q[2, 0] = x_target
    q[:, 0] = np.clip(q[:, 0], 0.0, float(w - 1))
    q[:, 1] = np.clip(q[:, 1], 0.0, float(h - 1))
    return pc.order_points(q)


def _trim_quad_right_edge(
    quad: np.ndarray,
    x_right: float,
    image_shape: Tuple[int, int],
    *,
    margin_px: float = 10.0,
) -> np.ndarray:
    """Przytnij prawą krawędź, gdy trapez wychodzi w tło (morph / zbyt szeroki ROI)."""
    h, w = image_shape[:2]
    q = pc.order_points(quad.astype(np.float32))
    cur_right = float(max(q[1, 0], q[2, 0]))
    x_target = float(np.clip(x_right + margin_px, 0.0, float(w - 1)))
    if cur_right <= x_target + 18.0:
        return q
    q = q.copy()
    q[1, 0] = min(q[1, 0], x_target)
    q[2, 0] = min(q[2, 0], x_target)
    q[:, 0] = np.clip(q[:, 0], 0.0, float(w - 1))
    q[:, 1] = np.clip(q[:, 1], 0.0, float(h - 1))
    return pc.order_points(q)


def _fuse_quad_bottom_edge(
    quad: np.ndarray,
    y_bottom: float,
    image_shape: Tuple[int, int],
    *,
    margin_px: float = 6.0,
    max_push_frac: float = 0.07,
) -> np.ndarray:
    """Domknij dół — dolny wiersz siatki nie może być ucięty."""
    h, w = image_shape[:2]
    q = pc.order_points(quad.astype(np.float32))
    cur_bottom = float(max(q[2, 1], q[3, 1]))
    gap = float(y_bottom) - cur_bottom
    if gap < 35.0:
        return q
    span_h = max(1.0, float(np.max(q[:, 1]) - np.min(q[:, 1])))
    push = min(max(0.0, gap - margin_px), span_h * max_push_frac)
    y_target = float(np.clip(cur_bottom + push, 0.0, float(h - 1)))
    if cur_bottom >= y_target - 2.0:
        return q
    q = q.copy()
    q[2, 1] = y_target
    q[3, 1] = y_target
    q[:, 1] = np.clip(q[:, 1], 0.0, float(h - 1))
    return pc.order_points(q)


def _pick_align_corners(
    rows: List[Dict[str, Any]],
    image_bgr: np.ndarray,
) -> Tuple[np.ndarray, str, bool, Optional[float]]:
    """Wybór zwycięzcy + fuzja lewej/prawej krawędzi z HSV/morph."""
    h, w = image_bgr.shape[:2]
    ref_left = _left_anchor_x_from_rows(rows)
    ref_right = _right_anchor_x_from_rows(rows, w)
    ref_top = _top_anchor_y_from_rows(rows)
    ref_bottom = _bottom_anchor_y_from_rows(rows)
    ref_w_frac = _ref_outer_width_frac(rows, w)

    for r in rows:
        r['_ref_left_x'] = ref_left
        r['_ref_right_x'] = ref_right
        r['_ref_top_y'] = ref_top
        r['_ref_bottom_y'] = ref_bottom
        r['_ref_w_frac'] = ref_w_frac
        r['rank_score'] = _score_align_candidate_row(r, w, ref_w_frac=ref_w_frac)

    best = min(rows, key=lambda r: float(r['rank_score']))
    best['chosen'] = True
    for r in rows:
        if r is not best:
            r['chosen'] = False

    corners = np.asarray(best['corners'], dtype=np.float32)
    label = str(best['label'])
    fused = False

    if ref_left is not None:
        cur_left = float(np.min(corners[:, 0]))
        if cur_left > ref_left + 45.0:
            corners = _fuse_quad_left_edge(corners, ref_left, (h, w))
            fused = True
            if '+fuse_left' not in label:
                label = f'{label}+fuse_left'

    if ref_right is not None:
        cur_right = float(np.max(corners[:, 0]))
        if cur_right < ref_right - 38.0:
            corners = _fuse_quad_right_edge(corners, ref_right, (h, w))
            fused = True
            if '+fuse_right' not in label:
                label = f'{label}+fuse_right'
        elif cur_right > ref_right + 52.0:
            corners = _trim_quad_right_edge(corners, ref_right, (h, w))
            fused = True
            if '+trim_right' not in label:
                label = f'{label}+trim_right'

    if ref_top is not None:
        cur_top = float(min(corners[:, 1]))
        if cur_top > ref_top + 40.0:
            corners = _fuse_quad_top_edge(corners, ref_top, (h, w))
            fused = True
            if '+fuse_top' not in label:
                label = f'{label}+fuse_top'

    if ref_bottom is not None:
        cur_bottom = float(np.max(corners[:, 1]))
        if cur_bottom < ref_bottom - 28.0:
            corners = _fuse_quad_bottom_edge(corners, ref_bottom, (h, w))
            fused = True
            if '+fuse_bottom' not in label:
                label = f'{label}+fuse_bottom'

    best['corners'] = corners
    rows.sort(key=lambda r: (0.0 if r.get('chosen') else 1.0, float(r.get('rank_score', 999))))
    return corners, label, fused, ref_left


def probe_alignment_corner_candidates(
    image_bgr: np.ndarray,
    k: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    *,
    fast: bool = True,
) -> Tuple[List[Dict[str, Any]], Optional[Tuple[int, int, int, int]], str]:
    """Kandydaci zewnętrznych rogów — bez niebieskiego ROI (drugi element zwracany = None)."""
    rows, label = probe_yellow_corners_baseline(image_bgr, k, dist, fast=fast)
    if not rows:
        return [], None, 'none'
    return rows, None, label


def _acquire_corners_align_hybrid(
    image_bgr: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    *,
    full_probe: bool = True,
) -> Tuple[Optional[np.ndarray], str, Dict[str, Any], List[Dict[str, Any]]]:
    """Live: zewnętrzne rogi siatki (outer_corners) — bez niebieskiego ROI."""
    from release.outer_corners import detect_outer_corners

    h, w = image_bgr.shape[:2]
    meta: Dict[str, Any] = {'method': 'align_hybrid', 'roi_source': 'none'}

    if (
        not full_probe
        and _TRACKER._last_good is not None
        and _TRACKER._misses == 0
        and _TRACKER._last_reproj < 26.0
    ):
        fast_c = _TRACKER._last_good.copy()
        meta.update({
            'reproj_mean_px': float(_TRACKER._last_reproj),
            'corner_score': float(_TRACKER._last_reproj),
            'tracker_fast': 1.0,
        })
        return fast_c, 'tracker_fast', meta, []

    from release.outer_corners import enhance_for_corner_probe

    probe_img = enhance_for_corner_probe(image_bgr)
    best_corners, best_label, debug_rows, pick_meta = detect_outer_corners(
        image_bgr, k, dist, fast=True, refine=False, probe_bgr=probe_img, enhance_probe=False,
    )
    meta.update(pick_meta)
    if best_corners is None or not debug_rows:
        meta['fail'] = pick_meta.get('fail', 'no_corners')
        return None, 'none', meta, debug_rows or []

    best_row = next((r for r in debug_rows if r.get('chosen')), debug_rows[0])
    fused_left = '+fuse_left' in best_label
    ref_left = best_row.get('_ref_left_x')
    best_reproj = float(best_row.get('reproj_mean_px', 999.0))

    if fused_left:
        c_can, detail = _evaluate_panel_roi_candidate(
            image_bgr, best_corners, k, dist,
            label=best_label, preserve_geometry=True,
        )
        if c_can is not None:
            best_corners = c_can
            best_reproj = float(detail.get('reproj_mean_px', best_reproj))

    meta['corner_align_label'] = best_label
    meta['yellow_source'] = 'outer_corners'
    meta['align_basis'] = best_row.get('align_basis', '')
    meta['align_confidence'] = best_row.get('align_confidence', 0.0)
    meta['panel_width_frac'] = _quad_width_frac_live(best_corners, w)
    meta['reproj_mean_px'] = best_reproj
    meta['corner_score'] = best_reproj
    meta['grid_structure_score'] = best_row.get('grid_structure_score')
    meta['panel_interior_score'] = best_row.get('panel_interior_score')
    meta['corner_shape'] = 'trapezoid' if not _is_near_axis_aligned_box(best_corners) else 'axis_box'
    if ref_left is not None:
        meta['grid_x_left'] = float(ref_left)
    if fused_left:
        meta['left_edge_fused'] = 1.0
        meta['external_left_gap_px'] = max(
            0.0, float(np.min(best_corners[:, 0])) - float(ref_left) - 10.0,
        )

    return best_corners.astype(np.float32), best_label, meta, debug_rows


def probe_line_grid_corner_candidates(
    image_bgr: np.ndarray,
    k: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """line_grid v3 rogi na obrazie roboczym (~1280 px), ocena na pełnym kadrze."""
    from module_geom.line_grid import _gather_corner_candidates

    h, w = image_bgr.shape[:2]
    if k is None or dist is None:
        k, dist = default_intrinsics((h, w))

    work, scale, ox, oy = _prepare_detection_image(image_bgr)
    wk, wdist = default_intrinsics(work.shape[:2])
    rows: List[Dict[str, Any]] = []
    seen: List[np.ndarray] = []

    for label, raw_work in _gather_corner_candidates(work, [], wk, wdist):
        full = _quad_to_full_image(raw_work, scale=scale, ox=ox, oy=oy)
        if any(float(np.max(np.abs(full - s))) < 12.0 for s in seen):
            continue
        tag = f'lg_{label}'
        c_can, detail = _evaluate_panel_roi_candidate(
            image_bgr, full, k, dist, label=tag, preserve_geometry=True,
        )
        if c_can is None:
            c_can, detail = _evaluate_candidate(image_bgr, full, k, dist, label=tag)
        if c_can is None:
            continue
        seen.append(c_can)
        rows.append({
            'label': tag,
            'corners': c_can,
            'rank_score': float(detail['reproj_mean_px']),
            'chosen': False,
            'reproj_mean_px': float(detail.get('reproj_mean_px', 999.0)),
            'grid_structure_score': detail.get('grid_structure_score', 0.0),
            'panel_interior_score': detail.get('panel_interior_score', 0.0),
            **detail,
        })

    if rows:
        best = min(rows, key=lambda r: float(r['reproj_mean_px']))
        if float(best['reproj_mean_px']) <= _MAX_LIVE_REPROJ_FALLBACK_PX:
            best['chosen'] = True
    rows.sort(key=lambda r: (0.0 if r.get('chosen') else 1.0, float(r['reproj_mean_px'])))
    return rows


def probe_line_grid_roi_candidates(
    image_bgr: np.ndarray,
    k: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Tuple[int, int, int, int]], str]:
    """
    line_grid v3 na wycinku ROI + trapez alignment + LSD zewnętrzne + ekspansja do ROI.
    """
    from module_geom.line_grid import _gather_corner_candidates
    from release.panel_roi import (
        crop_panel_roi,
        estimate_panel_roi_with_quad,
        map_quad_crop_to_full,
    )

    h, w = image_bgr.shape[:2]
    if k is None or dist is None:
        k, dist = default_intrinsics((h, w))

    roi, roi_src, align_quad = estimate_panel_roi_with_quad(image_bgr)
    if align_quad is not None:
        from release.grid_outer_quad import roi_bbox_from_quad
        roi = roi_bbox_from_quad(align_quad, (h, w), margin_frac=0.04)
    if roi is None:
        rows = probe_line_grid_corner_candidates(image_bgr, k, dist)
        for r in rows:
            r['label'] = f"lg_full_{r['label']}"
        return rows, None, roi_src

    rows: List[Dict[str, Any]] = []
    seen: List[np.ndarray] = []

    from release.panel_roi import roi_bounds_to_quad

    roi_box = roi_bounds_to_quad(roi)
    _try_add_corner_row(
        rows, seen, image_bgr, k, dist, roi_box, 'roi_bbox_quad',
        roi=roi, roi_src=roi_src,
    )

    if align_quad is not None:
        _try_add_corner_row(
            rows, seen, image_bgr, k, dist, align_quad, f'roi_{roi_src}_align_quad',
            roi=roi, roi_src=roi_src,
        )
        _append_outer_corner_variants(
            rows, seen, image_bgr, k, dist, align_quad, f'roi_{roi_src}_align_quad',
            roi=roi, roi_src=roi_src,
        )

    crop, rx0, ry0 = crop_panel_roi(image_bgr, roi)
    work, scale, _, _ = _prepare_detection_image(crop, top_frac=0.02, bottom_frac=0.03)
    wk, wdist = default_intrinsics(work.shape[:2])

    lsd_q = _quad_from_lsd_outer_grid(work)
    if lsd_q is not None:
        full_lsd = map_quad_crop_to_full(lsd_q, roi_x0=rx0, roi_y0=ry0, work_scale=scale)
        _try_add_corner_row(
            rows, seen, image_bgr, k, dist, full_lsd, 'lg_roi_lsd_outer',
            roi=roi, roi_src=roi_src,
        )
        _append_outer_corner_variants(
            rows, seen, image_bgr, k, dist, full_lsd, 'lg_roi_lsd_outer',
            roi=roi, roi_src=roi_src,
        )

    trap = _grid_trapezoid_from_white_profile(work)
    if trap is not None:
        full_trap = map_quad_crop_to_full(trap, roi_x0=rx0, roi_y0=ry0, work_scale=scale)
        _try_add_corner_row(
            rows, seen, image_bgr, k, dist, full_trap, 'lg_roi_white_trap',
            roi=roi, roi_src=roi_src,
        )

    for label, raw_work in _gather_corner_candidates(work, [], wk, wdist):
        raw_crop = raw_work.astype(np.float32)
        if scale > 1e-6 and abs(scale - 1.0) > 1e-4:
            raw_crop = raw_crop / scale
        full = map_quad_crop_to_full(raw_crop, roi_x0=rx0, roi_y0=ry0, work_scale=scale)
        tag = f'lg_roi_{label}'
        _try_add_corner_row(rows, seen, image_bgr, k, dist, full, tag, roi=roi, roi_src=roi_src)
        _append_outer_corner_variants(
            rows, seen, image_bgr, k, dist, full, tag, roi=roi, roi_src=roi_src,
        )

    _finalize_rows_with_roi(rows, roi)
    return rows, roi, roi_src


def probe_roi_hybrid_corner_candidates(
    image_bgr: np.ndarray,
    k: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Tuple[int, int, int, int]], str]:
    """Kandydaci do tablicy debug; wybór żółtego = _acquire_corners_roi_panel_first (nie ten merge)."""
    roi_rows, roi, roi_src = probe_line_grid_roi_candidates(image_bgr, k, dist)
    for r in roi_rows:
        r['chosen'] = False
    return roi_rows, roi, roi_src


def _quad_roi_margin_fracs(
    quad: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> Tuple[float, float, float, float]:
    """Luka żółty vs niebieski ROI jako ułamek szer./wys. ROI (0 = dotyka krawędzi)."""
    x0, y0, x1, y1 = roi
    rw = max(1.0, float(x1 - x0))
    rh = max(1.0, float(y1 - y0))
    q = quad.astype(np.float32)
    qx0, qy0 = float(q[:, 0].min()), float(q[:, 1].min())
    qx1, qy1 = float(q[:, 0].max()), float(q[:, 1].max())
    ml = max(0.0, (qx0 - float(x0)) / rw)
    mr = max(0.0, (float(x1) - qx1) / rw)
    mt = max(0.0, (qy0 - float(y0)) / rh)
    mb = max(0.0, (float(y1) - qy1) / rh)
    return ml, mr, mt, mb


def _expand_align_toward_roi(
    quad: np.ndarray,
    roi: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    *,
    image_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Trapez align → pełny panel; perspektywa zachowana (nie prostokąt osiowy)."""
    from release.panel_roi import expand_quad_adaptive

    h, w = image_shape[:2]
    if image_bgr is not None:
        return expand_quad_adaptive(image_bgr, quad, outer_margin_frac=0.05, roi=roi, for_corners=True)
    stub = np.zeros((h, w, 3), dtype=np.uint8)
    return expand_quad_adaptive(stub, quad, outer_margin_frac=0.05, roi=roi, for_corners=True)


def _score_roi_panel_candidate(
    reproj: float,
    coverage: float,
    *,
    label: str = '',
    grid_score: float = 0.0,
    external_penalty: float = 0.0,
    width_fill: float = 1.0,
) -> float:
    """Trapez z pełną szerokością siatki > niski reproj na wąskim wewnętrznym czworokącie."""
    cov = float(coverage)
    gap = max(0.0, 0.97 - cov)
    score = float(reproj) + gap * 120.0 + max(0.0, 0.88 - cov) * 28.0
    score += float(external_penalty)
    if width_fill < 0.82:
        score += (0.82 - width_fill) * 420.0
    elif width_fill < 0.92:
        score += (0.92 - width_fill) * 180.0
    if 'grid_outer' in label or 'lsd_outer' in label:
        score -= 22.0
    if 'hsv' in label or 'panel_align' in label or 'align' in label:
        score += 35.0
    if 'roi_bbox' in label or 'bbox_quad' in label:
        score += 25.0
    if cov >= 0.995 and external_penalty < 8.0:
        score -= 10.0
    score -= min(12.0, float(grid_score) * 8.0)
    return score


def _is_near_axis_aligned_box(quad: np.ndarray, tol_deg: float = 6.0) -> bool:
    """Czy czworokąt jest prawie osiowy prostokąt (nie trapez perspektywiczny)."""
    q = quad.astype(np.float32)
    for i in range(4):
        p0 = q[i]
        p1 = q[(i + 1) % 4]
        dx = abs(float(p1[0] - p0[0]))
        dy = abs(float(p1[1] - p0[1]))
        if dx < 2.0 and dy < 2.0:
            continue
        ang = abs(math.degrees(math.atan2(dy, max(dx, 1e-6))))
        if ang > tol_deg and abs(ang - 90.0) > tol_deg:
            return False
    return True


def _acquire_corners_roi_panel_first(
    image_bgr: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    *,
    try_lsd_on_crop: bool = True,
) -> Tuple[Optional[np.ndarray], str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Live: skrajne linie siatki → asymetryczny trapez → ROI = bbox quada.
    HSV/align/line_grid na cropie tylko jako kandydaci z karą zewnętrzną (bez expand od środka).
    """
    from release.grid_outer_quad import (
        detect_outer_grid_bounds,
        detect_panel_quad_and_roi,
        external_penalty_score,
        quad_from_outer_grid_bounds,
        quad_width_frac,
        roi_bbox_from_quad,
    )
    from release.panel_roi import (
        crop_panel_roi,
        map_quad_crop_to_full,
        quad_coverage_vs_roi,
        roi_bounds_to_quad,
    )

    h, w = image_bgr.shape[:2]
    img_shape = (h, w)
    meta: Dict[str, Any] = {'method': 'grid_outer_first'}

    grid_bounds = detect_outer_grid_bounds(image_bgr)
    grid_quad, _grid_roi, grid_src, _ = detect_panel_quad_and_roi(image_bgr)
    panel_quad = grid_quad

    debug_rows, _, _ = probe_line_grid_roi_candidates(image_bgr, k, dist)
    for r in debug_rows:
        r['chosen'] = False

    roi_src = grid_src
    min_w_frac = 0.42
    if grid_bounds is not None:
        meta['grid_x_left'] = float(grid_bounds.x_left_line)
        meta['grid_x_right'] = float(grid_bounds.x_right_line)

    best_corners: Optional[np.ndarray] = None
    best_label = 'none'
    best_reproj = 999.0
    best_cov = 0.0
    best_score = 1e9

    def _width_fill_vs_grid(quad: np.ndarray) -> float:
        if grid_bounds is None:
            return 1.0
        span = max(1.0, grid_bounds.x_right_line - grid_bounds.x_left_line)
        qx0, qx1 = float(quad[:, 0].min()), float(quad[:, 0].max())
        covered = max(0.0, min(qx1, grid_bounds.x_right_line) - max(qx0, grid_bounds.x_left_line))
        return float(np.clip(covered / span, 0.0, 1.0))

    def consider(q_raw: np.ndarray, label: str, *, preserve_geometry: bool = False) -> None:
        nonlocal best_corners, best_label, best_reproj, best_cov, best_score
        q_in = q_raw.astype(np.float32)
        if quad_width_frac(q_in, img_shape) < min_w_frac and 'dark_panel' not in label:
            return
        c_can, detail = _evaluate_panel_roi_candidate(
            image_bgr,
            q_in,
            k,
            dist,
            label=label,
            preserve_geometry=preserve_geometry or 'grid_outer' in label,
        )
        if c_can is None:
            return
        reproj_v = float(detail['reproj_mean_px'])
        if reproj_v > 135.0:
            return
        cand_roi = roi_bbox_from_quad(c_can, img_shape, margin_frac=0.04)
        cov_v = quad_coverage_vs_roi(c_can, cand_roi)
        grid_sc = float(detail.get('grid_structure_score', 0.0))
        ext_pen = external_penalty_score(c_can, grid_bounds)
        w_fill = _width_fill_vs_grid(c_can)
        score = _score_roi_panel_candidate(
            reproj_v,
            cov_v,
            label=label,
            grid_score=grid_sc,
            external_penalty=ext_pen,
            width_fill=w_fill,
        )
        if quad_width_frac(c_can, img_shape) < min_w_frac:
            score += 800.0
        if score < best_score:
            best_score = score
            best_corners = c_can
            best_label = label
            best_reproj = reproj_v
            best_cov = cov_v

    if grid_quad is not None:
        consider(grid_quad, 'grid_outer_lsd', preserve_geometry=True)
    if grid_bounds is not None:
        q_bounds = quad_from_outer_grid_bounds(grid_bounds, (h, w), cell_margin_frac=0.03)
        if q_bounds is not None:
            consider(q_bounds, 'grid_outer_bounds', preserve_geometry=True)

    crop_roi = roi_bbox_from_quad(grid_quad, img_shape) if grid_quad is not None else None
    if try_lsd_on_crop and crop_roi is not None:
        crop, rx0, ry0 = crop_panel_roi(image_bgr, crop_roi)
        work, scale, _, _ = _prepare_detection_image(crop, top_frac=0.02, bottom_frac=0.03)
        lsd_q = _quad_from_lsd_outer_grid(work)
        if lsd_q is not None:
            full_lsd = map_quad_crop_to_full(lsd_q, roi_x0=rx0, roi_y0=ry0, work_scale=scale)
            consider(full_lsd, 'panel_lsd_crop')

    for r in debug_rows:
        q = r.get('corners')
        if q is None:
            continue
        lbl = str(r.get('label', 'lg_roi'))
        consider(np.asarray(q, dtype=np.float32), lbl)

    if best_corners is None and grid_quad is not None:
        consider(grid_quad, f'{grid_src}_fallback', preserve_geometry=True)

    if best_corners is None:
        meta['fail'] = 'no_corners'
        return None, 'none', meta, debug_rows

    roi = roi_bbox_from_quad(best_corners, img_shape, margin_frac=0.04)
    meta['panel_roi'] = [int(x) for x in roi]
    meta['roi_source'] = roi_src
    meta['roi_unified'] = True
    best_cov = quad_coverage_vs_roi(best_corners, roi)

    meta['corner_shape'] = 'trapezoid' if not _is_near_axis_aligned_box(best_corners) else 'axis_box'
    if grid_bounds is not None:
        from release.grid_outer_quad import external_coverage_penalty
        lg, rg, tg, bg = external_coverage_penalty(best_corners, grid_bounds)
        meta['external_left_gap_px'] = lg
        meta['external_right_gap_px'] = rg

    debug_rows.append({
        'label': best_label,
        'corners': best_corners,
        'chosen': True,
        'rank_score': best_score,
        'reproj_mean_px': best_reproj,
        'roi_coverage': best_cov,
        'roi_source': roi_src,
    })
    debug_rows.sort(key=lambda r: (0.0 if r.get('chosen') else 1.0, float(r.get('rank_score', 999))))

    meta['reproj_mean_px'] = best_reproj
    meta['corner_score'] = best_reproj
    meta['roi_coverage'] = float(best_cov)
    meta['candidates_tried'] = len(debug_rows)
    return best_corners.astype(np.float32), best_label, meta, debug_rows


def probe_hybrid_corner_candidates(
    image_bgr: np.ndarray,
    k: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """live_corners + line_grid v3 — wybór min reproj (preferuj perspektywę nad profilem)."""
    live_rows = probe_all_corner_candidates(image_bgr, k, dist)
    lg_rows = probe_line_grid_corner_candidates(image_bgr, k, dist)
    merged: List[Dict[str, Any]] = []
    for row in live_rows + lg_rows:
        merged.append({**row, 'chosen': False})

    viable = [r for r in merged if float(r.get('reproj_mean_px', 999)) <= _MAX_LIVE_REPROJ_FALLBACK_PX]
    if not viable:
        merged.sort(key=lambda r: float(r['reproj_mean_px']))
        return merged

    def sort_key(r: Dict[str, Any]) -> Tuple[float, float]:
        reproj = float(r['reproj_mean_px'])
        pref = 0.0
        lbl = str(r['label'])
        if lbl.startswith('lg_') and 'geom_vp' in lbl:
            pref -= 2.0
        if lbl.startswith('lsd_outer') or lbl.startswith('white_trap'):
            pref -= 1.5
        if lbl == 'white_hlines_profile' and reproj > 28.0:
            pref += 8.0
        if lbl.startswith('lg_') and 'white_grid' in lbl:
            pref += 5.0
        return (reproj + pref, pref)

    viable.sort(key=sort_key)
    viable[0]['chosen'] = True
    for r in merged:
        r['chosen'] = r is viable[0]
    merged.sort(key=lambda r: (0.0 if r.get('chosen') else 1.0, float(r['reproj_mean_px'])))
    return merged


def acquire_corners_for_live(
    image_bgr: np.ndarray,
    k: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    *,
    corner_mode: str = 'live',
) -> Tuple[Optional[np.ndarray], str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    corner_mode:
      live — release/live_corners v4 (domyślnie)
      line_grid — detect_corners_line_grid v3 na obrazie roboczym
      auto — detect_panel_corners_for_module_b(prefer_line_grid=True)
      hybrid — konkurencja live + line_grid
    """
    mode = (corner_mode or 'live').strip().lower()
    if mode not in _LIVE_CORNER_MODES:
        raise ValueError(f'corner_mode must be one of {sorted(_LIVE_CORNER_MODES)}')

    meta: Dict[str, Any] = {'corner_mode': mode}
    h, w = image_bgr.shape[:2]
    if k is None or dist is None:
        k, dist = default_intrinsics((h, w))

    if mode == 'auto':
        from module_geom.pipeline import prepare_image_geom
        from module_panel.analyze import detect_panel_corners_for_module_b

        img_work, k_work, dist_work, und = prepare_image_geom(image_bgr, k, dist, calib_path=None)
        corners, src = detect_panel_corners_for_module_b(
            img_work, [], prefer_line_grid=True, k=k_work, dist=dist_work,
        )
        meta.update(und)
        rows: List[Dict[str, Any]] = []
        if corners is not None:
            c_can, detail = _evaluate_candidate(image_bgr, corners, k, dist, label=src or 'auto')
            if c_can is not None:
                corners = c_can
                rows.append({
                    'label': src or 'auto',
                    'corners': corners,
                    'chosen': True,
                    'rank_score': float(detail['reproj_mean_px']),
                    **detail,
                })
        if corners is None or not rows:
            meta['fail'] = 'no_corners'
            return None, 'none', meta, rows
        meta.update({k: v for k, v in rows[0].items() if k != 'corners'})
        return corners.astype(np.float32), str(src or 'auto'), meta, rows

    if mode in ('align_hybrid', 'outer_corners'):
        global _align_probe_frame_idx
        _align_probe_frame_idx += 1
        full_probe = (_align_probe_frame_idx % _ALIGN_FULL_PROBE_EVERY) == 0
        corners_out, label_out, meta, rows = _acquire_corners_align_hybrid(
            image_bgr, k, dist, full_probe=full_probe,
        )
        if corners_out is None:
            meta['fail'] = meta.get('fail', 'no_corners')
            return None, 'none', meta, rows
        return corners_out, label_out, meta, rows

    if mode == 'yolo_pose':
        from release.yolo_pose_live import detect_corners_yolo_pose

        corners_out, label_out, ymeta = detect_corners_yolo_pose(image_bgr)
        rows: List[Dict[str, Any]] = []
        if corners_out is not None:
            rows.append({
                'label': label_out,
                'corners': corners_out,
                'chosen': True,
                'rank_score': float(ymeta.get('reproj_mean_px', 999.0)),
                **ymeta,
            })
        meta.update(ymeta)
        if corners_out is None:
            meta['fail'] = ymeta.get('fail', 'no_corners')
            return None, 'none', meta, rows
        return corners_out, label_out, meta, rows

    if mode in ('roi_hybrid', 'roi_line_grid'):
        corners_out, label_out, meta, rows = _acquire_corners_roi_panel_first(
            image_bgr, k, dist, try_lsd_on_crop=(mode == 'roi_hybrid'),
        )
        if corners_out is None:
            meta['fail'] = meta.get('fail', 'no_corners')
            return None, 'none', meta, rows
        return corners_out, label_out, meta, rows

    roi_bounds: Optional[Tuple[int, int, int, int]] = None
    roi_source = 'none'

    if mode == 'line_grid':
        rows = probe_line_grid_corner_candidates(image_bgr, k, dist)
    elif mode == 'hybrid':
        rows = probe_hybrid_corner_candidates(image_bgr, k, dist)
    else:
        rows = probe_all_corner_candidates(image_bgr, k, dist)

    chosen = next((r for r in rows if r.get('chosen')), None)
    if chosen is None:
        meta['fail'] = 'no_corners'
        return None, 'none', meta, rows

    corners_out = chosen['corners'].astype(np.float32)
    label_out = str(chosen['label'])
    reproj_out = float(chosen['reproj_mean_px'])

    meta.update({k: v for k, v in chosen.items() if k not in ('corners', 'chosen')})
    meta['reproj_mean_px'] = reproj_out
    meta['corner_score'] = reproj_out
    meta['candidates_tried'] = len(rows)
    if roi_bounds is not None:
        meta['panel_roi'] = [int(x) for x in roi_bounds]
        meta['roi_source'] = roi_source
        if 'roi_coverage' not in meta:
            from release.panel_roi import quad_coverage_vs_roi
            meta['roi_coverage'] = float(quad_coverage_vs_roi(corners_out, roi_bounds))
    return corners_out, label_out, meta, rows


def detect_corners_live(
    image_bgr: np.ndarray,
    k: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    *,
    return_all_candidates: bool = False,
    use_tracker: bool = True,
    corner_mode: str = 'live',
) -> Tuple[Optional[np.ndarray], str, Dict[str, float | str | List[Dict[str, Any]]]]:
    meta: Dict[str, float | str | List[Dict[str, Any]]] = {
        'method': 'live_corners_v4' if corner_mode == 'live' else f'live_corners_v4_{corner_mode}',
    }
    corners, label, pick_meta, rows = acquire_corners_for_live(
        image_bgr, k, dist, corner_mode=corner_mode,
    )
    meta.update(pick_meta)
    if rows:
        meta['candidate_rows'] = rows
    if corners is None:
        meta['fail'] = meta.get('fail', 'no_live_corners')
        if use_tracker:
            held, _ = _TRACKER.apply(None, 999.0, image_shape=image_bgr.shape[:2])
            if held is not None:
                meta['tracker_held'] = 1.0
                roi = _TRACKER.panel_roi(image_bgr.shape[:2])
                if roi is not None:
                    meta['panel_roi'] = [int(x) for x in roi]
                    meta['roi_unified'] = True
                return held, 'tracker_hold', meta
        return None, 'none', meta

    reproj = float(meta.get('reproj_mean_px', 999.0))
    meta['corner_score'] = reproj
    if return_all_candidates:
        meta['all_candidates'] = [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
            for r in rows
        ]
    if use_tracker:
        corners, held = _TRACKER.apply(
            corners, reproj, image_shape=image_bgr.shape[:2],
        )
        if held:
            meta['tracker_held'] = 1.0
        if corners is None:
            meta['fail'] = 'no_live_corners'
            return None, 'none', meta
    return corners, str(label), meta
