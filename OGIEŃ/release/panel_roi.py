"""Dwustopniowość live: ROI panelu → line_grid v3 na wycinku (jak w literaturze landing pad)."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import pipeline_competition as pc

from module_pose.api import _rough_quad_from_white_hlines, strip_camera_overlays


def _bounds_from_quad(
    quad: np.ndarray,
    margin_frac: float = 0.10,
    *,
    margin_left: Optional[float] = None,
    margin_right: Optional[float] = None,
    margin_top: Optional[float] = None,
    margin_bottom: Optional[float] = None,
) -> Tuple[int, int, int, int]:
    q = quad.astype(np.float32)
    x0, y0 = float(q[:, 0].min()), float(q[:, 1].min())
    x1, y1 = float(q[:, 0].max()), float(q[:, 1].max())
    w_span = max(1.0, x1 - x0)
    h_span = max(1.0, y1 - y0)
    ml = margin_frac if margin_left is None else margin_left
    mr = margin_frac if margin_right is None else margin_right
    mt = margin_frac if margin_top is None else margin_top
    mb = margin_frac if margin_bottom is None else margin_bottom
    x0 -= ml * w_span
    x1 += mr * w_span
    y0 -= mt * h_span
    y1 += mb * h_span
    return (int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1)))


def _merge_bounds(
    boxes: List[Tuple[int, int, int, int]],
) -> Tuple[int, int, int, int]:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return x0, y0, x1, y1


def _clip_bounds(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x0 = int(np.clip(x0, 0, w - 2))
    y0 = int(np.clip(y0, 0, h - 2))
    x1 = int(np.clip(x1, x0 + 2, w - 1))
    y1 = int(np.clip(y1, y0 + 2, h - 1))
    return x0, y0, x1, y1


def _bbox_aspect_and_area(
    box: Tuple[int, int, int, int],
    w: int,
    h: int,
) -> Tuple[float, float]:
    x0, y0, x1, y1 = box
    bw = max(1.0, float(x1 - x0))
    bh = max(1.0, float(y1 - y0))
    asp = max(bw, bh) / bh if bw < bh else bw / bh
    area = bw * bh / max(1.0, float(w * h))
    return asp, area


def _roi_sane(
    box: Tuple[int, int, int, int],
    w: int,
    h: int,
    *,
    min_area_frac: float,
    max_area_frac: float = 0.96,
) -> bool:
    asp, area = _bbox_aspect_and_area(box, w, h)
    if area < min_area_frac or area > max_area_frac:
        return False
    return 1.25 <= asp <= 3.8


def _contains_point(
    box: Tuple[int, int, int, int],
    px: float,
    py: float,
) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= px <= x1 and y0 <= py <= y1


def _roi_bounds_align_adaptive(
    image_bgr: np.ndarray,
    align_quad: np.ndarray,
    *,
    outer_margin_frac: float = 0.06,
) -> Tuple[int, int, int, int]:
    """
    align_hybrid widzi wewnętrzną białą siatkę (~30–35% szer. kadru z bliska).
    Panel fizyczny (czarna ramka) jest szerszy — skaluj od środka wg. zajętości kadru.
    """
    h, w = image_bgr.shape[:2]
    q = align_quad.astype(np.float32)
    cur_w = float(q[:, 0].max() - q[:, 0].min()) / max(1.0, float(w))
    cur_h = float(q[:, 1].max() - q[:, 1].min()) / max(1.0, float(h))

    if cur_w > 0.62:
        scale_w, scale_h = 1.06, 1.05
    elif cur_w > 0.38:
        scale_w = min(2.25, 0.92 / max(0.12, cur_w))
        scale_h = min(1.22, 0.88 / max(0.12, cur_h))
    else:
        scale_w = min(3.4, 0.94 / max(0.12, cur_w))
        scale_h = min(1.35, 0.90 / max(0.12, cur_h))

    center = q.mean(axis=0, keepdims=True)
    out = center + (q - center) * np.array([scale_w, scale_h], dtype=np.float32)
    out[:, 0] = np.clip(out[:, 0], 0.0, float(w - 1))
    out[:, 1] = np.clip(out[:, 1], 0.0, float(h - 1))
    x0, y0 = float(out[:, 0].min()), float(out[:, 1].min())
    x1, y1 = float(out[:, 0].max()), float(out[:, 1].max())
    span_w = max(1.0, x1 - x0)
    span_h = max(1.0, y1 - y0)
    bx0 = int(np.floor(x0 - outer_margin_frac * span_w))
    by0 = int(np.floor(y0 - outer_margin_frac * span_h))
    bx1 = int(np.ceil(x1 + outer_margin_frac * span_w))
    by1 = int(np.ceil(y1 + outer_margin_frac * span_h))
    return _clip_bounds(bx0, by0, bx1, by1, w, h)


def _roi_bounds_from_white_detectors(
    image_bgr: np.ndarray,
    *,
    margin_frac: float = 0.06,
) -> List[Tuple[Tuple[int, int, int, int], str]]:
    """Biała siatka 10×10 — działa najlepiej z dalszej odległości."""
    h, w = image_bgr.shape[:2]
    out: List[Tuple[Tuple[int, int, int, int], str]] = []

    try:
        from module_pose.grid_corners import detect_corners_white_grid

        wg = detect_corners_white_grid(image_bgr)
        if wg is not None:
            box = _clip_bounds(*_bounds_from_quad(wg, margin_frac), w, h)
            out.append((box, 'white_grid'))
    except ImportError:
        pass

    work, ox, oy = strip_camera_overlays(image_bgr, top_frac=0.07, bottom_frac=0.10)
    wh = _rough_quad_from_white_hlines(work)
    if wh is not None:
        wh_full = wh.copy()
        wh_full[:, 0] += float(ox)
        wh_full[:, 1] += float(oy)
        quad = pc.order_points(wh_full.astype(np.float32))
        box = _clip_bounds(*_bounds_from_quad(quad, margin_frac), w, h)
        out.append((box, 'white_hlines'))
    return out


def _select_widest_panel_roi(
    candidates: List[Tuple[Tuple[int, int, int, int], str]],
    image_bgr: np.ndarray,
    *,
    align_quad: Optional[np.ndarray] = None,
    min_area_frac: float = 0.04,
) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    """Wybierz najszerszy sensowny ROI zawierający środek align (jeśli jest)."""
    h, w = image_bgr.shape[:2]
    seed_x = seed_y = None
    if align_quad is not None:
        seed_x = float(align_quad[:, 0].mean())
        seed_y = float(align_quad[:, 1].mean())

    valid: List[Tuple[Tuple[int, int, int, int], str, float]] = []
    for box, name in candidates:
        if not _roi_sane(box, w, h, min_area_frac=min_area_frac):
            continue
        if seed_x is not None and not _contains_point(box, seed_x, seed_y):
            continue
        valid.append((box, name, float(box[2] - box[0])))

    if not valid:
        return None, 'none'
    box, name, _ = max(valid, key=lambda t: t[2])
    return box, name


def roi_area(roi: Tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = roi
    return float(max(1, x1 - x0) * max(1, y1 - y0))


def quad_coverage_vs_roi(quad: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
    """
    Jak bardzo żółty box wypełnia niebieski ROI (0..1).
    Używa rozpiętości X/Y — trapez pod kątem ma małe pole, ale może obejmować cały panel.
    """
    x0, y0, x1, y1 = roi
    rw = max(1.0, float(x1 - x0))
    rh = max(1.0, float(y1 - y0))
    q = quad.astype(np.float32)
    qx0, qy0 = float(q[:, 0].min()), float(q[:, 1].min())
    qx1, qy1 = float(q[:, 0].max()), float(q[:, 1].max())
    w_fill = max(0.0, min(qx1, float(x1)) - max(qx0, float(x0))) / rw
    h_fill = max(0.0, min(qy1, float(y1)) - max(qy0, float(y0))) / rh
    return float(0.5 * (w_fill + h_fill))


def _adaptive_panel_scales(
    quad: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[float, float]:
    """Skala szer./wys. jak dla niebieskiego ROI — ta sama logika odległości."""
    h, w = image_shape[:2]
    q = quad.astype(np.float32)
    cur_w = float(q[:, 0].max() - q[:, 0].min()) / max(1.0, float(w))
    cur_h = float(q[:, 1].max() - q[:, 1].min()) / max(1.0, float(h))
    if cur_w > 0.62:
        return 1.06, 1.05
    if cur_w > 0.38:
        return min(2.25, 0.92 / max(0.12, cur_w)), min(1.22, 0.88 / max(0.12, cur_h))
    return (
        min(3.4, 0.94 / max(0.12, cur_w)),
        min(1.35, 0.90 / max(0.12, cur_h)),
    )


def expand_quad_adaptive(
    image_bgr: np.ndarray,
    quad: np.ndarray,
    *,
    outer_margin_frac: float = 0.05,
    roi: Optional[Tuple[int, int, int, int]] = None,
    for_corners: bool = False,
) -> np.ndarray:
    """
    Powiększ trapez od środka — zachowuje nachylenie (4 rogi + linie między nimi).

    for_corners=False: duży bbox pod niebieski ROI (szeroki prostokąt).
    for_corners=True: umiarkowane skalowanie — żółty trapez w perspektywie, bez przyklejenia do krawędzi kadru.
    """
    h, w = image_bgr.shape[:2]
    q = pc.order_points(quad.astype(np.float32))
    scale_w, scale_h = _adaptive_panel_scales(q, (h, w))
    if for_corners:
        scale_w = min(scale_w, 2.15)
        scale_h = min(scale_h, 1.18)
    else:
        scale_w = min(scale_w, 2.65)
        scale_h = min(scale_h, 1.22)

    if roi is not None:
        x0, y0, x1, y1 = roi
        inset = 0.05 if for_corners else 0.04
        rx0 = float(x0) + inset * (x1 - x0)
        rx1 = float(x1) - inset * (x1 - x0)
        ry0 = float(y0) + inset * (y1 - y0)
        ry1 = float(y1) - inset * (y1 - y0)
        tgt_w = max(1.0, rx1 - rx0)
        tgt_h = max(1.0, ry1 - ry0)
        cur_w = max(1.0, float(q[:, 0].max() - q[:, 0].min()))
        cur_h = max(1.0, float(q[:, 1].max() - q[:, 1].min()))
        scale_w = min(scale_w, tgt_w / cur_w)
        scale_h = min(scale_h, tgt_h / cur_h)

    center = q.mean(axis=0, keepdims=True)
    out = center + (q - center) * np.array([scale_w, scale_h], dtype=np.float32)
    if outer_margin_frac > 0.0:
        out = center + (out - center) * (1.0 + float(outer_margin_frac))

    if roi is not None and for_corners:
        x0, y0, x1, y1 = roi
        inset = 0.04
        rx0 = float(x0) + inset * (x1 - x0)
        rx1 = float(x1) - inset * (x1 - x0)
        ry0 = float(y0) + inset * (y1 - y0)
        ry1 = float(y1) - inset * (y1 - y0)
        for i in range(4):
            out[i, 0] = float(np.clip(out[i, 0], rx0, rx1))
            out[i, 1] = float(np.clip(out[i, 1], ry0, ry1))

    out[:, 0] = np.clip(out[:, 0], 0.0, float(w - 1))
    out[:, 1] = np.clip(out[:, 1], 0.0, float(h - 1))
    return pc.order_points(out)


def quad_apply_camera_taper(
    quad: np.ndarray,
    *,
    top_narrow: float = 0.055,
    bottom_wide: float = 0.035,
) -> np.ndarray:
    """
    Zamień osiowy prostokąt align na lekki trapez (kamera pod/nad panelem).
    Żółte linie idą między 4 rogami w perspektywie zamiast prostokąta.
    """
    q = pc.order_points(quad.astype(np.float32))
    cy = float(q[:, 1].mean())
    cx = float(q[:, 0].mean())
    out = q.copy()
    for i in range(4):
        if float(q[i, 1]) <= cy:
            out[i, 0] = cx + (float(q[i, 0]) - cx) * (1.0 - top_narrow)
        else:
            out[i, 0] = cx + (float(q[i, 0]) - cx) * (1.0 + bottom_wide)
    return pc.order_points(out)


def expand_quad_from_center(
    quad: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    scale: float,
) -> np.ndarray:
    h, w = image_shape[:2]
    q = quad.astype(np.float32)
    center = q.mean(axis=0, keepdims=True)
    out = center + (q - center) * float(scale)
    out[:, 0] = np.clip(out[:, 0], 0.0, float(w - 1))
    out[:, 1] = np.clip(out[:, 1], 0.0, float(h - 1))
    return pc.order_points(out)


def expand_quad_to_roi_coverage(
    quad: np.ndarray,
    roi: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    *,
    target_coverage: float = 0.88,
    max_scale: float = 1.42,
) -> np.ndarray:
    """Powiększ żółty czworokąt od środka, aż rozpiętość zbliży się do niebieskiego ROI."""
    h, w = image_shape[:2]
    q0 = quad.astype(np.float32)
    center = q0.mean(axis=0, keepdims=True)
    best = q0.copy()
    best_cov = quad_coverage_vs_roi(best, roi)
    if best_cov >= target_coverage:
        return pc.order_points(best)

    for s in np.linspace(1.03, max_scale, 22):
        q = center + (q0 - center) * float(s)
        q[:, 0] = np.clip(q[:, 0], 0.0, float(w - 1))
        q[:, 1] = np.clip(q[:, 1], 0.0, float(h - 1))
        q = pc.order_points(q)
        cov = quad_coverage_vs_roi(q, roi)
        if cov >= target_coverage:
            return q
        if cov > best_cov:
            best = q
            best_cov = cov

    x0, y0, x1, y1 = roi
    q = best.copy()
    margin_x = 0.02 * (x1 - x0)
    margin_y = 0.02 * (y1 - y0)
    for i in range(4):
        if q[i, 0] < x0 + margin_x:
            q[i, 0] = float(x0 + margin_x)
        if q[i, 0] > x1 - margin_x:
            q[i, 0] = float(x1 - margin_x)
        if q[i, 1] < y0 + margin_y:
            q[i, 1] = float(y0 + margin_y)
        if q[i, 1] > y1 - margin_y:
            q[i, 1] = float(y1 - margin_y)
    q[:, 0] = np.clip(q[:, 0], 0.0, float(w - 1))
    q[:, 1] = np.clip(q[:, 1], 0.0, float(h - 1))
    return pc.order_points(q)


def estimate_panel_roi_with_quad(
    image_bgr: np.ndarray,
    *,
    margin_frac: float = 0.12,
    min_area_frac: float = 0.04,
) -> Tuple[Optional[Tuple[int, int, int, int]], str, Optional[np.ndarray]]:
    """
    Live: najpierw skrajne białe linie (LSD) → żółty trapez → niebieski ROI = bbox quada.
    HSV/alignment tylko gdy linie się nie udały (potwierdzenie obecności panelu, nie kotwica X).
    """
    h, w = image_bgr.shape[:2]

    from release.grid_outer_quad import detect_panel_quad_and_roi, roi_bbox_from_quad

    grid_quad, grid_roi, grid_src, _bounds = detect_panel_quad_and_roi(
        image_bgr, roi_margin_frac=0.04,
    )
    if grid_quad is not None:
        grid_roi = roi_bbox_from_quad(grid_quad, (h, w), margin_frac=0.04)
        if _roi_sane(grid_roi, w, h, min_area_frac=min_area_frac):
            return grid_roi, grid_src, grid_quad

    from release.alignment_pipelines import pipeline_hsv_panel, pipeline_hybrid

    align_quad: Optional[np.ndarray] = None
    align_name = 'none'

    for name, fn in (('align_hybrid', pipeline_hybrid), ('align_hsv', pipeline_hsv_panel)):
        res = fn(image_bgr)
        if res.ok and res.quad is not None:
            align_quad = pc.order_points(res.quad.astype(np.float32))
            align_name = name
            break

    if align_quad is not None:
        fallback_roi = roi_bbox_from_quad(align_quad, (h, w), margin_frac=0.06)
        if _roi_sane(fallback_roi, w, h, min_area_frac=min_area_frac):
            return fallback_roi, f'{align_name}_hsv_fallback', align_quad

    candidates: List[Tuple[Tuple[int, int, int, int], str]] = []
    candidates.extend(_roi_bounds_from_white_detectors(image_bgr, margin_frac=0.06))

    if align_quad is not None:
        candidates.append((
            _roi_bounds_align_adaptive(image_bgr, align_quad, outer_margin_frac=0.06),
            'align_adaptive',
        ))

    roi, pick_src = _select_widest_panel_roi(
        candidates,
        image_bgr,
        align_quad=align_quad,
        min_area_frac=min_area_frac,
    )
    if roi is not None:
        src = f'{align_name}_{pick_src}' if align_quad is not None else pick_src
        quad_out = align_quad if align_quad is not None else roi_bounds_to_quad(roi)
        return roi, src, quad_out

    return None, 'none', None


def estimate_panel_roi(
    image_bgr: np.ndarray,
    *,
    margin_frac: float = 0.12,
    min_area_frac: float = 0.04,
) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    roi, src, _q = estimate_panel_roi_with_quad(
        image_bgr, margin_frac=margin_frac, min_area_frac=min_area_frac,
    )
    return roi, src


def crop_panel_roi(
    image_bgr: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, int, int]:
    x0, y0, x1, y1 = roi
    crop = image_bgr[y0:y1 + 1, x0:x1 + 1].copy()
    return crop, x0, y0


def map_quad_crop_to_full(
    quad_crop: np.ndarray,
    *,
    roi_x0: int,
    roi_y0: int,
    work_scale: float = 1.0,
) -> np.ndarray:
    q = quad_crop.astype(np.float32).copy()
    if work_scale > 1e-6 and abs(work_scale - 1.0) > 1e-4:
        q /= work_scale
    q[:, 0] += float(roi_x0)
    q[:, 1] += float(roi_y0)
    return pc.order_points(q)


def roi_bounds_to_quad(roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Czworokąt z niebieskiego ROI (osiowy) — pełny zasięg panelu w kadrze."""
    x0, y0, x1, y1 = roi
    return pc.order_points(
        np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32),
    )


def draw_roi_rect(image_bgr: np.ndarray, roi: Tuple[int, int, int, int], color=(255, 120, 0)) -> np.ndarray:
    vis = image_bgr.copy()
    x0, y0, x1, y1 = roi
    cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
    return vis
