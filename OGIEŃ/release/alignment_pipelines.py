from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pipeline_competition as pc
from module_pose.api import (
    _contour_to_quad,
    _grid_structure_score,
    _panel_color_quad,
    _quad_area_ratio,
    _quad_aspect,
    _rough_quad_from_white_hlines,
    _shift_quad,
    _white_marker_image_score,
    _white_marker_score,
    _warp_panel_candidate,
    strip_camera_overlays,
)


@dataclass
class AlignmentResult:
    name: str
    ok: bool
    confidence: float = 0.0
    quad: Optional[np.ndarray] = None
    center_px: Tuple[float, float] = (0.0, 0.0)
    offset_px: Tuple[float, float] = (0.0, 0.0)
    angle_deg: float = 0.0
    area_ratio: float = 0.0
    aspect: float = 0.0
    meta: Dict[str, float | str] = field(default_factory=dict)


PipelineFn = Callable[[np.ndarray], AlignmentResult]


def _strip_alignment_overlays(image_bgr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    # Live camera view has a bottom OSD/status strip that looks like the black panel.
    # Keep this local to alignment so pose/dataset evaluation is not silently cropped.
    return strip_camera_overlays(image_bgr, top_frac=0.07, bottom_frac=0.10)


def _result_from_quad(name: str, image_bgr: np.ndarray, quad: Optional[np.ndarray], *, base_conf: float = 1.0, meta: Optional[Dict[str, float | str]] = None) -> AlignmentResult:
    if quad is None or quad.shape != (4, 2):
        return AlignmentResult(name=name, ok=False, meta=dict(meta or {}))
    h, w = image_bgr.shape[:2]
    q = pc.order_points(quad.astype(np.float32))
    rect = cv2.minAreaRect(q.astype(np.float32))
    (cx, cy), (rw, rh), angle = rect
    long_side = max(float(rw), float(rh))
    short_side = max(1.0, min(float(rw), float(rh)))
    aspect = long_side / short_side
    area_ratio = _quad_area_ratio(q, float(h * w))
    aspect_score = max(0.0, 1.0 - abs(aspect - 2.0) / 1.4)
    area_score = 1.0 if 0.08 <= area_ratio <= 0.75 else 0.35
    confidence = float(np.clip(base_conf * (0.65 * aspect_score + 0.35 * area_score), 0.0, 1.0))
    return AlignmentResult(
        name=name,
        ok=True,
        confidence=confidence,
        quad=q,
        center_px=(float(cx), float(cy)),
        offset_px=(float(cx - w / 2.0), float(cy - h / 2.0)),
        angle_deg=float(angle),
        area_ratio=float(area_ratio),
        aspect=float(aspect),
        meta=dict(meta or {}),
    )


def _largest_quad_from_mask(mask: np.ndarray, image_shape: Tuple[int, int], *, min_area: float, max_area: float, min_aspect: float = 1.25, max_aspect: float = 3.8) -> Optional[np.ndarray]:
    h, w = image_shape[:2]
    img_area = float(h * w)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_q = None
    best_area = 0.0
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < min_area * img_area or area > max_area * img_area:
            continue
        q = _contour_to_quad(c)
        aspect = _quad_aspect(q)
        if min_aspect <= aspect <= max_aspect and area > best_area:
            best_q = q
            best_area = area
    return best_q


def _panel_hsv_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Live i Blender: panel bywa matowy (s~40–70), nie tylko nasycony zielono-niebieski.
    mask = (((h >= 68) & (h <= 118) & (s > 32) & (v < 165)).astype(np.uint8) * 255)
    mask[:max(1, int(0.03 * mask.shape[0])), :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask


def _profile_quad_from_mask(mask: np.ndarray, *, row_min_frac: float = 0.25, col_min_frac: float = 0.25) -> Optional[np.ndarray]:
    m = mask > 0
    h, w = mask.shape[:2]
    row_fill = m.mean(axis=1)
    ys = np.where(row_fill > row_min_frac)[0]
    if ys.size < 20:
        return None
    y0, y1 = int(ys[0]), int(ys[-1])
    col_fill = m[y0:y1 + 1, :].mean(axis=0)
    xs = np.where(col_fill > col_min_frac)[0]
    if xs.size < 20:
        return None
    x0, x1 = int(xs[0]), int(xs[-1])
    width = float(x1 - x0)
    height = float(y1 - y0)
    if width >= height and height > 1.0 and width / height < 1.65:
        # If supports/OSD leak into the mask, the bottom edge drifts down.
        # The horizontal banner is 2:1, so keep the reliable top edge and trim height.
        y1 = int(min(float(h - 1), float(y0) + width / 2.0))
    elif width >= height and height > 1.0 and width / height >= 1.85:
        # Szeroki profil HSV (live) — wymuś proporcje panelu 2:1.
        y1 = int(min(float(h - 1), float(y0) + width / 2.0))
    elif width >= height and height > 1.0 and width / height > 2.25:
        # Field lighting often makes the top rows less saturated, so the HSV
        # profile starts too low. Keep the bottom edge and restore the 2:1 shape.
        y0 = int(max(0.0, float(y1) - width / 2.0))
    elif height > width and width > 1.0 and height / width < 1.65:
        # Same idea for a vertical panel: keep the left edge and trim width.
        x1 = int(min(float(w - 1), float(x0) + height / 2.0))
    elif height > width and width > 1.0 and height / width > 2.25:
        x0 = int(max(0.0, float(x1) - height / 2.0))
    width = float(x1 - x0)
    height = float(y1 - y0)
    if width >= height and height > 1.0:
        # Live view tends to miss the top/right part of the tilted board while
        # the bottom edge is polluted by OSD/supports. Bias the margin upward
        # and rightward instead of expanding symmetrically.
        y0 = int(max(0.0, float(y0) - 0.07 * height))
        x0 = int(max(0.0, float(x0) - 0.05 * width))
        x1 = int(min(float(w - 1), float(x1) + 0.04 * width))
    elif height > width and width > 1.0:
        y0 = int(max(0.0, float(y0) - 0.04 * height))
        x0 = int(max(0.0, float(x0) - 0.04 * width))
        x1 = int(min(float(w - 1), float(x1) + 0.07 * width))
    q = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    if _quad_aspect(q) < 1.25 or _quad_aspect(q) > 3.4:
        return None
    area_ratio = _quad_area_ratio(q, float(h * w))
    if area_ratio < 0.06 or area_ratio > 0.85:
        return None
    return pc.order_points(q)


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num < 2:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask)
    out[labels == idx] = 255
    return out


def _hsv_profile_quad(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    mask = _largest_component_mask(_panel_hsv_mask(image_bgr))
    strict = _profile_quad_from_mask(mask)
    if strict is not None:
        return strict
    # In field lighting the upper/right rows can be dimmer and fail the dense
    # profile threshold. Relax before falling back to the less stable contour.
    relaxed = _profile_quad_from_mask(mask, row_min_frac=0.12, col_min_frac=0.12)
    if relaxed is not None:
        return relaxed
    return _profile_quad_from_mask(mask, row_min_frac=0.06, col_min_frac=0.06)


def _hsv_alignment_quad(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    profile = _hsv_profile_quad(image_bgr)
    contour = _panel_color_quad(image_bgr, float(image_bgr.shape[0] * image_bgr.shape[1]))
    if profile is None:
        # In live alignment the raw contour is unstable under perspective and
        # tends to lock onto the lower-left part of the panel/background.
        return (None, 'hsv_profile_missing')
    if contour is None:
        return (profile, 'hsv_profile')
    profile_area = _quad_area_ratio(profile, float(image_bgr.shape[0] * image_bgr.shape[1]))
    profile_aspect = _quad_aspect(profile)
    # In the current live setup the contour drifts to the lower-left part of
    # the panel. The profile is axis-aligned but much more stable for alignment.
    if 1.45 <= profile_aspect <= 2.75 and profile_area >= 0.18:
        return (profile, 'hsv_profile_preferred')

    contour_area = _quad_area_ratio(contour, float(image_bgr.shape[0] * image_bgr.shape[1]))
    contour_aspect = _quad_aspect(contour)
    profile_center = profile.astype(np.float32).mean(axis=0)
    contour_center = contour.astype(np.float32).mean(axis=0)
    center_dist = float(np.linalg.norm(profile_center - contour_center))
    center_close = center_dist <= 0.10 * float(max(image_bgr.shape[:2]))
    contour_rect = cv2.minAreaRect(contour.astype(np.float32))
    contour_angle = abs(float(contour_rect[2]))
    contour_tilted = 4.0 < min(contour_angle, abs(90.0 - contour_angle)) < 55.0

    # Profile is very stable for a frontal panel, but it is axis-aligned and
    # reacts poorly when the drone sees the banner under perspective/rotation.
    # Prefer the contour when it is not much larger and gives a sensible 2:1 box.
    contour_not_leaking = contour_area <= max(0.08, profile_area * 1.22)
    contour_more_panel_like = abs(contour_aspect - 2.0) + 0.12 < abs(profile_aspect - 2.0)
    if 1.65 <= profile_aspect <= 2.45 and profile_area >= 0.25 and not center_close:
        return (profile, 'hsv_profile_center_guard')
    if contour_tilted and contour_not_leaking and center_close and 1.45 <= contour_aspect <= 2.65:
        return (contour, 'hsv_contour_tilted')
    if contour_tilted and center_close and contour_area <= 0.65 and contour_area >= profile_area * 1.10 and 1.35 <= contour_aspect <= 2.85:
        return (contour, 'hsv_contour_tilted_area')
    if contour_not_leaking and center_close and contour_more_panel_like and 1.45 <= contour_aspect <= 2.65:
        return (contour, 'hsv_contour_aspect')
    return (profile, 'hsv_profile')


def _trapezoid_quad_from_hsv_roi(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    mask = _panel_hsv_mask(image_bgr)
    profile = _profile_quad_from_mask(mask, row_min_frac=0.08, col_min_frac=0.08)
    if profile is None:
        return None
    x0, y0, x1, y1 = _axis_bounds_quad(profile)
    h, w = mask.shape[:2]
    pad_x = 0.07 * (x1 - x0)
    pad_y = 0.04 * (y1 - y0)
    rx0 = int(max(0.0, x0 - pad_x))
    rx1 = int(min(float(w - 1), x1 + pad_x))
    ry0 = int(max(0.0, y0 - pad_y))
    ry1 = int(min(float(h - 1), y1 + pad_y))
    roi = np.zeros_like(mask)
    roi[ry0:ry1 + 1, rx0:rx1 + 1] = mask[ry0:ry1 + 1, rx0:rx1 + 1]
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.35 * cv2.contourArea(profile.astype(np.float32)):
        return None
    hull = cv2.convexHull(cnt).reshape(-1, 2).astype(np.float32)
    if hull.shape[0] < 4:
        return None
    s = hull[:, 0] + hull[:, 1]
    d = hull[:, 0] - hull[:, 1]
    q = np.array([
        hull[int(np.argmin(s))],
        hull[int(np.argmax(d))],
        hull[int(np.argmax(s))],
        hull[int(np.argmin(d))],
    ], dtype=np.float32)
    q = pc.order_points(q)
    aspect = _quad_aspect(q)
    area_ratio = _quad_area_ratio(q, float(h * w))
    if not (1.25 <= aspect <= 3.2 and 0.08 <= area_ratio <= 0.78):
        return None
    # If the contour shrinks too much, keep the profile. Otherwise this gives
    # the perspective trapezoid needed for precise XY/pose work.
    if cv2.contourArea(q) < 0.75 * cv2.contourArea(profile.astype(np.float32)):
        return None
    return q


def _grid_trapezoid_quad(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Estimate a perspective quad from the visible white grid line cloud."""
    mask = _panel_hsv_mask(image_bgr)
    profile = _profile_quad_from_mask(mask, row_min_frac=0.08, col_min_frac=0.08)
    if profile is None:
        return None
    x0, y0, x1, y1 = _axis_bounds_quad(profile)
    h, w = image_bgr.shape[:2]
    pad_x = 0.10 * (x1 - x0)
    pad_y = 0.06 * (y1 - y0)
    rx0 = int(max(0.0, x0 - pad_x))
    rx1 = int(min(float(w - 1), x1 + pad_x))
    ry0 = int(max(0.0, y0 - pad_y))
    ry1 = int(min(float(h - 1), y1 + pad_y))
    roi = image_bgr[ry0:ry1 + 1, rx0:rx1 + 1]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    white = (((gray > 105) & (hsv[:, :, 1] < 125)) | (gray > 155)).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white = cv2.dilate(white, np.ones((5, 5), np.uint8), iterations=1)
    pts = cv2.findNonZero(white)
    if pts is None or len(pts) < 120:
        return None
    pts2 = pts.reshape(-1, 2).astype(np.float32)
    # Drop tiny text/OSD-like speckles by keeping points inside the central
    # quantile envelope of the grid cloud.
    qx0, qx1 = np.percentile(pts2[:, 0], [1.0, 99.0])
    qy0, qy1 = np.percentile(pts2[:, 1], [1.0, 99.0])
    pts2 = pts2[(pts2[:, 0] >= qx0) & (pts2[:, 0] <= qx1) & (pts2[:, 1] >= qy0) & (pts2[:, 1] <= qy1)]
    if pts2.shape[0] < 80:
        return None
    rect = cv2.minAreaRect(pts2.reshape(-1, 1, 2))
    box = cv2.boxPoints(rect).astype(np.float32)
    box[:, 0] += float(rx0)
    box[:, 1] += float(ry0)
    box = pc.order_points(box)
    aspect = _quad_aspect(box)
    area_ratio = _quad_area_ratio(box, float(h * w))
    if not (1.35 <= aspect <= 3.1 and 0.08 <= area_ratio <= 0.78):
        return None
    return box


def _expand_quad(quad: np.ndarray, image_shape: Tuple[int, int], *, sx: float = 1.06, sy: float = 1.08) -> np.ndarray:
    h, w = image_shape[:2]
    q = quad.astype(np.float32)
    center = q.mean(axis=0, keepdims=True)
    scale = np.array([[sx, sy]], dtype=np.float32)
    out = center + (q - center) * scale
    out[:, 0] = np.clip(out[:, 0], 0.0, float(w - 1))
    out[:, 1] = np.clip(out[:, 1], 0.0, float(h - 1))
    return pc.order_points(out)


def _axis_bounds_quad(quad: np.ndarray) -> Tuple[float, float, float, float]:
    q = quad.astype(np.float32)
    return (float(np.min(q[:, 0])), float(np.min(q[:, 1])), float(np.max(q[:, 0])), float(np.max(q[:, 1])))


def _bounds_to_quad(x0: float, y0: float, x1: float, y1: float, image_shape: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    x0 = float(np.clip(x0, 0.0, float(w - 1)))
    x1 = float(np.clip(x1, 0.0, float(w - 1)))
    y0 = float(np.clip(y0, 0.0, float(h - 1)))
    y1 = float(np.clip(y1, 0.0, float(h - 1)))
    return pc.order_points(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32))


def _clamp_quad_to_reference(quad: np.ndarray, ref: np.ndarray, image_shape: Tuple[int, int], *, pad_frac: float = 0.04) -> np.ndarray:
    qx0, qy0, qx1, qy1 = _axis_bounds_quad(quad)
    rx0, ry0, rx1, ry1 = _axis_bounds_quad(ref)
    rw = rx1 - rx0
    rh = ry1 - ry0
    pad_x = pad_frac * rw
    pad_y = pad_frac * rh
    return _bounds_to_quad(
        max(qx0, rx0 - pad_x),
        max(qy0, ry0 - pad_y),
        min(qx1, rx1 + pad_x),
        min(qy1, ry1 + pad_y),
        image_shape,
    )


def _pad_bounds_asymmetric(quad: np.ndarray, image_shape: Tuple[int, int], *, left: float = 0.0, right: float = 0.0, top: float = 0.0, bottom: float = 0.0) -> np.ndarray:
    x0, y0, x1, y1 = _axis_bounds_quad(quad)
    width = x1 - x0
    height = y1 - y0
    return _bounds_to_quad(
        x0 - left * width,
        y0 - top * height,
        x1 + right * width,
        y1 + bottom * height,
        image_shape,
    )


def _dark_gray_mask(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask[:max(1, int(0.03 * mask.shape[0])), :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return mask


def _morph_panel_outer_quad(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Close white grid lines into one blob so inner grid cannot be selected.
    Returns outer quadrilateral of the merged panel region.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    bright = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -6,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    merged = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=3)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    merged = _largest_component_mask(merged)
    h, w = image_bgr.shape[:2]
    img_area = float(h * w)
    q = _largest_quad_from_mask(merged, (h, w), min_area=0.06, max_area=0.82, min_aspect=1.2, max_aspect=3.5)
    if q is not None:
        return q
    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if float(cv2.contourArea(cnt)) < 0.06 * img_area:
        return None
    return _contour_to_quad(cnt)


def pipeline_morph_blob(image_bgr: np.ndarray) -> AlignmentResult:
    work, ox, oy = _strip_alignment_overlays(image_bgr)
    q = _morph_panel_outer_quad(work)
    meta: Dict[str, float | str] = {'basis': 'morph close erases inner grid; outer contour'}
    if q is not None:
        ref, ref_basis = _hsv_alignment_quad(work)
        if ref is not None:
            w_frac = (float(np.max(q[:, 0])) - float(np.min(q[:, 0]))) / max(1.0, float(work.shape[1]))
            ref_w = (float(np.max(ref[:, 0])) - float(np.min(ref[:, 0]))) / max(1.0, float(work.shape[1]))
            if w_frac < ref_w * 0.88:
                q = _expand_quad(ref, work.shape[:2], sx=1.04, sy=1.05)
                meta['expanded_to_hsv_width'] = 1.0
        q = _shift_quad(q, ox, oy)
    return _result_from_quad('morph_blob', image_bgr, q, base_conf=0.88, meta=meta)


def pipeline_hsv_panel(image_bgr: np.ndarray) -> AlignmentResult:
    work, ox, oy = _strip_alignment_overlays(image_bgr)
    q, basis = _hsv_alignment_quad(work)
    if q is not None:
        q = _shift_quad(q, ox, oy)
    base_conf = 0.95 if 'profile' in basis else 0.45
    return _result_from_quad('hsv_panel', image_bgr, q, base_conf=base_conf, meta={'basis': basis})


def pipeline_dark_blob(image_bgr: np.ndarray) -> AlignmentResult:
    work, ox, oy = _strip_alignment_overlays(image_bgr)
    mask = _dark_gray_mask(work)
    q = _largest_quad_from_mask(mask, work.shape[:2], min_area=0.05, max_area=0.92)
    ref, ref_basis = _hsv_alignment_quad(work)
    meta: Dict[str, float | str] = {'basis': 'dark Otsu blob, constrained by panel color'}
    if q is None:
        prof = _profile_quad_from_mask(mask, row_min_frac=0.10, col_min_frac=0.10)
        if prof is not None:
            q = prof
            meta['basis'] = 'dark mask profile (full-frame blob)'
    if q is not None and ref is not None:
        q_area = _quad_area_ratio(q, float(work.shape[0] * work.shape[1]))
        ref_area = _quad_area_ratio(ref, float(work.shape[0] * work.shape[1]))
        if q_area > ref_area * 1.35:
            q = _expand_quad(ref, work.shape[:2], sx=1.02, sy=1.03)
            meta['replaced_huge_blob'] = 1.0
            meta['basis'] = f'dark huge -> {ref_basis}'
    elif q is not None and ref is None:
        wh = _rough_quad_from_white_hlines(work)
        if wh is not None:
            q = _clamp_quad_to_reference(
                q, _expand_quad(wh, work.shape[:2], sx=1.06, sy=1.08), work.shape[:2], pad_frac=0.04,
            )
            meta['basis'] = 'dark blob clamped to white_hlines'
    if q is not None:
        q = _shift_quad(q, ox, oy)
    return _result_from_quad('dark_blob', image_bgr, q, base_conf=0.62, meta=meta)


def pipeline_white_grid(image_bgr: np.ndarray) -> AlignmentResult:
    work, ox, oy = _strip_alignment_overlays(image_bgr)
    q = _rough_quad_from_white_hlines(work)
    ref, _basis = _hsv_alignment_quad(work)
    meta: Dict[str, float | str] = {'basis': 'white grid lines expanded, color constrained'}
    if q is not None:
        q = _expand_quad(q, work.shape[:2], sx=1.08, sy=1.10)
        if ref is not None:
            q = _clamp_quad_to_reference(q, _expand_quad(ref, work.shape[:2], sx=1.04, sy=1.05), work.shape[:2], pad_frac=0.02)
            meta['color_constrained'] = 1.0
            q = _pad_bounds_asymmetric(q, work.shape[:2], left=0.12, right=0.04, top=0.02, bottom=0.0)
        else:
            q = _pad_bounds_asymmetric(q, work.shape[:2], left=0.14, right=0.06, top=0.04, bottom=0.02)
        q = _shift_quad(q, ox, oy)
    # Siatka jest dobra jako walidacja, ale sama potrafi wybrać linie wewnętrzne.
    return _result_from_quad('white_grid', image_bgr, q, base_conf=0.55, meta=meta)


def pipeline_trapezoid(image_bgr: np.ndarray) -> AlignmentResult:
    work, ox, oy = _strip_alignment_overlays(image_bgr)
    q = _grid_trapezoid_quad(work)
    basis = 'white grid min-area trapezoid'
    if q is None:
        q = _trapezoid_quad_from_hsv_roi(work)
        basis = 'hsv contour inside profile ROI'
    if q is not None:
        q = _shift_quad(q, ox, oy)
    return _result_from_quad('trapezoid', image_bgr, q, base_conf=0.78, meta={'basis': basis})


def pipeline_current_scored(image_bgr: np.ndarray) -> AlignmentResult:
    work, ox, oy = _strip_alignment_overlays(image_bgr)
    q, basis = _hsv_alignment_quad(work)
    if q is not None:
        q = _shift_quad(q, ox, oy)
    else:
        return AlignmentResult(name='current_scored', ok=False, meta={'basis': basis})
    meta: Dict[str, float | str] = {'basis': f'current scored detector via {basis}'}
    if q is not None:
        try:
            warped = _warp_panel_candidate(image_bgr, q)
            meta['grid_score'] = float(_grid_structure_score(warped))
            meta['white_marker_warp'] = float(_white_marker_score(warped))
            meta['white_marker_img'] = float(_white_marker_image_score(image_bgr, q))
        except cv2.error:
            pass
    base_conf = 0.85 if 'profile' in basis else 0.45
    return _result_from_quad('current_scored', image_bgr, q, base_conf=base_conf, meta=meta)


def pipeline_hybrid(image_bgr: np.ndarray) -> AlignmentResult:
    candidates = [
        pipeline_morph_blob(image_bgr),
        pipeline_hsv_panel(image_bgr),
        pipeline_dark_blob(image_bgr),
        pipeline_white_grid(image_bgr),
        pipeline_trapezoid(image_bgr),
        pipeline_current_scored(image_bgr),
    ]
    ok = [r for r in candidates if r.ok and r.quad is not None]
    if not ok:
        return AlignmentResult(name='hybrid', ok=False, meta={'basis': 'best of all'})

    def score(r: AlignmentResult) -> float:
        q = r.quad
        assert q is not None
        extra = 0.0
        try:
            warped = _warp_panel_candidate(image_bgr, q)
            extra += 0.25 * _grid_structure_score(warped)
            extra += 0.35 * _white_marker_score(warped)
            extra += 0.35 * _white_marker_image_score(image_bgr, q)
        except cv2.error:
            pass
        if r.name == 'morph_blob':
            extra += 0.32
        if r.name == 'hsv_panel':
            extra += 0.25
        if r.name == 'trapezoid':
            extra += 0.18
        return float(r.confidence + extra)

    best = max(ok, key=score)
    out = _result_from_quad('hybrid', image_bgr, best.quad, base_conf=min(1.0, score(best)), meta={'basis': f'best={best.name}'})
    return out


PIPELINES: Dict[str, PipelineFn] = {
    'morph': pipeline_morph_blob,
    'hsv': pipeline_hsv_panel,
    'dark': pipeline_dark_blob,
    'grid': pipeline_white_grid,
    'trapezoid': pipeline_trapezoid,
    'scored': pipeline_current_scored,
    'hybrid': pipeline_hybrid,
}


def run_pipeline(name: str, image_bgr: np.ndarray) -> AlignmentResult:
    if name not in PIPELINES:
        raise ValueError(f'unknown_pipeline={name}')
    return PIPELINES[name](image_bgr)


def run_all_pipelines(image_bgr: np.ndarray) -> List[AlignmentResult]:
    return [fn(image_bgr) for fn in PIPELINES.values()]
