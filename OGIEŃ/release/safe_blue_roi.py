"""Makro niebieski ROI — zewnętrzna czarna ramka (HSV), żółty = linie siatki wewnątrz."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import pipeline_competition as pc

from release.alignment_pipelines import (
    _hsv_profile_quad,
    _largest_component_mask,
    _panel_hsv_mask,
    _strip_alignment_overlays,
)
from module_pose.api import _rough_quad_from_white_hlines


class BlueRoiStabilizer:
    """
    Wygładza niebieski ROI w czasie: góra nie „pływa”, zawsze obejmuje cały panel.
    y0 tylko w górę (mniejsze wartości), y1 tylko w dół — bez skracania kadru.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.14,
        top_release_px: float = 2.5,
        min_frames: int = 2,
    ) -> None:
        self.alpha = float(np.clip(alpha, 0.05, 0.45))
        self.top_release_px = float(top_release_px)
        self.min_frames = max(1, int(min_frames))
        self._smooth: Optional[Tuple[float, float, float, float]] = None
        self._frames = 0

    def reset(self) -> None:
        self._smooth = None
        self._frames = 0

    def update(
        self,
        raw: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        if raw is None:
            return self._smooth

        x0, y0, x1, y1 = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        if self._smooth is None:
            self._smooth = (x0, y0, x1, y1)
            self._frames = 1
            return self._clip_int(self._smooth)

        sx0, sy0, sx1, sy1 = self._smooth
        a = self.alpha
        # Góra: tylko w górę (min y); dół: tylko w dół (max y)
        ny0 = min(sy0, sy0 + a * (y0 - sy0))
        ny1 = max(sy1, sy1 + a * (y1 - sy1))
        nx0 = sx0 + a * (x0 - sx0)
        nx1 = sx1 + a * (x1 - sx1)
        # Powolne „puszczenie” góry w dół tylko gdy wiele klatek stabilnie niżej
        if y0 > sy0 + self.top_release_px:
            ny0 = sy0 + 0.04 * (y0 - sy0)
        else:
            ny0 = min(ny0, y0)

        self._smooth = (nx0, ny0, nx1, ny1)
        self._frames += 1
        return self._clip_int(self._smooth)

    @staticmethod
    def _clip_int(bounds: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = bounds
        return (
            int(np.floor(x0)),
            int(np.floor(y0)),
            int(np.ceil(x1)),
            int(np.ceil(y1)),
        )


_BLUE_ROI_STABILIZER = BlueRoiStabilizer(alpha=0.12, top_release_px=3.0)


def _bounds_from_quad(quad: np.ndarray) -> Tuple[float, float, float, float]:
    q = pc.order_points(quad.astype(np.float32))
    return (
        float(q[:, 0].min()),
        float(q[:, 1].min()),
        float(q[:, 0].max()),
        float(q[:, 1].max()),
    )


def _shift_quad_to_full(
    quad: np.ndarray,
    ox: float,
    oy: float,
) -> np.ndarray:
    out = quad.astype(np.float32).copy()
    out[:, 0] += float(ox)
    out[:, 1] += float(oy)
    return out


def _clip_roi(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    w: int,
    h: int,
) -> Tuple[int, int, int, int]:
    bx0 = int(np.floor(np.clip(x0, 0, w - 2)))
    by0 = int(np.floor(np.clip(y0, 0, h - 2)))
    bx1 = int(np.ceil(np.clip(x1, bx0 + 2, w - 1)))
    by1 = int(np.ceil(np.clip(y1, by0 + 2, h - 1)))
    return bx0, by0, bx1, by1


def _bbox_from_hsv_mask_pixels(
    image_bgr: np.ndarray,
    ox: float,
    oy: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Bez profilu wierszy — surowe piksele maski HSV (stabilniejsza góra)."""
    work, _, _ = _strip_alignment_overlays(image_bgr)
    mask = _largest_component_mask(_panel_hsv_mask(work))
    ys, xs = np.where(mask > 0)
    if ys.size < 500:
        return None
    x0 = float(xs.min()) + ox
    x1 = float(xs.max()) + ox
    y0 = float(ys.min()) + oy
    y1 = float(ys.max()) + oy
    return x0, y0, x1, y1


def _measure_blue_roi_raw(
    image_bgr: np.ndarray,
    *,
    pad_left: int = 36,
    pad_right: int = 36,
    pad_top: int = 44,
    pad_bottom: int = 44,
    outer_frac: float = 0.095,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Szeroki bbox = unia HSV (maska pikseli + profil), white_hlines rozszerzone do ramki.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    h, w = image_bgr.shape[:2]
    work, ox, oy = _strip_alignment_overlays(image_bgr)

    boxes: List[Tuple[float, float, float, float]] = []

    pix = _bbox_from_hsv_mask_pixels(image_bgr, ox, oy)
    if pix is not None:
        px0, py0, px1, py1 = pix
        psw, psh = max(1.0, px1 - px0), max(1.0, py1 - py0)
        boxes.append((
            px0 - 0.05 * psw - float(pad_left),
            py0 - 0.07 * psh - float(pad_top),
            px1 + 0.05 * psw + float(pad_right),
            py1 + 0.04 * psh + float(pad_bottom),
        ))

    hsv_q = _hsv_profile_quad(work)
    if hsv_q is not None:
        hq = _shift_quad_to_full(hsv_q, ox, oy)
        hx0, hy0, hx1, hy1 = _bounds_from_quad(hq)
        hsw, hsh = max(1.0, hx1 - hx0), max(1.0, hy1 - hy0)
        boxes.append((
            hx0 - 0.04 * hsw,
            hy0 - 0.10 * hsh - float(pad_top) * 0.35,
            hx1 + 0.04 * hsw,
            hy1 + 0.03 * hsh,
        ))

    wh_top: Optional[float] = None
    wh = _rough_quad_from_white_hlines(work)
    if wh is not None:
        wh = _shift_quad_to_full(wh.astype(np.float32), ox, oy)
        wh_top = float(np.min(wh[:, 1]))

    if not boxes:
        return None

    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    if wh_top is not None:
        y0 = min(y0, wh_top - float(pad_top) * 0.65)

    return _clip_roi(x0, y0, x1, y1, w, h)


def get_safe_blue_roi(
    image_bgr: np.ndarray,
    *,
    pad_left: int = 36,
    pad_right: int = 36,
    pad_top: int = 44,
    pad_bottom: int = 44,
    min_width_frac: float = 0.38,
    max_width_frac: float = 1.0,
    min_area_frac: float = 0.08,
    max_area_frac: float = 0.985,
    stabilize: bool = True,
) -> Optional[Tuple[int, int, int, int]]:
    """ROI zewnętrznej ramki; domyślnie ze stabilizatorem czasowym."""
    raw = _measure_blue_roi_raw(
        image_bgr,
        pad_left=pad_left,
        pad_right=pad_right,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
    )
    if raw is None:
        if stabilize:
            return _BLUE_ROI_STABILIZER.update(None)
        return None

    h, w = image_bgr.shape[:2]
    area_frac = float((raw[2] - raw[0]) * (raw[3] - raw[1])) / float(h * w)
    w_frac = float(raw[2] - raw[0]) / max(1.0, float(w))
    if area_frac < min_area_frac or area_frac > max_area_frac:
        return None
    if w_frac < min_width_frac or w_frac > max_width_frac:
        return None

    if stabilize:
        return _BLUE_ROI_STABILIZER.update(raw)
    return raw


def reset_blue_roi_stabilizer() -> None:
    _BLUE_ROI_STABILIZER.reset()


def expand_roi(
    roi: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    *,
    margin_frac: float = 0.10,
    pad_px: int = 24,
) -> Tuple[int, int, int, int]:
    """Poszerz ROI (np. żółty skan na pełniejszym obszarze niż niebieski)."""
    h, w = image_shape[:2]
    x0, y0, x1, y1 = (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))
    sw, sh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    m = float(margin_frac)
    return _clip_roi(
        x0 - m * sw - pad_px,
        y0 - m * sh - pad_px,
        x1 + m * sw + pad_px,
        y1 + m * sh + pad_px,
        w,
        h,
    )


def nudge_quad_inside_roi(
    quad: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
    *,
    inner_margin_px: float = 0.0,
    max_nudge_px: float = 28.0,
    tol_px: float = 8.0,
) -> np.ndarray:
    """
    Tylko lekkie przesunięcie wewnątrz niebieskiego — bez skalowania trapezu.
    Jeśli wystaje dużo, zwróć oryginał (żółty ważniejszy niż ciasny ROI).
    """
    if quad is None or quad.shape != (4, 2) or roi is None:
        return quad
    if quad_inside_roi(quad, roi, inner_margin_px=inner_margin_px, tol_px=tol_px):
        return pc.order_points(quad.astype(np.float32).copy())

    q = pc.order_points(quad.astype(np.float32).copy())
    bx0, by0, bx1, by1 = _roi_bounds(roi, inner_margin_px=inner_margin_px)
    dx_lo = bx0 - float(q[:, 0].min())
    dx_hi = bx1 - float(q[:, 0].max())
    dy_lo = by0 - float(q[:, 1].min())
    dy_hi = by1 - float(q[:, 1].max())
    dx = dy = 0.0
    if dx_lo > 0.0:
        dx = min(dx_lo, float(max_nudge_px))
    elif dx_hi < 0.0:
        dx = max(dx_hi, -float(max_nudge_px))
    if dy_lo > 0.0:
        dy = min(dy_lo, float(max_nudge_px))
    elif dy_hi < 0.0:
        dy = max(dy_hi, -float(max_nudge_px))
    q[:, 0] += dx
    q[:, 1] += dy
    if quad_inside_roi(q, roi, inner_margin_px=inner_margin_px, tol_px=tol_px):
        return q
    return pc.order_points(quad.astype(np.float32).copy())


def crop_by_roi(
    image_bgr: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, int, int]:
    x0, y0, x1, y1 = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
    return image_bgr[y0:y1, x0:x1].copy(), x0, y0


def clip_quad_to_image(
    quad: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Tylko krawędzie kadru — bez dopasowania do niebieskiego ROI."""
    if quad is None or quad.shape != (4, 2):
        return quad
    h, w = image_shape[:2]
    q = pc.order_points(quad.astype(np.float32).copy())
    q[:, 0] = np.clip(q[:, 0], 0.0, float(w - 1))
    q[:, 1] = np.clip(q[:, 1], 0.0, float(h - 1))
    return q


def merge_blue_roi_with_yellow(
    hsv_roi: Optional[Tuple[int, int, int, int]],
    yellow: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    margin_frac: float = 0.10,
    pad_px: int = 32,
    outer_margin_frac: float = 0.06,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Żółty trapez bez skalowania; niebieski = unia HSV + bbox żółtego + zapas na czarną ramkę.
    """
    from release.panel_labels import blue_roi_from_yellow

    yellow_out = clip_quad_to_image(yellow, image_shape)
    yellow_box = blue_roi_from_yellow(
        yellow_out, image_shape, margin_frac=margin_frac, pad_px=pad_px,
    )
    blue = ensure_roi_covers_quad(
        hsv_roi, yellow_out, image_shape, pad_px=pad_px, outer_margin_frac=outer_margin_frac,
    )
    if blue is None:
        blue = yellow_box
    else:
        x0 = min(int(blue[0]), int(yellow_box[0]))
        y0 = min(int(blue[1]), int(yellow_box[1]))
        x1 = max(int(blue[2]), int(yellow_box[2]))
        y1 = max(int(blue[3]), int(yellow_box[3]))
        h, w = image_shape[:2]
        blue = _clip_roi(float(x0), float(y0), float(x1), float(y1), w, h)
    return yellow_out, blue


def ensure_roi_covers_quad(
    roi: Optional[Tuple[int, int, int, int]],
    quad: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    pad_px: int = 12,
    outer_margin_frac: float = 0.055,
) -> Optional[Tuple[int, int, int, int]]:
    if quad is None or quad.shape != (4, 2):
        return roi
    h, w = image_shape[:2]
    q = quad.astype(np.float32)
    qx0, qy0 = float(q[:, 0].min()), float(q[:, 1].min())
    qx1, qy1 = float(q[:, 0].max()), float(q[:, 1].max())
    span_w = max(1.0, qx1 - qx0)
    span_h = max(1.0, qy1 - qy0)
    m = float(outer_margin_frac)
    outer_x0 = qx0 - m * span_w - float(pad_px)
    outer_x1 = qx1 + m * span_w + float(pad_px)
    outer_y0 = qy0 - m * span_h - float(pad_px)
    outer_y1 = qy1 + m * span_h + float(pad_px)

    if roi is None:
        return _clip_roi(outer_x0, outer_y0, outer_x1, outer_y1, w, h)

    x0, y0, x1, y1 = (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))
    x0 = min(x0, outer_x0)
    y0 = min(y0, outer_y0)
    x1 = max(x1, outer_x1)
    y1 = max(y1, outer_y1)
    return _clip_roi(x0, y0, x1, y1, w, h)


def _roi_bounds(
    roi: Tuple[int, int, int, int],
    *,
    inner_margin_px: float = 0.0,
) -> Tuple[float, float, float, float]:
    m = max(0.0, float(inner_margin_px))
    return (
        float(roi[0]) + m,
        float(roi[1]) + m,
        float(roi[2]) - m,
        float(roi[3]) - m,
    )


def quad_inside_roi(
    quad: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
    *,
    inner_margin_px: float = 0.0,
    tol_px: float = 1.5,
) -> bool:
    if roi is None or quad is None or quad.shape != (4, 2):
        return True
    bx0, by0, bx1, by1 = _roi_bounds(roi, inner_margin_px=inner_margin_px)
    if bx1 <= bx0 + 4.0 or by1 <= by0 + 4.0:
        return False
    q = quad.astype(np.float32)
    t = float(tol_px)
    return bool(
        float(q[:, 0].min()) >= bx0 - t
        and float(q[:, 0].max()) <= bx1 + t
        and float(q[:, 1].min()) >= by0 - t
        and float(q[:, 1].max()) <= by1 + t
    )


def quad_outside_roi_padded(
    quad: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
    *,
    pad_px: float = 8.0,
) -> bool:
    if roi is None or quad is None:
        return False
    return not quad_inside_roi(quad, roi, inner_margin_px=float(pad_px), tol_px=0.0)


def fit_quad_inside_roi(
    quad: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
    *,
    inner_margin_px: float = 0.0,
    max_scale_steps: int = 24,
    scale_step: float = 0.95,
    allow_shrink: bool = False,
) -> Optional[np.ndarray]:
    """Domyślnie nie zmniejsza trapezu — tylko nudge lub clip kadru (żółty > niebieski)."""
    if quad is None or quad.shape != (4, 2):
        return None
    q = clip_quad_to_image(quad, image_shape)

    if roi is None:
        return q

    if quad_inside_roi(q, roi, inner_margin_px=inner_margin_px, tol_px=2.0):
        return q

    nudged = nudge_quad_inside_roi(
        q, roi, image_shape, inner_margin_px=inner_margin_px, max_nudge_px=36.0, tol_px=2.0,
    )
    if quad_inside_roi(nudged, roi, inner_margin_px=inner_margin_px, tol_px=2.5):
        return nudged

    if not allow_shrink:
        return q

    bx0, by0, bx1, by1 = _roi_bounds(roi, inner_margin_px=inner_margin_px)
    if bx1 <= bx0 + 8.0 or by1 <= by0 + 8.0:
        return None

    def _inside(pts: np.ndarray) -> bool:
        return (
            float(pts[:, 0].min()) >= bx0
            and float(pts[:, 0].max()) <= bx1
            and float(pts[:, 1].min()) >= by0
            and float(pts[:, 1].max()) <= by1
        )

    q2 = pc.order_points(nudged.astype(np.float32).copy())
    center = q2.mean(axis=0)
    for _ in range(max_scale_steps):
        if _inside(q2):
            return pc.order_points(q2)
        q2 = center + (q2 - center) * float(scale_step)
    if _inside(q2):
        return pc.order_points(q2)
    return None


def clamp_quad_inside_roi(
    quad: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
    *,
    inner_margin_px: float = 0.0,
) -> np.ndarray:
    fitted = fit_quad_inside_roi(
        quad, roi, image_shape, inner_margin_px=inner_margin_px,
    )
    if fitted is not None:
        return fitted
    q = pc.order_points(quad.astype(np.float32).copy())
    h, w = image_shape[:2]
    if roi is None:
        q[:, 0] = np.clip(q[:, 0], 0.0, float(w - 1))
        q[:, 1] = np.clip(q[:, 1], 0.0, float(h - 1))
        return q
    bx0, by0, bx1, by1 = _roi_bounds(roi, inner_margin_px=inner_margin_px)
    q[:, 0] = np.clip(q[:, 0], bx0, bx1)
    q[:, 1] = np.clip(q[:, 1], by0, by1)
    return pc.order_points(q)
