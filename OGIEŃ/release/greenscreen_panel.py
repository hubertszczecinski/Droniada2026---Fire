"""
Greenscreen warsztatu: tło → zielony chroma; w panelu zostają czerń (wiele odcieni), biel i kartki.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import pipeline_competition as pc

from release.panel_black import (
    PanelBlackThresholds,
    calibrate_panel_black_from_corners,
    mask_keep_inside_panel,
    mask_keep_on_panel,
)

# BGR — klasyczny greenscreen (OpenCV)
GREEN_BGR = (0, 255, 0)


def _pixel_keep_mask(
    hsv: np.ndarray,
    thresholds: Optional[PanelBlackThresholds] = None,
) -> np.ndarray:
    return mask_keep_on_panel(hsv, thresholds)


def _panel_mask(
    shape: Tuple[int, int],
    corners_tltrbrbl: np.ndarray,
    *,
    shrink_frac: float = 0.0,
) -> np.ndarray:
    h, w = shape[:2]
    q = pc.order_points(corners_tltrbrbl.astype(np.float32))
    if shrink_frac > 0.0:
        c = q.mean(axis=0)
        q = c + (q - c) * (1.0 - float(shrink_frac))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, q.astype(np.int32), 255)
    return mask


def apply_workshop_greenscreen(
    image_bgr: np.ndarray,
    corners_tltrbrbl: Optional[np.ndarray] = None,
    *,
    shrink_panel_frac: float = 0.0,
    dilate_panel_px: int = 8,
    calibrate_black: bool = True,
) -> np.ndarray:
    """
    - Poza panelem: wszystko → green.
    - W panelu: wiele odcieni czerni + biel + kolor; reszta → green.
    - Z rogami: kalibracja progu czerni z percentyla wnętrza panelu.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    th: Optional[PanelBlackThresholds] = None
    if calibrate_black and corners_tltrbrbl is not None and corners_tltrbrbl.shape == (4, 2):
        th = calibrate_panel_black_from_corners(image_bgr, corners_tltrbrbl)

    out = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    if corners_tltrbrbl is not None and corners_tltrbrbl.shape == (4, 2):
        keep = mask_keep_inside_panel(image_bgr, corners_tltrbrbl, th, close_px=25)
        panel = _panel_mask((h, w), corners_tltrbrbl, shrink_frac=shrink_panel_frac)
        if dilate_panel_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_panel_px, dilate_panel_px))
            panel = cv2.dilate(panel, k, iterations=1)
        inside = panel > 0
        replace = np.ones((h, w), dtype=bool)
        replace[~inside] = True
        replace[inside] = keep[inside] == 0
    else:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        keep = _pixel_keep_mask(hsv, th)
        replace = keep == 0

    out[replace] = GREEN_BGR
    return out


def compose_preview_with_greenscreen(
    original_bgr: np.ndarray,
    greenscreen_bgr: np.ndarray,
    *,
    target_w: int = 1280,
) -> np.ndarray:
    """Oryginał | greenscreen obok siebie."""
    h = max(original_bgr.shape[0], greenscreen_bgr.shape[0])

    def _fit(img: np.ndarray) -> np.ndarray:
        scale = target_w / (2.0 * max(1, img.shape[1]))
        nw = max(1, int(round(img.shape[1] * scale)))
        nh = max(1, int(round(img.shape[0] * scale)))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    a = _fit(original_bgr)
    b = _fit(greenscreen_bgr)
    if a.shape[0] != b.shape[0]:
        nh = max(a.shape[0], b.shape[0])

        def _pad(x: np.ndarray) -> np.ndarray:
            if x.shape[0] == nh:
                return x
            p = np.zeros((nh, x.shape[1], 3), dtype=np.uint8)
            p[: x.shape[0]] = x
            return p

        a, b = _pad(a), _pad(b)
    gap = np.zeros((a.shape[0], 8, 3), dtype=np.uint8)
    cv2.putText(a, 'kamera', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(b, 'greenscreen', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return np.hstack([a, gap, b])
