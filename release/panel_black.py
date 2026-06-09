"""
Wiele odcieni czerni panelu (mat + cienie kamery) — wspólne progi dla greenscreen i odczytu kolorów.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pipeline_competition as pc


@dataclass
class PanelBlackThresholds:
    """Progi HSV — po kalibracji z wnętrza panelu."""

    v_black_max: float = 158.0
    s_black_max: float = 135.0
    v_shadow_max: float = 142.0
    s_shadow_max: float = 110.0
    v_white_min: float = 162.0
    s_white_max: float = 98.0
    s_color_min: float = 54.0
    v_color_min: float = 42.0
    v_color_max: float = 250.0
    gray_max: float = 172.0
    chroma_h_lo: int = 38
    chroma_h_hi: int = 88
    chroma_s_min: int = 78
    chroma_v_min: int = 78


def _panel_interior_mask(shape: Tuple[int, int], corners: np.ndarray) -> np.ndarray:
    h, w = shape[:2]
    q = pc.order_points(corners.astype(np.float32))
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(m, q.astype(np.int32), 255)
    return m > 0


def calibrate_panel_black_from_corners(
    image_bgr: np.ndarray,
    corners_tltrbrbl: np.ndarray,
    *,
    percentile: float = 99.0,
) -> PanelBlackThresholds:
    """
    Z wnętrza panelu: percentyl jasności ciemnych pikseli → górna granica „czerni”.
    Wyklucza białe linie siatki i nasycone kartki.
    """
    t = PanelBlackThresholds()
    if image_bgr is None or corners_tltrbrbl is None or corners_tltrbrbl.shape != (4, 2):
        return t

    inside = _panel_interior_mask(image_bgr.shape[:2], corners_tltrbrbl)
    if not np.any(inside):
        return t

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    white = inside & (v >= t.v_white_min) & (s <= t.s_white_max)
    color = inside & (s >= t.s_color_min) & (v >= t.v_color_min) & (v <= t.v_color_max)
    not_chroma = ~(((h >= t.chroma_h_lo) & (h <= t.chroma_h_hi)) & (s >= t.chroma_s_min) & (v >= t.chroma_v_min))
    color = color & not_chroma

    dark_candidate = inside & ~white & ~color & (s <= t.s_black_max + 15.0)
    vals = v[dark_candidate]
    if vals.size < 80:
        return t

    p = float(np.percentile(vals, percentile))
    t.v_black_max = float(np.clip(p + 18.0, 125.0, 198.0))
    t.v_shadow_max = float(np.clip(p + 10.0, 115.0, 188.0))
    t.s_black_max = float(np.clip(float(np.percentile(s[dark_candidate], 98)) + 18.0, 110.0, 165.0))
    t.gray_max = float(np.clip(float(np.percentile(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)[dark_candidate].astype(np.float32),
        99,
    )) + 14.0, 140.0, 205.0))
    return t


def mask_panel_black(
    hsv: np.ndarray,
    thresholds: Optional[PanelBlackThresholds] = None,
) -> np.ndarray:
    """
    Czerń + cienie (niska saturacja, umiarkowanie niski V, neutralny odcień szarości).
    """
    th = thresholds or PanelBlackThresholds()
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    gray = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY).astype(np.float32)

    core_black = (v <= th.v_black_max) & (s <= th.s_black_max)
    shadow = (v <= th.v_shadow_max) & (s <= th.s_shadow_max)
    matte = (gray <= th.gray_max) & (s <= th.s_black_max) & (v <= th.v_black_max + 8.0)
    neutral_dark = (v <= th.v_black_max) & (s <= 95.0)

    return (core_black | shadow | matte | neutral_dark).astype(np.uint8) * 255


def mask_panel_white(hsv: np.ndarray, thresholds: Optional[PanelBlackThresholds] = None) -> np.ndarray:
    th = thresholds or PanelBlackThresholds()
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    gray = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return (
        ((v >= th.v_white_min) & (s <= th.s_white_max))
        | (gray >= 158.0)
    ).astype(np.uint8) * 255


def mask_panel_color_card(hsv: np.ndarray, thresholds: Optional[PanelBlackThresholds] = None) -> np.ndarray:
    th = thresholds or PanelBlackThresholds()
    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    color = (s >= th.s_color_min) & (v >= th.v_color_min) & (v <= th.v_color_max)
    not_chroma = ~(
        ((h >= th.chroma_h_lo) & (h <= th.chroma_h_hi))
        & (s >= th.chroma_s_min)
        & (v >= th.chroma_v_min)
    )
    return (color & not_chroma).astype(np.uint8) * 255


def mask_keep_on_panel(
    hsv: np.ndarray,
    thresholds: Optional[PanelBlackThresholds] = None,
) -> np.ndarray:
    """Zostaw piksel: czern/cien + biel + kolor kartki."""
    bk = mask_panel_black(hsv, thresholds)
    wh = mask_panel_white(hsv, thresholds)
    co = mask_panel_color_card(hsv, thresholds)
    return np.maximum(np.maximum(bk, wh), co)


def mask_keep_inside_panel(
    image_bgr: np.ndarray,
    corners_tltrbrbl: np.ndarray,
    thresholds: Optional[PanelBlackThresholds] = None,
    *,
    close_px: int = 21,
) -> np.ndarray:
    """
    Wewnątrz trapezu: zostaw prawie wszystko co nie jest „tłem warsztatu”.
    Cienie = ciemne piksele względem percentyla w panelu + domknięcie morfologiczne.
    """
    th = thresholds or PanelBlackThresholds()
    if corners_tltrbrbl is None or corners_tltrbrbl.shape != (4, 2):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        return mask_keep_on_panel(hsv, th)

    inside = _panel_interior_mask(image_bgr.shape[:2], corners_tltrbrbl)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    white = mask_panel_white(hsv, th) > 0
    color = mask_panel_color_card(hsv, th) > 0
    neutral = inside & ~white & ~color

    v_hi = th.v_black_max
    g_hi = th.gray_max
    if np.count_nonzero(neutral) > 80:
        v_hi = float(np.percentile(v[neutral], 99.3)) + 10.0
        g_hi = float(np.percentile(gray[neutral], 99.3)) + 12.0
        v_hi = min(v_hi, 205.0)
        g_hi = min(g_hi, 210.0)

    dark = inside & (
        ((v <= v_hi) & (s <= th.s_black_max + 25.0))
        | (gray <= g_hi)
        | ((v <= v_hi + 12.0) & (s <= 85.0))
    )
    keep = white | color | dark

    if close_px > 0:
        dk = (dark.astype(np.uint8) * 255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        dk = cv2.morphologyEx(dk, cv2.MORPH_CLOSE, k)
        keep = keep | (dk > 0)

    return keep.astype(np.uint8) * 255


def patch_is_panel_cell(
    patch_bgr: np.ndarray,
    thresholds: Optional[PanelBlackThresholds] = None,
    *,
    min_black_frac: float = 0.22,
    median_v_max: float = 152.0,
) -> bool:
    """Komórka siatki: ciemna powierzchnia (w tym cienie) lub kolorowa kartka."""
    if patch_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    if float(np.max(hsv[:, :, 1])) >= (thresholds or PanelBlackThresholds()).s_color_min + 4.0:
        sat_mean = float(np.mean(hsv[:, :, 1]))
        if sat_mean >= 52.0 and float(np.mean(hsv[:, :, 2])) >= 40.0:
            return True
    black_m = mask_panel_black(hsv, thresholds)
    frac = float(np.count_nonzero(black_m)) / float(black_m.size)
    v = hsv[:, :, 2].astype(np.float32)
    if frac >= min_black_frac:
        return True
    if float(np.median(v)) <= median_v_max and frac >= 0.12:
        return True
    return False
