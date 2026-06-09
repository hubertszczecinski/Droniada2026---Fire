"""Detect colored cards on a warped panel (live test without YOLO weights)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from release.panel_black import PanelBlackThresholds

import cv2
import numpy as np
import pipeline_competition as pc
from release.card_color_profile import (
    active_color_ranges,
    is_calibrated_profile,
    load_active_profile,
    profile_val_floor,
    tight_range_from_centroid,
    hsv_centroid_distance,
    hue_distance,
    CENTROID_MAX_DIST,
)

_CLASS_MIN_SAT_MEAN: Dict[int, float] = {
    1: 48.0,
    2: 55.0,
    3: 58.0,
    4: 28.0,
    5: 55.0,
}

# Karta ma jasną piankę; ciemny panel z szumem H (zieleń/żółć) ma niskie V.
_CLASS_MIN_VAL_MEAN: Dict[int, float] = {
    0: 58.0,
    1: 55.0,
    2: 45.0,
    3: 50.0,
    4: 28.0,
    5: 58.0,
}

_MAX_CENTER_BLACK_FRAC = 0.62

_MIN_CELL_FILL = 0.14
_MIN_SAT_MEAN = 45.0
_MIN_SAT_MEAN_CARD = 55.0
_MAX_WHITE_S = 78.0
_MIN_WHITE_V = 148.0
_MAX_WHITE_FRAC = 0.28
_CELL_GRID_PAD_FRAC = 0.12
_CELL_COLOR_CORE_FRAC = 0.28
_CELL_COLOR_CORE_EDGE_FRAC = 0.18
_MIN_PANEL_BLACK_FRAC_COLORED = 0.34
_MIN_PANEL_BLACK_FRAC_RING = 0.22
_MIN_INSIDE_PANEL_PX = 12.0
_MIN_VALID_CELL_FRAC = 0.42
_MIN_ON_PANEL_RING_FRAC = 0.88
_MIN_COLOR_ON_VALID_FRAC = 0.14
_MIN_DETECTION_SCORE = 0.07
_PANEL_INSET_FRAC = 0.06
_MAX_CARDS = 4


def _grid_pad_frac(grid_row: int, grid_col: int) -> float:
    """Odsunięcie od białych linii siatki — nie mylić z próbką koloru kartki."""
    if _border_margin(grid_row, grid_col) <= 1:
        return 0.10
    return _CELL_GRID_PAD_FRAC


def _color_core_frac(grid_row: int, grid_col: int) -> float:
    """Środek komórki pod klasyfikację — brzeg panelu (np. żółć 1,2) = szerszy rdzeń."""
    if _border_margin(grid_row, grid_col) <= 1:
        return _CELL_COLOR_CORE_EDGE_FRAC
    return _CELL_COLOR_CORE_FRAC


def _max_cards_limit() -> int:
    return _MAX_CARDS


def _border_margin(grid_row: int, grid_col: int) -> int:
    """Odległość od brzegu siatki (0 = wiersz/kol. 1 lub 10)."""
    return int(min(grid_row - 1, 10 - grid_row, grid_col - 1, 10 - grid_col))


def _min_fill_for_margin(margin: int, cls_id: Optional[int] = None) -> float:
    """Im bliżej brzegu, tym większe pokrycie kolorem w środku komórki."""
    m = max(0, min(2, margin))
    base = 0.08 if is_calibrated_profile() else 0.11
    fill = base + 0.05 * (2 - m)
    if int(cls_id or -1) == 3 and is_calibrated_profile() and m <= 1:
        return max(0.06, fill * 0.72)
    return fill


def _min_valid_cell_frac(grid_row: int, grid_col: int) -> float:
    """Brzeg siatki (1,2 / 9,1) — maska panelu często ucina komórkę."""
    margin = _border_margin(grid_row, grid_col)
    if is_calibrated_profile():
        if margin <= 0:
            return 0.18
        if margin == 1:
            return 0.30
        return 0.38
    return _MIN_VALID_CELL_FRAC


def _enhance_patch_bgr_for_color(patch_bgr: np.ndarray) -> np.ndarray:
    """Lokalny CLAHE gdy kartka w kadrze jest ciemna (krawędź / cień)."""
    if patch_bgr.size == 0:
        return patch_bgr
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    v_med = float(np.median(v_ch))
    if v_med >= 95.0:
        return patch_bgr
    clip = 4.0 if v_med < 25.0 else (3.5 if v_med < 45.0 else 2.5)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4))
    v_eq = clahe.apply(v_ch)
    return cv2.cvtColor(cv2.merge([h_ch, s_ch, v_eq]), cv2.COLOR_HSV2BGR)


def _warp_bgr_for_color_detect(warped_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
    """Ciemny warp (np. nag5): CLAHE na kanale V — zachowuje H/S (szary BGR zerował S)."""
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    mean = float(np.mean(v_ch))
    if mean >= 38.0:
        return warped_bgr, mean
    clip = 4.2 if mean < 15.0 else 2.8
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    v_eq = clahe.apply(v_ch)
    out = cv2.cvtColor(cv2.merge([h_ch, s_ch, v_eq]), cv2.COLOR_HSV2BGR)
    return out, mean


def _min_sat_for_margin(margin: int, cls_id: int) -> float:
    base = _CLASS_MIN_SAT_MEAN.get(int(cls_id), _MIN_SAT_MEAN)
    m = max(0, min(2, margin))
    if int(cls_id) == 3 and is_calibrated_profile() and m <= 1:
        return max(36.0, base - 20.0)
    return max(base, base + 10.0 * (2 - m))


def _inner_panel_corners(corners_tltrbrbl: np.ndarray, inset_frac: float = _PANEL_INSET_FRAC) -> np.ndarray:
    c = corners_tltrbrbl.astype(np.float64)
    ctr = c.mean(axis=0, keepdims=True)
    scale = max(0.0, 1.0 - float(inset_frac))
    return (ctr + (c - ctr) * scale).astype(np.float32)


def _build_warp_panel_mask(
    warped_shape: Tuple[int, int],
    corners_tltrbrbl: np.ndarray,
    homography_img_to_warp: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    min_inside_px: float = _MIN_INSIDE_PANEL_PX,
    inset_frac: float = _PANEL_INSET_FRAC,
) -> np.ndarray:
    ih, iw = image_shape[:2]
    wh, ww = warped_shape[:2]
    img_mask = np.zeros((ih, iw), dtype=np.uint8)
    q = pc.order_points(_inner_panel_corners(corners_tltrbrbl, inset_frac))
    cv2.fillConvexPoly(img_mask, q.astype(np.int32), 255)
    if min_inside_px > 1.0:
        k = max(3, int(min_inside_px * 2) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
    warped_mask = cv2.warpPerspective(
        img_mask,
        homography_img_to_warp.astype(np.float32),
        (ww, wh),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_mask > 127


def _panel_black_fraction(
    patch_bgr: np.ndarray,
    black_thresholds: Optional['PanelBlackThresholds'] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    from release.panel_black import mask_panel_black

    if patch_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    bk = mask_panel_black(hsv, black_thresholds) > 0
    if valid_mask is not None and valid_mask.shape == bk.shape:
        n = int(np.count_nonzero(valid_mask))
        if n < 8:
            return 0.0
        return float(np.count_nonzero(bk & valid_mask)) / float(n)
    return float(np.count_nonzero(bk)) / float(bk.size)


def _panel_black_fraction_ring(
    patch_bgr: np.ndarray,
    black_thresholds: Optional['PanelBlackThresholds'] = None,
    valid_mask: Optional[np.ndarray] = None,
    *,
    ring_frac: float = 0.22,
) -> float:
    from release.panel_black import mask_panel_black

    if patch_bgr.size == 0:
        return 0.0
    ph, pw = patch_bgr.shape[:2]
    mx = max(1, int(pw * ring_frac))
    my = max(1, int(ph * ring_frac))
    if pw <= 2 * mx or ph <= 2 * my:
        return _panel_black_fraction(patch_bgr, black_thresholds)
    ring = np.zeros((ph, pw), dtype=bool)
    ring[:my, :] = True
    ring[ph - my :, :] = True
    ring[:, :mx] = True
    ring[:, pw - mx :] = True
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    bk = mask_panel_black(hsv, black_thresholds) > 0
    if valid_mask is not None and valid_mask.shape == ring.shape:
        ring = ring & valid_mask
    n = int(np.count_nonzero(ring))
    if n < 8:
        return 0.0
    return float(np.count_nonzero(bk & ring)) / float(n)


def _warp_center_inside_panel(
    warp_cx: float,
    warp_cy: float,
    corners_tltrbrbl: np.ndarray,
    homography_img_to_warp: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    min_inside_px: float = _MIN_INSIDE_PANEL_PX,
    inset_frac: float = _PANEL_INSET_FRAC,
) -> bool:
    try:
        h_inv = np.linalg.inv(homography_img_to_warp.astype(np.float64))
    except np.linalg.LinAlgError:
        return False
    src = np.array([[[float(warp_cx), float(warp_cy)]]], dtype=np.float32)
    pts = cv2.perspectiveTransform(src, h_inv.astype(np.float32))
    u, v = float(pts[0, 0, 0]), float(pts[0, 0, 1])
    ih, iw = image_shape[:2]
    if u < 0 or v < 0 or u >= iw or v >= ih:
        return False
    q = pc.order_points(_inner_panel_corners(corners_tltrbrbl, inset_frac))
    dist = cv2.pointPolygonTest(q.reshape(-1, 1, 2).astype(np.float32), (u, v), True)
    return float(dist) >= float(min_inside_px)


def _cell_has_saturated_card(patch_bgr: np.ndarray) -> bool:
    if patch_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    if float(np.max(sat)) < _MIN_SAT_MEAN_CARD:
        return False
    if float(np.mean(sat)) < _MIN_SAT_MEAN:
        return False
    return True


def _cell_on_black_panel(
    patch_bgr: np.ndarray,
    *,
    black_thresholds: Optional['PanelBlackThresholds'] = None,
    ring_patch_bgr: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    valid_ring_mask: Optional[np.ndarray] = None,
) -> bool:
    from release.panel_black import patch_is_panel_cell

    if patch_bgr.size == 0:
        return False
    if _cell_has_saturated_card(patch_bgr) and black_thresholds is None and valid_ring_mask is None:
        return True
    ring_src = ring_patch_bgr if ring_patch_bgr is not None and ring_patch_bgr.size else patch_bgr
    vm_ring = valid_ring_mask
    if vm_ring is not None and vm_ring.shape[:2] != ring_src.shape[:2]:
        vm_ring = None
    if _cell_has_saturated_card(patch_bgr):
        if vm_ring is not None and float(np.mean(vm_ring)) >= _MIN_ON_PANEL_RING_FRAC:
            return True
        black_frac = _panel_black_fraction(ring_src, black_thresholds, vm_ring)
        ring_frac = _panel_black_fraction_ring(ring_src, black_thresholds, vm_ring)
        if ring_frac < _MIN_PANEL_BLACK_FRAC_RING and black_frac < _MIN_PANEL_BLACK_FRAC_COLORED:
            return False
        hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        v_mean = float(np.mean(hsv[:, :, 2]))
        if v_mean > 128.0 and black_frac < 0.48 and ring_frac < 0.30:
            return False
        return True
    return patch_is_panel_cell(patch_bgr, black_thresholds)


def _color_fraction(
    hsv: np.ndarray,
    cls_id: int,
    *,
    valid_mask: Optional[np.ndarray] = None,
    white: Optional[np.ndarray] = None,
) -> float:
    for cid, (lo, hi) in active_color_ranges():
        if cid != cls_id:
            continue
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0
        if white is not None:
            mask = mask & ~white
        if valid_mask is not None:
            mask = mask & valid_mask
            denom = float(max(1, int(np.count_nonzero(valid_mask))))
        else:
            denom = float(mask.size)
        return float(np.count_nonzero(mask)) / denom
    return 0.0


def _hue_median_on_sat(
    hsv: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray] = None,
    sat_min: float = 28.0,
) -> Optional[float]:
    sat = hsv[:, :, 1].astype(np.float32)
    sel = sat >= sat_min
    if valid_mask is not None and valid_mask.shape == sel.shape:
        sel = sel & valid_mask
    if not bool(np.any(sel)):
        return None
    return float(np.median(hsv[:, :, 0][sel]))


def _resolve_blue_purple(
    hsv: np.ndarray,
    cls_id: int,
    *,
    valid_mask: Optional[np.ndarray] = None,
    white: Optional[np.ndarray] = None,
) -> int:
    """Fiolet w słabym świetle często ma H jak niebieski (V niskie) — rozstrzygamy po medianie H."""
    f_b = _color_fraction(hsv, 2, valid_mask=valid_mask, white=white)
    f_p = _color_fraction(hsv, 4, valid_mask=valid_mask, white=white)
    if f_b < 0.08 and f_p < 0.08:
        return cls_id
    h_med = _hue_median_on_sat(hsv, valid_mask=valid_mask)
    if h_med is not None:
        if h_med >= 115.0 and f_p >= 0.08:
            return 4
        if h_med <= 104.0 and f_b >= 0.10:
            return 2
    if f_p >= max(f_b * 0.38, 0.10) and f_p >= 0.08:
        return 4
    if f_b > f_p * 1.9 and f_b >= 0.14:
        return 2
    return cls_id


def _refine_color_class(
    hsv: np.ndarray,
    cls_id: int,
    *,
    valid_mask: Optional[np.ndarray] = None,
    white: Optional[np.ndarray] = None,
) -> int:
    if cls_id in (2, 4):
        return _resolve_blue_purple(hsv, cls_id, valid_mask=valid_mask, white=white)
    if cls_id in (3, 5):
        f_o = _color_fraction(hsv, 5, valid_mask=valid_mask, white=white)
        f_y = _color_fraction(hsv, 3, valid_mask=valid_mask, white=white)
        if cls_id == 3 and f_o >= 0.12 and f_o >= f_y * 0.55:
            return 5
        if cls_id == 5 and f_y > f_o * 1.4 and f_y >= 0.15:
            h_med = _hue_median_on_sat(hsv, valid_mask=valid_mask)
            if h_med is not None and float(h_med) <= 22.0:
                return 5
            return 3
    if cls_id in (1, 3):
        f_g = _color_fraction(hsv, 1, valid_mask=valid_mask, white=white)
        f_y = _color_fraction(hsv, 3, valid_mask=valid_mask, white=white)
        h_med = _hue_median_on_sat(hsv, valid_mask=valid_mask)
        if cls_id == 1 and f_y >= 0.10 and (f_y >= f_g * 0.42 or (h_med is not None and h_med <= 38.0)):
            return 3
        if cls_id == 3 and f_g > f_y * 1.35 and f_g >= 0.14 and (h_med is None or h_med >= 42.0):
            return 1
    return cls_id


def _white_grid_mask(hsv: np.ndarray) -> np.ndarray:
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    return (v >= _MIN_WHITE_V) & (s <= _MAX_WHITE_S)


def _center_crop_patch(
    patch_bgr: np.ndarray,
    frac: float = _CELL_COLOR_CORE_FRAC,
) -> np.ndarray:
    ph, pw = patch_bgr.shape[:2]
    if ph < 4 or pw < 4:
        return patch_bgr
    mx = max(1, int(pw * frac))
    my = max(1, int(ph * frac))
    return patch_bgr[my : ph - my, mx : pw - mx]


def _patch_median_hsv(
    hsv: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Optional[Tuple[float, float, float]]:
    """Mediana HSV z nasyconych pikseli — tylko środek komórki (już wycięty)."""
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    for smin, vmin in ((40.0, 35.0), (28.0, 22.0), (18.0, 16.0)):
        sel = (sat >= smin) & (val >= vmin)
        if valid_mask is not None and valid_mask.shape == sel.shape:
            sel = sel & valid_mask
        n = int(np.count_nonzero(sel))
        if n >= 8:
            return (
                float(np.median(hsv[:, :, 0][sel])),
                float(np.median(sat[sel])),
                float(np.median(val[sel])),
            )
    return None


def _color_fraction_tight(
    hsv: np.ndarray,
    lo: Tuple[int, int, int],
    hi: Tuple[int, int, int],
    *,
    valid_mask: Optional[np.ndarray] = None,
    white: Optional[np.ndarray] = None,
) -> float:
    mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0
    if white is not None:
        mask = mask & ~white
    if valid_mask is not None:
        mask = mask & valid_mask
        denom = float(max(1, int(np.count_nonzero(valid_mask))))
    else:
        denom = float(mask.size)
    return float(np.count_nonzero(mask)) / denom


def _reject_weak_detection(
    cls_id: int,
    color_frac: float,
    sat_mean: float,
    white_frac: float,
    grid_row: int,
    grid_col: int,
) -> bool:
    """Słabe / niepewne detekcje — progi zależą od odległości od brzegu, nie od (w,k)."""
    score = _detection_score(color_frac, sat_mean, white_frac=white_frac)
    if score < _MIN_DETECTION_SCORE:
        return True
    if white_frac >= _MAX_WHITE_FRAC:
        return True
    margin = _border_margin(grid_row, grid_col)
    if color_frac < _min_fill_for_margin(margin, cls_id):
        return True
    if sat_mean < _min_sat_for_margin(margin, cls_id):
        return True
    if cls_id in (1, 3) and white_frac > 0.18 and color_frac < 0.38:
        return True
    if cls_id == 1 and is_calibrated_profile():
        if color_frac < 0.18:
            return True
        if margin == 0:
            return True
        if margin >= 2 and color_frac < 0.28:
            return True
    if not is_calibrated_profile():
        # Brzeg warpu / ostatni wiersz — szum siatki bywa zielonkawy (tylko domyślne HSV).
        if cls_id == 1 and grid_row >= 9 and grid_col <= 4:
            if color_frac < 0.28 or sat_mean < 62.0:
                return True
    if cls_id == 1 and not is_calibrated_profile() and (grid_row >= 8 or grid_col <= 3):
        if color_frac < 0.22 or sat_mean < 60.0:
            return True
    if cls_id == 1 and is_calibrated_profile() and _border_margin(grid_row, grid_col) >= 3:
        if color_frac < 0.24 or sat_mean < 55.0:
            return True
    return False


def _centroid_match_ok(
    cls_id: int,
    mh: float,
    ms: float,
    mv: float,
    matched_cent: Tuple[float, float, float],
) -> bool:
    """Walidacja względem dopasowanego centroidu (nie globalnych progów pod jedno nagranie)."""
    ch, cs, cv = matched_cent
    cid = int(cls_id)
    h_tol = 24.0 if cid in (1, 2, 4) else 20.0
    if cid == 3:
        h_tol = 26.0
    if hue_distance(float(mh), float(ch)) > h_tol:
        return False
    min_v = max(32.0, float(cv) * 0.45)
    if float(mv) < min_v:
        return False
    if cid == 1:
        min_s = 85.0 if float(cs) >= 150.0 else 90.0
        if float(ms) < min_s:
            return False
        if float(cv) >= 100.0 and float(mv) < max(72.0, float(cv) * 0.58):
            return False
        if float(ch) <= 62.0 and float(mh) > 68.0:
            return False
        if float(ch) >= 70.0 and (float(mh) < 68.0 or float(mh) > 95.0):
            return False
        return True
    if cid == 2:
        if float(cv) >= 180.0 and float(mv) < max(100.0, float(cv) * 0.62):
            return False
        if hue_distance(float(mh), float(ch)) > 22.0:
            return False
        return True
    if cid == 3:
        if float(ch) <= 40.0 and float(mh) > 55.0:
            return False
        if float(ch) >= 55.0 and (float(mh) < 52.0 or float(mh) > 82.0):
            return False
        return True
    return True


def _resolve_yellow_green(
    mh: float,
    ms: float,
    mv: float,
    best_cls: int,
    best_dist: float,
    matched_cent: Tuple[float, float, float],
    prof,
) -> Tuple[int, float, Tuple[float, float, float]]:
    """Strefa H 52–88: żółty Test.mov (H~69) vs turkusowa zieleń (H~74–80)."""
    if not (52.0 <= float(mh) <= 88.0):
        return best_cls, best_dist, matched_cent
    y_id = int(pc.COLOR_TO_CLASS.get('ZOLTA', 3))
    g_id = int(pc.COLOR_TO_CLASS.get('ZIELONA', 1))
    y_cents = prof.centroids_by_cls.get(y_id, [])
    g_cents = prof.centroids_by_cls.get(g_id, [])
    if not y_cents or not g_cents:
        return best_cls, best_dist, matched_cent
    y_ranked = sorted((hsv_centroid_distance(mh, ms, mv, c), c) for c in y_cents)
    g_ranked = sorted((hsv_centroid_distance(mh, ms, mv, c), c) for c in g_cents)
    y_dist, y_cent = y_ranked[0]
    g_dist, g_cent = g_ranked[0]
    if float(mh) >= 72.0 and float(ms) >= 140.0 and float(g_cent[0]) >= 68.0:
        if g_dist <= y_dist + 0.22:
            return g_id, g_dist, g_cent
    if float(mh) <= 68.0 and float(y_cent[0]) >= 55.0 and float(ms) < 175.0:
        if y_dist <= g_dist + 0.15:
            return y_id, y_dist, y_cent
    return best_cls, best_dist, matched_cent


def _classify_patch_bgr(
    patch_bgr: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    *,
    grid_row: int = 5,
    grid_col: int = 5,
    dark_warp: bool = False,
) -> Optional[Tuple[int, float, float]]:
    if patch_bgr.size == 0:
        return None
    margin = _border_margin(grid_row, grid_col)
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    if valid_mask is not None and valid_mask.shape == sat.shape:
        n_valid = int(np.count_nonzero(valid_mask))
        if n_valid < 12:
            return None
        sat_mean = float(np.mean(sat[valid_mask]))
        val_mean = float(np.mean(val[valid_mask]))
        denom = float(n_valid)
    else:
        sat_mean = float(np.mean(sat))
        val_mean = float(np.mean(val))
        denom = float(sat.size)
        valid_mask = None
    min_sat_global = 28.0 if is_calibrated_profile() else (36.0 if dark_warp else _MIN_SAT_MEAN)
    min_val_global = 22.0 if is_calibrated_profile() else (26.0 if dark_warp else 38.0)
    if sat_mean < min_sat_global:
        return None
    if val_mean < min_val_global:
        return None
    if val_mean > 250 and sat_mean < 100:
        return None
    white = _white_grid_mask(hsv)
    if valid_mask is not None:
        white = white & valid_mask
        white_frac = float(np.count_nonzero(white)) / denom
    else:
        white_frac = float(np.count_nonzero(white)) / float(white.size)
    if white_frac >= _MAX_WHITE_FRAC:
        return None

    prof = load_active_profile()
    if is_calibrated_profile() and prof.has_centroids():
        med = _patch_median_hsv(hsv, valid_mask)
        if med is None:
            return None
        mh, ms, mv = med
        ranked: List[Tuple[float, int, Tuple[float, float, float]]] = []
        for cls_id, cents in prof.centroids_by_cls.items():
            for cent in cents:
                dist = hsv_centroid_distance(mh, ms, mv, cent)
                ranked.append((dist, int(cls_id), cent))
        if not ranked:
            return None
        ranked.sort(key=lambda x: x[0])
        best_dist, best_cls, matched_cent = ranked[0]
        if best_dist > CENTROID_MAX_DIST:
            return None
        # Pomarańcz vs żółć — przy niskim H (nag5) nie klasyfikuj pomarańczy jako żółtej.
        pom_id = int(pc.COLOR_TO_CLASS.get('POMARANCZOWA', 5))
        if int(best_cls) == 3 and float(mh) <= 21.0:
            pom_dists = [
                hsv_centroid_distance(mh, ms, mv, c)
                for c in prof.centroids_by_cls.get(pom_id, [])
            ]
            if pom_dists:
                pom_dist = min(pom_dists)
                if float(mh) <= 19.0 and pom_dist <= best_dist + 0.45:
                    best_cls = pom_id
                    best_dist = pom_dist
                elif pom_dist <= best_dist * 1.05:
                    best_cls = pom_id
                    best_dist = pom_dist
        best_cls, best_dist, matched_cent = _resolve_yellow_green(
            mh, ms, mv, int(best_cls), float(best_dist), matched_cent, prof,
        )
        if not _centroid_match_ok(int(best_cls), mh, ms, mv, matched_cent):
            return None
        if len(ranked) > 1:
            second_dist, second_cls, _ = ranked[1]
            if second_cls != best_cls and second_dist < best_dist * 1.12 and best_dist > 0.45:
                return None
        lo, hi = tight_range_from_centroid(best_cls, mh, ms, mv)
        best_frac = _color_fraction_tight(hsv, lo, hi, valid_mask=valid_mask, white=white)
        second_frac = 0.0
        if len(ranked) > 1:
            _, second_cls, _second_cent = ranked[1]
            if second_cls != best_cls:
                lo2, hi2 = tight_range_from_centroid(second_cls, mh, ms, mv)
                second_frac = _color_fraction_tight(hsv, lo2, hi2, valid_mask=valid_mask, white=white)
    else:
        best_cls: Optional[int] = None
        best_frac = 0.0
        second_frac = 0.0
        for cls_id, (lo, hi) in active_color_ranges():
            mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0
            mask = mask & ~white
            if valid_mask is not None:
                mask = mask & valid_mask
                frac = float(np.count_nonzero(mask)) / denom
            else:
                frac = float(np.count_nonzero(mask)) / float(mask.size)
            if frac > best_frac:
                second_frac = best_frac
                best_frac = frac
                best_cls = cls_id
            elif frac > second_frac:
                second_frac = frac

    if best_cls is None or best_frac < _min_fill_for_margin(margin, best_cls):
        return None
    if best_frac < second_frac * 1.15 and best_frac < 0.22:
        return None
    min_sat_cls = _min_sat_for_margin(margin, int(best_cls))
    if sat_mean < min_sat_cls:
        return None
    cls_out = _refine_color_class(hsv, int(best_cls), valid_mask=valid_mask, white=white)
    if int(best_cls) == 2 and int(cls_out) == 4:
        best_frac = max(best_frac, _color_fraction(hsv, 4, valid_mask=valid_mask, white=white))
    min_val = _CLASS_MIN_VAL_MEAN.get(int(cls_out), 45.0)
    if is_calibrated_profile():
        floor = profile_val_floor(int(cls_out))
        if floor is not None:
            min_val = min(min_val, floor)
        if int(cls_out) == 3:
            min_val = min(min_val, 36.0)
    if val_mean < min_val:
        return None
    if int(cls_out) == 1:
        med_v = _patch_median_hsv(hsv, valid_mask)
        if med_v is not None:
            mh2, ms2, mv2 = med_v
            if float(mv2) < 32.0:
                return None
            if int(grid_row) >= 9 and float(mh2) < 75.0 and float(ms2) >= 200.0:
                return None
    return (cls_out, float(best_frac), float(sat_mean))


def _detection_score(color_frac: float, sat_mean: float, *, white_frac: float = 0.0) -> float:
    return float(color_frac) * (float(sat_mean) / 100.0) * (1.0 - 0.65 * float(white_frac))


def detect_cards_on_warped(
    warped_bgr: np.ndarray,
    *,
    min_cells: int = 1,
    max_cells: int = 0,
    black_thresholds: Optional['PanelBlackThresholds'] = None,
    corners_tltrbrbl: Optional[np.ndarray] = None,
    homography_img_to_warp: Optional[np.ndarray] = None,
    image_shape: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    if max_cells <= 0:
        max_cells = _max_cards_limit()
    warped_bgr, warp_mean = _warp_bgr_for_color_detect(warped_bgr)
    dark_warp = float(warp_mean) < 22.0
    h, w = warped_bgr.shape[:2]
    cw = w / 10.0
    ch = h / 10.0
    from release.snapshot_cell_color import cell_rect_from_grid_lines, grid_lines_for_warped

    glx, gly, grid_src = grid_lines_for_warped(warped_bgr)
    use_line_grid = grid_src == 'detected'
    panel_mask: Optional[np.ndarray] = None
    if (
        corners_tltrbrbl is not None
        and homography_img_to_warp is not None
        and image_shape is not None
    ):
        inset = 0.02 if is_calibrated_profile() else _PANEL_INSET_FRAC
        panel_mask = _build_warp_panel_mask(
            warped_bgr.shape[:2],
            corners_tltrbrbl,
            homography_img_to_warp,
            image_shape,
            inset_frac=inset,
        )
    found: List[Dict[str, Any]] = []
    for grid_row_est in range(1, 11):
        for grid_col_est in range(1, 11):
            pad_frac = _grid_pad_frac(grid_row_est, grid_col_est)
            core_frac = _color_core_frac(grid_row_est, grid_col_est)
            if use_line_grid:
                try:
                    lx0, ly0, lx1, ly1 = cell_rect_from_grid_lines(
                        glx, gly, grid_row_est, grid_col_est,
                    )
                except ValueError:
                    continue
                pw = max(1, lx1 - lx0)
                ph = max(1, ly1 - ly0)
                pad_x = max(1, int(pw * pad_frac))
                pad_y = max(1, int(ph * pad_frac))
                x0 = lx0 + pad_x
                y0 = ly0 + pad_y
                x1 = lx1 - pad_x
                y1 = ly1 - pad_y
            else:
                row = 10 - grid_row_est
                col = grid_col_est - 1
                pad_x = max(2, int(cw * pad_frac))
                pad_y = max(2, int(ch * pad_frac))
                x0 = int(col * cw) + pad_x
                y0 = int(row * ch) + pad_y
                x1 = int((col + 1) * cw) - pad_x
                y1 = int((row + 1) * ch) - pad_y
            if x1 <= x0 or y1 <= y0:
                continue
            patch_full = warped_bgr[y0:y1, x0:x1]
            valid_full: Optional[np.ndarray] = None
            if panel_mask is not None:
                valid_full = panel_mask[y0:y1, x0:x1]
                if float(np.mean(valid_full)) < _min_valid_cell_frac(grid_row_est, grid_col_est):
                    continue
            patch_full = _enhance_patch_bgr_for_color(patch_full)
            patch = _center_crop_patch(patch_full, core_frac)
            valid_patch: Optional[np.ndarray] = None
            if valid_full is not None:
                valid_patch = _center_crop_patch(valid_full.astype(np.uint8), core_frac) > 0
                if int(np.count_nonzero(valid_patch)) < 12:
                    if _border_margin(grid_row_est, grid_col_est) <= 1 and is_calibrated_profile():
                        valid_patch = None
                    else:
                        continue
            meta = _classify_patch_bgr(
                patch, valid_patch, grid_row=grid_row_est, grid_col=grid_col_est, dark_warp=dark_warp,
            )
            if meta is None:
                continue
            cls_id, color_frac, sat_mean = meta
            hsv_c = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            white_c = _white_grid_mask(hsv_c)
            if valid_patch is not None:
                white_c = white_c & valid_patch
                wf = float(np.count_nonzero(white_c)) / float(max(1, int(np.count_nonzero(valid_patch))))
            else:
                wf = float(np.count_nonzero(white_c)) / float(white_c.size)
            if not _cell_on_black_panel(
                patch,
                black_thresholds=black_thresholds,
                ring_patch_bgr=patch_full,
                valid_mask=valid_patch,
                valid_ring_mask=valid_full,
            ):
                continue
            center_black = _panel_black_fraction(patch, black_thresholds, valid_patch)
            if center_black > _MAX_CENTER_BLACK_FRAC:
                continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            mask = np.zeros(patch.shape[:2], dtype=np.uint8)
            white_px = _white_grid_mask(hsv)
            for cid, (lo, hi) in active_color_ranges():
                if cid != cls_id:
                    continue
                mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                mask[white_px] = 0
                break
            if valid_patch is not None:
                mask = cv2.bitwise_and(mask, (valid_patch.astype(np.uint8) * 255))
            if cv2.countNonZero(mask) > 0:
                m = cv2.moments(mask)
                px = float(m['m10'] / m['m00'])
                py = float(m['m01'] / m['m00'])
            else:
                px = (patch.shape[1] - 1) * 0.5
                py = (patch.shape[0] - 1) * 0.5
            gx = float(x0 + px)
            gy = float(y0 + py)
            if use_line_grid:
                grid_col = int(grid_col_est)
                grid_row = int(grid_row_est)
            else:
                grid_col = int(np.clip(round(gx / cw - 0.5) + 1, 1, 10))
                grid_row = int(np.clip(10 - round(gy / ch - 0.5), 1, 10))
            if _reject_weak_detection(cls_id, color_frac, sat_mean, wf, grid_row, grid_col):
                continue
            if (
                corners_tltrbrbl is not None
                and homography_img_to_warp is not None
                and image_shape is not None
                and not _warp_center_inside_panel(
                    gx, gy, corners_tltrbrbl, homography_img_to_warp, image_shape,
                )
            ):
                if not (
                    is_calibrated_profile()
                    and _border_margin(grid_row, grid_col) <= 1
                ):
                    continue
            found.append({
                'cls_id': cls_id,
                'warp_cx': gx,
                'warp_cy': gy,
                'grid_col': grid_col,
                'grid_row': grid_row,
                'color': pc.CLASS_TO_COLOR.get(cls_id, 'UNKNOWN'),
                'color_frac': color_frac,
                'sat_mean': sat_mean,
                'score': _detection_score(color_frac, sat_mean, white_frac=wf),
            })
    by_cell: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for d in found:
        key = (int(d['grid_row']), int(d['grid_col']))
        if key not in by_cell or float(d['score']) > float(by_cell[key]['score']):
            by_cell[key] = d
    ranked = sorted(by_cell.values(), key=lambda d: -float(d['score']))
    return ranked[:max_cells] if len(ranked) >= min_cells else ranked


def warped_detections_to_yolo(
    warped_dets: List[Dict[str, Any]],
    homography_img_to_warp: np.ndarray,
    img_w: int,
    img_h: int,
) -> List[Tuple[int, float, float, float, float]]:
    if not warped_dets:
        return []
    try:
        h_inv = np.linalg.inv(homography_img_to_warp.astype(np.float64))
    except np.linalg.LinAlgError:
        return []
    out: List[Tuple[int, float, float, float, float]] = []
    cell_w_n = 0.08
    cell_h_n = 0.08
    for d in warped_dets:
        src = np.array([[[d['warp_cx'], d['warp_cy']]]], dtype=np.float32)
        pts = cv2.perspectiveTransform(src, h_inv.astype(np.float32))
        u, v = float(pts[0, 0, 0]), float(pts[0, 0, 1])
        out.append((
            int(d['cls_id']),
            float(np.clip(u / img_w, 0.0, 1.0)),
            float(np.clip(v / img_h, 0.0, 1.0)),
            cell_w_n,
            cell_h_n,
        ))
    return out


def warped_dets_to_predictions(warped_dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in warped_dets:
        out.append({
            'x': int(d['grid_col']),
            'y': int(d['grid_row']),
            'color': str(d.get('color', 'UNKNOWN')),
        })
    return out


def detect_cards_live(
    image_bgr: np.ndarray,
    corners_tltrbrbl: np.ndarray,
    homography_img_to_warp: np.ndarray,
    warped_bgr: np.ndarray,
    *,
    black_thresholds: Optional['PanelBlackThresholds'] = None,
) -> Tuple[List[Tuple[int, float, float, float, float]], List[Dict[str, Any]]]:
    h, w = image_bgr.shape[:2]
    warped_dets = detect_cards_on_warped(
        warped_bgr,
        black_thresholds=black_thresholds,
        corners_tltrbrbl=corners_tltrbrbl,
        homography_img_to_warp=homography_img_to_warp,
        image_shape=(h, w),
    )
    yolo = warped_detections_to_yolo(warped_dets, homography_img_to_warp, w, h)
    return yolo, warped_dets
