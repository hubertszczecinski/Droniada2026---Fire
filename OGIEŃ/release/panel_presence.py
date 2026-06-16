"""
Czy panel jest realnie w kadrze (nie: tracker hold / fałszywy quad YOLO).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

# Progi (env nadpisują na Jetsonie / nag5)
def _env_float(name: str, default: float) -> float:
    import os
    raw = os.environ.get(name, '').strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def assess_panel_presence(
    corners_px: Optional[np.ndarray],
    corner_meta: Optional[Dict[str, Any]],
    corner_src: str,
    *,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Zwraca (present, reason, detail).

    present=False gdy brak rogów, sam hold trackera, słaba detekcja YOLO lub zły reproj/rozmiar.
    """
    meta = dict(corner_meta or {})
    src = str(corner_src or 'none')

    if corners_px is None or corners_px.shape != (4, 2):
        return False, 'no_corners', meta

    if meta.get('tracker_held') or 'hold' in src:
        return False, 'tracker_hold', meta

    if meta.get('fail') in ('no_detection', 'low_kpt_conf'):
        return False, str(meta.get('fail')), meta

    reproj = float(meta.get('reproj_mean_px', 999.0))
    max_reproj = _env_float('DRONIADA_PANEL_MAX_REPROJ_PX', 42.0)
    if reproj > max_reproj:
        return False, 'high_reproj', {**meta, 'reproj_mean_px': reproj}

    if str(meta.get('method', '')) == 'yolo_pose':
        det_conf = float(meta.get('det_conf', 0.0))
        kpt_min = float(meta.get('kpt_conf_min', 0.0))
        min_det = _env_float('DRONIADA_YOLO_MIN_DET_CONF', 0.22)
        min_kpt = _env_float('DRONIADA_YOLO_MIN_KPT_CONF', 0.18)
        if det_conf < min_det and kpt_min < min_kpt:
            return False, 'low_yolo_conf', {
                **meta,
                'det_conf': det_conf,
                'kpt_conf_min': kpt_min,
            }

    if image_shape is not None:
        h, w = int(image_shape[0]), int(image_shape[1])
        if h > 0 and w > 0:
            pts = corners_px.reshape(-1, 2).astype(np.float64)
            area = float(cv2_quad_area(pts)) if pts.shape[0] == 4 else 0.0
            min_frac = _env_float('DRONIADA_PANEL_MIN_AREA_FRAC', 0.04)
            if area / float(h * w) < min_frac:
                return False, 'quad_too_small', {**meta, 'area_frac': area / float(h * w)}

    return True, 'ok', meta


def resolve_panel_presence(
    corners_px: Optional[np.ndarray],
    corner_meta: Optional[Dict[str, Any]],
    corner_src: str,
    *,
    image_shape: Optional[Tuple[int, int]] = None,
    reliable_b: bool = False,
    reproj_b: float = 999.0,
    max_reproj_b: float = 18.0,
    analyze_err: Optional[str] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Łączy heurystykę rogów YOLO z wynikiem modułu B.

    Gdy moduł B ma grid_xy_reliable (lub niski reproj po analizie), uznajemy panel
    za obecny — nawet jeśli reproj w meta rogów YOLO jest zawyżony (high_reproj).
    """
    if analyze_err == 'no_corners' or corners_px is None or corners_px.shape != (4, 2):
        return False, 'no_corners', dict(corner_meta or {})

    present, reason, detail = assess_panel_presence(
        corners_px,
        corner_meta,
        corner_src,
        image_shape=image_shape,
    )
    hard_absent = reason in ('no_corners', 'tracker_hold', 'no_detection', 'low_kpt_conf')
    if hard_absent:
        return present, reason, detail

    if reliable_b:
        return True, 'reliable_b', {**detail, 'reproj_b': float(reproj_b)}

    if float(reproj_b) <= float(max_reproj_b):
        return True, 'ok_reproj_b', {**detail, 'reproj_b': float(reproj_b)}

    return present, reason, detail


def cv2_quad_area(pts: np.ndarray) -> float:
    import cv2
    return float(abs(cv2.contourArea(pts.astype(np.float32))))
