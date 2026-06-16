"""Intrinsics for Tarot T10X-2A gimbal camera (10x optical zoom, 1080p HDMI)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

_PROFILE_PATH = Path(__file__).resolve().parents[1] / 'config' / 'tarot_t10x_2a.json'

LENS_MM_WIDE = 4.9
LENS_MM_TELE = 49.0
FOV_H_WIDE_DEG = 66.6
FOV_H_TELE_DEG = 7.2
OPTICAL_ZOOM = 10.0


def _sensor_width_mm() -> float:
    """Effective horizontal sensor size from wide-end FOV and focal length."""
    return 2.0 * LENS_MM_WIDE * math.tan(math.radians(FOV_H_WIDE_DEG * 0.5))


def focal_mm_from_zoom_ratio(zoom_ratio: float) -> float:
    z = float(zoom_ratio)
    z = max(1.0, min(OPTICAL_ZOOM, z))
    return LENS_MM_WIDE * z


def fov_horizontal_deg_from_zoom_ratio(zoom_ratio: float) -> float:
    f_mm = focal_mm_from_zoom_ratio(zoom_ratio)
    sw = _sensor_width_mm()
    return math.degrees(2.0 * math.atan(sw / (2.0 * f_mm)))


def intrinsics_from_tarot_t10x(
    image_shape: Tuple[int, int],
    *,
    zoom_ratio: float = 1.0,
    focal_mm: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pinhole K for Tarot T10X-2A at given zoom.

    zoom_ratio: 1.0 (wide, f≈4.9 mm) … 10.0 (tele, f≈49 mm).
    focal_mm: overrides zoom_ratio when set.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    f_mm = float(focal_mm) if focal_mm is not None else focal_mm_from_zoom_ratio(zoom_ratio)
    sw = _sensor_width_mm()
    sh = sw * (h / float(w))
    fx = f_mm / sw * w
    fy = f_mm / sh * h
    cx, cy = w / 2.0, h / 2.0
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    dist = np.zeros((5, 1), dtype=np.float32)
    return k, dist


def load_profile() -> Dict[str, Any]:
    if not _PROFILE_PATH.is_file():
        return {}
    with _PROFILE_PATH.open(encoding='utf-8') as fh:
        return json.load(fh)


def intrinsics_for_preset(
    image_shape: Tuple[int, int],
    preset: str,
) -> Tuple[np.ndarray, np.ndarray]:
    data = load_profile()
    presets = data.get('presets') or {}
    if preset not in presets:
        raise KeyError(f'unknown tarot preset: {preset!r} (have {list(presets)})')
    pr = presets[preset]
    zr = float(pr.get('zoom_ratio', 1.0))
    f_mm = pr.get('focal_mm')
    return intrinsics_from_tarot_t10x(
        image_shape,
        zoom_ratio=zr,
        focal_mm=float(f_mm) if f_mm is not None else None,
    )
