"""Camera calibration load/save and undistortion."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

_DEFAULT_CALIB = Path(__file__).resolve().parents[1] / 'config' / 'camera_calibration.npz'


def default_intrinsics(image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_shape[:2]
    k = np.array([[1000.0, 0.0, w / 2.0], [0.0, 1000.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    dist = np.zeros((5, 1), dtype=np.float32)
    return k, dist


def resolve_intrinsics(
    image_shape: Tuple[int, int],
    *,
    profile: Optional[str] = None,
    zoom_ratio: float = 1.0,
    calib_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Priority: camera_calibration.npz > tarot_t10x_2a profile > default fx=1000.

    profile: None | 'tarot_t10x_2a' | 'tarot_t10x_2a:wide' | 'tarot_t10x_2a:mid' | 'tarot_t10x_2a:tele'
    """
    meta: Dict[str, Any] = {'source': 'default'}
    k_npz, dist_npz, cmeta = load_calibration(calib_path)
    if k_npz is not None and dist_npz is not None:
        meta = {'source': 'calibration_npz', **cmeta}
        return k_npz, dist_npz, meta

    if profile and profile.startswith('tarot'):
        from module_geom.tarot_t10x import intrinsics_for_preset, intrinsics_from_tarot_t10x

        preset = 'wide'
        if ':' in profile:
            _, preset = profile.split(':', 1)
        if preset in ('wide', 'mid', 'tele'):
            k, dist = intrinsics_for_preset(image_shape, preset)
            meta = {'source': 'tarot_t10x_2a', 'preset': preset}
        else:
            k, dist = intrinsics_from_tarot_t10x(image_shape, zoom_ratio=zoom_ratio)
            meta = {'source': 'tarot_t10x_2a', 'zoom_ratio': zoom_ratio}
        return k, dist, meta

    k, dist = default_intrinsics(image_shape)
    return k, dist, meta


def save_calibration(path: str, k: np.ndarray, dist: np.ndarray, *, meta: Optional[Dict[str, Any]] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {'camera_matrix': k.astype(np.float64), 'dist_coeffs': dist.astype(np.float64).reshape(-1)}
    if meta:
        payload['meta_json'] = np.array([json.dumps(meta, ensure_ascii=False)])
    np.savez_compressed(str(p), **payload)


def load_calibration(path: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    p = Path(path) if path else _DEFAULT_CALIB
    meta: Dict[str, Any] = {'path': str(p), 'loaded': False}
    if not p.is_file():
        return None, None, meta
    data = np.load(str(p), allow_pickle=True)
    k = np.asarray(data['camera_matrix'], dtype=np.float32)
    dist = np.asarray(data['dist_coeffs'], dtype=np.float32).reshape(-1, 1)
    meta['loaded'] = True
    if 'meta_json' in data:
        try:
            meta.update(json.loads(str(data['meta_json'][0])))
        except (json.JSONDecodeError, IndexError, TypeError):
            pass
    return k, dist, meta


def undistort_image(
    image_bgr: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    *,
    alpha: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return undistorted BGR and new camera matrix (for geometry on rectified image)."""
    h, w = image_bgr.shape[:2]
    d = dist.astype(np.float32)
    if d.size == 0 or float(np.max(np.abs(d))) < 1e-7:
        return image_bgr.copy(), k.astype(np.float32)
    kf = k.astype(np.float32)
    new_k, _roi = cv2.getOptimalNewCameraMatrix(kf, d, (w, h), alpha, (w, h))
    out = cv2.undistort(image_bgr, kf, d, None, new_k)
    return out, new_k.astype(np.float32)
