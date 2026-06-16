"""End-to-end geometry pipeline: undistort -> VP corners -> grid homography -> cards."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import pipeline_competition as pc
from module_geom.camera import load_calibration, undistort_image
from module_geom.grid_homography import build_grid_homography
from module_geom.lines_vp import detect_corners_geom_vp as _detect_corners_vp_lsd
from module_geom.map_cards import map_yolo_to_cells_geom
from module_pose.api import canonicalize_corners_by_white_anchor


def prepare_image_geom(
    image_bgr: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    *,
    calib_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Undistort using chessboard calib file if present, else supplied k/dist.
    Returns (image, k_used, dist_used, meta).
    """
    meta: Dict[str, Any] = {'undistort': False}
    ck, cd, cm = load_calibration(calib_path)
    if ck is not None and cd is not None:
        out, new_k = undistort_image(image_bgr, ck, cd)
        meta.update({'undistort': True, 'undistort_source': 'chessboard_file', **cm})
        return out, new_k, np.zeros((5, 1), np.float32), meta
    out, new_k = undistort_image(image_bgr, k, dist)
    if meta.get('undistort') is False and float(np.max(np.abs(dist))) >= 1e-7:
        meta['undistort'] = True
        meta['undistort_source'] = 'pose_or_runtime_intrinsics'
    return out, new_k, np.zeros((5, 1), np.float32), meta


def detect_corners_geom_vp(
    image_bgr: np.ndarray,
) -> Tuple[Optional[np.ndarray], str, Dict[str, Any]]:
    raw, lmeta = _detect_corners_vp_lsd(image_bgr)
    if raw is None:
        return None, 'geom_vp_miss', lmeta
    c, _anc = canonicalize_corners_by_white_anchor(image_bgr, raw.astype(np.float32))
    meta = dict(lmeta)
    meta['canonicalized'] = True
    return c.astype(np.float32), 'geom_vp', meta


def analyze_cards_geom(
    image_bgr: np.ndarray,
    yolo_det: List[Tuple[int, float, float, float, float]],
    corners_px: np.ndarray,
    *,
    src_wh: Tuple[int, int],
) -> Tuple[List[Dict[str, Any]], bool, Dict[str, Any]]:
    w, h = src_wh
    h_mat, ok, gmeta = build_grid_homography(image_bgr, corners_px)
    if not ok or h_mat is None:
        return [], False, gmeta
    preds, mmeta = map_yolo_to_cells_geom(yolo_det, h_mat, w, h, use_contact_point=False)
    meta = {**gmeta, **mmeta, 'homography_img_to_rect': True}
    return preds, True, meta
