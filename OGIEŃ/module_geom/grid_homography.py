"""Over-constrained homography from internal 10x10 grid line intersections."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pipeline_competition as pc


def _rect_dst_grid(n: int = 10) -> np.ndarray:
    rw, rh = pc.RECT_W, pc.RECT_H
    step_x = rw / float(n)
    step_y = rh / float(n)
    pts = []
    for j in range(n + 1):
        for i in range(n + 1):
            pts.append([i * step_x, j * step_y])
    return np.array(pts, dtype=np.float32)


def _rough_img_to_rect(corners_tltrbrbl: np.ndarray) -> np.ndarray:
    dst = np.array(
        [[0, 0], [pc.RECT_W - 1, 0], [pc.RECT_W - 1, pc.RECT_H - 1], [0, pc.RECT_H - 1]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(corners_tltrbrbl.astype(np.float32), dst)


def build_grid_homography(
    image_bgr: np.ndarray,
    corners_tltrbrbl: np.ndarray,
    *,
    min_inliers: int = 12,
) -> Tuple[Optional[np.ndarray], bool, Dict[str, Any]]:
    """
    Image -> ideal rect homography using RANSAC on grid line crossings.
    H_total maps image pixels to [0,RECT_W] x [0,RECT_H].
    """
    meta: Dict[str, Any] = {'method': 'grid_ransac_homography'}
    h_i2w = _rough_img_to_rect(corners_tltrbrbl)
    warped = cv2.warpPerspective(image_bgr, h_i2w, (pc.RECT_W, pc.RECT_H))
    from module_panel.grid import detect_grid_lines_warped
    xs, ys, ok_lines, line_meta = detect_grid_lines_warped(warped)
    meta.update(line_meta)
    if not ok_lines:
        meta['fail'] = 'grid_lines'
        return None, False, meta
    src_warp: List[List[float]] = []
    dst_ideal: List[List[float]] = []
    step_x = pc.RECT_W / 10.0
    step_y = pc.RECT_H / 10.0
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            src_warp.append([float(x), float(y)])
            dst_ideal.append([float(i * step_x), float(j * step_y)])
    src_w = np.array(src_warp, dtype=np.float32).reshape(-1, 1, 2)
    dst_i = np.array(dst_ideal, dtype=np.float32).reshape(-1, 1, 2)
    if src_w.shape[0] < min_inliers:
        meta['fail'] = 'too_few_points'
        return None, False, meta
    h_w2r, mask = cv2.findHomography(src_w, dst_i, cv2.RANSAC, 4.0)
    if h_w2r is None or not np.isfinite(h_w2r).all():
        meta['fail'] = 'findHomography'
        return None, False, meta
    inliers = int(mask.ravel().sum()) if mask is not None else 0
    meta['homography_inliers'] = inliers
    meta['homography_total_pts'] = int(src_w.shape[0])
    if inliers < min_inliers:
        meta['fail'] = 'ransac_inliers'
        return None, False, meta
    h_i2r = h_w2r @ h_i2w
    meta['grid_lines_x'] = [float(v) for v in xs]
    meta['grid_lines_y'] = [float(v) for v in ys]
    return h_i2r.astype(np.float64), True, meta


def rect_px_to_model_xy(px: float, py: float) -> Tuple[float, float]:
    """Map ideal rect pixel to panel model coords used by model_xy_to_cell."""
    lx = -1.0 + 2.0 * float(px) / max(1.0, float(pc.RECT_W - 1))
    ly = -0.5 + float(py) / max(1.0, float(pc.RECT_H - 1))
    return lx, ly
