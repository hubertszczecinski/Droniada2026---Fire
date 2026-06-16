from typing import Optional, Tuple

import cv2
import numpy as np
import pipeline_competition as pc


def _warp_flip_matrix(w: int, h: int, transform: str) -> np.ndarray:
    """Macierz 3×3 flipu w układzie warpu (nie obrotu 90° — panel zostaje 2:1)."""
    if transform == 'fx':
        return np.array([[-1.0, 0.0, float(w - 1)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    if transform == 'fy':
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, float(h - 1)], [0.0, 0.0, 1.0]], dtype=np.float64)
    if transform == 'fxy':
        return np.array(
            [[-1.0, 0.0, float(w - 1)], [0.0, -1.0, float(h - 1)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    raise ValueError(f'unsupported_warp_flip={transform}')


def orient_warped_panel_by_white_anchor(
    warped_bgr: np.ndarray,
    h_mat: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """
    Po homografii wymuś biały kwadrat (1,1) w lewym dolnym rogu warpu.

    Używa flipów (nie obrotu 90°), żeby zachować proporcje 1000×500.
    """
    t = pc.detect_white_corner_transform(warped_bgr)
    if t == 'id':
        return warped_bgr, h_mat, t
    h_px, w_px = warped_bgr.shape[:2]
    if t == 'fx':
        out = cv2.flip(warped_bgr, 1)
    elif t == 'fy':
        out = cv2.flip(warped_bgr, 0)
    else:
        out = cv2.flip(warped_bgr, -1)
    new_h = h_mat
    if h_mat is not None:
        f = _warp_flip_matrix(w_px, h_px, t)
        new_h = (f @ np.asarray(h_mat, dtype=np.float64)).astype(np.float32)
    return out, new_h, t


def warp_panel_rect(
    image_bgr: np.ndarray,
    corners_px: np.ndarray,
    out_wh: Tuple[int, int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if out_wh is None:
        out_wh = (pc.RECT_W, pc.RECT_H)
    w, h = out_wh
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    h_mat = cv2.getPerspectiveTransform(corners_px.astype(np.float32), dst)
    warped = cv2.warpPerspective(image_bgr, h_mat, (w, h))
    warped, h_mat, _t = orient_warped_panel_by_white_anchor(warped, h_mat)
    return warped, h_mat
