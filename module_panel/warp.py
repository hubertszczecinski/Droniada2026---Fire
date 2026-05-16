from typing import Tuple
import cv2
import numpy as np
import pipeline_competition as pc

def warp_panel_rect(image_bgr: np.ndarray, corners_px: np.ndarray, out_wh: Tuple[int, int]=None) -> Tuple[np.ndarray, np.ndarray]:
    if out_wh is None:
        out_wh = (pc.RECT_W, pc.RECT_H)
    w, h = out_wh
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    h_mat = cv2.getPerspectiveTransform(corners_px.astype(np.float32), dst)
    warped = cv2.warpPerspective(image_bgr, h_mat, (w, h))
    return (warped, h_mat)
