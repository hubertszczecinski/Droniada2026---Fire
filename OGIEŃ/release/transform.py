from __future__ import annotations

import cv2
import numpy as np

_ROTATE_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def apply_rotate(bgr: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return bgr
    code = _ROTATE_MAP.get(degrees)
    if code is None:
        raise ValueError(f'unsupported_rotate={degrees}')
    return cv2.rotate(bgr, code)
