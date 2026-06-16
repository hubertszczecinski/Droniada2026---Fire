"""Moduł A — most do YOLO-Pose modułu B (oddzielna ścieżka PnP, ten sam detektor rogów)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from release.yolo_pose_live import detect_corners_yolo_pose, kpts_to_quad

__all__ = [
    'acquire_corners_for_pose',
    'detect_corners_yolo_pose',
    'kpts_to_quad',
]


def acquire_corners_for_pose(
    image_bgr: np.ndarray,
    k: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    *,
    use_tracker: bool = True,
) -> Tuple[Optional[np.ndarray], str, Dict[str, Any]]:
    """
    Te same rogi co moduł B przy ``corner_mode=yolo_pose``:
    YOLO-Pose → bias → opcjonalnie EMA/hold z ``live_corners``.
    """
    if use_tracker:
        from release.live_corners import detect_corners_live

        return detect_corners_live(
            image_bgr,
            k,
            dist,
            corner_mode='yolo_pose',
            use_tracker=True,
        )
    return detect_corners_yolo_pose(image_bgr)
