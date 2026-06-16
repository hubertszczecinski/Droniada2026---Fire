"""Map YOLO detections to grid cells via inverse homography (contact point)."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pipeline_competition as pc
from module_geom.grid_homography import rect_px_to_model_xy
from module_panel.grid import model_xy_to_cell


def yolo_contact_point_px(
    det: Tuple[int, float, float, float, float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float]:
    """Bottom-edge center on the panel plane (x=center, y=max of bbox)."""
    _cls, cx_n, cy_n, _bw, bh_n = det
    x = float(cx_n * img_w)
    y = float((cy_n + 0.5 * bh_n) * img_h)
    return x, y


def yolo_center_point_px(
    det: Tuple[int, float, float, float, float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float]:
    _cls, cx_n, cy_n, _bw, _bh = det
    return float(cx_n * img_w), float(cy_n * img_h)


def map_yolo_to_cells_geom(
    yolo_det: List[Tuple[int, float, float, float, float]],
    homography_img_to_rect: np.ndarray,
    img_w: int,
    img_h: int,
    *,
    use_contact_point: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    point_fn = yolo_contact_point_px if use_contact_point else yolo_center_point_px
    meta: Dict[str, Any] = {
        'xy_source': 'geom_contact_point' if use_contact_point else 'geom_bbox_center',
        'contact_point': bool(use_contact_point),
    }
    for det in yolo_det:
        cls_id = det[0]
        u, v = point_fn(det, img_w, img_h)
        src = np.array([[[u, v]]], dtype=np.float32)
        try:
            pts = cv2.perspectiveTransform(src, homography_img_to_rect.astype(np.float32))
        except cv2.error:
            col, row = (5, 5)
        else:
            px, py = float(pts[0, 0, 0]), float(pts[0, 0, 1])
            lx, ly = rect_px_to_model_xy(px, py)
            col, row = model_xy_to_cell(lx, ly)
        out.append({'x': col, 'y': row, 'color': pc.CLASS_TO_COLOR.get(cls_id, 'UNKNOWN')})
    return out, meta
