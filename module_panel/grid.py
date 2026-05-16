from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import pipeline_competition as pc

def model_xy_to_cell(X: float, Y: float) -> Tuple[int, int]:
    col = int(round(X / 0.2 + 5.5))
    row = int(round(Y / 0.1 + 5.5))
    col = int(pc.clamp(col, 1, 10))
    row = int(pc.clamp(row, 1, 10))
    return (col, row)

def map_yolo_to_cells_corners_homography(yolo_det: List[Tuple[int, float, float, float, float]], corners_img_tltrbrbl: np.ndarray, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    obj2d = np.array([[-1.0, -0.5], [1.0, -0.5], [1.0, 0.5], [-1.0, 0.5]], dtype=np.float32)
    h_obj_to_img = cv2.getPerspectiveTransform(obj2d, corners_img_tltrbrbl.astype(np.float32))
    try:
        hi = np.linalg.inv(h_obj_to_img)
    except np.linalg.LinAlgError:
        hi = np.eye(3, dtype=np.float64)
    out: List[Dict[str, Any]] = []
    for cls_id, cx_n, cy_n, _w, _h in yolo_det:
        u = float(cx_n * img_w)
        v = float(cy_n * img_h)
        p = hi @ np.array([u, v, 1.0], dtype=np.float64)
        if abs(p[2]) < 1e-09:
            col, row = (5, 5)
        else:
            lx, ly = (float(p[0] / p[2]), float(p[1] / p[2]))
            col, row = model_xy_to_cell(lx, ly)
        out.append({'x': col, 'y': row, 'color': pc.CLASS_TO_COLOR.get(cls_id, 'UNKNOWN')})
    return out
