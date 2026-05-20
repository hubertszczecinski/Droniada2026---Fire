"""Detect colored cards on a warped panel (live test without YOLO weights)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pipeline_competition as pc

# HSV (OpenCV): H 0-180, S,V 0-255 — loose ranges for printed / foam cards under room light.
_COLOR_RANGES: List[Tuple[int, Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = [
    (0, ((0, 80, 80), (12, 255, 255))),      # CZERWONA
    (0, ((168, 80, 80), (180, 255, 255))),    # CZERWONA (wrap)
    (1, ((35, 60, 60), (88, 255, 255))),     # ZIELONA
    (2, ((95, 60, 60), (128, 255, 255))),     # NIEBIESKA
    (3, ((18, 80, 80), (38, 255, 255))),      # ZOLTA
    (4, ((128, 40, 60), (162, 255, 255))),    # FIOLETOWA
    (5, ((8, 80, 80), (22, 255, 255))),       # POMARANCZOWA
]

_MIN_CELL_FILL = 0.12
_MIN_SAT_MEAN = 55.0


def _classify_patch_bgr(patch_bgr: np.ndarray) -> Optional[int]:
    if patch_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    if float(np.mean(sat)) < _MIN_SAT_MEAN:
        return None
    if float(np.mean(val)) < 45 or float(np.mean(val)) > 250:
        return None
    best_cls: Optional[int] = None
    best_frac = 0.0
    for cls_id, lo, hi in _COLOR_RANGES:
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        frac = float(np.count_nonzero(mask)) / float(mask.size)
        if frac > best_frac:
            best_frac = frac
            best_cls = cls_id
    if best_cls is None or best_frac < _MIN_CELL_FILL:
        return None
    return best_cls


def detect_cards_on_warped(
    warped_bgr: np.ndarray,
    *,
    min_cells: int = 1,
    max_cells: int = 8,
) -> List[Dict[str, Any]]:
    """
    Scan 10x10 grid on rectified panel; return detections with warped pixel centers.
    """
    h, w = warped_bgr.shape[:2]
    cw = w / 10.0
    ch = h / 10.0
    pad_x = max(2, int(cw * 0.22))
    pad_y = max(2, int(ch * 0.22))
    found: List[Dict[str, Any]] = []
    for row in range(10):
        for col in range(10):
            x0 = int(col * cw) + pad_x
            y0 = int(row * ch) + pad_y
            x1 = int((col + 1) * cw) - pad_x
            y1 = int((row + 1) * ch) - pad_y
            if x1 <= x0 or y1 <= y0:
                continue
            patch = warped_bgr[y0:y1, x0:x1]
            cls_id = _classify_patch_bgr(patch)
            if cls_id is None:
                continue
            cx = (col + 0.5) * cw
            cy = (row + 0.5) * ch
            # Wiersz 1 = dol siatki (jak w raporcie / Blender), wiersz 10 = gora.
            found.append({
                'cls_id': cls_id,
                'warp_cx': float(cx),
                'warp_cy': float(cy),
                'grid_col': col + 1,
                'grid_row': 10 - row,
                'color': pc.CLASS_TO_COLOR.get(cls_id, 'UNKNOWN'),
            })
    found.sort(key=lambda d: -d['warp_cx'])
    return found[:max_cells] if len(found) >= min_cells else found


def warped_detections_to_yolo(
    warped_dets: List[Dict[str, Any]],
    homography_img_to_warp: np.ndarray,
    img_w: int,
    img_h: int,
) -> List[Tuple[int, float, float, float, float]]:
    """Convert warped cell centers to normalized YOLO boxes in image space."""
    if not warped_dets:
        return []
    try:
        h_inv = np.linalg.inv(homography_img_to_warp.astype(np.float64))
    except np.linalg.LinAlgError:
        return []
    out: List[Tuple[int, float, float, float, float]] = []
    cell_w_n = 0.08
    cell_h_n = 0.08
    for d in warped_dets:
        src = np.array([[[d['warp_cx'], d['warp_cy']]]], dtype=np.float32)
        pts = cv2.perspectiveTransform(src, h_inv.astype(np.float32))
        u, v = float(pts[0, 0, 0]), float(pts[0, 0, 1])
        out.append((
            int(d['cls_id']),
            float(np.clip(u / img_w, 0.0, 1.0)),
            float(np.clip(v / img_h, 0.0, 1.0)),
            cell_w_n,
            cell_h_n,
        ))
    return out


def detect_cards_live(
    image_bgr: np.ndarray,
    corners_tltrbrbl: np.ndarray,
    homography_img_to_warp: np.ndarray,
    warped_bgr: np.ndarray,
) -> Tuple[List[Tuple[int, float, float, float, float]], List[Dict[str, Any]]]:
    h, w = image_bgr.shape[:2]
    warped_dets = detect_cards_on_warped(warped_bgr)
    yolo = warped_detections_to_yolo(warped_dets, homography_img_to_warp, w, h)
    return yolo, warped_dets
