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


def _white_grid_mask(warped_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = (((v > 115) & (s < 105)) | (gray > 175)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def _profile_line_model(profile: np.ndarray, expected: int, min_hits: int, min_strength: float) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    n = int(profile.shape[0])
    if n <= 0:
        return (np.linspace(0.0, 1.0, expected + 1), False, {'hits': 0})
    prof = profile.astype(np.float32)
    mx = float(np.max(prof))
    if mx <= 1e-6:
        return (np.linspace(0.0, float(n - 1), expected + 1), False, {'hits': 0, 'max_strength': 0.0})
    prof /= mx
    step = n / float(expected)
    idxs: List[float] = []
    poss: List[float] = []
    strengths: List[float] = []
    for i in range(1, expected):
        center = i * step
        radius = max(4, int(round(step * 0.36)))
        lo = max(0, int(round(center - radius)))
        hi = min(n, int(round(center + radius + 1)))
        if hi <= lo:
            continue
        local = prof[lo:hi]
        j = int(np.argmax(local)) + lo
        strength = float(prof[j])
        if strength >= min_strength:
            idxs.append(float(i))
            poss.append(float(j))
            strengths.append(strength)
    if len(poss) < min_hits:
        return (np.linspace(0.0, float(n - 1), expected + 1), False, {'hits': len(poss), 'max_strength': mx})
    coeff = np.polyfit(np.asarray(idxs, dtype=np.float32), np.asarray(poss, dtype=np.float32), 1)
    slope = float(coeff[0])
    intercept = float(coeff[1])
    ok_slope = 0.55 * step <= slope <= 1.45 * step
    lines = slope * np.arange(0, expected + 1, dtype=np.float32) + intercept
    ok_bounds = lines[0] > -0.75 * step and lines[-1] < (n - 1) + 0.75 * step
    ok = bool(ok_slope and ok_bounds)
    lines = np.clip(lines, 0.0, float(n - 1))
    lines = np.maximum.accumulate(lines)
    return (lines.astype(np.float32), ok, {'hits': len(poss), 'mean_strength': float(np.mean(strengths)), 'slope': slope})


def detect_grid_lines_warped(warped_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
    h, w = warped_bgr.shape[:2]
    mask = _white_grid_mask(warped_bgr)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(15, h // 12)))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 16), 3))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, v_kernel)
    hori = cv2.morphologyEx(mask, cv2.MORPH_OPEN, h_kernel)
    x_profile = np.mean(vert > 0, axis=0)
    y_profile = np.mean(hori > 0, axis=1)
    xs, ok_x, mx = _profile_line_model(x_profile, 10, min_hits=5, min_strength=0.18)
    ys, ok_y, my = _profile_line_model(y_profile, 10, min_hits=5, min_strength=0.18)
    ok = bool(ok_x and ok_y)
    meta = {'grid_line_ok': ok, 'grid_x': mx, 'grid_y': my}
    return (xs, ys, ok, meta)


def _cell_from_grid_lines(px: float, py: float, xs: np.ndarray, ys: np.ndarray) -> Tuple[int, int]:
    col = int(np.searchsorted(xs, px, side='right'))
    row = int(np.searchsorted(ys, py, side='right'))
    col = int(pc.clamp(col, 1, 10))
    row = int(pc.clamp(row, 1, 10))
    return (col, row)


def map_yolo_to_cells_warp_grid(
    yolo_det: List[Tuple[int, float, float, float, float]],
    warped_bgr: np.ndarray,
    homography_img_to_warp: np.ndarray,
    img_w: int,
    img_h: int,
) -> Tuple[List[Dict[str, Any]], bool, Dict[str, Any]]:
    xs, ys, ok, meta = detect_grid_lines_warped(warped_bgr)
    if not ok:
        return ([], False, meta)
    src = np.array([[[cx * img_w, cy * img_h]] for _, cx, cy, _, _ in yolo_det], dtype=np.float32)
    if len(src):
        pts = cv2.perspectiveTransform(src, homography_img_to_warp).reshape(-1, 2)
    else:
        pts = np.zeros((0, 2), dtype=np.float32)
    out: List[Dict[str, Any]] = []
    for (cls_id, *_rest), (px, py) in zip(yolo_det, pts):
        col, row = _cell_from_grid_lines(float(px), float(py), xs, ys)
        out.append({'x': col, 'y': row, 'color': pc.CLASS_TO_COLOR.get(cls_id, 'UNKNOWN')})
    meta.update({'grid_lines_x': [float(x) for x in xs], 'grid_lines_y': [float(y) for y in ys]})
    return (out, True, meta)
