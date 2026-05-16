import math
from typing import Optional, Tuple
import cv2
import numpy as np
PANEL_WIDTH_M = 2.0
PANEL_HEIGHT_M = 1.0
PANEL_GRID_COLS = 10
PANEL_GRID_ROWS = 10
PANEL_INTERNAL_LINES_PER_AXIS = PANEL_GRID_COLS - 1
PANEL_CELL_WIDTH_M = PANEL_WIDTH_M / float(PANEL_GRID_COLS)
PANEL_CELL_HEIGHT_M = PANEL_HEIGHT_M / float(PANEL_GRID_ROWS)
PANEL_OBJECT_PTS = np.array([[-1.0, -0.5, 0.0], [1.0, -0.5, 0.0], [1.0, 0.5, 0.0], [-1.0, 0.5, 0.0]], dtype=np.float32)

def rotation_matrix_to_euler_deg(rmat: np.ndarray) -> Tuple[float, float, float]:
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-06
    if not singular:
        roll = math.atan2(rmat[2, 1], rmat[2, 2])
        pitch = math.atan2(-rmat[2, 0], sy)
        yaw = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        roll = math.atan2(-rmat[1, 2], rmat[1, 1])
        pitch = math.atan2(-rmat[2, 0], sy)
        yaw = 0.0
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

def _reproj_mean_px(obj: np.ndarray, img_obs: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray, dist: np.ndarray) -> float:
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k.astype(np.float32), dist.astype(np.float32))
    proj = proj.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - img_obs.astype(np.float64), axis=1)))

def solve_panel_pose(corners_px: np.ndarray, k: np.ndarray, dist: Optional[np.ndarray]=None, refine_lm: bool=True) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float]:
    if corners_px is None or corners_px.shape != (4, 2):
        return (False, None, None, 0.0)
    img_pts = corners_px.astype(np.float32).reshape(4, 1, 2)
    if dist is None:
        dist = np.zeros((4, 1), dtype=np.float32)
    kf = k.astype(np.float32)
    df = dist.astype(np.float32)
    try:
        ok_g, rvecs, tvecs, _ = cv2.solvePnPGeneric(PANEL_OBJECT_PTS, img_pts, kf, df, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        ok_g = False
        rvecs, tvecs = (None, None)
    candidates = []
    if ok_g and rvecs is not None and (tvecs is not None):
        for rv, tv in zip(rvecs, tvecs):
            rvf = np.asarray(rv, dtype=np.float64).reshape(3, 1)
            tvf = np.asarray(tv, dtype=np.float64).reshape(3, 1)
            e = _reproj_mean_px(PANEL_OBJECT_PTS, corners_px.reshape(4, 2), rvf, tvf, kf, df)
            rmat, _ = cv2.Rodrigues(rvf)
            n_cam = rmat @ np.array([[0.0], [0.0], [1.0]], dtype=np.float64)
            front_score = float(np.dot((-tvf).reshape(3), n_cam.reshape(3)))
            z_score = float(tvf[2, 0])
            candidates.append((rvf, tvf, e, front_score, z_score))
    else:
        ok, rv, tv = cv2.solvePnP(PANEL_OBJECT_PTS, img_pts, kf, df, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return (False, None, None, 0.0)
        rvf = np.asarray(rv, dtype=np.float64).reshape(3, 1)
        tvf = np.asarray(tv, dtype=np.float64).reshape(3, 1)
        e = _reproj_mean_px(PANEL_OBJECT_PTS, corners_px.reshape(4, 2), rvf, tvf, kf, df)
        rmat, _ = cv2.Rodrigues(rvf)
        n_cam = rmat @ np.array([[0.0], [0.0], [1.0]], dtype=np.float64)
        front_score = float(np.dot((-tvf).reshape(3), n_cam.reshape(3)))
        z_score = float(tvf[2, 0])
        candidates.append((rvf, tvf, e, front_score, z_score))
    candidates.sort(key=lambda c: (c[2], -c[3], -c[4]))
    rvec = candidates[0][0]
    tvec = candidates[0][1]
    if refine_lm:
        try:
            ok2, rvec2, tvec2 = cv2.solvePnP(PANEL_OBJECT_PTS, img_pts, kf, df, rvec.astype(np.float32), tvec.astype(np.float32), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok2:
                rv2 = np.asarray(rvec2, dtype=np.float64).reshape(3, 1)
                tv2 = np.asarray(tvec2, dtype=np.float64).reshape(3, 1)
                e0 = _reproj_mean_px(PANEL_OBJECT_PTS, corners_px.reshape(4, 2), rvec, tvec, kf, df)
                e1 = _reproj_mean_px(PANEL_OBJECT_PTS, corners_px.reshape(4, 2), rv2, tv2, kf, df)
                if e1 <= e0 * 1.02:
                    rvec, tvec = (rv2, tv2)
        except cv2.error:
            pass
    err = _reproj_mean_px(PANEL_OBJECT_PTS, corners_px.reshape(4, 2), rvec, tvec, kf, df)
    return (True, rvec.astype(np.float32), tvec.astype(np.float32), err)
