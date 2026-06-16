from __future__ import annotations
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
ANGLE_CLASSES: Tuple[int, ...] = (0, 45, 90)
CATEGORY_BY_ANGLE: Dict[int, str] = {0: 'horizontal', 45: '45_deg', 90: 'vertical'}
_CALIB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'angle_linear_rmat.json')

def _default_theta_thresholds() -> Tuple[float, float]:
    return (32.0, 58.0)

def report_angle_from_rmat_theta(rmat: np.ndarray, t_low: float, t_high: float) -> int:
    r = np.asarray(rmat, dtype=np.float64).reshape(3, 3)
    c = float(np.clip(abs(r[2, 2]), 0.0, 1.0))
    theta = math.degrees(math.acos(c))
    if theta < t_low:
        return 0
    if theta < t_high:
        return 45
    return 90

def report_angle_from_rmat_linear(rmat: np.ndarray, reproj_px: float, cal: Dict[str, Any]) -> int:
    W = np.asarray(cal['W'], dtype=np.float64)
    f = np.concatenate([rmat.reshape(-1)[:9], [reproj_px / 25.0], [1.0]])
    scores = W @ f
    idx = int(np.argmax(scores))
    return int(ANGLE_CLASSES[idx])

def load_angle_calibration(path: Optional[str]=None) -> Optional[Dict[str, Any]]:
    p = path or _CALIB_PATH
    if not os.path.isfile(p):
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def estimate_report_angle_and_category(rmat: np.ndarray, reproj_px: float=0.0, calibration_path: Optional[str]=None, mode: str='rmat_linear') -> Tuple[int, str, Dict[str, Any]]:
    meta: Dict[str, Any] = {'angle_mode': mode}
    cal = load_angle_calibration(calibration_path)
    if mode == 'rmat_linear' and cal is not None and ('W' in cal):
        ang = report_angle_from_rmat_linear(rmat, reproj_px, cal)
        meta['calibration'] = os.path.basename(calibration_path or _CALIB_PATH)
    else:
        t0, t1 = _default_theta_thresholds()
        if cal is not None and 'theta_t01' in cal and ('theta_t12' in cal):
            t0 = float(cal['theta_t01'])
            t1 = float(cal['theta_t12'])
        ang = report_angle_from_rmat_theta(rmat, t0, t1)
        meta['theta_thresholds'] = [t0, t1]
    cat = CATEGORY_BY_ANGLE.get(ang, 'horizontal')
    return (ang, cat, meta)
