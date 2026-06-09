"""Klasyfikacja ustawienia panelu na stojaku (poziomy / 45° / pionowy) — moduł A."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pipeline_competition as pc

STAND_CATEGORIES: Tuple[str, ...] = ('horizontal', '45_deg', 'vertical')
CATEGORY_TO_ANGLE: Dict[str, int] = {'horizontal': 0, '45_deg': 45, 'vertical': 90}
STAND_LABEL_PL: Dict[str, str] = {
    'horizontal': 'poziomy',
    '45_deg': 'poziomy-przechylony',
    'vertical': 'pionowy',
}

_DEFAULT_CALIB = Path(__file__).resolve().parent / 'data' / 'panel_stand_linear.json'


def _quad_wh_ratio(corners_tltrbrbl: np.ndarray) -> float:
    """Krótszy bok / dłuższy bok quadu w kadrze."""
    q = pc.order_points(np.asarray(corners_tltrbrbl, dtype=np.float64).reshape(4, 2))
    sides = [
        float(np.linalg.norm(q[0] - q[1])),
        float(np.linalg.norm(q[1] - q[2])),
        float(np.linalg.norm(q[2] - q[3])),
        float(np.linalg.norm(q[3] - q[0])),
    ]
    long_s = max(sides)
    short_s = min(sides)
    return float(short_s / max(long_s, 1e-6))


def stand_feature_vector(
    rmat: np.ndarray,
    reproj_px: float,
    corners_tltrbrbl: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Cechy do klasyfikatora liniowego (zgodne z treningiem w calibrate_panel_stand)."""
    r = np.asarray(rmat, dtype=np.float64).reshape(3, 3)
    h, w = int(image_shape[0]), int(image_shape[1])
    q = pc.order_points(np.asarray(corners_tltrbrbl, dtype=np.float64).reshape(4, 2))
    img_area = max(1.0, float(h * w))
    area_frac = float(cv2.contourArea(q.reshape(-1, 1, 2).astype(np.float32))) / img_area
    wh = _quad_wh_ratio(q)
    aspect = 1.0 / max(wh, 1e-6)
    return np.asarray(
        [
            *r.reshape(-1)[:9],
            float(reproj_px) / 25.0,
            wh,
            aspect,
            area_frac,
            1.0,
        ],
        dtype=np.float64,
    )


def load_stand_calibration(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    p = Path(path) if path else _DEFAULT_CALIB
    if not p.is_file():
        return None
    with p.open(encoding='utf-8') as fh:
        return json.load(fh)


def save_stand_calibration(cal: Dict[str, Any], path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as fh:
        json.dump(cal, fh, ensure_ascii=False, indent=2)
    return str(p)


def train_stand_classifier(
    X: np.ndarray,
    y_categories: List[str],
) -> Dict[str, Any]:
    """Wieloklasowy regresja liniowa: scores = X @ W.T."""
    Xf = np.asarray(X, dtype=np.float64)
    n, _f = Xf.shape
    k = len(STAND_CATEGORIES)
    Y = np.zeros((n, k), dtype=np.float64)
    for i, cat in enumerate(y_categories):
        if cat not in STAND_CATEGORIES:
            raise ValueError(f'unknown category {cat!r}')
        Y[i, STAND_CATEGORIES.index(cat)] = 1.0
    coef, _, _, _ = np.linalg.lstsq(Xf, Y, rcond=1e-4)
    pred = np.argmax(Xf @ coef, axis=1)
    y_idx = [STAND_CATEGORIES.index(c) for c in y_categories]
    train_acc = float(np.mean(pred == np.asarray(y_idx, dtype=np.int64)))
    return {
        'version': 1,
        'W': coef.T.tolist(),
        'n_features': int(Xf.shape[1]),
        'n_samples': int(n),
        'categories': list(STAND_CATEGORIES),
        'train_acc_linear': round(train_acc, 4),
    }


def _linear_scores(
    feats: np.ndarray,
    cal: Dict[str, Any],
) -> Tuple[np.ndarray, int]:
    W = np.asarray(cal['W'], dtype=np.float64)
    scores = feats.reshape(1, -1) @ W.T
    return scores.reshape(-1), int(np.argmax(scores))


def estimate_panel_stand(
    rmat: np.ndarray,
    reproj_px: float,
    corners_tltrbrbl: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    calibration_path: Optional[str] = None,
) -> Tuple[int, str, float, Dict[str, Any]]:
    """
    Zwraca: report_angle_deg, panel_angle_category, stand_confidence, meta.
    """
    meta: Dict[str, Any] = {}
    wh = _quad_wh_ratio(corners_tltrbrbl)
    meta['quad_wh_ratio'] = round(wh, 4)

    cal = load_stand_calibration(calibration_path)
    feats = stand_feature_vector(rmat, reproj_px, corners_tltrbrbl, image_shape)

    if cal is not None and 'W' in cal:
        scores, idx = _linear_scores(feats, cal)
        cat = STAND_CATEGORIES[idx]
        meta['stand_source'] = 'linear_calibration'
        meta['calibration'] = os.path.basename(str(calibration_path or _DEFAULT_CALIB))
        meta['stand_scores'] = {c: round(float(scores[i]), 4) for i, c in enumerate(STAND_CATEGORIES)}
        exp_s = np.exp(scores - np.max(scores))
        probs = exp_s / max(float(np.sum(exp_s)), 1e-9)
        conf = float(probs[idx])
        ang = CATEGORY_TO_ANGLE.get(cat, 0)
        return ang, cat, conf, meta

    from module_panel.angle_from_pose import report_angle_from_rmat_theta

    ang = report_angle_from_rmat_theta(rmat, 32.0, 58.0)
    cat = {0: 'horizontal', 45: '45_deg', 90: 'vertical'}.get(ang, 'horizontal')
    meta['stand_source'] = 'rmat_theta_fallback'
    return int(ang), str(cat), 0.45, meta


def integration_dict_from_pose_fields(
    *,
    ok: bool,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    distance_m: float,
    report_angle_deg: int,
    panel_angle_category: str,
    stand_confidence: float,
    reproj_mean_px: float,
    method: str,
    panel_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        'ok': bool(ok),
        'panel_id': panel_id,
        'distance_camera_to_panel_center_m': float(distance_m),
        'roll_deg': float(roll_deg),
        'pitch_deg': float(pitch_deg),
        'yaw_deg': float(yaw_deg),
        'report_angle_deg': int(report_angle_deg),
        'panel_angle_category': str(panel_angle_category),
        'panel_stand_label_pl': STAND_LABEL_PL.get(str(panel_angle_category), str(panel_angle_category)),
        'stand_confidence': float(stand_confidence),
        'reproj_mean_px': float(reproj_mean_px),
        'method': str(method),
    }
