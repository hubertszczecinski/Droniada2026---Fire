"""YOLOv8-Pose — 4 zewnętrzne rogi panelu (live / video)."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pipeline_competition as pc

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DEFAULT_WEIGHTS = os.path.join(
    _ROOT, 'runs', 'pose', 'droniada_panel_corners', 'weights', 'best.pt',
)

_MODEL = None
_MODEL_PATH: Optional[str] = None


def _load_model(weights: Optional[str] = None):
    global _MODEL, _MODEL_PATH
    path = weights or os.environ.get('DRONIADA_YOLO_POSE_WEIGHTS', _DEFAULT_WEIGHTS)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'Brak wag YOLO-Pose: {path}. Uruchom: '
            '.venv_yolo/bin/python -m release.eval_yolo_pose_corners --train',
        )
    if _MODEL is None or _MODEL_PATH != path:
        from ultralytics import YOLO

        from release.device import device_label, resolve_yolo_device

        device = resolve_yolo_device()
        _MODEL = YOLO(path)
        _MODEL.to(device)
        _MODEL_PATH = path
        _MODEL._droniada_device = device  # type: ignore[attr-defined]
        print(f'[yolo_pose] weights={path} device={device_label(device)}', flush=True)
    return _MODEL


def kpts_to_quad(kpts: np.ndarray, conf_thr: float = 0.2) -> Optional[np.ndarray]:
    if kpts is None or kpts.shape[0] < 4:
        return None
    pts = []
    for i in range(4):
        x, y, c = float(kpts[i, 0]), float(kpts[i, 1]), float(kpts[i, 2])
        if c < conf_thr or not np.isfinite(x) or not np.isfinite(y):
            return None
        pts.append([x, y])
    return pc.order_points(np.asarray(pts, dtype=np.float32))


def detect_corners_yolo_pose(
    image_bgr: np.ndarray,
    *,
    weights: Optional[str] = None,
    conf: float = 0.2,
    imgsz: int = 640,
) -> Tuple[Optional[np.ndarray], str, Dict[str, Any]]:
    """Jeden quad z najlepszej detekcji YOLO-Pose."""
    from module_pose.api import default_intrinsics
    from release.live_corners import _evaluate_panel_roi_candidate

    from release.device import resolve_yolo_device

    model = _load_model(weights)
    device = getattr(model, '_droniada_device', resolve_yolo_device())
    res = model.predict(image_bgr, verbose=False, conf=conf, imgsz=imgsz, device=device)[0]
    meta: Dict[str, Any] = {'method': 'yolo_pose', 'weights': _MODEL_PATH or weights}

    if res.keypoints is None or len(res.keypoints.data) == 0:
        meta['fail'] = 'no_detection'
        return None, 'yolo_none', meta

    best_i = 0
    if res.boxes is not None and len(res.boxes) > 1:
        best_i = int(np.argmax(res.boxes.conf.cpu().numpy()))
        meta['det_conf'] = float(res.boxes.conf.cpu().numpy()[best_i])

    kpts = res.keypoints.data[best_i].cpu().numpy()
    raw = kpts_to_quad(kpts, conf_thr=0.15)
    if raw is None:
        meta['fail'] = 'low_kpt_conf'
        return None, 'yolo_low_kpt', meta

    h, w = image_bgr.shape[:2]
    k, dist = default_intrinsics((h, w))
    corners, detail = _evaluate_panel_roi_candidate(
        image_bgr, raw, k, dist, label='yolo_pose', preserve_geometry=True,
    )
    if corners is None:
        corners = raw
        detail = {'reproj_mean_px': 999.0, 'grid_structure_score': 0.0}
    area_frac = float(detail.get('area_ratio', 0.0))
    if area_frac > 0 and area_frac < 0.11:
        from release.panel_roi import expand_quad_from_center

        corners = expand_quad_from_center(corners, (h, w), scale=1.10)
        detail['yolo_outer_expand'] = 1.0
    meta.update(detail)
    meta['kpt_conf_min'] = float(np.min(kpts[:4, 2]))
    if os.environ.get('DRONIADA_YOLO_BIAS', '1').strip().lower() not in ('0', 'false', 'no'):
        from release.yolo_corner_bias import apply_bias_correction, load_bias_correction

        delta = load_bias_correction()
        if float(np.max(np.abs(delta))) > 0.5:
            corners = apply_bias_correction(corners, delta)
            meta['yolo_bias_corrected'] = 1.0
    return corners.astype(np.float32), 'yolo_pose', meta
