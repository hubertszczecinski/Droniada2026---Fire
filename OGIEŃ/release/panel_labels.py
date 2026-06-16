"""Etykiety ręczne: żółty trapez (4 rogi siatki) + opcjonalnie niebieski ROI."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pipeline_competition as pc

LABEL_VERSION = 1
DEFAULT_LABELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'dataset',
    'panel_labels',
)

CORNER_NAMES = ('TL', 'TR', 'BR', 'BL')


def labels_dir(root: Optional[str] = None) -> str:
    d = root or DEFAULT_LABELS_DIR
    os.makedirs(d, exist_ok=True)
    return d


def label_path_for_image(image_path: str, root: Optional[str] = None) -> str:
    stem = Path(image_path).stem
    return os.path.join(labels_dir(root), f'{stem}.json')


def save_label(
    image_path: str,
    yellow_corners: np.ndarray,
    *,
    blue_roi: Optional[Sequence[int]] = None,
    root: Optional[str] = None,
    notes: str = '',
    rotate_deg: int = 0,
) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    y = pc.order_points(np.asarray(yellow_corners, dtype=np.float32).reshape(4, 2))
    payload: Dict[str, Any] = {
        'version': LABEL_VERSION,
        'image_path': os.path.abspath(image_path),
        'image_size': [int(w), int(h)],
        'yellow_corners': y.tolist(),
        'blue_roi': list(map(int, blue_roi)) if blue_roi is not None else None,
        'rotate_deg': int(rotate_deg),
        'notes': str(notes),
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }
    out = label_path_for_image(image_path, root=root)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
    return out


def load_label(path: str) -> Dict[str, Any]:
    with open(path, encoding='utf-8') as fh:
        data = json.load(fh)
    if 'yellow_corners' in data:
        data['yellow_corners'] = np.asarray(data['yellow_corners'], dtype=np.float32)
    if data.get('blue_roi') is not None:
        data['blue_roi'] = tuple(int(x) for x in data['blue_roi'])
    return data


def iter_labels(root: Optional[str] = None) -> List[str]:
    d = labels_dir(root)
    return sorted(
        os.path.join(d, name)
        for name in os.listdir(d)
        if name.endswith('.json')
    )


def blue_roi_from_yellow(
    yellow: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    margin_frac: float = 0.10,
    pad_px: int = 28,
) -> Tuple[int, int, int, int]:
    """Niebieski bbox = żółty + zapas na czarną ramkę."""
    h, w = image_shape[:2]
    q = pc.order_points(yellow.astype(np.float32))
    x0, y0 = float(q[:, 0].min()), float(q[:, 1].min())
    x1, y1 = float(q[:, 0].max()), float(q[:, 1].max())
    sw, sh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    m = float(margin_frac)
    bx0 = int(np.floor(x0 - m * sw - pad_px))
    by0 = int(np.floor(y0 - m * sh - pad_px))
    bx1 = int(np.ceil(x1 + m * sw + pad_px))
    by1 = int(np.ceil(y1 + m * sh + pad_px))
    bx0 = max(0, bx0)
    by0 = max(0, by0)
    bx1 = min(w - 1, bx1)
    by1 = min(h - 1, by1)
    return bx0, by0, bx1, by1


def corner_errors_px(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Tuple[float, List[float]]:
    p = pc.order_points(pred.astype(np.float32))
    g = pc.order_points(gt.astype(np.float32))
    d = np.linalg.norm(p - g, axis=1)
    return float(np.mean(d)), [float(x) for x in d]


def draw_label_overlay(
    bgr: np.ndarray,
    yellow: Optional[np.ndarray] = None,
    blue_roi: Optional[Tuple[int, int, int, int]] = None,
    *,
    pred: Optional[np.ndarray] = None,
) -> np.ndarray:
    vis = bgr.copy()
    if blue_roi is not None:
        x0, y0, x1, y1 = blue_roi
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 160, 0), 2)
    for quad, color in ((yellow, (0, 255, 255)), (pred, (0, 180, 255))):
        if quad is None:
            continue
        q = pc.order_points(quad.astype(np.float32))
        pts = q.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, color, 2)
        for i, p in enumerate(q.astype(np.int32)):
            cv2.circle(vis, tuple(p), 6, color, -1)
            cv2.putText(
                vis, CORNER_NAMES[i], (int(p[0]) + 5, int(p[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )
    return vis
