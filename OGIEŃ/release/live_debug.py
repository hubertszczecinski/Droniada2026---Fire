"""Zapis i wizualizacja debug live — kandydaci rogów, overlay, JSON."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import pipeline_competition as pc

_CANDIDATE_COLORS = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 80, 255),
    (255, 255, 80),
    (255, 80, 255),
    (80, 255, 255),
    (200, 140, 60),
    (140, 60, 200),
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    raise TypeError(type(obj))


def new_session_dir(root: str) -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(root, ts)
    os.makedirs(path, exist_ok=True)
    return path


def draw_candidates_board(
    image_bgr: np.ndarray,
    candidates: List[Dict[str, Any]],
    *,
    winner_label: str = '',
) -> np.ndarray:
    vis = image_bgr.copy()
    h, w = vis.shape[:2]
    for i, cand in enumerate(candidates):
        q = cand.get('corners')
        if q is None:
            continue
        q = np.asarray(q, dtype=np.float32).reshape(4, 2)
        color = _CANDIDATE_COLORS[i % len(_CANDIDATE_COLORS)]
        thick = 4 if cand.get('label') == winner_label else 2
        pts = q.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, color, thick)
        lbl = str(cand.get('label', '?'))
        reproj = cand.get('reproj_mean_px', 999)
        sc = cand.get('rank_score', 0)
        tag = f"{lbl} r={reproj:.1f} g={cand.get('grid_structure_score', 0):.2f}"
        if cand.get('chosen'):
            tag += ' *WIN*'
        cv2.putText(
            vis, tag,
            (int(q[0, 0]), max(20, int(q[0, 1]) - 8 - i * 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA,
        )
    cv2.putText(vis, f'candidates={len(candidates)} winner={winner_label}', (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2)
    return vis


def save_live_frame_bundle(
    session_dir: str,
    frame_id: str,
    *,
    raw_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    warped_bgr: np.ndarray,
    candidates_board: np.ndarray,
    record: Dict[str, Any],
) -> str:
    stem = frame_id.replace('/', '_')
    base = os.path.join(session_dir, stem)
    cv2.imwrite(base + '_raw.jpg', raw_bgr)
    cv2.imwrite(base + '_overlay.jpg', overlay_bgr)
    cv2.imwrite(base + '_warped.jpg', warped_bgr)
    cv2.imwrite(base + '_candidates.jpg', candidates_board)
    with open(base + '.json', 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=_json_default)
    return base


def append_session_index(session_dir: str, row: Dict[str, Any]) -> None:
    path = os.path.join(session_dir, 'index.jsonl')
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + '\n')
