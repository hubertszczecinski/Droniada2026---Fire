#!/usr/bin/env python3
"""Porównanie trybów rogów na zapisanej sesji live_debug (A/B)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from module_panel.analyze import analyze_panel_image
from release.live_corners import acquire_corners_for_live, default_intrinsics

_MODES = ('roi_hybrid', 'roi_line_grid', 'live', 'line_grid', 'auto', 'hybrid')


def _stats(values: list[float]) -> dict:
    valid = [float(v) for v in values if float(v) < 9000.0]
    if not valid:
        return {'ok': 0, 'mean': None, 'median': None, 'min': None, 'max': None, 'le_18': 0, 'le_22': 0}
    arr = np.array(valid, dtype=np.float64)
    return {
        'ok': len(valid),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'le_18': int((arr <= 18.0).sum()),
        'le_22': int((arr <= 22.0).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='A/B corner modes on live_debug session')
    ap.add_argument('--session', required=True, help='np. live_debug/20260520_164437')
    ap.add_argument('--xy-mode', default='line_grid')
    ap.add_argument('--max-reproj-reliable', type=float, default=18.0)
    ap.add_argument('--out', default='dataset/results/compare_live_corners.json')
    args = ap.parse_args()

    session = args.session if os.path.isabs(args.session) else os.path.join(_ROOT, args.session)
    frames = sorted(f for f in os.listdir(session) if f.endswith('_raw.jpg'))
    if not frames:
        raise SystemExit(f'Brak *_raw.jpg w {session}')

    reproj_by_mode: dict[str, list[float]] = {m: [] for m in _MODES}
    reliable_by_mode: dict[str, list[bool]] = {m: [] for m in _MODES}
    labels_by_mode: dict[str, dict[str, int]] = {m: {} for m in _MODES}
    per_frame: list[dict] = []

    for fn in frames:
        fid = fn.replace('_raw.jpg', '')
        bgr = cv2.imread(os.path.join(session, fn))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        k, dist = default_intrinsics((h, w))
        row: dict = {'frame_id': fid}
        for mode in _MODES:
            corners, src, meta, _rows = acquire_corners_for_live(bgr, k, dist, corner_mode=mode)
            reproj = float(meta.get('reproj_mean_px', 9999.0)) if corners is not None else 9999.0
            reproj_by_mode[mode].append(reproj)
            labels_by_mode[mode][src] = labels_by_mode[mode].get(src, 0) + 1
            rel = False
            if corners is not None:
                pan = analyze_panel_image(
                    bgr,
                    [],
                    k=k,
                    dist=dist,
                    xy_mode=args.xy_mode,
                    max_reproj_px=float(args.max_reproj_reliable),
                    allowed_orbit_steps=None,
                    corners_px=corners,
                    corner_source=src,
                )
                rel = bool(pan.meta.get('grid_xy_reliable', False))
                reproj = float(pan.meta.get('reproj_mean_px', reproj))
                reproj_by_mode[mode][-1] = reproj
            reliable_by_mode[mode].append(rel)
            row[mode] = {
                'corner_source': src,
                'reproj_mean_px': None if reproj >= 9000 else reproj,
                'grid_xy_reliable': rel,
            }
        per_frame.append(row)

    wins = {m: 0 for m in _MODES}
    wins['none'] = 0
    for row in per_frame:
        scores = {
            m: row[m]['reproj_mean_px']
            for m in _MODES
            if row[m]['reproj_mean_px'] is not None
        }
        if not scores:
            wins['none'] += 1
            continue
        best = min(scores.values())
        winners = [m for m, v in scores.items() if abs(v - best) < 0.5]
        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            wins['tie'] = wins.get('tie', 0) + 1

    summary = {
        'session': session,
        'frames': len(per_frame),
        'xy_mode': args.xy_mode,
        'reproj': {m: _stats(reproj_by_mode[m]) for m in _MODES},
        'reliable': {m: int(sum(reliable_by_mode[m])) for m in _MODES},
        'corner_labels': labels_by_mode,
        'reproj_wins': wins,
        'per_frame': per_frame,
    }

    out_path = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(json.dumps({
        'session': os.path.basename(session),
        'frames': summary['frames'],
        'reproj': summary['reproj'],
        'reliable': summary['reliable'],
        'reproj_wins': summary['reproj_wins'],
        'out': out_path,
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
