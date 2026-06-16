#!/usr/bin/env python3
"""Eval zewnętrznych rogów + oracle (najlepszy kandydat z probe) — do pętli tuningu."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pipeline_competition as pc
from module_pose.api import default_intrinsics
from release.panel_labels import CORNER_NAMES, corner_errors_px, iter_labels, load_label
from release.outer_corners import detect_outer_corners


def _err_vs_gt(pred: np.ndarray, gt: np.ndarray) -> tuple[float, list[float]]:
    pred = pc.order_points(pred.astype(np.float32))
    gt = pc.order_points(gt.astype(np.float32))
    return corner_errors_px(pred, gt)


def eval_one(
    image_bgr: np.ndarray,
    gt: np.ndarray,
    *,
    fast: bool = True,
) -> dict:
    k, dist = default_intrinsics(image_bgr.shape[:2])
    pick, label, rows, meta = detect_outer_corners(
        image_bgr, k, dist, fast=fast, refine=False,
    )
    oracle_err = 9999.0
    oracle_lbl = 'none'
    per_cand: list[dict] = []
    for r in rows:
        c = r.get('corners')
        if c is None:
            continue
        mean_e, _ = _err_vs_gt(np.asarray(c, dtype=np.float32), gt)
        lbl = str(r.get('label', ''))
        gs = float(r.get('grid_structure_score', 0.0))
        per_cand.append({'label': lbl, 'mean_px': mean_e, 'grid_s': gs})
        if mean_e < oracle_err:
            oracle_err = mean_e
            oracle_lbl = lbl

    pick_err = 9999.0
    if pick is not None:
        pick_err, _ = _err_vs_gt(pick, gt)

    gap = pick_err - oracle_err if pick_err < 9000 and oracle_err < 9000 else None
    return {
        'pick_err': pick_err,
        'pick_label': label,
        'oracle_err': oracle_err,
        'oracle_label': oracle_lbl,
        'gap_px': gap,
        'n_cands': len(per_cand),
        'candidates': sorted(per_cand, key=lambda x: x['mean_px'])[:6],
        'meta': meta,
    }


def run_eval(labels_dir: str | None, fast: bool) -> dict:
    pick_errs: list[float] = []
    oracle_errs: list[float] = []
    gaps: list[float] = []
    worst: list[tuple[float, str, dict]] = []
    fail = 0

    for lp in iter_labels(labels_dir):
        data = load_label(lp)
        img = cv2.imread(str(data['image_path']))
        if img is None:
            continue
        gt = pc.order_points(data['yellow_corners'])
        rec = eval_one(img, gt, fast=fast)
        base = os.path.basename(data['image_path'])
        if rec['pick_err'] >= 9000:
            fail += 1
            print(f'{base}: BRAK')
            continue
        pick_errs.append(rec['pick_err'])
        if rec['oracle_err'] < 9000:
            oracle_errs.append(rec['oracle_err'])
        if rec['gap_px'] is not None:
            gaps.append(rec['gap_px'])
        print(
            f'{base}: pick={rec["pick_err"]:.0f} ({rec["pick_label"]}) '
            f'oracle={rec["oracle_err"]:.0f} ({rec["oracle_label"]}) gap={rec["gap_px"]:.0f}'
            if rec['gap_px'] is not None else
            f'{base}: pick={rec["pick_err"]:.0f} ({rec["pick_label"]})'
        )
        worst.append((rec['pick_err'], base, rec))

    worst.sort(reverse=True)
    print('---')
    n = len(pick_errs)
    print(f'OK={n} BRAK={fail}')
    if n:
        print(f'PICK: mean={np.mean(pick_errs):.1f} med={np.median(pick_errs):.1f} p90={np.percentile(pick_errs, 90):.1f}')
    if oracle_errs:
        print(f'ORACLE: mean={np.mean(oracle_errs):.1f} med={np.median(oracle_errs):.1f}')
    if gaps:
        print(f'GAP pick-oracle: mean={np.mean(gaps):.1f} med={np.median(gaps):.1f} max={max(gaps):.1f}')
    print('Worst 5:')
    for e, b, r in worst[:5]:
        print(f'  {b}: {e:.0f}px {r["pick_label"]} oracle={r["oracle_err"]:.0f}')

    return {
        'n': n,
        'pick_median': float(np.median(pick_errs)) if pick_errs else None,
        'oracle_median': float(np.median(oracle_errs)) if oracle_errs else None,
        'gap_median': float(np.median(gaps)) if gaps else None,
        'worst': [(b, e) for e, b, _ in worst[:8]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels-dir', default=None)
    ap.add_argument('--no-fast', action='store_true')
    ap.add_argument('--log', default='release/CORNERS_EXPERIMENTS.jsonl')
    ap.add_argument('--note', default='')
    args = ap.parse_args()

    summary = run_eval(args.labels_dir, fast=not args.no_fast)
    entry = {
        'ts': datetime.now(timezone.utc).isoformat(),
        'note': args.note,
        'fast': not args.no_fast,
        **summary,
    }
    log_path = os.path.join(_ROOT, args.log) if not os.path.isabs(args.log) else args.log
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f'[log] {log_path}')


if __name__ == '__main__':
    main()
