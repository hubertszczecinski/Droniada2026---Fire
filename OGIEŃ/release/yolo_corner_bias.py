"""Kalibracja i korekcja systematycznego biasu YOLO-Pose (pred−GT na panel_labels).

Jedna komenda (po treningu / zmianie etykiet):

  .venv_yolo/bin/python -m release.yolo_corner_bias
  ./scripts/calibrate_yolo_corner_bias.sh

Live ładuje ``module_panel/data/yolo_corner_bias.json`` gdy ``DRONIADA_YOLO_BIAS=1`` (domyślnie).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pipeline_competition as pc

from release.panel_labels import CORNER_NAMES, corner_errors_px, iter_labels, load_label

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BIAS_PATH = _ROOT / 'module_panel' / 'data' / 'yolo_corner_bias.json'
_DEFAULT_WEIGHTS = _ROOT / 'runs' / 'pose' / 'droniada_real_finetune' / 'weights' / 'best.pt'
_FRAMES_ROTATED = _ROOT / 'dataset' / 'panel_labels' / '_frames_rotated'


def load_bias_correction(path: Optional[str] = None) -> np.ndarray:
    """Wektor (4,2) px do dodania do rogów w kolejności TL,TR,BR,BL."""
    p = Path(path or os.environ.get('DRONIADA_YOLO_BIAS_JSON', str(_DEFAULT_BIAS_PATH)))
    if p.is_file():
        data = json.loads(p.read_text(encoding='utf-8'))
        corners = data.get('corners') or {}
        out = np.zeros((4, 2), dtype=np.float32)
        for i, name in enumerate(CORNER_NAMES):
            v = corners.get(name, [0.0, 0.0])
            out[i] = (float(v[0]), float(v[1]))
        return out
    return np.zeros((4, 2), dtype=np.float32)


def apply_bias_correction(corners: np.ndarray, delta: np.ndarray) -> np.ndarray:
    q = pc.order_points(corners.astype(np.float32))
    out = q + delta.reshape(4, 2)
    return out.astype(np.float32)


def resolve_label_image_path(data: Dict, label_json_path: str) -> Optional[Path]:
    """Ścieżka obrazu z JSON lub domyślny katalog _frames_rotated/."""
    ip = data.get('image_path')
    if ip:
        p = Path(str(ip))
        if p.is_file():
            return p
    stem = Path(label_json_path).stem
    for base in (_FRAMES_ROTATED, _ROOT / 'dataset' / 'panel_labels'):
        for ext in ('.jpg', '.jpeg', '.png'):
            cand = base / f'{stem}{ext}'
            if cand.is_file():
                return cand
    return None


def eval_with_optional_bias(
    weights: str,
    labels_dir: str | None,
    *,
    apply_bias: bool,
) -> Tuple[int, int, List[float], Dict[str, List[float]]]:
    """Zwraca (n_ok, n_fail, mean_corner_errs, per_corner_mean_dx_dy)."""
    import cv2
    from ultralytics import YOLO

    from release.eval_yolo_pose_corners import _eval_yolo_pose

    model = YOLO(weights)
    delta = load_bias_correction() if apply_bias else np.zeros((4, 2), dtype=np.float32)
    errs: List[float] = []
    per_corner: Dict[str, List[float]] = {n: [] for n in CORNER_NAMES}
    n_fail = 0
    n_ok = 0
    for lp in iter_labels(labels_dir):
        data = load_label(lp)
        img_path = resolve_label_image_path(data, lp)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gt = pc.order_points(data['yellow_corners'])
        pred, _ = _eval_yolo_pose(img, model)
        if pred is None:
            n_fail += 1
            continue
        pred = pc.order_points(pred)
        if apply_bias and float(np.max(np.abs(delta))) > 0.5:
            pred = apply_bias_correction(pred, delta)
        n_ok += 1
        m, per = corner_errors_px(pred, gt)
        errs.append(m)
        for i, name in enumerate(CORNER_NAMES):
            per_corner[name].append(float(per[i]))
    return n_ok, n_fail, errs, per_corner


def calibrate_bias_from_labels(
    weights: str,
    labels_dir: str | None = None,
    *,
    out_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, list]]:
    from ultralytics import YOLO
    from release.eval_yolo_pose_corners import _eval_yolo_pose
    import cv2

    model = YOLO(weights)
    acc: Dict[str, list] = {n: [] for n in CORNER_NAMES}
    n_ok = 0
    n_skip = 0
    for lp in iter_labels(labels_dir):
        data = load_label(lp)
        img_path = resolve_label_image_path(data, lp)
        if img_path is None:
            n_skip += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            n_skip += 1
            continue
        gt = pc.order_points(data['yellow_corners'])
        pred, _ = _eval_yolo_pose(img, model)
        if pred is None:
            continue
        pred = pc.order_points(pred)
        n_ok += 1
        for i, name in enumerate(CORNER_NAMES):
            acc[name].append((float(pred[i, 0] - gt[i, 0]), float(pred[i, 1] - gt[i, 1])))
    correction = np.zeros((4, 2), dtype=np.float32)
    payload_corners: Dict[str, list] = {}
    for i, name in enumerate(CORNER_NAMES):
        if not acc[name]:
            payload_corners[name] = [0.0, 0.0]
            continue
        arr = np.asarray(acc[name], dtype=np.float64)
        mean_err = arr.mean(axis=0)
        correction[i] = (-mean_err[0], -mean_err[1])
        payload_corners[name] = [round(float(correction[i, 0]), 1), round(float(correction[i, 1]), 1)]
    out = Path(out_path or _DEFAULT_BIAS_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                'source': f'calibrate n={n_ok} weights={weights}',
                'note': 'correction_px dodawane do pred (pred + correction ≈ GT)',
                'corners': payload_corners,
            },
            indent=2,
            ensure_ascii=False,
        )
        + '\n',
        encoding='utf-8',
    )
    print(f'calibrated n={n_ok} skip={n_skip} -> {out}')
    for name in CORNER_NAMES:
        c = payload_corners[name]
        print(f'  {name}: correction=({c[0]:+.1f}, {c[1]:+.1f}) px')
    print(
        'Live: export DRONIADA_YOLO_BIAS=1  (domyślnie w scripts/run_live_*.sh)\n'
        f'      plik: {out}',
        flush=True,
    )
    return correction, acc


def _default_weights() -> str:
    env = os.environ.get('DRONIADA_YOLO_POSE_WEIGHTS', '').strip()
    if env and Path(env).is_file():
        return env
    if _DEFAULT_WEIGHTS.is_file():
        return str(_DEFAULT_WEIGHTS)
    return str(_ROOT / 'runs' / 'pose' / 'droniada_panel_corners' / 'weights' / 'best.pt')


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description='Kalibracja biasu YOLO-Pose (pred−GT na panel_labels) → yolo_corner_bias.json',
    )
    ap.add_argument(
        '--weights',
        default=None,
        help=f'wagi .pt (domyślnie DRONIADA_YOLO_POSE_WEIGHTS lub {_DEFAULT_WEIGHTS.name})',
    )
    ap.add_argument('--labels-dir', default=None, help='dataset/panel_labels (domyślnie)')
    ap.add_argument(
        '--out',
        default=None,
        help=f'ścieżka JSON (domyślnie { _DEFAULT_BIAS_PATH.relative_to(_ROOT) })',
    )
    ap.add_argument(
        '--report-only',
        action='store_true',
        help='tylko pokaż obecny plik bias + ewaluację przed/po (bez przeliczania)',
    )
    ap.add_argument(
        '--no-eval',
        action='store_true',
        help='po kalibracji nie drukuj mediany błędu przed/po',
    )
    args = ap.parse_args(argv)

    weights = args.weights or _default_weights()
    if not Path(weights).is_file():
        print(f'Brak wag: {weights}', file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else _DEFAULT_BIAS_PATH

    if not args.report_only:
        calibrate_bias_from_labels(
            weights,
            args.labels_dir,
            out_path=str(out_path),
        )
    elif not out_path.is_file():
        print(f'Brak pliku bias: {out_path} — uruchom bez --report-only', file=sys.stderr)
        return 1
    else:
        data = json.loads(out_path.read_text(encoding='utf-8'))
        print(f'Bias: {out_path}')
        print(f'  source: {data.get("source", "?")}')
        for name in CORNER_NAMES:
            c = (data.get('corners') or {}).get(name, [0, 0])
            print(f'  {name}: ({c[0]:+.1f}, {c[1]:+.1f}) px')

    if args.no_eval:
        return 0

    print('--- eval na panel_labels ---')
    n0, f0, e0, _ = eval_with_optional_bias(weights, args.labels_dir, apply_bias=False)
    n1, f1, e1, _ = eval_with_optional_bias(weights, args.labels_dir, apply_bias=True)
    if e0:
        print(
            f'bez bias:  n={n0} fail={f0} mean={np.mean(e0):.1f}px med={np.median(e0):.1f}px',
        )
    if e1:
        print(
            f'z bias:    n={n1} fail={f1} mean={np.mean(e1):.1f}px med={np.median(e1):.1f}px',
        )
    if e0 and e1:
        print(f'poprawa:   mean {np.mean(e0) - np.mean(e1):+.1f}px  med {np.median(e0) - np.median(e1):+.1f}px')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
