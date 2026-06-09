#!/usr/bin/env python3
"""
Porównanie rogów: YOLO-Pose (zero-shot / fine-tuned) vs release.outer_corners (CV).

Użycie:
  .venv_yolo/bin/python -m release.export_yolo_pose_dataset
  .venv_yolo/bin/python -m release.eval_yolo_pose_corners --train --epochs 60
  .venv_yolo/bin/python -m release.eval_yolo_pose_corners --weights runs/pose/.../weights/best.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pipeline_competition as pc

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from release.panel_labels import CORNER_NAMES, corner_errors_px, iter_labels, load_label


def _kpts_to_quad(kpts: np.ndarray, conf_thr: float = 0.25) -> Optional[np.ndarray]:
    """kpts: (4, 3) x,y,conf"""
    if kpts is None or kpts.shape[0] < 4:
        return None
    pts = []
    for i in range(4):
        x, y, c = float(kpts[i, 0]), float(kpts[i, 1]), float(kpts[i, 2])
        if c < conf_thr or not np.isfinite(x) or not np.isfinite(y):
            return None
        pts.append([x, y])
    return pc.order_points(np.asarray(pts, dtype=np.float32))


def _eval_cv(img: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    from module_pose.api import default_intrinsics
    from release.outer_corners import detect_outer_corners, enhance_for_corner_probe

    k, dist = default_intrinsics(img.shape[:2])
    probe = enhance_for_corner_probe(img)
    corners, label, _, _ = detect_outer_corners(
        img, k, dist, fast=True, refine=False, probe_bgr=probe, enhance_probe=False,
    )
    return corners, label or 'none'


def _eval_yolo_pose(
    img: np.ndarray,
    model,
    conf: float = 0.2,
) -> Tuple[Optional[np.ndarray], str]:
    res = model.predict(img, verbose=False, conf=conf)[0]
    if res.keypoints is None or len(res.keypoints.data) == 0:
        return None, 'yolo_none'
    # najwyższy conf detekcji
    boxes = res.boxes
    best_i = 0
    if boxes is not None and len(boxes) > 1:
        confs = boxes.conf.cpu().numpy()
        best_i = int(np.argmax(confs))
    kpts = res.keypoints.data[best_i].cpu().numpy()
    quad = _kpts_to_quad(kpts, conf_thr=0.15)
    if quad is None:
        return None, 'yolo_low_kpt'
    return quad, 'yolo_pose'


def run_eval(
    weights: Optional[str],
    labels_dir: str | None,
    *,
    mode: str = 'cv',
) -> None:
    from ultralytics import YOLO

    model = None
    if mode in ('yolo', 'both'):
        w = weights or 'yolov8n-pose.pt'
        model = YOLO(w)

    cv_errs: List[float] = []
    yolo_errs: List[float] = []
    yolo_fail = 0
    cv_fail = 0

    for lp in iter_labels(labels_dir):
        data = load_label(lp)
        ip = data.get('image_path')
        if not ip:
            continue
        img = cv2.imread(str(ip))
        if img is None:
            continue
        gt = pc.order_points(data['yellow_corners'])

        if mode in ('cv', 'both'):
            pred, lbl = _eval_cv(img)
            if pred is None:
                cv_fail += 1
            else:
                m, _ = corner_errors_px(pred, gt)
                cv_errs.append(m)
                print(f'{Path(data["image_path"]).name}: CV={m:.0f}px ({lbl})')

        if mode in ('yolo', 'both') and model is not None:
            pred, lbl = _eval_yolo_pose(img, model)
            if pred is None:
                yolo_fail += 1
            else:
                m, _ = corner_errors_px(pred, gt)
                yolo_errs.append(m)
                if mode == 'yolo':
                    print(f'{Path(data["image_path"]).name}: YOLO={m:.0f}px ({lbl})')

    print('---')
    if cv_errs:
        print(f'CV: n={len(cv_errs)} fail={cv_fail} mean={np.mean(cv_errs):.1f} med={np.median(cv_errs):.1f}')
    if yolo_errs:
        print(f'YOLO: n={len(yolo_errs)} fail={yolo_fail} mean={np.mean(yolo_errs):.1f} med={np.median(yolo_errs):.1f}')
    elif mode in ('yolo', 'both'):
        print(f'YOLO: brak detekcji na wszystkich klatkach (fail={yolo_fail})')


def _training_device() -> str:
    import torch
    if torch.cuda.is_available():
        return '0'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def train_pose(
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    project: str,
    *,
    weights: str = 'yolov8n-pose.pt',
    name: str = 'droniada_panel_corners',
    lr0: Optional[float] = None,
    device: Optional[str] = None,
) -> Path:
    from ultralytics import YOLO

    dev = device or _training_device()
    model = YOLO(weights)
    kw: dict = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name=name,
        patience=15,
        exist_ok=True,
        verbose=True,
        device=dev,
    )
    if lr0 is not None:
        kw['lr0'] = lr0
    print(f'[train] device={dev} init={weights}')
    model.train(**kw)
    best = Path(project) / name / 'weights' / 'best.pt'
    print(f'[train] init={weights} -> {best}')
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels-dir', default=None)
    ap.add_argument('--mode', default='both', choices=['cv', 'yolo', 'both'])
    ap.add_argument('--weights', default=None, help='ścieżka do .pt (domyślnie yolov8n-pose.pt)')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--data-yaml', default=str(_ROOT / 'dataset' / 'droniada_pose' / 'droniada_pose.yaml'))
    ap.add_argument('--project', default=str(_ROOT / 'runs' / 'pose'))
    ap.add_argument(
        '--train-weights',
        default='yolov8n-pose.pt',
        help='checkpoint startowy: yolov8n-pose.pt (pretrain) lub best.pt z etapu 1 (fine-tune)',
    )
    ap.add_argument('--train-name', default='droniada_panel_corners', help='podkatalog w --project')
    ap.add_argument(
        '--lr0',
        type=float,
        default=None,
        help='learning rate (np. 0.001 przy fine-tune na realnych)',
    )
    ap.add_argument(
        '--calibrate-bias',
        action='store_true',
        help='policz korekcję pred−GT i zapisz module_panel/data/yolo_corner_bias.json',
    )
    ap.add_argument('--no-bias-eval', action='store_true', help='eval YOLO bez korekcji bias')
    ap.add_argument(
        '--no-post-train-eval',
        action='store_true',
        help='po --train nie uruchamiaj run_eval na panel_labels',
    )
    args = ap.parse_args()

    if args.calibrate_bias:
        from release.yolo_corner_bias import calibrate_bias_from_labels

        w = args.weights or str(_ROOT / 'runs' / 'pose' / 'droniada_real_finetune' / 'weights' / 'best.pt')
        calibrate_bias_from_labels(w, args.labels_dir)
        return

    data_yaml = Path(args.data_yaml)
    if args.train:
        if not data_yaml.is_file():
            from release.export_yolo_pose_dataset import export_dataset
            export_dataset(data_yaml.parent, args.labels_dir)
        best = train_pose(
            data_yaml,
            args.epochs,
            args.imgsz,
            args.project,
            weights=args.train_weights,
            name=args.train_name,
            lr0=args.lr0,
        )
        args.weights = str(best)
        if args.mode == 'both':
            args.mode = 'yolo'

    if args.train and args.weights and not args.no_post_train_eval:
        run_eval(args.weights, args.labels_dir, mode=args.mode)
    elif not args.train:
        run_eval(args.weights, args.labels_dir, mode=args.mode)


if __name__ == '__main__':
    main()
