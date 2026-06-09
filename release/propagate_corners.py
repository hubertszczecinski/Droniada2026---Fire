#!/usr/bin/env python3
"""
Propagacja 4 rogów panelu między keyframe'ami (homografia ORB + fallback liniowy).

Workflow:
  1) Ręcznie zaznacz keyframe'y: annotate_corners (co N-tą klatkę).
  2) Uruchom propagację — wypełni klatki pomiędzy istniejącymi JSON w panel_labels.

Przykład (nag4, obrót jak live):
  python3 -m release.propagate_corners \\
    --frames dataset/panel_labels/frames/Droniada_nag4 \\
    --rotate 180
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pipeline_competition as pc
from release.panel_labels import (
    blue_roi_from_yellow,
    label_path_for_image,
    load_label,
    save_label,
)
from release.transform import apply_rotate

_FRAME_NUM_RE = re.compile(r'_f(\d+)(?:_r\d+)?(?:\.(?:jpg|png|jpeg))?$', re.I)


def _frame_index(path: str) -> int:
    stem = Path(path).stem
    m = _FRAME_NUM_RE.search(stem)
    if m:
        return int(m.group(1))
    m2 = re.search(r'f(\d+)', stem)
    if m2:
        return int(m2.group(1))
    return 0


def _rotated_image_path(raw_path: str, rotate_deg: int, labels_root: str) -> str:
    if not rotate_deg:
        return raw_path
    stem = Path(raw_path).stem
    out = os.path.join(labels_root, '_frames_rotated', f'{stem}_r{rotate_deg}.jpg')
    return out


def _ensure_rotated_bgr(raw_path: str, rotate_deg: int, labels_root: str) -> Tuple[np.ndarray, str]:
    """Wczytaj klatkę; zapisz obróconą kopię jeśli rotate != 0."""
    bgr = cv2.imread(raw_path)
    if bgr is None:
        raise FileNotFoundError(raw_path)
    if not rotate_deg:
        return bgr, raw_path
    bgr = apply_rotate(bgr, rotate_deg)
    save_path = _rotated_image_path(raw_path, rotate_deg, labels_root)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.isfile(save_path):
        cv2.imwrite(save_path, bgr)
    return bgr, save_path


def _roi_mask_from_quad(shape: Tuple[int, int], quad: np.ndarray, margin_frac: float = 0.15) -> np.ndarray:
    h, w = shape[:2]
    q = pc.order_points(quad.astype(np.float32))
    cx, cy = float(q[:, 0].mean()), float(q[:, 1].mean())
    scale = 1.0 + float(margin_frac)
    q_exp = np.zeros_like(q)
    for i in range(4):
        q_exp[i, 0] = cx + (q[i, 0] - cx) * scale
        q_exp[i, 1] = cy + (q[i, 1] - cy) * scale
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, q_exp.astype(np.int32), 255)
    return mask


def _estimate_homography(
    bgr0: np.ndarray,
    bgr1: np.ndarray,
    quad0: np.ndarray,
) -> Tuple[Optional[np.ndarray], int]:
    """H: mapuje punkty z obrazu0 → obraz1."""
    g0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2GRAY)
    mask = _roi_mask_from_quad(bgr0.shape, quad0, margin_frac=0.2)
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=12)
    kp0, des0 = orb.detectAndCompute(g0, mask)
    kp1, des1 = orb.detectAndCompute(g1, mask)
    if des0 is None or des1 is None or len(kp0) < 12 or len(kp1) < 12:
        return None, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = bf.knnMatch(des0, des1, k=2)
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 8:
        return None, len(good)
    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(pts0, pts1, cv2.RANSAC, 4.0)
    nin = int(inl.sum()) if inl is not None else 0
    if H is None or nin < 8:
        return None, nin
    return H, nin


def _warp_corners(quad: np.ndarray, H: np.ndarray) -> np.ndarray:
    q = pc.order_points(quad.astype(np.float32).reshape(4, 2))
    pts = q.reshape(1, 4, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(4, 2)
    return pc.order_points(out.astype(np.float32))


def _lerp_corners(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    a = pc.order_points(q0.astype(np.float32))
    b = pc.order_points(q1.astype(np.float32))
    return pc.order_points((1.0 - t) * a + t * b)


def _quad_in_bounds(quad: np.ndarray, w: int, h: int, margin: int = 8) -> bool:
    q = quad.reshape(4, 2)
    if np.any(q[:, 0] < -margin) or np.any(q[:, 1] < -margin):
        return False
    if np.any(q[:, 0] > w + margin) or np.any(q[:, 1] > h + margin):
        return False
    area = cv2.contourArea(q.astype(np.float32))
    if area < 5000:
        return False
    return True


def propagate_segment(
    paths: List[str],
    i0: int,
    i1: int,
    corners0: np.ndarray,
    corners1: np.ndarray,
    *,
    rotate_deg: int,
    labels_root: str,
    min_inliers: int,
) -> List[Tuple[int, np.ndarray, str]]:
    """Propaguj rogi na klatkach (i0, i1) — bez końców (mają GT)."""
    if i1 <= i0 + 1:
        return []
    bgr0, save0 = _ensure_rotated_bgr(paths[i0], rotate_deg, labels_root)
    bgr1_end, _ = _ensure_rotated_bgr(paths[i1], rotate_deg, labels_root)
    h, w = bgr0.shape[:2]
    out: List[Tuple[int, np.ndarray, str]] = []
    for j in range(i0 + 1, i1):
        t = float(j - i0) / float(i1 - i0)
        bgr_j, save_j = _ensure_rotated_bgr(paths[j], rotate_deg, labels_root)
        H_fwd, nin_fwd = _estimate_homography(bgr0, bgr_j, corners0)
        H_end, nin_end = _estimate_homography(bgr_j, bgr1_end, corners1)
        method = 'lerp'
        corners_j: Optional[np.ndarray] = None
        if H_fwd is not None and nin_fwd >= min_inliers:
            cand = _warp_corners(corners0, H_fwd)
            if _quad_in_bounds(cand, w, h):
                corners_j = cand
                method = f'homography_fwd(inl={nin_fwd})'
        if corners_j is None and H_end is not None and nin_end >= min_inliers:
            try:
                H_inv = np.linalg.inv(H_end)
                cand = _warp_corners(corners1, H_inv)
                if _quad_in_bounds(cand, w, h):
                    corners_j = cand
                    method = f'homography_inv(inl={nin_end})'
            except np.linalg.LinAlgError:
                pass
        if corners_j is None:
            corners_j = _lerp_corners(corners0, corners1, t)
            method = 'lerp_fallback'
        out.append((j, corners_j, method))
    return out


def collect_frame_paths(frames_dir: str) -> List[str]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    paths = []
    for name in os.listdir(frames_dir):
        if Path(name).suffix.lower() in exts:
            paths.append(os.path.join(frames_dir, name))
    paths.sort(key=_frame_index)
    return paths


def load_keyframe_corners(
    save_image_path: str,
    labels_root: str,
) -> Optional[np.ndarray]:
    lp = label_path_for_image(save_image_path, root=labels_root)
    if not os.path.isfile(lp):
        return None
    data = load_label(lp)
    notes = str(data.get('notes', ''))
    if notes.startswith('skip'):
        return None
    y = data.get('yellow_corners')
    if y is None:
        return None
    return np.asarray(y, dtype=np.float32)


def _json_prefixes_for_video(video_stem: str) -> Tuple[str, ...]:
    """Droniada_nag2.mov vs etykiety Droniad_nag2_* (literówka w starych plikach)."""
    prefixes = [video_stem]
    if video_stem.startswith('Droniada_'):
        prefixes.append('Droniad_' + video_stem[len('Droniada_') :])
    elif video_stem.startswith('Droniad_'):
        prefixes.append('Droniada_' + video_stem[len('Droniad_') :])
    return tuple(dict.fromkeys(prefixes))


def load_manual_keyframes(labels_root: str, video_stem: str) -> Dict[int, np.ndarray]:
    """frame_index → yellow_corners (tylko ręczne / bez notes propagated)."""
    out: Dict[int, np.ndarray] = {}
    prefixes = _json_prefixes_for_video(video_stem)
    d = labels_root if os.path.isabs(labels_root) else os.path.join(_ROOT, labels_root)
    for name in sorted(os.listdir(d)):
        if not name.endswith('.json') or not any(name.startswith(p) for p in prefixes):
            continue
        lp = os.path.join(d, name)
        data = load_label(lp)
        notes = str(data.get('notes', ''))
        if notes.startswith('propagated:'):
            continue
        y = data.get('yellow_corners')
        if y is None:
            continue
        img = str(data.get('image_path', name))
        out[_frame_index(img)] = np.asarray(y, dtype=np.float32)
    return out


def _save_propagated_frame(
    bgr_rot: np.ndarray,
    save_path: str,
    corners: np.ndarray,
    labels_root: str,
    method: str,
    *,
    dry_run: bool,
    review_dir: Optional[str],
) -> bool:
    if dry_run:
        return True
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, bgr_rot)
    blue = blue_roi_from_yellow(corners, bgr_rot.shape[:2], margin_frac=0.12, pad_px=40)
    save_label(
        save_path,
        corners,
        blue_roi=blue,
        root=labels_root,
        notes=f'propagated:{method}',
    )
    if review_dir:
        from release.panel_labels import draw_label_overlay
        os.makedirs(review_dir, exist_ok=True)
        vis = draw_label_overlay(bgr_rot, yellow=corners, blue_roi=blue)
        cv2.putText(vis, method, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(review_dir, f'{Path(save_path).stem}_prop.jpg'), vis)
    return True


def run_propagate_video(
    video_path: str,
    *,
    labels_root: str,
    rotate_deg: int,
    every_n: int,
    min_inliers: int,
    dry_run: bool,
    force: bool,
    review_dir: Optional[str],
    save_raw_frames_dir: Optional[str],
) -> int:
    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        print('Brak wideo:', video_path)
        return 1
    stem = Path(video_path).stem
    keyframes = load_manual_keyframes(labels_root, stem)
    if len(keyframes) < 2:
        print(f'Potrzebujesz ≥2 ręcznych keyframe JSON ({stem}_f*_r180.json), masz {len(keyframes)}')
        return 1
    kf_nums = sorted(keyframes.keys())
    print(f'Wideo: {video_path}')
    print(f'Ręczne keyframe: {len(kf_nums)} → {kf_nums}')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Nie można otworzyć wideo')
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    written = 0
    skipped = 0

    def _read_frame(idx: int) -> Optional[np.ndarray]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, fr = cap.read()
        return fr if ok else None

    for seg in range(len(kf_nums) - 1):
        f0, f1 = kf_nums[seg], kf_nums[seg + 1]
        c0, c1 = keyframes[f0], keyframes[f1]
        bgr0_raw = _read_frame(f0)
        if bgr0_raw is None:
            print(f'  skip segment: brak klatki {f0}')
            continue
        bgr0 = apply_rotate(bgr0_raw, rotate_deg) if rotate_deg else bgr0_raw
        h, w = bgr0.shape[:2]
        targets = list(range(f0 + every_n, f1, every_n))
        print(f'Segment f{f0}→f{f1}: {len(targets)} klatek (co {every_n})')
        for fi in targets:
            save_p = os.path.join(
                labels_root, '_frames_rotated', f'{stem}_f{fi:06d}_r{rotate_deg}.jpg',
            ) if rotate_deg else os.path.join(
                labels_root, 'frames', stem, f'{stem}_f{fi:06d}.jpg',
            )
            lp = label_path_for_image(save_p, root=labels_root)
            if os.path.isfile(lp) and not force:
                data = load_label(lp)
                if not str(data.get('notes', '')).startswith('propagated:'):
                    skipped += 1
                    continue
            bgr_i_raw = _read_frame(fi)
            if bgr_i_raw is None:
                continue
            bgr_i = apply_rotate(bgr_i_raw, rotate_deg) if rotate_deg else bgr_i_raw
            t = float(fi - f0) / float(f1 - f0)
            H_fwd, nin_fwd = _estimate_homography(bgr0, bgr_i, c0)
            method = 'lerp'
            corners_i: Optional[np.ndarray] = None
            if H_fwd is not None and nin_fwd >= min_inliers:
                cand = _warp_corners(c0, H_fwd)
                if _quad_in_bounds(cand, w, h):
                    corners_i = cand
                    method = f'homography_fwd(inl={nin_fwd})'
            if corners_i is None:
                corners_i = _lerp_corners(c0, c1, t)
                method = 'lerp_fallback'
            if dry_run:
                print(f'  [dry] f{fi} {method}')
                written += 1
                continue
            if save_raw_frames_dir:
                os.makedirs(save_raw_frames_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(save_raw_frames_dir, f'{stem}_f{fi:06d}.jpg'),
                    bgr_i_raw,
                )
            if _save_propagated_frame(bgr_i, save_p, corners_i, labels_root, method, dry_run=False, review_dir=review_dir):
                written += 1

    cap.release()
    print(f'Gotowe: zapisano={written}, pominięto (ręczne)={skipped}, klatek wideo≈{total}')
    return 0


def list_keyframe_gaps(paths: List[str], labels_root: str, rotate_deg: int) -> None:
    labeled = []
    for p in paths:
        save_p = _rotated_image_path(p, rotate_deg, labels_root) if rotate_deg else p
        if load_keyframe_corners(save_p, labels_root) is not None:
            labeled.append(_frame_index(p))
    print(f'Keyframe z etykietą: {len(labeled)} / {len(paths)} klatek')
    if labeled:
        print('  indeksy:', labeled)
    unlabeled = [_frame_index(p) for p in paths if _frame_index(p) not in labeled]
    if unlabeled:
        print(f'Brak etykiety ({len(unlabeled)}): pierwsze 20 → {unlabeled[:20]}...')


def run_propagate(
    frames_dir: str,
    *,
    labels_root: str,
    rotate_deg: int,
    min_inliers: int,
    dry_run: bool,
    force: bool,
    review_dir: Optional[str],
) -> int:
    paths = collect_frame_paths(frames_dir)
    if len(paths) < 2:
        print('Za mało klatek w', frames_dir)
        return 1

    key_indices: List[int] = []
    key_corners: Dict[int, np.ndarray] = {}
    for i, p in enumerate(paths):
        if rotate_deg:
            save_p = _rotated_image_path(p, rotate_deg, labels_root)
        else:
            save_p = p
        c = load_keyframe_corners(save_p, labels_root)
        if c is not None:
            key_indices.append(i)
            key_corners[i] = c

    if len(key_indices) < 2:
        print(
            f'Potrzebujesz ≥2 keyframe z JSON (masz {len(key_indices)}). '
            f'Najpierw: python3 -m release.annotate_corners {frames_dir} --rotate {rotate_deg}',
        )
        return 1

    print(f'Klatki: {len(paths)}, keyframe: {len(key_indices)} (idx {key_indices})')
    written = 0
    skipped = 0

    for seg in range(len(key_indices) - 1):
        i0 = key_indices[seg]
        i1 = key_indices[seg + 1]
        c0 = key_corners[i0]
        c1 = key_corners[i1]
        n_between = i1 - i0 - 1
        print(
            f'Segment f{_frame_index(paths[i0])} → f{_frame_index(paths[i1])} '
            f'({n_between} plików między — jeśli 0, użyj --video)',
        )
        propagated = propagate_segment(
            paths, i0, i1, c0, c1,
            rotate_deg=rotate_deg,
            labels_root=labels_root,
            min_inliers=min_inliers,
        )
        for j, corners_j, method in propagated:
            if rotate_deg:
                save_p = _rotated_image_path(paths[j], rotate_deg, labels_root)
                _ensure_rotated_bgr(paths[j], rotate_deg, labels_root)
            else:
                save_p = paths[j]
            lp = label_path_for_image(save_p, root=labels_root)
            if os.path.isfile(lp) and not force:
                data = load_label(lp)
                notes = str(data.get('notes', ''))
                if not notes.startswith('propagated:'):
                    skipped += 1
                    continue
            if dry_run:
                print(f'  [dry] f{_frame_index(paths[j])} {method}')
                written += 1
                continue
            blue = blue_roi_from_yellow(corners_j, cv2.imread(save_p).shape[:2], margin_frac=0.12, pad_px=40)
            save_label(
                save_p,
                corners_j,
                blue_roi=blue,
                root=labels_root,
                notes=f'propagated:{method}',
            )
            written += 1
            if review_dir:
                os.makedirs(review_dir, exist_ok=True)
                bgr = cv2.imread(save_p)
                from release.panel_labels import draw_label_overlay
                vis = draw_label_overlay(bgr, yellow=corners_j, blue_roi=blue)
                cv2.putText(
                    vis, method, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                )
                cv2.imwrite(
                    os.path.join(review_dir, f'{Path(save_p).stem}_prop.jpg'),
                    vis,
                )

    print(f'Gotowe: zapisano/propozycja={written}, pominięto (ręczne)={skipped}')
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description='Propagacja rogów między keyframe panel_labels.')
    ap.add_argument(
        '--frames',
        default=None,
        help='Folder z klatkami (tylko gdy między plikami są pośrednie klatki)',
    )
    ap.add_argument(
        '--video',
        default=None,
        help='Źródło wideo — propagacja klatek między keyframe (zalecane dla nag4)',
    )
    ap.add_argument('--labels-dir', default=None)
    ap.add_argument('--rotate', type=int, default=180, choices=[0, 90, 180, 270])
    ap.add_argument(
        '--every-n',
        type=int,
        default=5,
        help='Co która klatka wideo zapisać między keyframe (tryb --video)',
    )
    ap.add_argument('--min-inliers', type=int, default=12, help='Min. inlierów RANSAC homografii')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true', help='Nadpisz także ręczne etykiety')
    ap.add_argument('--list-keyframes', action='store_true', help='Tylko pokaż które klatki mają JSON')
    ap.add_argument(
        '--review-dir',
        default=None,
        help='Zapis podglądu propagacji (np. dataset/panel_labels/propagated_review/nag4)',
    )
    ap.add_argument(
        '--save-raw-frames',
        default=None,
        help='Opcjonalnie zapisz surowe JPG z wideo (np. dataset/panel_labels/frames/Droniada_nag4)',
    )
    args = ap.parse_args()

    labels_root = args.labels_dir or os.path.join(_ROOT, 'dataset', 'panel_labels')

    if args.video:
        if args.list_keyframes:
            kf = load_manual_keyframes(labels_root, Path(args.video).stem)
            print('Ręczne keyframe:', sorted(kf.keys()))
            return
        run_propagate_video(
            args.video,
            labels_root=labels_root,
            rotate_deg=args.rotate,
            every_n=max(1, args.every_n),
            min_inliers=args.min_inliers,
            dry_run=args.dry_run,
            force=args.force,
            review_dir=args.review_dir,
            save_raw_frames_dir=args.save_raw_frames,
        )
        return

    if not args.frames:
        ap.error('Podaj --video (zalecane) albo --frames')

    frames_dir = os.path.abspath(args.frames)
    if not os.path.isdir(frames_dir):
        ap.error(f'Brak folderu: {frames_dir}')

    paths = collect_frame_paths(frames_dir)
    if args.list_keyframes:
        list_keyframe_gaps(paths, labels_root, args.rotate)
        return

    run_propagate(
        frames_dir,
        labels_root=labels_root,
        rotate_deg=args.rotate,
        min_inliers=args.min_inliers,
        dry_run=args.dry_run,
        force=args.force,
        review_dir=args.review_dir,
    )


if __name__ == '__main__':
    main()
