#!/usr/bin/env python3
"""Kliknij 4 rogi żółtej siatki (TL→TR→BR→BL). Zapis do dataset/panel_labels/."""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from release.panel_labels import (
    CORNER_NAMES,
    blue_roi_from_yellow,
    draw_label_overlay,
    label_path_for_image,
    load_label,
    save_label,
)
from release.transform import apply_rotate

_WIN = 'annotate_panel'
_IMG_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
_VIDEO_SUFFIXES = ('.mov', '.mp4', '.avi', '.mkv', '.webm', '.m4v')


def _collect_images(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for ext in _IMG_EXTS:
                out.extend(glob.glob(os.path.join(p, '**', ext), recursive=True))
        elif os.path.isfile(p):
            low = p.lower()
            if any(low.endswith(e[1:]) for e in _IMG_EXTS):
                out.append(os.path.abspath(p))
    return sorted(set(out))


def _collect_videos(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.lower().endswith(_VIDEO_SUFFIXES):
                    out.append(os.path.abspath(os.path.join(p, name)))
        elif os.path.isfile(p) and p.lower().endswith(_VIDEO_SUFFIXES):
            out.append(os.path.abspath(p))
    return sorted(set(out))


def _resolve_video_path(video_arg: str, search_dirs: List[str]) -> str:
    if os.path.isabs(video_arg) and os.path.isfile(video_arg):
        return video_arg
    candidates = [
        os.path.abspath(video_arg),
        os.path.join(_ROOT, video_arg),
    ]
    for d in search_dirs:
        if os.path.isdir(d):
            candidates.append(os.path.join(os.path.abspath(d), os.path.basename(video_arg)))
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f'Nie znaleziono pliku wideo: {video_arg}\n'
        f'Sprawdź ścieżkę, np. dataset/my_capture/Droniada_nag1.mov',
    )


def _extract_video_frames(
    video_path: str,
    out_dir: str,
    *,
    every_n: int,
    max_frames: int,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Nie można otworzyć wideo: {video_path}')
    paths: List[str] = []
    idx = 0
    saved = 0
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            path = os.path.join(out_dir, f'{Path(video_path).stem}_f{idx:06d}.jpg')
            cv2.imwrite(path, frame)
            paths.append(path)
            saved += 1
        idx += 1
    cap.release()
    return paths


class Annotator:
    def __init__(self, images: List[str], *, rotate_deg: int, labels_root: str) -> None:
        self.images = images
        self.rotate_deg = int(rotate_deg)
        self.labels_root = labels_root
        self.i = 0
        self.clicks: List[Tuple[float, float]] = []
        self.blue_roi: Optional[Tuple[int, int, int, int]] = None
        self._bgr: Optional[np.ndarray] = None
        self._path = ''

    def _load(self) -> bool:
        if self.i < 0 or self.i >= len(self.images):
            return False
        self._path = self.images[self.i]
        bgr = cv2.imread(self._path)
        if bgr is None:
            return False
        self._bgr = apply_rotate(bgr, self.rotate_deg) if self.rotate_deg else bgr
        self.clicks = []
        self.blue_roi = None
        lp = label_path_for_image(self._path, root=self.labels_root)
        if os.path.isfile(lp):
            data = load_label(lp)
            y = data.get('yellow_corners')
            if y is not None:
                self.clicks = [(float(p[0]), float(p[1])) for p in y]
            br = data.get('blue_roi')
            if br is not None:
                self.blue_roi = tuple(int(x) for x in br)
        return True

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _ud) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self._bgr is None:
            return
        if len(self.clicks) >= 4:
            self.clicks = []
        self.clicks.append((float(x), float(y)))

    def _vis(self) -> np.ndarray:
        assert self._bgr is not None
        yellow = None
        if len(self.clicks) == 4:
            yellow = np.asarray(self.clicks, dtype=np.float32)
        vis = draw_label_overlay(self._bgr, yellow=yellow, pred=None)
        n = len(self.clicks)
        hint = CORNER_NAMES[n] if n < 4 else 'OK'
        lines = [
            f'{self.i + 1}/{len(self.images)} {os.path.basename(self._path)}',
            f'Klik: {n}/4 ({hint}) | s=zapisz n=next p=prev r=reset q=quit',
        ]
        y = 24
        for line in lines:
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 26
        return vis

    def run(self) -> None:
        if not self.images:
            print('Brak obrazów.')
            return
        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(_WIN, self._on_mouse)
        if not self._load():
            print('Nie wczytano pierwszego obrazu.')
            return
        while True:
            cv2.imshow(_WIN, self._vis())
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('r'):
                self.clicks = []
                self.blue_roi = None
            if key == ord('b') and len(self.clicks) == 4:
                y = np.asarray(self.clicks, dtype=np.float32)
                assert self._bgr is not None
                self.blue_roi = blue_roi_from_yellow(y, self._bgr.shape[:2], margin_frac=0.12, pad_px=40)
            if key == ord('s') and len(self.clicks) == 4:
                y = np.asarray(self.clicks, dtype=np.float32)
                save_path = self._path
                if self.rotate_deg and self._bgr is not None:
                    fr_dir = os.path.join(self.labels_root, '_frames_rotated')
                    os.makedirs(fr_dir, exist_ok=True)
                    stem = Path(self._path).stem
                    save_path = os.path.join(fr_dir, f'{stem}_r{self.rotate_deg}.jpg')
                    cv2.imwrite(save_path, self._bgr)
                out = save_label(
                    save_path,
                    y,
                    blue_roi=self.blue_roi,
                    root=self.labels_root,
                    rotate_deg=0,
                )
                print(f'Zapisano: {out}')
            if key == ord('n'):
                self.i = min(self.i + 1, len(self.images) - 1)
                self._load()
            if key == ord('p'):
                self.i = max(self.i - 1, 0)
                self._load()
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Zaznacz 4 rogi żółtej siatki (TL, TR, BR, BL).',
        epilog=(
            'Przykład (Twoje nagrania, obrót jak w live):\n'
            '  python3 -m release.annotate_corners dataset/my_capture/ --rotate 180\n'
            '  python3 -m release.annotate_corners --video dataset/my_capture/Droniada_nag1.mov --rotate 180'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        'inputs', nargs='*', default=[],
        help='Opcjonalnie: folder ze zdjęciami lub .mov (w folderze sam wyciągnie klatki)',
    )
    ap.add_argument('--labels-dir', default=None, help='domyślnie dataset/panel_labels')
    ap.add_argument('--rotate', type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument('--video', default=None, help='Jeden plik .mov/.mp4 — wyciąga klatki przed klikaniem')
    ap.add_argument('--every-n', type=int, default=15, help='Co która klatka wideo (15 = co ~0.5 s przy 30 fps)')
    ap.add_argument('--max-frames', type=int, default=80)
    args = ap.parse_args()

    if not args.inputs and not args.video:
        ap.error(
            'Podaj folder (np. dataset/my_capture/) albo --video ścieżka.mov\n'
            'Uruchamiaj z katalogu projektu Droniada (tam gdzie jest folder release/).',
        )

    root = args.labels_dir or os.path.join(_ROOT, 'dataset', 'panel_labels')
    frames_root = os.path.join(root, 'frames')
    images: List[str] = []

    if args.video:
        vpath = _resolve_video_path(args.video, args.inputs)
        out_dir = os.path.join(frames_root, Path(vpath).stem)
        images = _extract_video_frames(
            vpath, out_dir,
            every_n=max(1, args.every_n),
            max_frames=max(1, args.max_frames),
        )
        print(f'Wyciągnięto {len(images)} klatek z {vpath}')
        print(f'Klatki: {out_dir}')
    else:
        images = _collect_images(args.inputs)
        if not images:
            videos = _collect_videos(args.inputs)
            if videos:
                print(f'W folderze nie ma .jpg — są filmy ({len(videos)}). Wyciągam klatki…')
                for vpath in videos:
                    out_dir = os.path.join(frames_root, Path(vpath).stem)
                    part = _extract_video_frames(
                        vpath, out_dir,
                        every_n=max(1, args.every_n),
                        max_frames=max(1, args.max_frames),
                    )
                    images.extend(part)
                    print(f'  {Path(vpath).name} → {len(part)} klatek → {out_dir}')
            else:
                print('Brak obrazów (.jpg/.png) i filmów (.mov/.mp4) w podanej ścieżce.')
                print('Użyj np.: python3 -m release.annotate_corners dataset/my_capture/ --rotate 180')
                return

    Annotator(images, rotate_deg=args.rotate, labels_root=root).run()


if __name__ == '__main__':
    main()
