from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Dict, Iterable
import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from release.alignment_pipelines import PIPELINES, AlignmentResult, run_all_pipelines, run_pipeline
from release.camera_source import CameraConfig, CameraSource
from release.transform import apply_rotate
from release.video_source import VideoConfig, VideoSource


_COLORS = {
    'hsv_panel': (0, 255, 0),
    'dark_blob': (0, 180, 255),
    'white_grid': (255, 180, 0),
    'current_scored': (255, 0, 255),
    'hybrid': (0, 255, 255),
}


class ResultStabilizer:
    def __init__(self, alpha: float, hold_frames: int) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.hold_frames = max(0, int(hold_frames))
        self._last: Dict[str, AlignmentResult] = {}
        self._misses: Dict[str, int] = {}

    def update(self, result: AlignmentResult) -> AlignmentResult:
        prev = self._last.get(result.name)
        if not result.ok or result.quad is None:
            missed = self._misses.get(result.name, 0) + 1
            self._misses[result.name] = missed
            if prev is not None and missed <= self.hold_frames:
                held = AlignmentResult(
                    name=result.name,
                    ok=True,
                    confidence=max(0.0, prev.confidence * 0.85),
                    quad=prev.quad.copy() if prev.quad is not None else None,
                    center_px=prev.center_px,
                    offset_px=prev.offset_px,
                    angle_deg=prev.angle_deg,
                    area_ratio=prev.area_ratio,
                    aspect=prev.aspect,
                    meta={**prev.meta, 'stabilized': 'held_after_miss'},
                )
                self._last[result.name] = held
                return held
            return result
        self._misses[result.name] = 0
        if prev is None or prev.quad is None or not prev.ok:
            self._last[result.name] = result
            return result
        q = (1.0 - self.alpha) * prev.quad.astype(np.float32) + self.alpha * result.quad.astype(np.float32)
        smoothed = _copy_with_quad(result, q, meta={**result.meta, 'stabilized': 'ema'})
        self._last[result.name] = smoothed
        return smoothed


def _copy_with_quad(result: AlignmentResult, quad: np.ndarray, *, meta: Dict[str, float | str]) -> AlignmentResult:
    if result.quad is None:
        return result
    q = quad.astype(np.float32)
    cx, cy = q.mean(axis=0)
    h_est = max(1.0, float(np.linalg.norm(q[0] - q[3])))
    w_est = max(1.0, float(np.linalg.norm(q[0] - q[1])))
    aspect = max(w_est, h_est) / max(1.0, min(w_est, h_est))
    # Offset needs image dimensions; recover them from previous center/offset.
    img_cx = result.center_px[0] - result.offset_px[0]
    img_cy = result.center_px[1] - result.offset_px[1]
    area = float(cv2.contourArea(q.astype(np.float32)))
    img_area = max(1.0, float((2.0 * img_cx) * (2.0 * img_cy)))
    return AlignmentResult(
        name=result.name,
        ok=True,
        confidence=result.confidence,
        quad=q,
        center_px=(float(cx), float(cy)),
        offset_px=(float(cx - img_cx), float(cy - img_cy)),
        angle_deg=result.angle_deg,
        area_ratio=float(area / img_area),
        aspect=float(aspect),
        meta=meta,
    )


def _draw_result(vis: np.ndarray, result: AlignmentResult, *, thickness: int = 2) -> None:
    color = _COLORS.get(result.name, (200, 200, 200))
    if result.ok and result.quad is not None:
        pts = result.quad.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, color, thickness)
        cx, cy = result.center_px
        cv2.circle(vis, (int(cx), int(cy)), 5, color, -1)
        label = f'{result.name} conf={result.confidence:.2f} area={result.area_ratio:.2f} asp={result.aspect:.2f}'
        cv2.putText(vis, label, tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    else:
        cv2.putText(vis, f'{result.name}: MISS', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _draw_hud(vis: np.ndarray, results: Iterable[AlignmentResult], frame_id: str, pipeline: str) -> None:
    cv2.putText(vis, f'{frame_id} pipeline={pipeline} q=quit', (12, vis.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    y = 24
    for r in results:
        status = 'OK' if r.ok else 'MISS'
        line = f'{r.name:14s} {status:4s} conf={r.confidence:.2f} off=({r.offset_px[0]:.0f},{r.offset_px[1]:.0f})'
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLORS.get(r.name, (230, 230, 230)), 1)
        y += 20


def _log_result(frame_id: str, result: AlignmentResult) -> None:
    if not result.ok:
        print(f'[{frame_id}] {result.name} MISS', flush=True)
        return
    print(
        f'[{frame_id}] {result.name} ok conf={result.confidence:.2f} '
        f'center=({result.center_px[0]:.0f},{result.center_px[1]:.0f}) '
        f'offset=({result.offset_px[0]:.0f},{result.offset_px[1]:.0f}) '
        f'angle={result.angle_deg:.1f} area={result.area_ratio:.2f} asp={result.aspect:.2f} '
        f'{result.meta}',
        flush=True,
    )


def _log_control(frame_id: str, result: AlignmentResult) -> None:
    if not result.ok:
        print(f'[{frame_id}] CONTROL ok=0', flush=True)
        return
    print(
        f'[{frame_id}] CONTROL ok=1 conf={result.confidence:.2f} '
        f'offset_x={result.offset_px[0]:.0f} offset_y={result.offset_px[1]:.0f} '
        f'area={result.area_ratio:.3f} angle={result.angle_deg:.1f}',
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=int, default=1)
    ap.add_argument('--video', default=None)
    ap.add_argument('--rotate', type=int, default=180, choices=[0, 90, 180, 270])
    ap.add_argument('--width', type=int, default=0)
    ap.add_argument('--height', type=int, default=0)
    ap.add_argument('--pipeline', choices=[*PIPELINES.keys(), 'all'], default='hybrid')
    ap.add_argument('--interval-ms', type=int, default=300)
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--no-loop', action='store_true')
    ap.add_argument('--no-stabilize', action='store_true')
    ap.add_argument('--smooth-alpha', type=float, default=0.35)
    ap.add_argument('--hold-frames', type=int, default=3)
    ap.add_argument('--control-output', action='store_true', help='print compact alignment line for flight control')
    ap.add_argument('--save-dir', default=None, help='optional directory for sampled debug frames')
    ap.add_argument('--save-every', type=int, default=15, help='save every N processed frames when --save-dir is set')
    args = ap.parse_args()

    print(f'[alignment] start pipeline={args.pipeline} rotate={args.rotate} camera={args.camera} video={args.video}', flush=True)
    processed = 0
    src = None
    stabilizer = None if args.no_stabilize else ResultStabilizer(args.smooth_alpha, args.hold_frames)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    try:
        if args.video:
            src = VideoSource(VideoConfig(path=os.path.abspath(args.video), rotate_deg=args.rotate, loop=not args.no_loop))
        else:
            src = CameraSource(CameraConfig(device_index=args.camera, width=args.width, height=args.height))
        with src:
            last = 0.0
            while True:
                ok, bgr, fid = src.read()
                if not ok:
                    if args.video:
                        break
                    continue
                if args.rotate and args.video is None:
                    bgr = apply_rotate(bgr, args.rotate)
                now = time.monotonic()
                if (now - last) * 1000.0 < args.interval_ms:
                    if args.preview:
                        cv2.imshow('alignment_test', bgr)
                        if cv2.waitKey(1) & 255 == ord('q'):
                            break
                    continue
                last = now
                processed += 1
                if args.pipeline == 'all':
                    results = run_all_pipelines(bgr)
                else:
                    results = [run_pipeline(args.pipeline, bgr)]
                if stabilizer is not None:
                    results = [stabilizer.update(r) for r in results]
                for r in results:
                    _log_result(fid, r)
                if args.control_output and results:
                    best = next((r for r in results if r.name == 'hybrid'), results[0])
                    _log_control(fid, best)
                if args.preview:
                    vis = bgr.copy()
                    for r in results:
                        _draw_result(vis, r, thickness=3 if r.name == 'hybrid' else 2)
                    _draw_hud(vis, results, fid, args.pipeline)
                    if args.save_dir and args.save_every > 0 and processed % args.save_every == 0:
                        cv2.imwrite(os.path.join(args.save_dir, f'{fid}_{args.pipeline}.jpg'), vis)
                    cv2.imshow('alignment_test', vis)
                    if cv2.waitKey(1) & 255 == ord('q'):
                        break
                if args.max_frames > 0 and processed >= args.max_frames:
                    break
    except KeyboardInterrupt:
        print('[alignment] Ctrl+C', flush=True)
    finally:
        cv2.destroyAllWindows()
        print(f'[alignment] stop processed={processed}', flush=True)


if __name__ == '__main__':
    main()
