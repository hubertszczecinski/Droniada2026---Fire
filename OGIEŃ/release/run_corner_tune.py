from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Dict, List, Optional, TextIO
import cv2
import numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from release.camera_source import CameraConfig, CameraSource
from release.corner_probes import ProbeStats, build_probes, format_probe_line, pick_winner, run_all_probes, summary_table
from release.transform import apply_rotate
from release.video_source import VideoConfig, VideoSource

_COLORS = [(0, 255, 0), (0, 200, 255), (255, 200, 0), (255, 0, 200), (200, 255, 200)]

def _log(fh: Optional[TextIO], msg: str) -> None:
    print(msg, flush=True)
    if fh:
        fh.write(msg + '\n')
        fh.flush()

def _draw_winner(vis: np.ndarray, bgr: np.ndarray, winner_name: str, probes: list) -> np.ndarray:
    for pname, fn in probes:
        if pname != winner_name:
            continue
        c = fn(bgr)
        if c is not None and c.shape == (4, 2):
            pts = c.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
            for i, p in enumerate(c.astype(np.int32)):
                cv2.circle(vis, tuple(p), 8, (0, 255, 255), -1)
                cv2.putText(vis, str(i), (int(p[0]) + 6, int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        break
    return vis

def _overlay_hud(vis: np.ndarray, winner: Optional[str], r) -> None:
    lines = [f'BEST: {winner or "none"}', 'q = quit']
    if r is not None and r.found:
        plausible = r.area_ratio < 0.55 and 1.2 <= r.aspect <= 3.5
        tag = 'OK' if r.pnp_ok and r.reproj_px < 12.0 and plausible else 'WEAK'
        lines.insert(1, f'{tag} reproj={r.reproj_px:.1f}px area={r.area_ratio:.2f} asp={r.aspect:.2f}')
        lines.insert(2, f'anchor={r.anchor} white={r.white_t}')
    y = 28
    for line in lines:
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        y += 28

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=int, default=1)
    ap.add_argument('--video', default=None)
    ap.add_argument('--rotate', type=int, default=180, choices=[0, 90, 180, 270])
    ap.add_argument('--interval-ms', type=int, default=800)
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--log-file', default=None)
    ap.add_argument('--summary-every', type=int, default=15)
    ap.add_argument('--max-cycles', type=int, default=0)
    ap.add_argument('--probes', choices=['live', 'full'], default='live', help='live=ta sama detekcja co run_live (img_panel); full=strojenie offline')
    args = ap.parse_args()
    log_fh = open(args.log_file, 'a', encoding='utf-8') if args.log_file else None
    probes = build_probes(args.probes)
    stats: Dict[str, ProbeStats] = {n: ProbeStats() for n, _ in probes}
    k = np.eye(3, dtype=np.float32)
    dist = np.zeros((4, 1), dtype=np.float32)
    _log(log_fh, f'[tune] start probes={len(probes)} rotate={args.rotate} camera={args.camera} video={args.video}')
    _log(log_fh, '[tune] zamknij QuickTime jesli uzywasz kamery')
    cycles = 0
    try:
        if args.video:
            src = VideoSource(VideoConfig(path=os.path.abspath(args.video), rotate_deg=args.rotate, loop=True))
        else:
            src = CameraSource(CameraConfig(device_index=args.camera))
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
                        cv2.imshow('corner_tune', bgr)
                        if cv2.waitKey(1) & 255 == ord('q'):
                            break
                    continue
                last = now
                cycles += 1
                h, w = bgr.shape[:2]
                k[0, 0] = 1000.0
                k[1, 1] = 1000.0
                k[0, 2] = w / 2.0
                k[1, 2] = h / 2.0
                results = run_all_probes(bgr, k, dist, probes)
                winner = pick_winner(results)
                _log(log_fh, f'[{fid}] cycle={cycles}')
                for r in sorted(results, key=lambda x: (not x.pnp_ok, x.reproj_px)):
                    _log(log_fh, format_probe_line(r))
                    stats[r.name].record(r, r.name == winner)
                _log(log_fh, f'  >>> BEST={winner or "none"}')
                if args.preview:
                    vis = bgr.copy()
                    win_r = next((x for x in results if x.name == winner), None)
                    if winner:
                        vis = _draw_winner(vis, bgr, winner, probes)
                    _overlay_hud(vis, winner, win_r)
                    cv2.imshow('corner_tune', vis)
                    if cv2.waitKey(1) & 255 == ord('q'):
                        break
                if args.summary_every > 0 and cycles % args.summary_every == 0:
                    _log(log_fh, summary_table(stats))
                if args.max_cycles > 0 and cycles >= args.max_cycles:
                    break
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        _log(log_fh, f'[tune] error {e}')
    finally:
        _log(log_fh, summary_table(stats))
        if log_fh:
            log_fh.close()
        cv2.destroyAllWindows()
        _log(None, f'[tune] stop cycles={cycles}')

if __name__ == '__main__':
    main()
