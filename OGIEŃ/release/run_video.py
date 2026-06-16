from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Optional, TextIO
import cv2
import numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from release.panel_runtime import PanelConfig, PanelRuntime
from release.pose_runtime import PoseConfig, PoseRuntime
from release.video_source import VideoConfig, VideoSource

def _log_line(fh: Optional[TextIO], msg: str) -> None:
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + '\n')
        fh.flush()

def _draw_pose_overlay(img: np.ndarray, corners: Optional[np.ndarray]) -> np.ndarray:
    vis = img.copy()
    if corners is not None and corners.shape == (4, 2):
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
    return vis

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--rotate', type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument('--mode', default='both', choices=['pose', 'panel', 'both'])
    ap.add_argument('--interval-ms', type=int, default=500)
    ap.add_argument('--panel-id', default='A')
    ap.add_argument('--log-file', default=None)
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--no-loop', action='store_true')
    ap.add_argument('--max-frames', type=int, default=0)
    args = ap.parse_args()
    path = os.path.abspath(args.video)
    if not os.path.isfile(path):
        print(json.dumps({'ok': False, 'reason': 'no_video', 'path': path}))
        return
    log_fh = open(args.log_file, 'a', encoding='utf-8') if args.log_file else None
    pose_rt = PoseRuntime(PoseConfig()) if args.mode in ('pose', 'both') else None
    panel_rt = PanelRuntime(PanelConfig(panel_id=args.panel_id, angle_source='rmat_linear')) if args.mode in ('panel', 'both') else None
    empty_det: list = []
    _log_line(log_fh, f'[video] {path} rotate={args.rotate} mode={args.mode}')
    processed = 0
    try:
        with VideoSource(VideoConfig(path=path, rotate_deg=args.rotate, loop=not args.no_loop)) as vid:
            last_proc = 0.0
            while True:
                ok, bgr, fid = vid.read()
                if not ok:
                    break
                if args.preview:
                    delay = max(1, int(1000 / 30))
                    key = cv2.waitKey(delay) & 255
                    if key == ord('q'):
                        break
                now = time.monotonic()
                if (now - last_proc) * 1000.0 < args.interval_ms:
                    if args.preview:
                        cv2.imshow('droniada_video', bgr)
                    continue
                last_proc = now
                processed += 1
                vis = bgr
                if pose_rt is not None:
                    po = pose_rt.process_bgr(bgr, fid, det=empty_det, record=False)
                    if po.ok:
                        _log_line(log_fh, f'[{fid}] POSE ok roll={po.roll_deg:.1f} pitch={po.pitch_deg:.1f} yaw={po.yaw_deg:.1f} dist={po.distance_m:.2f}m reproj={po.reproj_mean_px:.2f} conf={po.confidence:.2f}')
                    else:
                        reason = po.meta.get('reason', po.meta.get('err', '?'))
                        _log_line(log_fh, f'[{fid}] POSE fail reason={reason}')
                    if args.preview and po.ok:
                        vis = _draw_pose_overlay(vis, pose_rt._corners_buf)
                if panel_rt is not None:
                    pan = panel_rt.process_bgr(bgr, fid, det=empty_det, record=False)
                    if pan.ok:
                        _log_line(log_fh, f'[{fid}] PANEL ok angle={pan.report_angle_deg} cat={pan.panel_angle_category} cards={len(pan.predictions)} grid_ok={pan.grid_xy_reliable}')
                    else:
                        _log_line(log_fh, f'[{fid}] PANEL fail err={pan.meta.get("err", "?")}')
                if args.preview:
                    cv2.imshow('droniada_video', vis)
                    if cv2.waitKey(1) & 255 == ord('q'):
                        break
                if args.max_frames > 0 and processed >= args.max_frames:
                    break
    except KeyboardInterrupt:
        _log_line(log_fh, '[video] Ctrl+C')
    except RuntimeError as e:
        _log_line(log_fh, f'[video] error {e}')
    finally:
        if log_fh is not None:
            log_fh.close()
        cv2.destroyAllWindows()
        _log_line(None, f'[video] stop processed={processed}')

if __name__ == '__main__':
    main()
