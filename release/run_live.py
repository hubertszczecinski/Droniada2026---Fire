from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Optional, TextIO
import cv2
import numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from release.camera_source import CameraConfig, CameraSource
from release.panel_runtime import PanelConfig, PanelRuntime
from release.pose_runtime import PoseConfig, PoseRuntime
from release.transform import apply_rotate
from module_geom.camera import resolve_intrinsics

def _log_line(fh: Optional[TextIO], msg: str) -> None:
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + '\n')
        fh.flush()

def _draw_pose_overlay(
    img: np.ndarray,
    po,
    *,
    ok: bool,
) -> np.ndarray:
    from module_pose.panel_stand import STAND_LABEL_PL

    vis = img.copy()
    if po is not None and po.corners_px is not None and po.corners_px.shape == (4, 2):
        color = (0, 255, 0) if ok else (0, 180, 255)
        pts = po.corners_px.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, color, 2, cv2.LINE_AA)
    else:
        cv2.putText(vis, 'brak panelu', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if po is not None and po.ok:
        pl = STAND_LABEL_PL.get(po.panel_angle_category, po.panel_angle_category)
        lines = [
            f'Modul A OK  reproj={po.reproj_mean_px:.1f}px',
            f'd={po.distance_m:.2f}m  {pl} ({po.report_angle_deg}°)',
            f'roll={po.roll_deg:.0f}  pitch={po.pitch_deg:.0f}  yaw={po.yaw_deg:.0f}',
        ]
        y = 28
        for line in lines:
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2, cv2.LINE_AA)
            y += 22
    elif po is not None:
        reason = str(po.meta.get('reason', po.meta.get('fail', '?')))
        cv2.putText(vis, f'A FAIL: {reason}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
    return vis

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=int, default=1)
    ap.add_argument('--width', type=int, default=0)
    ap.add_argument('--height', type=int, default=0)
    ap.add_argument('--mode', default='both', choices=['pose', 'panel', 'both'])
    ap.add_argument('--interval-ms', type=int, default=500, help='min czas miedzy analizami (ms)')
    ap.add_argument('--panel-id', default='A')
    ap.add_argument('--log-file', default=None)
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--max-frames', type=int, default=0, help='0 = bez limitu (Ctrl+C)')
    ap.add_argument('--rotate', type=int, default=180, choices=[0, 90, 180, 270], help='obrot kadru z kamery (Continuity czesto 180)')
    ap.add_argument(
        '--camera-profile',
        default='tarot_t10x_2a:wide',
        help='profil intrinsics (tarot_t10x_2a:wide|mid|tele); pusty = fx=1000',
    )
    ap.add_argument('--zoom-ratio', type=float, default=1.0)
    ap.add_argument('--no-stabilize', action='store_true', help='wylacz EMA rogow YOLO')
    args = ap.parse_args()
    log_fh = open(args.log_file, 'a', encoding='utf-8') if args.log_file else None
    profile = args.camera_profile.strip() or None
    cam_calib = os.path.join(_ROOT, 'config', 'camera_calibration.npz')
    if not os.path.isfile(cam_calib):
        cam_calib = None
    pose_rt = (
        PoseRuntime(PoseConfig(
            corner_source='yolo_pose',
            yolo_pose_use_tracker=not args.no_stabilize,
        ))
        if args.mode in ('pose', 'both') else None
    )
    panel_rt = PanelRuntime(PanelConfig(panel_id=args.panel_id, angle_source='rmat_linear')) if args.mode in ('panel', 'both') else None
    empty_det: list = []
    _log_line(log_fh, f'[live] start camera={args.camera} mode={args.mode} rotate={args.rotate} profile={profile} interval_ms={args.interval_ms}')
    _log_line(log_fh, '[live] zamknij QuickTime — kamera moze byc tylko w jednym programie')
    if pose_rt is not None:
        _log_line(log_fh, '[live] Modul A: YOLO-Pose + PnP (zielony trapez)')
    processed = 0
    try:
        with CameraSource(CameraConfig(device_index=args.camera, width=args.width, height=args.height)) as cam:
            last_proc = 0.0
            while True:
                ok, bgr, fid = cam.read()
                if not ok:
                    _log_line(log_fh, f'[{fid}] capture_fail')
                    if args.preview:
                        cv2.imshow('droniada_live', bgr if bgr is not None else np.zeros((480, 640, 3), np.uint8))
                        if cv2.waitKey(1) & 255 == ord('q'):
                            break
                    continue
                if args.rotate:
                    bgr = apply_rotate(bgr, args.rotate)
                now = time.monotonic()
                if (now - last_proc) * 1000.0 < args.interval_ms:
                    if args.preview:
                        cv2.imshow('droniada_live', bgr)
                        if cv2.waitKey(1) & 255 == ord('q'):
                            break
                    continue
                last_proc = now
                processed += 1
                h, w = bgr.shape[:2]
                k, dist, _intr = resolve_intrinsics(
                    (h, w),
                    profile=profile,
                    zoom_ratio=float(args.zoom_ratio),
                    calib_path=cam_calib,
                )
                vis = bgr
                if pose_rt is not None:
                    po = pose_rt.process_bgr(bgr, fid, det=empty_det, k=k, dist=dist, record=False)
                    if po.ok:
                        integ = po.to_integration_dict()
                        _log_line(
                            log_fh,
                            f'[{fid}] POSE ok roll={po.roll_deg:.1f} pitch={po.pitch_deg:.1f} '
                            f'yaw={po.yaw_deg:.1f} dist={po.distance_m:.2f}m '
                            f'stand={po.panel_angle_category} ({po.report_angle_deg}°) '
                            f'reproj={po.reproj_mean_px:.2f} conf={po.confidence:.2f}',
                        )
                        _log_line(log_fh, f'[{fid}] POSE dict {integ}')
                    else:
                        reason = po.meta.get('reason', po.meta.get('err', '?'))
                        _log_line(log_fh, f'[{fid}] POSE fail reason={reason} method={po.method}')
                    if args.preview:
                        vis = _draw_pose_overlay(vis, po, ok=po.ok)
                if panel_rt is not None:
                    pan = panel_rt.process_bgr(bgr, fid, det=empty_det, record=False)
                    if pan.ok:
                        _log_line(log_fh, f'[{fid}] PANEL ok panel={pan.panel_id} angle={pan.report_angle_deg} cat={pan.panel_angle_category} cards={len(pan.predictions)} grid_ok={pan.grid_xy_reliable}')
                        for line in pan.report_lines:
                            _log_line(log_fh, f'[{fid}]   {line}')
                    else:
                        err = pan.meta.get('err', '?')
                        _log_line(log_fh, f'[{fid}] PANEL fail err={err}')
                if args.preview:
                    cv2.imshow('droniada_live', vis)
                    if cv2.waitKey(1) & 255 == ord('q'):
                        break
                if args.max_frames > 0 and processed >= args.max_frames:
                    break
    except KeyboardInterrupt:
        _log_line(log_fh, '[live] przerwano Ctrl+C')
    except RuntimeError as e:
        _log_line(log_fh, f'[live] error {e}')
    finally:
        if log_fh is not None:
            log_fh.close()
        cv2.destroyAllWindows()
        _log_line(None, f'[live] stop processed={processed}')

if __name__ == '__main__':
    main()
