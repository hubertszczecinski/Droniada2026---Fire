"""
Live test modułu B (line_grid v3 baseline) — kamera + kolorowe kartki na panelu.

Zamknij QuickTime / inne apki używające kamery. Podgląd: klawisz q.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, TextIO, Tuple

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pipeline_competition as pc
from module_panel.analyze import analyze_panel_image, detect_panel_corners_for_module_b
from module_panel.report import predictions_to_report_lines
from module_panel.warp import warp_panel_rect
from release.camera_source import CameraConfig, CameraSource
from release.live_card_detect import detect_cards_live
from release.transform import apply_rotate

_BGR_BY_COLOR = {
    'CZERWONA': (40, 40, 230),
    'ZIELONA': (50, 190, 50),
    'NIEBIESKA': (230, 120, 40),
    'ZOLTA': (20, 220, 240),
    'FIOLETOWA': (200, 60, 200),
    'POMARANCZOWA': (30, 130, 240),
    'UNKNOWN': (180, 180, 180),
}


def _log(fh: Optional[TextIO], msg: str) -> None:
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + '\n')
        fh.flush()


def _default_intrinsics(w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    k = np.array([[1000.0, 0.0, w / 2.0], [0.0, 1000.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    dist = np.zeros((4, 1), dtype=np.float32)
    return k, dist


def _draw_status_block(
    vis: np.ndarray,
    *,
    reliable: bool,
    corner_source: str,
    xy_backend: str,
    reproj: float,
    n_cards: int,
    angle: int,
    category: str,
) -> None:
    lines = [
        ('Droniada LIVE panel B (line_grid v3)', (30, 30, 30)),
        (f'reliable={"TAK" if reliable else "NIE"}  reproj={reproj:.1f}px', (0, 140, 0) if reliable else (0, 0, 200)),
        (f'corner={corner_source}  xy={xy_backend}', (50, 50, 50)),
        (f'kat={angle}  cat={category}  det={n_cards}', (50, 50, 50)),
        ('q = wyjscie', (100, 100, 100)),
    ]
    y = 12
    for text, color in lines:
        cv2.putText(vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
        y += 24


def _draw_panel_overlay(
    vis: np.ndarray,
    corners: Optional[np.ndarray],
    det: List[Tuple[int, float, float, float, float]],
    preds: List[dict],
    *,
    ok_corners: bool,
) -> None:
    h, w = vis.shape[:2]
    if corners is not None and corners.shape == (4, 2):
        color = (0, 255, 255) if ok_corners else (0, 180, 255)
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, color, 3)
        for idx, p in enumerate(corners.astype(np.int32)):
            cv2.circle(vis, tuple(p), 6, color, -1)
            cv2.putText(vis, str(idx), (int(p[0]) + 6, int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for idx, (cls_id, cx_n, cy_n, bw_n, bh_n) in enumerate(det):
        name = pc.CLASS_TO_COLOR.get(cls_id, 'UNKNOWN')
        bgr = _BGR_BY_COLOR.get(name, _BGR_BY_COLOR['UNKNOWN'])
        cx = int(round(cx_n * w))
        cy = int(round(cy_n * h))
        bw = max(8, int(round(bw_n * w)))
        bh = max(8, int(round(bh_n * h)))
        p1 = (max(0, cx - bw // 2), max(0, cy - bh // 2))
        p2 = (min(w - 1, cx + bw // 2), min(h - 1, cy + bh // 2))
        cv2.rectangle(vis, p1, p2, bgr, 2)
        pred = preds[idx] if idx < len(preds) else None
        if pred:
            label = f"{name[:3]} X{pred['x']} Y{pred['y']}"
            cv2.putText(vis, label, (p1[0], max(16, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)


def _draw_warped_inset(warped: np.ndarray, preds: List[dict], warped_dets: List[dict]) -> np.ndarray:
    thumb = warped.copy()
    h, w = thumb.shape[:2]
    for i in range(11):
        x = int(round(i * w / 10.0))
        y = int(round(i * h / 10.0))
        cv2.line(thumb, (x, 0), (x, h - 1), (200, 200, 200), 1)
        cv2.line(thumb, (0, y), (w - 1, y), (200, 200, 200), 1)
    for d in warped_dets:
        p = (int(round(d['warp_cx'])), int(round(d['warp_cy'])))
        bgr = _BGR_BY_COLOR.get(str(d['color']), _BGR_BY_COLOR['UNKNOWN'])
        cv2.circle(thumb, p, 10, bgr, -1)
    for p in preds:
        gx = int(round((float(p['x']) - 0.5) * w / 10.0))
        gy = int(round((float(p['y']) - 0.5) * h / 10.0))
        cv2.drawMarker(thumb, (gx, gy), (0, 0, 0), cv2.MARKER_CROSS, 16, 2)
    return thumb


def _compose_preview(main: np.ndarray, warped_thumb: np.ndarray, target_w: int = 1280) -> np.ndarray:
    h, w = main.shape[:2]
    tw = warped_thumb.shape[1]
    scale = min(1.0, target_w / float(w + tw + 24))
    if scale < 1.0:
        main = cv2.resize(main, (int(w * scale), int(h * scale)))
        warped_thumb = cv2.resize(
            warped_thumb,
            (int(warped_thumb.shape[1] * scale), int(warped_thumb.shape[0] * scale)),
        )
    gap = np.full((main.shape[0], 12, 3), 40, dtype=np.uint8)
    return np.hstack([main, gap, warped_thumb])


def main() -> None:
    ap = argparse.ArgumentParser(description='Live test modulu B — line_grid v3 + detekcja kolorow na siatce')
    ap.add_argument('--camera', type=int, default=1, help='indeks kamery (0 lub 1 na Macu)')
    ap.add_argument('--width', type=int, default=0)
    ap.add_argument('--height', type=int, default=0)
    ap.add_argument('--rotate', type=int, default=180, choices=[0, 90, 180, 270])
    ap.add_argument('--interval-ms', type=int, default=450, help='min. odstep miedzy analizami (ms)')
    ap.add_argument('--panel-id', default='A')
    ap.add_argument('--xy-mode', default='line_grid', choices=['grid_geom', 'grid_geom_white', 'warp_grid', 'geom_grid', 'line_grid'])
    ap.add_argument('--angle-source', default='rmat_linear', choices=['rmat_linear', 'rmat_theta', 'geom', 'pnp'])
    ap.add_argument('--require-reliable', action='store_true', help='raport kart tylko gdy grid_xy_reliable')
    ap.add_argument('--no-color-detect', action='store_true', help='tylko rogi/siatka, bez skanowania kolorow')
    ap.add_argument('--preview', action='store_true', help='okno podgladu (zalecane)')
    ap.add_argument('--preview-width', type=int, default=1280)
    ap.add_argument('--log-file', default=None)
    ap.add_argument('--save-dir', default=None, help='zapisz klatki gdy reliable=TAK')
    ap.add_argument('--max-frames', type=int, default=0)
    args = ap.parse_args()

    log_fh = open(args.log_file, 'a', encoding='utf-8') if args.log_file else None
    calib_path = os.path.join(_ROOT, 'module_panel', 'data', 'angle_linear_rmat.json')
    if not os.path.isfile(calib_path):
        calib_path = None
    cam_calib = os.path.join(_ROOT, 'config', 'camera_calibration.npz')
    if not os.path.isfile(cam_calib):
        cam_calib = None

    _log(log_fh, f'[live_panel] start camera={args.camera} xy_mode={args.xy_mode} rotate={args.rotate}')
    _log(log_fh, '[live_panel] zamknij QuickTime — kamera tylko w jednym programie')
    _log(log_fh, '[live_panel] zawies kolorowe kartki na siatce 10x10')

    processed = 0
    try:
        with CameraSource(CameraConfig(device_index=args.camera, width=args.width, height=args.height)) as cam:
            last_proc = 0.0
            while True:
                ok_cap, bgr, fid = cam.read()
                if not ok_cap or bgr is None:
                    _log(log_fh, f'[{fid}] capture_fail')
                    if args.preview:
                        cv2.imshow('droniada_live_panel', np.zeros((480, 640, 3), np.uint8))
                        if cv2.waitKey(1) & 255 == ord('q'):
                            break
                    continue

                if args.rotate:
                    bgr = apply_rotate(bgr, args.rotate)

                now = time.monotonic()
                if (now - last_proc) * 1000.0 < args.interval_ms:
                    if args.preview:
                        cv2.imshow('droniada_live_panel', bgr)
                        if cv2.waitKey(1) & 255 == ord('q'):
                            break
                    continue
                last_proc = now
                processed += 1

                h, w = bgr.shape[:2]
                k, dist = _default_intrinsics(w, h)
                det: List[Tuple[int, float, float, float, float]] = []
                warped_dets: List[dict] = []
                corners_px, corner_src = detect_panel_corners_for_module_b(
                    bgr,
                    det,
                    prefer_line_grid=(args.xy_mode == 'line_grid'),
                    k=k,
                    dist=dist,
                )

                if corners_px is None:
                    pan_meta = {'err': 'no_corners'}
                    warped = bgr.copy()
                    preds: List[dict] = []
                    reliable = False
                    reproj = 999.0
                    xy_back = '-'
                    angle = 0
                    category = 'horizontal'
                else:
                    warped, h_mat = warp_panel_rect(bgr, corners_px)
                    if not args.no_color_detect:
                        det, warped_dets = detect_cards_live(bgr, corners_px, h_mat, warped)

                    pan = analyze_panel_image(
                        bgr,
                        det,
                        k=k,
                        dist=dist,
                        xy_mode=args.xy_mode,
                        angle_source=args.angle_source,
                        angle_calibration_path=calib_path,
                        camera_calib_path=cam_calib,
                        allowed_orbit_steps=None,
                    )
                    pan_meta = pan.meta
                    warped = pan.warped_bgr
                    preds = list(pan.predictions)
                    reliable = bool(pan.meta.get('grid_xy_reliable', False))
                    reproj = float(pan.meta.get('reproj_mean_px', 999.0))
                    corner_src = str(pan.meta.get('corner_source', corner_src))
                    xy_back = str(pan.meta.get('xy_backend_selected', pan.meta.get('xy_source', '?')))
                    angle = int(pan.report_angle_deg)
                    category = str(pan.panel_angle_category)

                if args.require_reliable and not reliable:
                    preds = []

                lines = predictions_to_report_lines(args.panel_id, angle, preds)

                if pan_meta.get('err') == 'no_corners':
                    _log(log_fh, f'[{fid}] brak panelu')
                else:
                    _log(log_fh, f'[{fid}] reliable={reliable} reproj={reproj:.2f} corner={corner_src} xy={xy_back} det={len(det)} preds={len(preds)}')
                    for line in lines:
                        _log(log_fh, f'[{fid}]   {line}')

                vis = bgr.copy()
                ok_corners = pan_meta.get('err') != 'no_corners'
                _draw_panel_overlay(vis, corners_px, det, preds, ok_corners=ok_corners)
                _draw_status_block(
                    vis,
                    reliable=reliable,
                    corner_source=corner_src,
                    xy_backend=xy_back,
                    reproj=reproj,
                    n_cards=len(preds),
                    angle=angle,
                    category=category,
                )

                if args.preview:
                    thumb = _draw_warped_inset(warped, preds, warped_dets)
                    show = _compose_preview(vis, thumb, target_w=args.preview_width)
                    cv2.imshow('droniada_live_panel', show)
                    if cv2.waitKey(1) & 255 == ord('q'):
                        break

                if args.save_dir and reliable and preds:
                    os.makedirs(args.save_dir, exist_ok=True)
                    ts = datetime.now().strftime('%H%M%S')
                    path = os.path.join(args.save_dir, f'{ts}_{fid}.png')
                    cv2.imwrite(path, vis)

                if args.max_frames > 0 and processed >= args.max_frames:
                    break
    except KeyboardInterrupt:
        _log(log_fh, '[live_panel] Ctrl+C')
    except RuntimeError as e:
        _log(log_fh, f'[live_panel] error {e}')
    finally:
        if log_fh is not None:
            log_fh.close()
        cv2.destroyAllWindows()
        _log(None, f'[live_panel] stop processed={processed}')


if __name__ == '__main__':
    main()
