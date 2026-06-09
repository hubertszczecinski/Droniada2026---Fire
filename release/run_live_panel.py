"""
Live test modułu B (line_grid v3 baseline) — kamera + kolorowe kartki na panelu.

Zamknij QuickTime / inne apki używające kamery. Podgląd: klawisz q.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TextIO, Tuple

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pipeline_competition as pc
from module_panel.analyze import analyze_panel_image
from release.live_corners import (
    DEFAULT_LIVE_CORNER_MODE,
    LIVE_MAX_REPROJ_RELIABLE_PX,
    _LIVE_CORNER_MODES,
    _TRACKER,
    detect_corners_live,
    probe_all_corner_candidates,
    probe_hybrid_corner_candidates,
    probe_line_grid_corner_candidates,
    probe_line_grid_roi_candidates,
)
from release.live_debug import (
    append_session_index,
    draw_candidates_board,
    new_session_dir,
    save_live_frame_bundle,
)
from module_panel.report import predictions_to_report_lines
from module_panel.warp import warp_panel_rect
from module_geom.camera import resolve_intrinsics
from release.camera_source import CameraConfig, CameraSource
from release.cxy_latch import CxyLatch
from release.cxy_latch_preview import (
    compose_latch_dashboard,
    draw_warped_panel_preview,
    save_cxy_latch_artifacts,
)
from release.live_card_detect import detect_cards_live, warped_dets_to_predictions
from release.live_dashboard import (
    LiveSnapshotStore,
    compose_unified_dashboard,
    new_dashboard_session,
    show_dashboard_window,
    snapshot_frame_eligible,
)
from release.snapshot_cxy_competition import (
    predictions_from_report_lines,
    update_session_competition,
)
from release.snapshot_cell_color import load_panel_color_layout
from release.pose_runtime import PoseConfig, PoseFrameOutput, PoseRuntime
from release.transform import apply_rotate
from release.video_source import VideoConfig, VideoSource

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


def _refresh_snapshot_store_limits(store: Optional[LiveSnapshotStore], gate: dict, args: argparse.Namespace) -> None:
    if store is None:
        return
    store.min_stable_frames = max(1, int(gate.get('snapshot_min_stable', args.snapshot_min_stable)))
    store.max_reproj = float(gate.get('snapshot_max_reproj', args.snapshot_max_reproj))


def _snapshot_competition_thresholds(
    args: argparse.Namespace,
    *,
    n_snapshots: int,
) -> Tuple[int, float]:
    """Progi głosowania CXY — przy panelu WWW i 5 migawkach: większość (3/5)."""
    min_votes = int(args.snapshot_competition_min_votes)
    min_ratio = float(args.snapshot_competition_min_ratio)
    if int(args.web_port) > 0 and min_ratio <= 0.0:
        min_ratio = 0.5
    if int(args.snapshot_competition_min_votes) == 2 and n_snapshots >= 5:
        min_votes = max(min_votes, 3)
    return min_votes, min_ratio


def _default_intrinsics(w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    k = np.array([[1000.0, 0.0, w / 2.0], [0.0, 1000.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    dist = np.zeros((4, 1), dtype=np.float32)
    return k, dist


def _analyze_cxy_from_corners(
    work_bgr: np.ndarray,
    corners_px: np.ndarray,
    corner_src: str,
    k: np.ndarray,
    dist: np.ndarray,
    *,
    panel_id: str,
    xy_mode: str,
    angle_source: str,
    calib_path: Optional[str],
    cam_calib: Optional[str],
    max_reproj_px: float,
    min_homography_inliers: int,
    no_color_detect: bool,
) -> Tuple[object, List[dict], List, List[dict], float, bool, str, int, str, dict]:
    """Pełna ścieżka: warp + (opcjonalnie) detekcja kolorów + analyze_panel_image → CXY + reliable v2."""
    from release.panel_black import calibrate_panel_black_from_corners

    warped, h_mat = warp_panel_rect(work_bgr, corners_px)
    black_th = calibrate_panel_black_from_corners(work_bgr, corners_px)
    det: List[Tuple[int, float, float, float, float]] = []
    warped_dets: List[dict] = []
    if not no_color_detect:
        det, warped_dets = detect_cards_live(
            work_bgr, corners_px, h_mat, warped, black_thresholds=black_th,
        )
    pan = analyze_panel_image(
        work_bgr,
        det,
        k=k,
        dist=dist,
        xy_mode=xy_mode,
        angle_source=angle_source,
        angle_calibration_path=calib_path,
        camera_calib_path=cam_calib,
        allowed_orbit_steps=None,
        min_homography_inliers=min_homography_inliers,
        max_reproj_px=max_reproj_px,
        corners_px=corners_px,
        corner_source=corner_src,
    )
    preds = list(pan.predictions)
    if warped_dets:
        preds = warped_dets_to_predictions(warped_dets)
        pan.meta['xy_source'] = 'live_warp_grid_hsv'
        pan.meta['xy_backend_selected'] = 'live_warp_grid_hsv'
    reliable = bool(pan.meta.get('grid_xy_reliable', False))
    reproj = float(pan.meta.get('reproj_mean_px', 999.0))
    xy_back = str(pan.meta.get('xy_backend_selected', pan.meta.get('xy_source', '?')))
    angle = int(pan.report_angle_deg)
    category = str(pan.panel_angle_category)
    return (
        pan, preds, det, warped_dets, reproj, reliable, xy_back, angle, category, dict(pan.meta),
    )


def _frame_color_detect(
    work_bgr: np.ndarray,
    corners_px: np.ndarray,
    warped: np.ndarray,
    h_mat: np.ndarray,
    black_th: object,
    *,
    panel_id: str,
    angle: int,
    pan_meta: Optional[dict] = None,
) -> Tuple[List, List[dict], List[dict], List[str]]:
    from release.snapshot_cell_color import (
        detect_gt_cells_on_warped,
        gt_cells_from_config,
        load_panel_color_layout,
    )

    gt_cfg = load_panel_color_layout()
    meta = pan_meta or {}
    glx = meta.get('grid_lines_x')
    gly = meta.get('grid_lines_y')
    if gt_cfg and glx and gly and len(glx) >= 11 and len(gly) >= 11:
        cells = gt_cells_from_config(gt_cfg)
        raw = detect_gt_cells_on_warped(warped, glx, gly, cells)
        live_preds = [{'x': int(p['x']), 'y': int(p['y']), 'color': str(p['color'])} for p in raw]
        live_lines = predictions_to_report_lines(panel_id, angle, live_preds)
        return [], [], live_preds, live_lines

    det, warped_dets = detect_cards_live(
        work_bgr, corners_px, h_mat, warped, black_thresholds=black_th,
    )
    live_preds = warped_dets_to_predictions(warped_dets)
    live_lines = predictions_to_report_lines(panel_id, angle, live_preds)
    return det, warped_dets, live_preds, live_lines


def _refresh_saved_snapshot_colors(
    *,
    snap_path: str,
    snapshot_store: LiveSnapshotStore,
    work_bgr: np.ndarray,
    warped: np.ndarray,
    corners_px: np.ndarray,
    h_mat: np.ndarray,
    black_th: object,
    vis: np.ndarray,
    panel_id: str,
    angle: int,
    module_a_po: Optional[PoseFrameOutput],
    reliable: bool,
    reproj: float,
    hom_inliers: int,
    corner_src: str,
    xy_back: str,
    category: str,
    latched_preds: Optional[List[dict]],
    latched_lines: Optional[List[str]],
    latch_txt: Optional[str],
    latch_locked: bool,
    opencv_grid_x: Optional[List[float]],
    opencv_grid_y: Optional[List[float]],
    hough_grid_x: Optional[List[float]],
    hough_grid_y: Optional[List[float]],
    grid_overlap_ratio: float,
    pan_meta: Optional[dict] = None,
) -> Tuple[List, List[dict], List[dict], List[str], np.ndarray]:
    det, warped_dets, live_preds, live_lines = _frame_color_detect(
        work_bgr, corners_px, warped, h_mat, black_th, panel_id=panel_id, angle=angle,
        pan_meta=pan_meta,
    )
    dash = compose_unified_dashboard(
        vis,
        warped,
        live_preds,
        warped_dets,
        module_a=module_a_po,
        panel_id=panel_id,
        reliable=reliable,
        reproj_b=reproj,
        homography_inliers=hom_inliers,
        corner_source=corner_src,
        xy_backend=xy_back,
        angle=angle,
        category=category,
        live_report_lines=live_lines,
        latched_preds=latched_preds,
        latched_report_lines=latched_lines,
        latch_txt=latch_txt,
        latch_locked=latch_locked,
        snapshot_entries=snapshot_store.ranked_entries,
        snapshot_preds=live_preds,
        opencv_grid_x=opencv_grid_x,
        opencv_grid_y=opencv_grid_y,
        hough_grid_x=hough_grid_x,
        hough_grid_y=hough_grid_y,
        grid_overlap_ratio=grid_overlap_ratio,
    )
    snapshot_store.refresh_snapshot_artifacts(
        snap_path,
        dash,
        record_updates={
            'module_b': {
                'live_predictions': live_preds,
                'live_report_lines': live_lines,
            },
            'color_detect': {
                'mode': 'on_snapshot_gt' if load_panel_color_layout() else 'on_snapshot',
                'n_cards': len(live_preds),
            },
        },
    )
    return det, warped_dets, live_preds, live_lines, dash


def _draw_module_a_overlay(
    vis: np.ndarray,
    po: Optional[PoseFrameOutput],
    *,
    panel_present: bool = True,
) -> None:
    """Moduł A — trapez (YOLO-Pose, współdzielony z B) + ustawienie panelu."""
    if po is None or not panel_present:
        return
    if po.corners_px is not None and po.corners_px.shape == (4, 2):
        color = (0, 220, 0) if po.ok else (0, 160, 255)
        pts = po.corners_px.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, color, 2, cv2.LINE_AA)
    if po.ok:
        from module_pose.panel_stand import STAND_LABEL_PL
        label = STAND_LABEL_PL.get(po.panel_angle_category, po.panel_angle_category)
        txt = f'A: {label} {po.report_angle_deg}°  d={po.distance_m:.2f}m'
        from release.cv_text import put_text_utf8
        put_text_utf8(vis, txt, (12, vis.shape[0] - 14), (0, 200, 0), scale=0.55, thickness=2)


def _draw_vis_overlay(
    vis: np.ndarray,
    *,
    module_a_po: Optional[PoseFrameOutput],
    corners_px: Optional[np.ndarray],
    det: List[Tuple[int, float, float, float, float]],
    live_preds: List[dict],
    panel_present: bool,
    tracker_hold: bool,
    reliable: bool,
    corner_src: str,
    xy_back: str,
    reproj: float,
    angle: int,
    category: str,
    corner_meta: dict,
    hom_inliers: int,
    grid_overlap_ratio: float,
    h_mat: Optional[np.ndarray],
    opencv_grid_x: Optional[List[float]],
    opencv_grid_y: Optional[List[float]],
    warped_shape: Tuple[int, int],
    latch_txt: Optional[str],
    analysis_paused: bool,
    flight_controller: Optional[object],
) -> None:
    """Kamera + overlay YOLO (bez pełnego dashboardu)."""
    if analysis_paused:
        from release.cv_text import put_text_utf8

        _phase_lbl = (
            flight_controller.phase.value
            if flight_controller is not None else 'pauza'
        )
        _spd_lbl = (
            f'{flight_controller.speed:.2f}'
            if flight_controller is not None else '—'
        )
        put_text_utf8(
            vis,
            f'ANALIZA OFF · {_phase_lbl} · speed={_spd_lbl}',
            (12, 40),
            (0, 90, 255),
            scale=0.72,
            thickness=2,
        )
    ok_corners = panel_present and corners_px is not None
    _draw_module_a_overlay(vis, module_a_po, panel_present=panel_present)
    _draw_panel_overlay(
        vis,
        corners_px,
        det,
        live_preds,
        ok_corners=ok_corners,
        panel_present=panel_present,
        tracker_hold=tracker_hold,
    )
    if (
        h_mat is not None
        and opencv_grid_x is not None
        and opencv_grid_y is not None
    ):
        from module_panel.grid_overlap import draw_opencv_grid_on_camera

        draw_opencv_grid_on_camera(
            vis,
            opencv_grid_x,
            opencv_grid_y,
            h_mat,
            warp_w=int(warped_shape[1]),
            warp_h=int(warped_shape[0]),
        )
    _draw_status_block(
        vis,
        reliable=reliable,
        corner_source=corner_src,
        xy_backend=xy_back,
        reproj=reproj,
        n_cards=len(live_preds),
        angle=angle,
        category=category,
        roi_coverage=float(corner_meta['roi_coverage'])
        if corner_meta.get('roi_coverage') is not None else None,
        cxy_latch=latch_txt,
        homography_inliers=hom_inliers,
        grid_overlap_ratio=grid_overlap_ratio,
        module_a=module_a_po,
    )


def _push_stream_vis_preview(
    web_publisher: object,
    bgr: np.ndarray,
    overlay_state: Optional[dict],
) -> None:
    """Płynny MJPEG między klatkami YOLO — ostatni overlay na świeżej klatce."""
    if overlay_state is None:
        return
    vis = bgr.copy()
    _draw_vis_overlay(vis, **overlay_state)
    web_publisher.set_stream_vis_frame(vis)


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
    roi_coverage: Optional[float] = None,
    cxy_latch: Optional[str] = None,
    homography_inliers: Optional[int] = None,
    grid_overlap_ratio: Optional[float] = None,
    module_a: Optional[PoseFrameOutput] = None,
) -> None:
    cov_txt = f'  cov_roi={roi_coverage:.0%}' if roi_coverage is not None else ''
    inl_txt = f'  inl={homography_inliers}' if homography_inliers is not None else ''
    header = 'Droniada LIVE A+B' if module_a is not None else 'Droniada LIVE B'
    lines = [
        (f'{header}  corner={corner_source}', (30, 30, 30)),
    ]
    if module_a is not None:
        from module_pose.panel_stand import STAND_LABEL_PL
        a_ok = module_a.ok
        pl = STAND_LABEL_PL.get(module_a.panel_angle_category, module_a.panel_angle_category)
        lines.append((
            f'Modul A: {"OK" if a_ok else "FAIL"}  {pl} ({module_a.report_angle_deg}°)  '
            f'd={module_a.distance_m:.2f}m  reproj={module_a.reproj_mean_px:.1f}px',
            (0, 160, 0) if a_ok else (0, 120, 220),
        ))
        if a_ok:
            lines.append((
                f'  roll={module_a.roll_deg:.0f} pitch={module_a.pitch_deg:.0f} yaw={module_a.yaw_deg:.0f}',
                (60, 60, 60),
            ))
    grid_txt = ''
    if grid_overlap_ratio is not None:
        pct = 100.0 * float(grid_overlap_ratio)
        grid_txt = f'  siatka={pct:.0f}%'
    lines.extend([
        (f'Modul B: reliable={"TAK" if reliable else "NIE"}  reproj={reproj:.1f}px{inl_txt}{cov_txt}{grid_txt}',
         (0, 140, 0) if reliable else (0, 0, 200)),
        (f'  xy={xy_backend}  kat={angle}  cat={category}  det={n_cards}', (50, 50, 50)),
        ('  migawka: szara+pomarancz+zielona musza sie zgadzac', (90, 90, 90)),
    ])
    if cxy_latch:
        lines.append((cxy_latch, (0, 200, 0) if 'ZAMKNIETE' in cxy_latch else (0, 120, 220)))
    lines.extend([
        ('s = zlap CXY  r = reset  q = wyjscie  (zatrzask = osobne okno)', (100, 100, 100)),
    ])
    from release.cv_text import put_text_utf8
    y = 12
    for text, color in lines:
        put_text_utf8(vis, text, (12, y), color, scale=0.52, thickness=2)
        y += 24


def _draw_quad_grid_lines(vis: np.ndarray, corners: np.ndarray, color: Tuple[int, int, int]) -> None:
    """Siatka 10×10 w perspektywie — linie między rogami (homografia)."""
    if corners is None or corners.shape != (4, 2):
        return
    dst = np.array(
        [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
        dtype=np.float32,
    )
    src = corners.astype(np.float32)
    try:
        h_mat = cv2.getPerspectiveTransform(src, dst)
        h_inv = np.linalg.inv(h_mat)
    except cv2.error:
        return
    h_vis, w_vis = vis.shape[:2]
    for i in range(11):
        t = i / 10.0
        seg = np.array([[[t * 1000.0, 0.0]], [[t * 1000.0, 1000.0]]], dtype=np.float32)
        pts = cv2.perspectiveTransform(seg, h_inv).reshape(-1, 2).astype(np.int32)
        cv2.line(vis, tuple(pts[0]), tuple(pts[1]), color, 1, cv2.LINE_AA)
        seg = np.array([[[0.0, t * 1000.0]], [[1000.0, t * 1000.0]]], dtype=np.float32)
        pts = cv2.perspectiveTransform(seg, h_inv).reshape(-1, 2).astype(np.int32)
        cv2.line(vis, tuple(pts[0]), tuple(pts[1]), color, 1, cv2.LINE_AA)


def _draw_panel_overlay(
    vis: np.ndarray,
    corners: Optional[np.ndarray],
    det: List[Tuple[int, float, float, float, float]],
    preds: List[dict],
    *,
    ok_corners: bool,
    panel_present: bool = True,
    tracker_hold: bool = False,
) -> None:
    h, w = vis.shape[:2]
    show_quad = corners is not None and corners.shape == (4, 2)
    if show_quad and (panel_present or tracker_hold):
        if tracker_hold:
            color = (0, 200, 255)
            grid_color = (0, 160, 220)
        else:
            color = (0, 255, 255) if ok_corners else (0, 180, 255)
            grid_color = (0, 220, 220) if ok_corners else (0, 140, 200)
        _draw_quad_grid_lines(vis, corners, grid_color)
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, color, 3)
        from release.panel_labels import CORNER_NAMES

        q = pc.order_points(corners.astype(np.float32))
        for i, name in enumerate(CORNER_NAMES):
            p = q[i].astype(np.int32)
            cv2.circle(vis, tuple(p), 7, color, -1)
            cv2.putText(
                vis, name, (int(p[0]) + 6, int(p[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

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
    return draw_warped_panel_preview(warped, preds, warped_dets)


_LIVE_WINDOW = 'droniada_live_panel'
_LATCH_WINDOW = 'droniada_cxy_zatrzask'


def _scale_preview_image(img: np.ndarray, target_w: int) -> np.ndarray:
    tw = max(720, int(target_w))
    if img.shape[1] <= tw:
        return img
    scale = tw / float(img.shape[1])
    return cv2.resize(
        img,
        (tw, max(1, int(round(img.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _show_latch_window(latched_vis: np.ndarray, target_w: int) -> None:
    """Osobne okno ze zatrzaśniętym panelem i listą CXY (utrzymywane do resetu)."""
    show = _scale_preview_image(latched_vis, target_w)
    cv2.namedWindow(_LATCH_WINDOW, cv2.WINDOW_NORMAL)
    cv2.imshow(_LATCH_WINDOW, show)
    cv2.resizeWindow(_LATCH_WINDOW, show.shape[1], show.shape[0])


def _hide_latch_window() -> None:
    try:
        cv2.destroyWindow(_LATCH_WINDOW)
    except cv2.error:
        pass


def _compose_preview(main: np.ndarray, warped_thumb: np.ndarray, target_w: int = 1280) -> np.ndarray:
    h, w = main.shape[:2]
    th, tw = warped_thumb.shape[:2]
    if th != h:
        tw = max(1, int(round(tw * h / float(th))))
        warped_thumb = cv2.resize(warped_thumb, (tw, h), interpolation=cv2.INTER_AREA)
    total_w = w + 12 + tw
    scale = min(1.0, target_w / float(total_w))
    if scale < 1.0:
        nh, nw = int(round(h * scale)), int(round(w * scale))
        main = cv2.resize(main, (nw, nh), interpolation=cv2.INTER_AREA)
        warped_thumb = cv2.resize(
            warped_thumb,
            (int(round(tw * scale)), nh),
            interpolation=cv2.INTER_AREA,
        )
    gap = np.full((main.shape[0], 12, 3), 40, dtype=np.uint8)
    return np.hstack([main, gap, warped_thumb])


def main() -> None:
    from release.pipeline_threads import limit_blas_threads

    limit_blas_threads()

    ap = argparse.ArgumentParser(description='Live test modulu B — line_grid v3 + detekcja kolorow na siatce')
    ap.add_argument('--camera', type=int, default=1, help='indeks kamery (0 lub 1 na Macu)')
    ap.add_argument(
        '--cap-pipeline',
        default=None,
        help='GStreamer pipeline (OpenCV CAP_GSTREAMER); nadpisuje DRONIADA_CAP_PIPELINE',
    )
    ap.add_argument('--video', default=None, help='plik wideo zamiast kamery (np. Droniad_nag2.mov)')
    ap.add_argument('--no-loop', action='store_true', help='przy --video: zakoncz na koncu pliku')
    ap.add_argument('--width', type=int, default=0)
    ap.add_argument('--height', type=int, default=0)
    ap.add_argument(
        '--camera-profile',
        default=None,
        help='profil intrinsics: tarot_t10x_2a, tarot_t10x_2a:wide|mid|tele (gimbal T10X-2A)',
    )
    ap.add_argument(
        '--zoom-ratio',
        type=float,
        default=1.0,
        help='zoom optyczny 1..10 dla tarot_t10x_2a (gdy bez presetu wide/mid/tele)',
    )
    ap.add_argument('--rotate', type=int, default=180, choices=[0, 90, 180, 270])
    ap.add_argument('--interval-ms', type=int, default=450, help='min. odstep miedzy analizami (ms)')
    ap.add_argument('--panel-id', default='A')
    ap.add_argument('--xy-mode', default='line_grid', choices=['grid_geom', 'grid_geom_white', 'warp_grid', 'geom_grid', 'line_grid'])
    ap.add_argument(
        '--corner-mode',
        default=DEFAULT_LIVE_CORNER_MODE,
        choices=sorted(_LIVE_CORNER_MODES),
        help='yolo_pose = YOLOv8-Pose (4 kropki); outer_corners/align_hybrid = CV line_grid+border_scan',
    )
    ap.add_argument(
        '--out-video',
        default=None,
        help='zapisz podglad do MP4 (np. dataset/debug_yolo_nag3.mp4)',
    )
    ap.add_argument('--max-reproj-reliable', type=float, default=LIVE_MAX_REPROJ_RELIABLE_PX,
                    help='prog reproj dla grid_xy_reliable na live (domyslnie 18)')
    ap.add_argument('--no-stabilize', action='store_true', help='wylacz EMA rogow')
    ap.add_argument(
        '--smooth-alpha',
        type=float,
        default=0.28,
        help='EMA rogow 0..1 (wyzsze=szybsza reakcja ramki, nizsze=spokojniejsza)',
    )
    ap.add_argument('--hold-frames', type=int, default=14, help='klatek hold po zgubieniu panelu')
    ap.add_argument(
        '--tracker-good-reproj',
        type=float,
        default=28.0,
        help='reproj px uznawany za dobry quad (szybsze skoki trackera)',
    )
    ap.add_argument('--angle-source', default='rmat_linear', choices=['rmat_linear', 'rmat_theta', 'geom', 'pnp'])
    ap.add_argument('--require-reliable', action='store_true', help='raport kart tylko gdy grid_xy_reliable')
    ap.add_argument(
        '--cxy-latch',
        action='store_true',
        help='sciagnij CXY dopiero gdy siatka idealna (grid_xy_reliable przez N klatek)',
    )
    ap.add_argument(
        '--cxy-stable-frames',
        type=int,
        default=5,
        help='ile kolejnych klatek z identycznym wynikiem CXY, zanim zatrzasnąć wynik',
    )
    ap.add_argument(
        '--cxy-latch-dir',
        default='dataset/debug_cxy_latch',
        help='katalog zapisu podglądu zatrzaśniętej klatki (panel + lista CXY)',
    )
    ap.add_argument(
        '--min-homography-inliers',
        type=int,
        default=12,
        help='min. inlierów RANSAC siatki dla grid_xy_reliable v2',
    )
    ap.add_argument('--no-color-detect', action='store_true', help='tylko rogi/siatka, bez skanowania kolorow')
    ap.add_argument(
        '--live-color-detect',
        action='store_true',
        help='HSV na kazdej klatce (domyslnie kolory dopiero przy migawce / zatrzasnieciu CXY)',
    )
    ap.add_argument(
        '--corners-only',
        action='store_true',
        help='tylko overlay rogow (bez analyze_panel — szybciej dla yolo_pose + video)',
    )
    ap.add_argument(
        '--greenscreen',
        action='store_true',
        help='tlo i szum → zielony chroma; w panelu zostaja czern/biel i kolorowe kartki (odczyt kolorow)',
    )
    ap.add_argument(
        '--greenscreen-split',
        action='store_true',
        help='podglad: kamera | greenscreen obok siebie (wymaga --greenscreen)',
    )
    ap.add_argument('--preview', action='store_true', help='okno podgladu (zalecane)')
    ap.add_argument(
        '--headless',
        action='store_true',
        help='bez okien OpenCV/tkinter (Jetson/Docker); logi + migawki na dysku',
    )
    ap.add_argument('--preview-width', type=int, default=1280)
    ap.add_argument('--log-file', default=None)
    ap.add_argument('--save-dir', default=None, help='dodatkowo zapisz overlay gdy reliable=TAK')
    ap.add_argument('--debug-dir', default='live_debug', help='katalog sesji debug (JSON + JPG); pusty = wylacz')
    ap.add_argument('--no-debug', action='store_true', help='nie zapisuj live_debug')
    ap.add_argument('--debug-every', type=int, default=3,
                    help='zapisz pelny debug co N-ta analizowana klatka (1=kazda)')
    ap.add_argument('--align-full-every', type=int, default=2,
                    help='pelny align_hybrid co N-ta klatke; miedzy = tracker_fast')
    ap.add_argument('--debug-html', action='store_true', help='po zakonczeniu generuj index.html')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument(
        '--module-a',
        action='store_true',
        help='modul A (PnP + ustawienie panelu + odleglosc) rownolegle z modulem B',
    )
    ap.add_argument(
        '--bench',
        action='store_true',
        help='skrot: --dashboard --module-a --cxy-latch --preview --no-debug --corner-mode yolo_pose',
    )
    ap.add_argument(
        '--dashboard',
        action='store_true',
        help='jeden panel: modul A+B + sidebar + galeria migawek (niski reproj)',
    )
    ap.add_argument(
        '--dashboard-dir',
        default='dataset/live_dashboard',
        help='katalog sesji migawek (index.html po zakonczeniu)',
    )
    ap.add_argument('--snapshot-max-reproj', type=float, default=15.0,
                    help='max reproj B (px) dla migawki; nie musi być grid_xy_reliable')
    ap.add_argument('--snapshot-max', type=int, default=8, help='max migawek w galerii')
    ap.add_argument('--snapshot-min-stable', type=int, default=2,
                    help='klatek z pokryciem siatki >= min-grid-overlap z rzędu przed zapisem')
    ap.add_argument('--snapshot-min-grid-overlap', type=float, default=0.75,
                    help='min. zgodność 3 siatek: szara (rogow) + CLAHE + OpenCV (0–1)')
    ap.add_argument('--snapshot-grid-line-tol', type=float, default=0.14,
                    help='tolerancja (ułamek komórki) — wszystkie 3 siatki w tym paśmie')
    ap.add_argument('--snapshot-replace-margin', type=float, default=0.05,
                    help='gdy galeria pełna: zamień gorszą migawkę jeśli nowa reproj o tyle mniejsza (px)')
    ap.add_argument('--snapshot-max-reproj-a', type=float, default=18.0,
                    help='max reproj modułu A (px) dla migawki')
    ap.add_argument('--snapshot-require-reliable', action=argparse.BooleanOptionalAction, default=False,
                    help='migawka tylko przy grid_xy_reliable (domyślnie wyłączone na nag5)')
    ap.add_argument('--snapshot-block-unreliable-zero-inl', action=argparse.BooleanOptionalAction,
                    default=False, help='przy reliable=NIE i inliers=0: ostrzejsze progi (domyślnie wł.)')
    ap.add_argument('--snapshot-min-grid-overlap-unreliable', type=float, default=0.78,
                    help='min. 3siatki gdy reliable=NIE i inliers=0')
    ap.add_argument('--snapshot-min-homography-inliers', type=int, default=0,
                    help='opcjonalny min. inlierów RANSAC (0=nie wymagaj)')
    ap.add_argument('--snapshot-min-warp-coverage', type=float, default=0.20,
                    help='min. pokrycie panelu na warpie (0=wył.; blokada tylko skrajnego tła)')
    ap.add_argument('--snapshot-competition-min-votes', type=int, default=2,
                    help='konkurs CXY: min. migawek z tą samą kartą (odrzuca jednorazowe)')
    ap.add_argument('--snapshot-competition-min-ratio', type=float, default=0.0,
                    help='konkurs CXY: min. ułamek migawek (np. 0.5); 0 = tylko min-votes')
    ap.add_argument(
        '--mission-panels',
        default=None,
        help='misja wielopanelowa, np. A,B,C — osobny limit migawek na panel',
    )
    ap.add_argument(
        '--snapshots-per-panel',
        type=int,
        default=0,
        help='ile migawek zebrać na każdy panel (0 = użyj --snapshot-max)',
    )
    ap.add_argument(
        '--report-send-pause-s',
        type=float,
        default=5.0,
        help='pauza (s) po Wyślij — bez zbierania migawek przed następnym panelem',
    )
    ap.add_argument(
        '--report-mode',
        choices=('live', 'preset'),
        default=None,
        help='live=raport z CXY/edycja; preset=raporty z pliku + skróty 1/2/3 (A/B/C)',
    )
    ap.add_argument(
        '--preset-reports',
        default=None,
        help='ścieżka do config/preset_reports.json (tryb preset)',
    )
    ap.add_argument(
        '--mjpeg-port',
        type=int,
        default=0,
        help='serwer HTTP MJPEG (Jetson headless), np. 8088',
    )
    ap.add_argument(
        '--mjpeg-host',
        default='0.0.0.0',
        help='adres nasłuchu MJPEG',
    )
    ap.add_argument(
        '--web-port',
        type=int,
        default=0,
        help='podgląd live (kamera + start/stop); wymaga --dashboard',
    )
    ap.add_argument(
        '--web-control-port',
        type=int,
        default=-1,
        help='panel sterowania (parametry, raport, progi migawek); domyślnie web-port+1',
    )
    ap.add_argument(
        '--web-host',
        default='0.0.0.0',
        help='adres nasłuchu panelu WWW',
    )
    args = ap.parse_args()
    _report_mode_raw = (
        (args.report_mode or os.environ.get('DRONIADA_REPORT_MODE', '') or 'live')
        .strip()
        .lower()
    )
    if _report_mode_raw not in ('live', 'preset'):
        raise SystemExit(f'Nieznany --report-mode / DRONIADA_REPORT_MODE: {_report_mode_raw!r}')
    args.report_mode = _report_mode_raw
    preset_reports: Dict[str, List[str]] = {}
    preset_reports_path = ''
    if args.report_mode == 'preset':
        from release.preset_reports import load_preset_reports_file

        _preset_path = (
            args.preset_reports
            or os.environ.get('DRONIADA_PRESET_REPORTS', '')
            or 'config/preset_reports.json'
        ).strip()
        preset_reports_path = os.path.abspath(
            _preset_path if os.path.isabs(_preset_path) else os.path.join(_ROOT, _preset_path),
        )
        try:
            preset_reports = load_preset_reports_file(preset_reports_path, root=_ROOT)
        except (OSError, ValueError) as exc:
            raise SystemExit(f'Tryb preset: błąd wczytania raportów: {exc}') from exc
    if args.cap_pipeline and str(args.cap_pipeline).strip():
        os.environ['DRONIADA_CAP_PIPELINE'] = str(args.cap_pipeline).strip()
    if int(args.web_port) > 0 and int(args.mjpeg_port) > 0 and int(args.web_port) != int(args.mjpeg_port):
        raise SystemExit('Użyj jednego portu: --web-port (podgląd) albo --mjpeg-port (sam strumień).')
    if int(args.web_port) > 0:
        if int(args.web_control_port) < 0:
            args.web_control_port = int(args.web_port) + 1
        if int(args.web_control_port) == int(args.web_port):
            raise SystemExit('--web-control-port musi być inny niż --web-port')
    if args.bench:
        args.module_a = True
        args.cxy_latch = True
        args.preview = True
        args.no_debug = True
        args.corner_mode = 'yolo_pose'
        args.dashboard = True
        if args.interval_ms == 450:
            args.interval_ms = 350
        if float(args.snapshot_max_reproj) == 15.0:
            args.snapshot_max_reproj = float(args.max_reproj_reliable)
        if int(args.snapshot_min_stable) == 2:
            args.snapshot_min_stable = 1
        if float(args.snapshot_max_reproj_a) == 15.0:
            args.snapshot_max_reproj_a = float(args.max_reproj_reliable)

    log_fh = None
    if args.log_file:
        log_path = args.log_file if os.path.isabs(args.log_file) else os.path.join(_ROOT, args.log_file)
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
        log_fh = open(log_path, 'a', encoding='utf-8')
    calib_path = os.path.join(_ROOT, 'module_panel', 'data', 'angle_linear_rmat.json')
    if not os.path.isfile(calib_path):
        calib_path = None
    cam_calib = os.path.join(_ROOT, 'config', 'camera_calibration.npz')
    if not os.path.isfile(cam_calib):
        cam_calib = None
    profile = args.camera_profile
    if profile is None and args.video:
        profile = 'tarot_t10x_2a:wide'

    _log(log_fh, f'[live_panel] start camera={args.camera} video={args.video} xy_mode={args.xy_mode} corner_mode={args.corner_mode} rotate={args.rotate} profile={profile}')
    if not args.video:
        _log(log_fh, '[live_panel] zamknij QuickTime — kamera tylko w jednym programie')
    _log(log_fh, '[live_panel] zawies kolorowe kartki na siatce 10x10')
    from release.card_color_profile import active_profile_label, load_active_profile
    load_active_profile(force=True)
    _log(log_fh, f'[live_panel] kolory kartek: {active_profile_label()}')
    _log(log_fh, f'[live_panel] rogi: corner_mode={args.corner_mode} max_reproj_rel={args.max_reproj_reliable}')
    show_preview = bool(args.preview or args.cxy_latch or args.bench or args.dashboard)
    show_window = show_preview and not args.headless
    live_window = 'droniada_dashboard' if args.dashboard else ('droniada_live_bench' if args.module_a else _LIVE_WINDOW)
    if args.corner_mode == 'yolo_pose' or args.module_a:
        from release.device import device_label, resolve_yolo_device

        _log(log_fh, f'[live_panel] YOLO device={device_label(resolve_yolo_device())}')
    if args.headless:
        _log(log_fh, '[live_panel] headless: brak okien GUI, wyniki w logu i dataset/live_dashboard')
    web_publisher: Optional[object] = None
    web_httpd: Optional[object] = None
    web_control_httpd: Optional[object] = None
    ws_publisher: Optional[object] = None
    mjpeg_server: Optional[object] = None
    vis_preview_worker: Optional[object] = None
    live_frame_buf: Optional[object] = None
    live_overlay_cache: Optional[object] = None
    analysis_worker: Optional[object] = None
    snapshot_store: Optional[LiveSnapshotStore] = None
    snapshot_paths: List[str] = []
    browser_proc = None
    dashboard_session: Optional[str] = None
    runtime_gate: dict = {
        'snapshot_min_grid_overlap': float(args.snapshot_min_grid_overlap),
        'snapshot_min_grid_overlap_unreliable': float(args.snapshot_min_grid_overlap_unreliable),
        'snapshot_min_stable': int(args.snapshot_min_stable),
        'snapshot_max_reproj': float(args.snapshot_max_reproj),
        'snapshot_min_warp_coverage': float(args.snapshot_min_warp_coverage),
    }
    if args.dashboard:
        dash_root = args.dashboard_dir if os.path.isabs(args.dashboard_dir) else os.path.join(_ROOT, args.dashboard_dir)
        dashboard_session = new_dashboard_session(dash_root)
        snapshot_store = LiveSnapshotStore(
            dashboard_session,
            max_snapshots=int(args.snapshot_max),
            max_reproj=float(args.snapshot_max_reproj),
            min_stable_frames=int(args.snapshot_min_stable),
            require_module_a=bool(args.module_a),
            replace_margin=float(args.snapshot_replace_margin),
        )
        _log(log_fh, f'[live_panel] Dashboard: okno {live_window}')
        _log(
            log_fh,
            f'[live_panel] Migawki: {dashboard_session} '
            f'(3 siatki >= {100 * args.snapshot_min_grid_overlap:.0f}%, '
            f'reproj B<={args.snapshot_max_reproj}px'
            + (f', reproj A<={args.snapshot_max_reproj_a}px' if args.module_a else '')
            + (', reliable=TAK' if args.snapshot_require_reliable else '')
            + (', unreli+inl0:ostrzej' if args.snapshot_block_unreliable_zero_inl else '')
            + (f', warp>={args.snapshot_min_warp_coverage:.0%}' if args.snapshot_min_warp_coverage > 0 else '')
            + f', {args.snapshot_min_stable} klatek z rzędu)',
        )
        _log(log_fh, f'[live_panel] Po sesji: open {dashboard_session}/index.html')
    if int(args.web_port) > 0:
        if not dashboard_session:
            raise SystemExit('--web-port wymaga --dashboard (np. --bench)')
        from release.web_dashboard import LiveWebPublisher, start_split_web_servers
        from release.web_runtime_settings import runtime_snapshot

        web_publisher = LiveWebPublisher(
            dashboard_session,
            report_mode=args.report_mode,
            preset_reports=preset_reports,
            preset_reports_path=preset_reports_path,
        )
        _rt_boot = {
            'snapshot_min_grid_overlap': float(args.snapshot_min_grid_overlap),
            'snapshot_min_grid_overlap_unreliable': float(args.snapshot_min_grid_overlap_unreliable),
            'snapshot_min_stable': int(args.snapshot_min_stable),
            'snapshot_max_reproj': float(args.snapshot_max_reproj),
            'snapshot_min_warp_coverage': float(args.snapshot_min_warp_coverage),
        }
        if os.path.isfile(web_publisher.runtime_path):
            runtime_gate = runtime_snapshot(web_publisher.read_runtime_settings())
        else:
            runtime_gate = runtime_snapshot(_rt_boot)
            web_publisher.write_runtime_settings(runtime_gate)
        web_publisher._runtime_mtime = os.path.getmtime(web_publisher.runtime_path)
        web_publisher, web_httpd, web_control_httpd = start_split_web_servers(
            dashboard_session,
            host=str(args.web_host),
            view_port=int(args.web_port),
            control_port=int(args.web_control_port),
            publisher=web_publisher,
        )
        _log(
            log_fh,
            f'[live_panel] Podgląd WWW: http://<host>:{int(args.web_port)}/',
        )
        _log(
            log_fh,
            f'[live_panel] Sterowanie WWW: http://<host>:{int(args.web_control_port)}/ '
            f'(parametry, raport, progi migawek) · sesja {dashboard_session}',
        )
        if args.report_mode == 'preset' and preset_reports:
            _panels = ','.join(
                f'{pid}({len(preset_reports.get(pid, []))})'
                for pid in ('A', 'B', 'C')
                if pid in preset_reports
            )
            _log(
                log_fh,
                f'[live_panel] Raporty preset (:8088): {preset_reports_path} · panele {_panels}',
            )
        elif int(args.web_control_port) > 0:
            _log(
                log_fh,
                f'[live_panel] Raporty :8088 z migawek (CXY) · autonomia hold_stopped lub klawisze 1/2/3 na :{int(args.web_control_port)}/',
            )
        _refresh_snapshot_store_limits(snapshot_store, runtime_gate, args)
        from release.pipeline_threads import (
            DropQueueWorker,
            LatestFrameBuffer,
            OverlayCache,
            VisPreviewWorker,
        )

        live_frame_buf = LatestFrameBuffer()
        live_overlay_cache = OverlayCache()

        def _vis_draw_push(bgr_in, state: dict, *, _wp=web_publisher) -> None:
            _push_stream_vis_preview(_wp, bgr_in, state)

        from release.gst_mjpeg_camera import should_use_gst_capture, wants_stream_passthrough

        _stream_vis_on = os.environ.get(
            'DRONIADA_STREAM_SOURCE', 'vis',
        ).strip().lower() in ('vis', 'dashboard')
        _gst_mjpeg_pass = wants_stream_passthrough() if should_use_gst_capture() else False
        vis_preview_worker = VisPreviewWorker(
            live_frame_buf,
            live_overlay_cache,
            _vis_draw_push,
            interval_s=float(os.environ.get('DRONIADA_VIS_PREVIEW_INTERVAL_S', '0.033') or 0.033),
            stream_enabled=_stream_vis_on and not _gst_mjpeg_pass,
        )
        vis_preview_worker.start()
        _log(
            log_fh,
            '[live_panel] Wątki: kamera · vis-preview · MJPEG · HTTP · YOLO-worker '
            f'(GStreamer passthrough={_gst_mjpeg_pass})',
        )
    from release.integration_ws import IntegrationWsPublisher, IntegrationWsSubscriber

    ws_publisher = IntegrationWsPublisher.from_env()
    if ws_publisher is not None:
        _log(log_fh, f'[live_panel] WebSocket integracja: {os.environ.get("DRONIADA_WS_URL", "")}')
    ws_subscriber = IntegrationWsSubscriber.from_env()
    if ws_subscriber is not None:
        from release.integration_ws import autonomy_analysis_pause_s as _autonomy_pause_default

        _log(
            log_fh,
            f'[live_panel] Autonomia WS (edge-trigger hold_started/stopped): '
            f'{os.environ.get("DRONIADA_AUTONOMY_WS_URL", "")} · '
            f'raport po hold_stopped, pauza analizy {_autonomy_pause_default():.0f}s',
        )
    if int(args.mjpeg_port) > 0:
        from release.mjpeg_stream import start_mjpeg_stream

        mjpeg_server = start_mjpeg_stream(
            str(args.mjpeg_host),
            int(args.mjpeg_port),
            log=lambda msg: _log(log_fh, msg),
        )
    mission_manager: Optional[object] = None
    panel_review_ui: Optional[object] = None
    mission_panel_ids: List[str] = []
    if args.mission_panels:
        mission_panel_ids = [
            p.strip().upper() for p in str(args.mission_panels).split(',') if p.strip()
        ]
    elif args.dashboard and int(args.web_port) > 0:
        mission_panel_ids = ['A', 'B', 'C']
    if mission_panel_ids and dashboard_session:
        from release.mission_panels import MultiPanelMissionManager

        if int(args.snapshots_per_panel) > 0:
            per_panel = int(args.snapshots_per_panel)
        elif int(args.web_port) > 0:
            per_panel = 5
        else:
            per_panel = int(args.snapshot_max)
        mission_manager = MultiPanelMissionManager(
            dashboard_session,
            mission_panel_ids,
            snapshots_per_panel=per_panel,
            store_kwargs={
                'max_reproj': float(args.snapshot_max_reproj),
                'min_stable_frames': int(args.snapshot_min_stable),
                'require_module_a': bool(args.module_a),
                'replace_margin': float(args.snapshot_replace_margin),
            },
        )
        snapshot_store = mission_manager.current_store()
        _log(
            log_fh,
            f'[live_panel] Misja paneli: {",".join(mission_panel_ids)} '
            f'po {per_panel} migawek; start={mission_manager.current_panel}',
        )

        if show_window:
            from release.panel_review_ui import PanelReviewUI

            panel_review_ui = PanelReviewUI(
                lambda pid, lines: apply_report_send(pid, lines, source='tk'),
            )
            panel_review_ui.bind_panel_id(mission_manager.current_panel)
    flight_controller: Optional[object] = None
    if int(args.web_port) > 0 or ws_publisher is not None:
        from release.flight_mission import FlightMissionController
        from release.mission_settings import merge_mission_settings

        _snap_n = 5.0
        if mission_manager is not None:
            _snap_n = float(mission_manager.state.snapshots_per_panel)
        elif int(args.snapshots_per_panel) > 0:
            _snap_n = float(args.snapshots_per_panel)
        _saved_mission = (
            web_publisher.read_mission_settings() if web_publisher is not None else {}
        )
        _mission_boot_patch: dict = {}
        if not _saved_mission:
            _mission_boot_patch = {
                'report_send_pause_s': float(args.report_send_pause_s),
                'snapshots_per_panel': _snap_n,
            }
        _mission_init = merge_mission_settings(_saved_mission, _mission_boot_patch)
        flight_controller = FlightMissionController(settings=_mission_init)
        if web_publisher is not None:
            web_publisher.write_mission_settings(_mission_init)
        if mission_manager is not None:
            _snap_n_init = max(1, int(_mission_init.get('snapshots_per_panel', 5)))
            mission_manager.state.snapshots_per_panel = _snap_n_init
            for _st in mission_manager.stores.values():
                _st.max_snapshots = _snap_n_init
            if int(args.web_port) > 0:
                for _st in mission_manager.stores.values():
                    _refresh_snapshot_store_limits(_st, runtime_gate, args)
        _log(
            log_fh,
            '[live_panel] Misja drona: WebSocket speed∈[0,1] (1=stój), ustawienia w panelu WWW',
        )
    color_on_snapshot = (
        not args.no_color_detect
        and snapshot_store is not None
        and not args.live_color_detect
    )
    run_live_color = not args.no_color_detect and (
        bool(args.live_color_detect) or snapshot_store is None
    )
    if color_on_snapshot:
        _log(
            log_fh,
            '[live_panel] Kolory HSV: po zapisie migawki (live bez skanowania komorek); '
            f'konkurs CXY min_votes={args.snapshot_competition_min_votes}',
        )
    pose_rt: Optional[PoseRuntime] = None
    if args.module_a:
        pose_rt = PoseRuntime(PoseConfig(
            corner_source='yolo_pose',
            yolo_pose_use_tracker=not args.no_stabilize,
            yolo_pose_fallback_cv=True,
        ))
        _log(log_fh, '[live_panel] Modul A: YOLO-Pose + PnP (zielony trapez, te same rogi co B)')
    if args.cxy_latch:
        _log(log_fh, f'[live_panel] CXY latch: same_result_frames={args.cxy_stable_frames} min_inl={args.min_homography_inliers}')
        if args.cxy_latch and not args.preview:
            _log(log_fh, '[live_panel] --cxy-latch: wlaczam okno podgladu (live + osobne okno zatrzasku)')
        if args.corners_only:
            _log(log_fh, '[live_panel] uwaga: --cxy-latch wylacza --corners-only (potrzebna detekcja CXY)')
    cxy_latch: Optional[CxyLatch] = None
    if args.cxy_latch:
        cxy_latch = CxyLatch(min_stable_frames=max(1, int(args.cxy_stable_frames)))
    cxy_locked_announced = False
    cxy_latch_dir = (
        args.cxy_latch_dir if os.path.isabs(args.cxy_latch_dir)
        else os.path.join(_ROOT, args.cxy_latch_dir)
    )
    if args.cxy_latch:
        os.makedirs(cxy_latch_dir, exist_ok=True)
    latched_vis: Optional[np.ndarray] = None
    if args.greenscreen:
        _log(log_fh, '[live_panel] greenscreen: tlo→zielony; czern panelu z kalibracji cieni (percentyl)')
    if not args.no_stabilize:
        _TRACKER.reset()
        from release.tracker_tuning import init_from_args

        init_from_args(
            smooth_alpha=float(np.clip(args.smooth_alpha, 0.05, 1.0)),
            hold_frames=int(args.hold_frames),
            tracker_good_reproj=float(args.tracker_good_reproj),
            interval_ms=int(args.interval_ms),
        )
        _log(
            log_fh,
            f'[live_panel] EMA alpha={_TRACKER.alpha:.3f} hold_frames={_TRACKER.hold_frames} '
            f'good_reproj={_TRACKER.good_reproj:.1f}px interval={args.interval_ms}ms',
        )
    if web_publisher is not None and not args.no_stabilize:
        from release.tracker_tuning import snapshot as tracker_snapshot

        web_publisher.write_tracker_settings(tracker_snapshot())
    import release.live_corners as _lc
    _lc._ALIGN_FULL_PROBE_EVERY = max(1, int(args.align_full_every))

    debug_session: Optional[str] = None
    if args.debug_dir and not args.no_debug:
        debug_root = args.debug_dir if os.path.isabs(args.debug_dir) else os.path.join(_ROOT, args.debug_dir)
        debug_session = new_session_dir(debug_root)
        _log(log_fh, f'[live_panel] debug zapis: {debug_session}')
        _log(log_fh, '[live_panel] po sesji: python3 -m pipelines.analyze_live_debug --root live_debug --html')

    processed = 0
    last_show: Optional[np.ndarray] = None
    last_snap_preds: List[dict] = []
    last_snap_lines: List[str] = []
    consensus_preds: List[dict] = []
    consensus_lines: List[str] = []
    writer: Optional[cv2.VideoWriter] = None
    report_draft_seeded = False
    report_pause_until = 0.0
    autonomy_pause_until = 0.0
    autonomy_hold_active = False

    def autonomy_pause_remaining() -> float:
        return max(0.0, float(autonomy_pause_until) - time.monotonic())

    def report_pause_remaining() -> float:
        if flight_controller is not None:
            return flight_controller.depart_remaining_s()
        return max(0.0, float(report_pause_until) - time.monotonic())

    def vision_pause_remaining() -> float:
        return max(autonomy_pause_remaining(), report_pause_remaining())

    def analysis_paused() -> bool:
        if mission_manager is not None and mission_manager.state.mission_done():
            return True
        if time.monotonic() < autonomy_pause_until:
            return True
        if flight_controller is not None:
            return not bool(flight_controller.vision_active)
        return report_pause_remaining() > 0.0

    def snapshots_paused() -> bool:
        return analysis_paused()

    def _tick_flight_controller() -> None:
        if flight_controller is None:
            return
        _dist_fc: Optional[float] = None
        if module_a_po is not None and getattr(module_a_po, 'ok', False):
            _dist_fc = float(module_a_po.distance_m)
        _reliable_fc = False
        if _run_analysis and reliable:
            _reliable_fc = True
        flight_controller.update(
            distance_m=_dist_fc,
            panel_full=(
                mission_manager.panel_is_full() if mission_manager else False
            ),
            snap_count=(
                mission_manager.snapshot_count() if mission_manager else 0
            ),
            snap_max=(
                int(mission_manager.state.snapshots_per_panel)
                if mission_manager else 5
            ),
            mission_done=(
                mission_manager.state.mission_done() if mission_manager else False
            ),
            panel_reliable=_reliable_fc,
        )

    def _apply_mission_settings_patch(patch: dict) -> None:
        if flight_controller is None or not patch:
            return
        from release.mission_settings import merge_mission_settings

        merged = merge_mission_settings(flight_controller.settings, patch)
        flight_controller.apply_settings(merged)
        if mission_manager is not None and 'snapshots_per_panel' in merged:
            n = max(1, int(merged['snapshots_per_panel']))
            mission_manager.state.snapshots_per_panel = n
            for _st in mission_manager.stores.values():
                _st.max_snapshots = n

    def apply_report_send(
        panel_id: str,
        lines: List[str],
        *,
        source: str = 'web',
        advance_flight: bool = True,
    ) -> bool:
        nonlocal consensus_preds, consensus_lines, last_snap_preds, last_snap_lines
        nonlocal snapshot_store, report_draft_seeded, report_pause_until
        from module_panel.competition_report import validate_competition_report_lines

        ok, errors = validate_competition_report_lines(
            lines,
            min_cards=0,
            max_cards=4,
            expected_panel=str(panel_id).upper()[:1],
            allow_empty=True,
        )
        if not ok:
            for err in errors:
                _log(log_fh, f'[report] odrzucono ({source}): {err}')
            return False
        if mission_manager is not None:
            preds = predictions_from_report_lines(lines) if lines else list(consensus_preds)
            next_panel, n_cleared = mission_manager.submit_panel(
                panel_id,
                report_lines=lines,
                predictions=preds,
                meta={'operator_edited': True, 'source': source},
            )
            consensus_preds = []
            consensus_lines = []
            last_snap_preds = []
            last_snap_lines = []
            snapshot_store = mission_manager.current_store()
            _log(
                log_fh,
                f'[mission] Wyslano panel {panel_id} ({source}): usunieto {n_cleared} migawek',
            )
            if next_panel:
                _log(log_fh, f'[mission] Nastepny panel: {next_panel}')
                if panel_review_ui is not None:
                    panel_review_ui.bind_panel_id(next_panel)
                    panel_review_ui.set_report_lines([], force=True)
            else:
                _log(log_fh, '[mission] Wszystkie panele zakonczone')
        if web_publisher is not None:
            web_publisher.write_draft_lines([])
        report_draft_seeded = False
        if flight_controller is not None and advance_flight:
            flight_controller.on_report_sent()
            _log(
                log_fh,
                '[mission] Wyślij: analiza OFF, lot do następnego panelu '
                f'(cruise={flight_controller.speed:.2f})',
            )
        elif flight_controller is None:
            pause_s = max(0.0, float(args.report_send_pause_s))
            if pause_s > 0.0:
                report_pause_until = time.monotonic() + pause_s
                _log(log_fh, f'[report] Pauza {pause_s:.0f}s po wysyłce — bez nowych migawek')
        return True

    def handle_autonomy_hold_started(timeout_from_host: Optional[float] = None) -> None:
        """hold_started (edge): start holdu — zbieraj migawki, bez raportu."""
        nonlocal autonomy_hold_active
        from release.integration_ws import hold_duration_s

        duration_s = hold_duration_s(timeout_from_host)
        autonomy_hold_active = True
        _log(
            log_fh,
            f'[autonomy] hold_started — zbieranie migawek (~{duration_s:.0f}s, lokalny timer)',
        )

    def handle_autonomy_hold_finished(*, trigger: str) -> None:
        """Koniec holdu: raport z migawek na :8088, reset migawek, pauza analizy."""
        nonlocal autonomy_pause_until, autonomy_hold_active
        nonlocal consensus_preds, consensus_lines
        nonlocal last_snap_preds, last_snap_lines, snapshot_store, report_draft_seeded

        from release.integration_ws import autonomy_analysis_pause_s

        if not autonomy_hold_active:
            _log(log_fh, f'[autonomy] {trigger} — pominięto (brak aktywnego holdu)')
            return
        autonomy_hold_active = False

        pause_s = autonomy_analysis_pause_s()
        autonomy_pause_until = time.monotonic() + pause_s

        panel = (
            str(mission_manager.current_panel)
            if mission_manager is not None
            else str(args.panel_id)
        ).upper()[:1]

        lines: List[str] = []
        if args.report_mode == 'preset':
            from release.web_dashboard import resolve_report_lines_at_click

            lines = resolve_report_lines_at_click(list(preset_reports.get(panel, [])))
        else:
            lines = list(consensus_lines)
            if web_publisher is not None:
                draft = web_publisher.read_draft_lines()
                if draft:
                    lines = list(draft)

        if not lines:
            _log(log_fh, f'[autonomy] {trigger} — brak raportu z migawek (panel {panel})')
        else:
            if web_publisher is not None:
                web_publisher.add_broadcast_report_lines(panel, lines)
                _log(
                    log_fh,
                    f'[autonomy] {trigger} → raport z migawek ({len(lines)} linii) na :8088',
                )
            if apply_report_send(
                panel,
                lines,
                source='autonomy-hold',
                advance_flight=False,
            ):
                _log(
                    log_fh,
                    f'[autonomy] {trigger} → migawki wyzerowane, detekcja OFF na {pause_s:.0f}s',
                )
            else:
                _log(log_fh, f'[autonomy] {trigger} — raport z migawek odrzucony (walidacja)')

    def handle_autonomy_hold_stopped(*, reason: str = 'server') -> None:
        """hold_stopped (edge): koniec holdu — raport + pauza analizy."""
        nonlocal autonomy_hold_active
        if reason == 'disconnect':
            autonomy_hold_active = False
            _log(log_fh, '[autonomy] WS rozłączony — anulowano hold bez raportu')
            return
        handle_autonomy_hold_finished(trigger='hold_stopped')

    def poll_autonomy_events() -> None:
        if ws_subscriber is None:
            return
        while True:
            ev = ws_subscriber.poll()
            if ev is None:
                break
            name = str(ev.get('event', ''))
            if name == 'hold_started':
                handle_autonomy_hold_started(ev.get('timeout'))
            elif name == 'hold_stopped':
                handle_autonomy_hold_stopped(
                    reason=str(ev.get('reason', 'server')),
                )
            elif name == 'hold_expired':
                handle_autonomy_hold_finished(trigger='hold_expired')

    def maybe_seed_report_draft() -> None:
        nonlocal report_draft_seeded
        if args.report_mode == 'preset':
            return
        if web_publisher is None or not consensus_lines:
            return
        if mission_manager is not None and not mission_manager.panel_is_full():
            return
        if report_draft_seeded:
            return
        if web_publisher.read_draft_lines():
            report_draft_seeded = True
            return
        web_publisher.write_draft_lines(list(consensus_lines))
        report_draft_seeded = True
        n_cards = len(consensus_preds) if consensus_preds else len(consensus_lines)
        _log(
            log_fh,
            f'[report] Szkic raportu ({n_cards} kart) — gotowy do edycji w WWW',
        )
    if args.out_video and args.corner_mode != 'yolo_pose':
        _log(log_fh, '[live_panel] hint: dla nag3 z YOLO uzyj --corner-mode yolo_pose')
    camera_rotation_in_feed = False
    try:
        _stream_dash: List[Optional[np.ndarray]] = [None]
        if args.video:
            vpath = os.path.abspath(args.video)
            src = VideoSource(VideoConfig(path=vpath, rotate_deg=args.rotate, loop=not args.no_loop))
        else:
            from release.camera_source import camera_config_from_env

            use_env_cam = bool(
                (os.environ.get('DRONIADA_CAP_PIPELINE') or '').strip()
                or os.environ.get('DRONIADA_CAMERA_DEVICE')
                or os.environ.get('DRONIADA_CAMERA_FOURCC')
                or os.environ.get('DRONIADA_CAMERA_BRIGHTNESS')
                or os.environ.get('DRONIADA_CAMERA_WIDTH')
                or os.environ.get('DRONIADA_CAMERA_HEIGHT')
            )
            if use_env_cam:
                cam_cfg = camera_config_from_env(
                    default_device=args.camera,
                    default_width=args.width,
                    default_height=args.height,
                )
            else:
                cam_cfg = CameraConfig(
                    device=args.camera,
                    width=args.width,
                    height=args.height,
                )
            _preview_w = int(args.preview_width)
            _rotate_in_feed = False
            _cam_cb = None
            if web_publisher is not None and live_frame_buf is not None:
                def _cam_cb(
                    ok: bool,
                    bgr,
                    fid: str,
                    *,
                    _wp=web_publisher,
                    _fb=live_frame_buf,
                ) -> None:
                    try:
                        if not ok or bgr is None:
                            return
                        frame = (
                            apply_rotate(bgr, args.rotate)
                            if (args.rotate and not _rotate_in_feed)
                            else bgr
                        )
                        _fb.publish(True, frame, fid)
                        _wp.set_stream_camera_frame(frame)
                    except Exception:
                        pass
            from release.gst_mjpeg_camera import (
                GstMjpegCameraFeed,
                should_use_gst_capture,
                wants_stream_passthrough,
            )

            _stream_src = os.environ.get('DRONIADA_STREAM_SOURCE', 'vis').strip().lower()
            use_gst_cam = should_use_gst_capture()
            if use_gst_cam:
                _rotate = int(args.rotate or 0)
                _fps = int(os.environ.get('DRONIADA_CAMERA_FPS', '30') or 30)
                _dev = cam_cfg.device
                if isinstance(_dev, int):
                    _dev = os.environ.get('DRONIADA_CAMERA_DEVICE') or f'/dev/video{_dev}'
                _passthrough = wants_stream_passthrough()

                def _mjpeg_cb(data: bytes, *, _wp=web_publisher) -> None:
                    try:
                        _wp.push_stream_jpeg_bytes(data)
                    except Exception:
                        pass

                def _gst_bgr_cb(bgr, *, _cb=_cam_cb) -> None:
                    if _cb is not None:
                        _cb(True, bgr, 'gst')

                src = GstMjpegCameraFeed(
                    str(_dev),
                    width=int(cam_cfg.width or 0),
                    height=int(cam_cfg.height or 0),
                    fps=_fps,
                    rotate=_rotate,
                    on_mjpeg=_mjpeg_cb if (_passthrough and web_publisher is not None) else None,
                    on_bgr=_gst_bgr_cb,
                )
                if _passthrough and web_publisher is not None:
                    web_publisher.enable_stream_passthrough(enabled=True)
                if _rotate:
                    _rotate_in_feed = True
                    camera_rotation_in_feed = True
                _gst_meta = getattr(src, 'open_meta', {})
                _log(
                    log_fh,
                    f'[live_panel] GStreamer MJPEG: passthrough={_passthrough} '
                    f'hw_jpeg={_gst_meta.get("gstreamer_hw_jpeg")} '
                    f'rotate={_rotate} stream={_stream_src}',
                )
            else:
                from release.camera_feed_thread import SharedCameraFeed
                src = SharedCameraFeed(cam_cfg, on_frame=_cam_cb)
        use_camera_feed = not args.video
        with src:
            if use_camera_feed:
                import time as _time
                for _ in range(50):
                    meta = getattr(src, 'open_meta', None)
                    if meta:
                        _log(log_fh, f'[live_panel] kamera: {meta}')
                        if use_gst_cam and meta.get('error'):
                            _log(log_fh, f'[live_panel] GStreamer błąd: {meta["error"]}')
                        if use_gst_cam and meta.get('gstreamer_pipeline'):
                            _log(log_fh, f'[live_panel] GST pipeline: {meta["gstreamer_pipeline"]}')
                        break
                    _time.sleep(0.05)
            elif getattr(src, 'open_meta', None):
                _log(log_fh, f'[live_panel] kamera: {src.open_meta}')
            last_proc = 0.0
            overlay_state: Optional[Dict[str, Any]] = None
            video_fps = float(getattr(src, 'fps', 25.0)) if args.video else 25.0
            frame_period_ms = max(1, int(1000.0 / max(video_fps, 1.0)))
            video_realtime = bool(args.video and (show_preview or args.out_video))
            analysis_pause_logged = False

            def _overlay_state_from_locals() -> Dict[str, Any]:
                return {
                    'module_a_po': module_a_po,
                    'corners_px': corners_px.copy() if corners_px is not None else None,
                    'det': list(det),
                    'live_preds': list(live_preds),
                    'panel_present': panel_present,
                    'tracker_hold': tracker_hold,
                    'reliable': reliable,
                    'corner_src': corner_src,
                    'xy_back': xy_back,
                    'reproj': reproj,
                    'angle': angle,
                    'category': category,
                    'corner_meta': dict(corner_meta),
                    'hom_inliers': hom_inliers,
                    'grid_overlap_ratio': grid_overlap_ratio,
                    'h_mat': h_mat.copy() if h_mat is not None else None,
                    'opencv_grid_x': list(opencv_grid_x) if opencv_grid_x is not None else None,
                    'opencv_grid_y': list(opencv_grid_y) if opencv_grid_y is not None else None,
                    'warped_shape': (int(warped.shape[0]), int(warped.shape[1])),
                    'latch_txt': latch_txt,
                    'analysis_paused': not _run_analysis,
                    'flight_controller': flight_controller,
                }

            def _publish_overlay_cache() -> None:
                nonlocal overlay_state
                overlay_state = _overlay_state_from_locals()
                if live_overlay_cache is not None:
                    live_overlay_cache.update(overlay_state)

            _async_analysis = os.environ.get(
                'DRONIADA_ASYNC_ANALYSIS', '1',
            ).strip().lower() not in ('0', 'false', 'no', 'off')
            _analysis_ctx: Dict[str, Any] = {}

            def _ingest_analysis_pkg(pkg: Dict[str, Any]) -> None:
                nonlocal module_a_po, corners_px, corner_src, corner_meta
                nonlocal det, warped_dets, cand_rows, pan_meta, warped, work_bgr, gs_bgr
                nonlocal preds, reliable, reproj, xy_back, angle, category, hom_inliers
                nonlocal grid_overlap_ratio, opencv_grid_x, opencv_grid_y
                nonlocal hough_grid_x, hough_grid_y, live_preds, live_lines
                nonlocal latched_preds, latched_lines, latch_txt
                nonlocal panel_present, panel_presence_reason, tracker_hold
                nonlocal h_mat, black_th, gs_bgr, cxy_locked_announced, fid, panel_id_live

                fid = str(pkg['fid'])
                panel_id_live = str(pkg['panel_id_live'])
                work_bgr = pkg['work_bgr']
                warped = pkg['warped']
                module_a_po = pkg['module_a_po']
                corners_px = pkg['corners_px']
                corner_src = str(pkg['corner_src'])
                corner_meta = dict(pkg['corner_meta'])
                det = list(pkg['det'])
                warped_dets = list(pkg['warped_dets'])
                cand_rows = list(pkg['cand_rows'])
                pan_meta = dict(pkg['pan_meta'])
                live_preds = list(pkg['live_preds'])
                live_lines = list(pkg['live_lines'])
                latched_preds = pkg['latched_preds']
                latched_lines = pkg['latched_lines']
                latch_txt = pkg['latch_txt']
                reliable = bool(pkg['reliable'])
                reproj = float(pkg['reproj'])
                xy_back = str(pkg['xy_back'])
                angle = int(pkg['angle'])
                category = str(pkg['category'])
                hom_inliers = int(pkg['hom_inliers'])
                grid_overlap_ratio = float(pkg['grid_overlap_ratio'])
                opencv_grid_x = pkg['opencv_grid_x']
                opencv_grid_y = pkg['opencv_grid_y']
                hough_grid_x = pkg['hough_grid_x']
                hough_grid_y = pkg['hough_grid_y']
                panel_present = bool(pkg['panel_present'])
                panel_presence_reason = str(pkg['panel_presence_reason'])
                tracker_hold = bool(pkg['tracker_hold'])
                h_mat = pkg['h_mat']
                black_th = pkg['black_th']
                gs_bgr = pkg.get('gs_bgr')
                for ev in pkg.get('latch_events') or []:
                    if ev == 'cxy_locked':
                        cxy_locked_announced = True
                    elif str(ev).startswith('latch_line:'):
                        _log(log_fh, f'[{fid}]   latch {str(ev)[11:]}')

            if _async_analysis:
                from release.live_frame_analysis import run_frame_analysis

                def _analysis_job(bgr_in, fid_in, panel_in):
                    _analysis_ctx['cxy_locked_announced'] = cxy_locked_announced
                    _analysis_ctx['last_snap_preds'] = last_snap_preds
                    _analysis_ctx['last_snap_lines'] = last_snap_lines
                    return run_frame_analysis(_analysis_ctx, bgr_in, fid_in, panel_in)

                analysis_worker = DropQueueWorker(_analysis_job, name='droniada-yolo')
                analysis_worker.start()

            while True:
                loop_t0 = time.monotonic()
                analysis_ready = False
                if analysis_worker is not None:
                    _pkg = analysis_worker.poll_result()
                    if _pkg is not None:
                        _ingest_analysis_pkg(_pkg)
                        analysis_ready = True
                    _aerr = analysis_worker.poll_error()
                    if _aerr is not None:
                        _log(log_fh, f'[live_panel] analiza (wątek): {_aerr}')
                poll_autonomy_events()
                if web_publisher is not None:
                    from release.tracker_tuning import apply as apply_tracker_tuning

                    _ru = web_publisher.poll_runtime_update()
                    if _ru:
                        runtime_gate.update(_ru)
                        _refresh_snapshot_store_limits(snapshot_store, runtime_gate, args)
                        if mission_manager is not None:
                            for _st in mission_manager.stores.values():
                                _refresh_snapshot_store_limits(_st, runtime_gate, args)
                    _tune = web_publisher.poll_tracker_update()
                    if _tune:
                        apply_tracker_tuning(_tune, log_fn=lambda msg: _log(log_fh, msg))
                    _ms_top = web_publisher.poll_mission_update()
                    if _ms_top:
                        _apply_mission_settings_patch(_ms_top)
                if use_camera_feed:
                    ok_cap, bgr, fid = src.get_latest()
                else:
                    ok_cap, bgr, fid = src.read()
                if not ok_cap or bgr is None:
                    if args.video:
                        break
                    if not use_camera_feed:
                        _log(log_fh, f'[{fid}] capture_fail')
                    if show_window:
                        cv2.imshow(live_window, np.zeros((480, 640, 3), np.uint8))
                        if cv2.waitKey(1) & 255 == ord('q'):
                            break
                    if not use_camera_feed:
                        time.sleep(0.05)
                    else:
                        time.sleep(0.02)
                    continue

                if args.rotate and args.video is None and not camera_rotation_in_feed:
                    bgr = apply_rotate(bgr, args.rotate)

                if web_publisher is not None and bgr is not None:
                    web_publisher.set_stream_camera_frame(bgr)
                    if live_frame_buf is not None:
                        live_frame_buf.publish(True, bgr, fid)

                now = time.monotonic()

                from release.tracker_tuning import get_interval_ms

                live_interval_ms = get_interval_ms()
                eff_interval = 0 if (args.out_video or video_realtime) else live_interval_ms
                if (
                    eff_interval > 0
                    and (now - last_proc) * 1000.0 < eff_interval
                    and not analysis_ready
                ):
                    if show_window and not video_realtime:
                        if last_show is not None:
                            cv2.imshow(live_window, last_show)
                        else:
                            cv2.imshow(live_window, _compose_preview(bgr, bgr, target_w=args.preview_width))
                    continue
                last_proc = now

                processed += 1
                panel_id_live = str(args.panel_id)
                if mission_manager is not None:
                    panel_id_live = str(mission_manager.current_panel)
                    snapshot_store = mission_manager.current_store()

                _run_analysis = not analysis_paused()
                if not _run_analysis and not analysis_pause_logged:
                    _phase = (
                        flight_controller.phase.value
                        if flight_controller is not None else 'pauza'
                    )
                    _log(
                        log_fh,
                        f'[live_panel] ANALIZA OFF (faza={_phase}) — bez YOLO/skanu, lot dalej',
                    )
                    analysis_pause_logged = True
                elif _run_analysis:
                    analysis_pause_logged = False

                module_a_po = None
                corners_px = None
                corner_src = 'paused'
                corner_meta: dict = {}
                det: List[Tuple[int, float, float, float, float]] = []
                warped_dets: List[dict] = []
                cand_rows: List[dict] = []
                pan_meta: dict = {'err': 'analysis_paused'}
                warped = bgr.copy()
                work_bgr = bgr
                gs_bgr: Optional[np.ndarray] = None
                preds: List[dict] = []
                reliable = False
                reproj = 999.0
                xy_back = '-'
                angle = 0
                category = 'horizontal'
                hom_inliers = 0
                grid_overlap_ratio = 0.0
                opencv_grid_x: Optional[List[float]] = None
                opencv_grid_y: Optional[List[float]] = None
                hough_grid_x: Optional[List[float]] = None
                hough_grid_y: Optional[List[float]] = None
                live_preds: List[dict] = []
                live_lines: List[str] = []
                latched_preds: Optional[List[dict]] = None
                latched_lines: Optional[List[str]] = None
                latch_txt: Optional[str] = None
                panel_present = False
                panel_presence_reason = 'analysis_paused'
                tracker_hold = False
                h_mat = None
                black_th = None

                if _run_analysis and analysis_worker is not None:
                    _analysis_ctx.update({
                        'args': args,
                        'pose_rt': pose_rt,
                        'profile': profile,
                        'cam_calib': cam_calib,
                        'calib_path': calib_path,
                        'run_live_color': run_live_color,
                        'cxy_latch': cxy_latch,
                        'log_fh': log_fh,
                        'color_on_snapshot': color_on_snapshot,
                        'cxy_latch_dir': cxy_latch_dir,
                        'show_window': show_window,
                    })
                    analysis_worker.submit(bgr.copy(), fid, panel_id_live)
                elif _run_analysis:
                    h, w = bgr.shape[:2]
                    k, dist, intr_meta = resolve_intrinsics(
                        (h, w),
                        profile=profile,
                        zoom_ratio=float(args.zoom_ratio),
                        calib_path=cam_calib,
                    )
                    det: List[Tuple[int, float, float, float, float]] = []
                    warped_dets: List[dict] = []
                    cand_rows: List[dict] = []
                    module_a_po: Optional[PoseFrameOutput] = None
                    corners_px: Optional[np.ndarray] = None
                    corner_src = 'none'
                    corner_meta: dict = {}

                    a_yolo = pose_rt is not None and pose_rt.cfg.corner_source == 'yolo_pose'
                    b_yolo = args.corner_mode == 'yolo_pose'
                    shared_yolo = a_yolo and b_yolo

                    if shared_yolo:
                        corners_px, corner_src, corner_meta = detect_corners_live(
                            bgr,
                            k,
                            dist,
                            return_all_candidates=False,
                            corner_mode='yolo_pose',
                            use_tracker=not args.no_stabilize,
                        )
                        raw_cand = corner_meta.get('candidate_rows')
                        if isinstance(raw_cand, list):
                            cand_rows = raw_cand
                        if corner_meta.get('tracker_held'):
                            corner_src = f'{corner_src}+hold'

                    if pose_rt is not None:
                        module_a_po = pose_rt.process_bgr(
                            bgr,
                            fid,
                            det=[],
                            k=k,
                            dist=dist,
                            corners_px=corners_px if shared_yolo else None,
                            corners_meta=corner_meta if shared_yolo else None,
                            record=False,
                        )
                        if module_a_po.ok:
                            integ = module_a_po.to_integration_dict(panel_id=panel_id_live)
                            _log(
                                log_fh,
                                f'[{fid}] MODUL A ok stand={module_a_po.panel_angle_category} '
                                f'd={module_a_po.distance_m:.2f}m reproj={module_a_po.reproj_mean_px:.1f}',
                            )
                            _log(log_fh, f'[{fid}] MODUL A dict {integ}')
                        else:
                            reason = module_a_po.meta.get('reason', '?')
                            _log(log_fh, f'[{fid}] MODUL A fail reason={reason}')

                    if not shared_yolo:
                        corners_px, corner_src, corner_meta = detect_corners_live(
                            bgr,
                            k,
                            dist,
                            return_all_candidates=False,
                            corner_mode=args.corner_mode,
                            use_tracker=not args.no_stabilize,
                        )
                        raw_cand = corner_meta.get('candidate_rows')
                        if isinstance(raw_cand, list):
                            cand_rows = raw_cand
                        if corner_meta.get('tracker_held'):
                            corner_src = f'{corner_src}+hold'

                    if (
                        corners_px is None
                        and module_a_po is not None
                        and module_a_po.ok
                        and module_a_po.corners_px is not None
                        and module_a_po.corners_px.shape == (4, 2)
                    ):
                        corners_px = module_a_po.corners_px.astype(np.float32).copy()
                        corner_src = (
                            f'{module_a_po.meta.get("corner_source", "yolo_pose")}+module_a'
                        )
                        corner_meta = dict(module_a_po.meta or {})
                        corner_meta.setdefault(
                            'reproj_mean_px',
                            float(module_a_po.reproj_mean_px),
                        )
                        corner_meta.setdefault('method', 'yolo_pose')

                    work_bgr = bgr
                    gs_bgr: Optional[np.ndarray] = None
                    black_th = None
                    if corners_px is not None:
                        from release.panel_black import calibrate_panel_black_from_corners

                        black_th = calibrate_panel_black_from_corners(bgr, corners_px)
                    if args.greenscreen:
                        from release.greenscreen_panel import apply_workshop_greenscreen

                        gs_bgr = apply_workshop_greenscreen(
                            bgr, corners_px if corners_px is not None else None,
                            calibrate_black=True,
                        )
                        work_bgr = gs_bgr

                    h_mat = None
                    grid_overlap_ratio = 0.0
                    opencv_grid_x: Optional[List[float]] = None
                    opencv_grid_y: Optional[List[float]] = None
                    hough_grid_x: Optional[List[float]] = None
                    hough_grid_y: Optional[List[float]] = None

                    if corners_px is None:
                        pan_meta = {'err': 'no_corners'}
                        warped = work_bgr.copy()
                        preds: List[dict] = []
                        reliable = False
                        reproj = 999.0
                        xy_back = '-'
                        angle = 0
                        category = 'horizontal'
                    elif (
                        not args.cxy_latch
                        and (
                            args.corners_only
                            or (args.corner_mode == 'yolo_pose' and args.out_video)
                        )
                    ):
                        warped, h_mat = warp_panel_rect(work_bgr, corners_px)
                        preds = []
                        reproj = float(corner_meta.get('reproj_mean_px', 0.0))
                        pan_meta = {
                            'reproj_mean_px': reproj,
                            'grid_xy_reliable': reproj < args.max_reproj_reliable,
                            'xy_backend_selected': 'yolo_pose',
                            'corner_source': corner_src,
                        }
                        reliable = bool(pan_meta['grid_xy_reliable'])
                        xy_back = 'yolo_pose'
                        angle = 0
                        category = 'horizontal'
                    else:
                        warped, h_mat = warp_panel_rect(work_bgr, corners_px)
                        if run_live_color:
                            det, warped_dets = detect_cards_live(
                                work_bgr, corners_px, h_mat, warped, black_thresholds=black_th,
                            )

                        if args.corner_mode == 'yolo_pose' and args.cxy_latch:
                            pan, preds, det, warped_dets, reproj, reliable, xy_back, angle, category, pan_meta = (
                                _analyze_cxy_from_corners(
                                    work_bgr,
                                    corners_px,
                                    corner_src,
                                    k,
                                    dist,
                                    panel_id=panel_id_live,
                                    xy_mode=args.xy_mode,
                                    angle_source=args.angle_source,
                                    calib_path=calib_path,
                                    cam_calib=cam_calib,
                                    max_reproj_px=float(args.max_reproj_reliable),
                                    min_homography_inliers=int(args.min_homography_inliers),
                                    no_color_detect=not run_live_color,
                                )
                            )
                            warped = pan.warped_bgr
                        else:
                            pan = analyze_panel_image(
                                work_bgr,
                                det,
                                k=k,
                                dist=dist,
                                xy_mode=args.xy_mode,
                                angle_source=args.angle_source,
                                angle_calibration_path=calib_path,
                                camera_calib_path=cam_calib,
                                allowed_orbit_steps=None,
                                min_homography_inliers=int(args.min_homography_inliers),
                                max_reproj_px=float(args.max_reproj_reliable),
                                corners_px=corners_px,
                                corner_source=corner_src,
                            )
                            pan_meta = pan.meta
                            warped = pan.warped_bgr
                            preds = list(pan.predictions)
                            if warped_dets:
                                live_preds = warped_dets_to_predictions(warped_dets)
                                preds = live_preds
                                pan_meta['xy_source'] = 'live_warp_grid_hsv'
                                pan_meta['xy_backend_selected'] = 'live_warp_grid_hsv'
                            reliable = bool(pan.meta.get('grid_xy_reliable', False))
                            reproj = float(pan.meta.get('reproj_mean_px', 999.0))
                            corner_src = str(pan.meta.get('corner_source', corner_src))
                            if corner_meta.get('reproj_mean_px') is not None and args.corner_mode != 'yolo_pose':
                                reproj = float(corner_meta.get('reproj_mean_px', reproj))
                            xy_back = str(pan.meta.get('xy_backend_selected', pan.meta.get('xy_source', '?')))
                            angle = int(pan.report_angle_deg)
                            category = str(pan.panel_angle_category)

                    hom_inliers = int(pan_meta.get('homography_inliers', 0))
                    if corners_px is not None and pan_meta.get('err') != 'no_corners':
                        from module_panel.grid_overlap import measure_panel_opencv_grid_overlap

                        grid_overlap_ratio, g_ov = measure_panel_opencv_grid_overlap(
                            warped,
                            grid_lines_x=pan_meta.get('grid_lines_x'),
                            grid_lines_y=pan_meta.get('grid_lines_y'),
                            line_tol_frac=float(args.snapshot_grid_line_tol),
                        )
                        pan_meta['grid_overlap_ratio'] = float(grid_overlap_ratio)
                        pan_meta['grid_overlap_method'] = str(g_ov.get('overlap_method', 'triple_grid'))
                        pan_meta['triple_consensus'] = float(g_ov.get('triple_consensus', 0.0))
                        pan_meta['spacing_min'] = float(g_ov.get('spacing_min', 0.0))
                        pan_meta['warp_panel_coverage'] = float(g_ov.get('warp_panel_coverage', 0.0))
                        pan_meta['grid_cell_iou'] = float(g_ov.get('cell_iou', 0.0))
                        pan_meta['grid_line_match_ratio'] = float(g_ov.get('line_match_ratio', 0.0))
                        pan_meta['hough_line_ok'] = bool(g_ov.get('hough_line_ok', False))
                        if g_ov.get('hough_lines_x') and g_ov.get('hough_lines_y'):
                            hough_grid_x = list(g_ov['hough_lines_x'])
                            hough_grid_y = list(g_ov['hough_lines_y'])
                            pan_meta['hough_lines_x'] = hough_grid_x
                            pan_meta['hough_lines_y'] = hough_grid_y
                        if g_ov.get('grid_lines_x') and g_ov.get('grid_lines_y'):
                            opencv_grid_x = list(g_ov['grid_lines_x'])
                            opencv_grid_y = list(g_ov['grid_lines_y'])
                            pan_meta['grid_lines_x'] = opencv_grid_x
                            pan_meta['grid_lines_y'] = opencv_grid_y
                    live_preds = list(preds)
                    live_lines = predictions_to_report_lines(panel_id_live, angle, live_preds)

                    if args.require_reliable and not reliable:
                        live_lines = []
                        if not args.cxy_latch:
                            live_preds = []

                    latch_txt: Optional[str] = None
                    latched_preds: Optional[List[dict]] = None
                    latched_lines: Optional[List[str]] = None
                    if cxy_latch is not None:
                        was_locked = cxy_latch.locked
                        latch_meta = cxy_latch.update(
                            frame_id=fid,
                            reliable=reliable,
                            reproj_mean_px=reproj,
                            homography_inliers=hom_inliers,
                            predictions=live_preds,
                            report_lines=live_lines,
                            meta=pan_meta,
                        )
                        pan_meta.update(latch_meta)
                        snap = cxy_latch.snapshot
                        if cxy_latch.locked and snap is not None:
                            latched_preds = list(snap.predictions)
                            latched_lines = list(snap.report_lines)
                            latch_txt = (
                                f'ZAMKNIĘTE @ {snap.frame_id}  reproj={snap.reproj_mean_px:.1f}px  '
                                f'inl={snap.homography_inliers}'
                            )
                        elif snap is not None:
                            latch_txt = (
                                f'Oczekiwanie… best @ {snap.frame_id} '
                                f'({int(latch_meta.get("cxy_stable_run", 0))}/{cxy_latch.min_stable_frames})'
                            )
                        if cxy_latch.locked and not cxy_locked_announced:
                            cxy_locked_announced = True
                            _log(log_fh, f'[{snap.frame_id}] === CXY ZATRZAŚNIĘTE (siatka idealna) ===')
                            for line in latched_lines or []:
                                _log(log_fh, f'[{snap.frame_id}]   {line}')
                            if snap is not None and corners_px is not None and warped is not None:
                                latch_preds_out = list(last_snap_preds) if color_on_snapshot else list(snap.predictions)
                                latch_lines_out = list(last_snap_lines) if color_on_snapshot else list(snap.report_lines)
                                latch_dets_out = list(warped_dets) if run_live_color else []
                                try:
                                    paths = save_cxy_latch_artifacts(
                                        cxy_latch_dir,
                                        frame_id=snap.frame_id,
                                        work_bgr=work_bgr,
                                        corners_tltrbrbl=corners_px,
                                        warped_bgr=warped,
                                        preds=latch_preds_out,
                                        warped_dets=latch_dets_out,
                                        reproj_px=float(snap.reproj_mean_px),
                                        homography_inliers=int(snap.homography_inliers),
                                        report_lines=latch_lines_out,
                                        meta=dict(snap.meta or {}),
                                    )
                                    _log(log_fh, f'[{snap.frame_id}] podglad latch: {paths.get("dashboard")}')
                                    latched_vis = compose_latch_dashboard(
                                        work_bgr,
                                        corners_px,
                                        warped,
                                        latch_preds_out,
                                        warped_dets=latch_dets_out,
                                        frame_id=snap.frame_id,
                                        reproj_px=float(snap.reproj_mean_px),
                                        homography_inliers=int(snap.homography_inliers),
                                        report_lines=latch_lines_out,
                                    )
                                    if show_window and not args.dashboard:
                                        _show_latch_window(latched_vis, args.preview_width)
                                        _log(log_fh, f'[{snap.frame_id}] okno podgladu: {_LATCH_WINDOW}')
                                except Exception as exc:
                                    _log(log_fh, f'[{snap.frame_id}] blad zapisu latch preview: {exc}')

                    from release.panel_presence import resolve_panel_presence

                    panel_present, panel_presence_reason, _presence_detail = resolve_panel_presence(
                        corners_px,
                        corner_meta,
                        corner_src,
                        image_shape=bgr.shape[:2],
                        reliable_b=bool(reliable),
                        reproj_b=float(reproj),
                        max_reproj_b=float(args.max_reproj_reliable),
                        analyze_err=str(pan_meta.get('err') or ''),
                    )
                    tracker_hold = bool(corner_meta.get('tracker_held')) or 'hold' in str(corner_src)

                    if not panel_present:
                        _log(log_fh, f'[{fid}] brak panelu w kadrze ({panel_presence_reason})')
                    elif pan_meta.get('err') == 'no_corners':
                        _log(log_fh, f'[{fid}] brak rogów (analyze)')
                    else:
                        _log(
                            log_fh,
                            f'[{fid}] reliable={reliable} reproj={reproj:.2f} corner={corner_src} '
                            f'xy={xy_back} det={len(det)} live_preds={len(live_preds)}',
                        )
                        for line in live_lines:
                            _log(log_fh, f'[{fid}]   live {line}')
                        if latched_lines:
                            for line in latched_lines:
                                _log(log_fh, f'[{fid}]   latch {line}')
                    analysis_ready = True
                else:
                    analysis_ready = True

                if not analysis_ready:
                    if show_window and not video_realtime and last_show is not None:
                        cv2.imshow(live_window, last_show)
                    continue

                _tick_flight_controller()

                vis = work_bgr.copy()
                _draw_vis_overlay(
                    vis,
                    module_a_po=module_a_po,
                    corners_px=corners_px,
                    det=det,
                    live_preds=live_preds,
                    panel_present=panel_present,
                    tracker_hold=tracker_hold,
                    reliable=reliable,
                    corner_src=corner_src,
                    xy_back=xy_back,
                    reproj=reproj,
                    angle=angle,
                    category=category,
                    corner_meta=corner_meta,
                    hom_inliers=hom_inliers,
                    grid_overlap_ratio=grid_overlap_ratio,
                    h_mat=h_mat,
                    opencv_grid_x=opencv_grid_x,
                    opencv_grid_y=opencv_grid_y,
                    warped_shape=(int(warped.shape[0]), int(warped.shape[1])),
                    latch_txt=latch_txt,
                    analysis_paused=not _run_analysis,
                    flight_controller=flight_controller,
                )
                _publish_overlay_cache()
                if web_publisher is not None:
                    web_publisher.set_stream_vis_frame(vis)

                thumb = _draw_warped_inset(warped, live_preds, warped_dets)
                if args.dashboard:
                    latch_locked = bool(cxy_latch.locked) if cxy_latch is not None else False
                    dash = compose_unified_dashboard(
                        vis,
                        warped,
                        live_preds,
                        warped_dets,
                        module_a=module_a_po,
                        panel_id=panel_id_live,
                        reliable=reliable,
                        reproj_b=reproj,
                        homography_inliers=hom_inliers,
                        corner_source=corner_src,
                        xy_backend=xy_back,
                        angle=angle,
                        category=category,
                        live_report_lines=live_lines,
                        latched_preds=latched_preds,
                        latched_report_lines=latched_lines,
                        latch_txt=latch_txt,
                        latch_locked=latch_locked,
                        snapshot_entries=snapshot_store.ranked_entries if snapshot_store else None,
                        target_w=args.preview_width,
                        opencv_grid_x=opencv_grid_x,
                        opencv_grid_y=opencv_grid_y,
                        hough_grid_x=hough_grid_x,
                        hough_grid_y=hough_grid_y,
                        grid_overlap_ratio=grid_overlap_ratio,
                        grid_line_match_ratio=pan_meta.get('grid_line_match_ratio'),
                        roi_coverage=corner_meta.get('roi_coverage'),
                        warp_panel_coverage=pan_meta.get('warp_panel_coverage'),
                        snapshot_preds=last_snap_preds if color_on_snapshot else None,
                        consensus_preds=consensus_preds if consensus_preds else None,
                        panel_present=panel_present,
                        panel_presence_reason=panel_presence_reason,
                    )
                    last_show = dash
                    if snapshot_store is not None and not snapshots_paused() and (
                        mission_manager is None or mission_manager.can_accept_snapshots()
                    ):
                        pnp_ok = bool(pan_meta.get('pnp_ok', corners_px is not None))
                        mod_a_ok = bool(module_a_po.ok) if module_a_po is not None else True
                        reproj_a = (
                            float(module_a_po.reproj_mean_px)
                            if module_a_po is not None and np.isfinite(module_a_po.reproj_mean_px)
                            else 999.0
                        )
                        warp_cov = float(pan_meta.get('warp_panel_coverage', 0.0))
                        snap_ok, snap_reason = snapshot_frame_eligible(
                            reproj_b=float(reproj),
                            max_reproj_px=float(runtime_gate.get('snapshot_max_reproj', args.snapshot_max_reproj)),
                            grid_overlap_ratio=float(grid_overlap_ratio),
                            min_grid_overlap=float(runtime_gate.get('snapshot_min_grid_overlap', args.snapshot_min_grid_overlap)),
                            pnp_ok=pnp_ok,
                            require_module_a=bool(args.module_a),
                            module_a_ok=mod_a_ok,
                            reproj_a=reproj_a,
                            max_reproj_a_px=float(args.snapshot_max_reproj_a),
                            reliable=bool(reliable),
                            require_reliable=bool(args.snapshot_require_reliable),
                            homography_inliers=int(hom_inliers),
                            min_homography_inliers=int(args.snapshot_min_homography_inliers),
                            block_unreliable_zero_inl=bool(args.snapshot_block_unreliable_zero_inl),
                            min_grid_overlap_if_unreliable=float(
                                runtime_gate.get(
                                    'snapshot_min_grid_overlap_unreliable',
                                    args.snapshot_min_grid_overlap_unreliable,
                                ),
                            ),
                            warp_panel_coverage=warp_cov,
                            min_warp_panel_coverage=float(
                                runtime_gate.get('snapshot_min_warp_coverage', args.snapshot_min_warp_coverage),
                            ),
                        )
                        a_record: dict = {}
                        if module_a_po is not None and module_a_po.ok:
                            a_record = module_a_po.to_integration_dict(panel_id=panel_id_live)
                        snap_path = snapshot_store.maybe_save(
                            frame_id=fid,
                            dashboard_bgr=last_show if last_show is not None else dash,
                            reproj_b=reproj,
                            snapshot_ok=snap_ok,
                            reliable=reliable,
                            snapshot_reason=snap_reason,
                            module_a_ok=mod_a_ok,
                            grid_overlap_ratio=float(grid_overlap_ratio),
                            record={
                                'module_a': a_record,
                                'snapshot_gates': {
                                    'reproj_a_px': float(reproj_a),
                                    'warp_panel_coverage': float(warp_cov),
                                    'require_reliable': bool(args.snapshot_require_reliable),
                                },
                                'module_b': {
                                    'reliable': reliable,
                                    'grid_overlap_ratio': float(grid_overlap_ratio),
                                    'reproj_mean_px': reproj,
                                    'homography_inliers': hom_inliers,
                                    'corner_source': corner_src,
                                    'xy_backend': xy_back,
                                    'live_predictions': live_preds,
                                    'live_report_lines': live_lines,
                                    'latched_predictions': latched_preds,
                                    'latched_report_lines': latched_lines,
                                    'panel_id': panel_id_live,
                                    'report_angle_deg': angle,
                                    'panel_angle_category': category,
                                },
                                'pan_meta': {k: v for k, v in pan_meta.items() if k != 'homography'},
                            },
                        )
                        if snap_path:
                            _log(
                                log_fh,
                                f'[{fid}] migawka zapisana: {snap_path} '
                                f'(3siatki={100 * grid_overlap_ratio:.0f}% reprojA={reproj_a:.1f} '
                                f'reprojB={reproj:.1f} warp={warp_cov:.2f})',
                            )
                            if (
                                color_on_snapshot
                                and corners_px is not None
                                and h_mat is not None
                                and black_th is not None
                            ):
                                det, warped_dets, live_preds, live_lines, dash = (
                                    _refresh_saved_snapshot_colors(
                                        snap_path=snap_path,
                                        snapshot_store=snapshot_store,
                                        work_bgr=work_bgr,
                                        warped=warped,
                                        corners_px=corners_px,
                                        h_mat=h_mat,
                                        black_th=black_th,
                                        vis=vis,
                                        panel_id=panel_id_live,
                                        angle=angle,
                                        module_a_po=module_a_po,
                                        reliable=reliable,
                                        reproj=reproj,
                                        hom_inliers=hom_inliers,
                                        corner_src=corner_src,
                                        xy_back=xy_back,
                                        category=category,
                                        latched_preds=latched_preds,
                                        latched_lines=latched_lines,
                                        latch_txt=latch_txt,
                                        latch_locked=latch_locked,
                                        opencv_grid_x=opencv_grid_x,
                                        opencv_grid_y=opencv_grid_y,
                                        hough_grid_x=hough_grid_x,
                                        hough_grid_y=hough_grid_y,
                                        grid_overlap_ratio=grid_overlap_ratio,
                                        pan_meta=pan_meta,
                                    )
                                )
                                last_snap_preds = list(live_preds)
                                last_snap_lines = list(live_lines)
                                comp_min_v, comp_min_r = _snapshot_competition_thresholds(
                                    args,
                                    n_snapshots=len(snapshot_store.entries),
                                )
                                comp = update_session_competition(
                                    snapshot_store.session_dir,
                                    min_votes=comp_min_v,
                                    min_support_ratio=comp_min_r,
                                    max_cards=4,
                                )
                                if comp is not None:
                                    consensus_preds = list(comp.predictions)
                                    consensus_lines = list(comp.report_lines)
                                    dash = compose_unified_dashboard(
                                        vis,
                                        warped,
                                        [],
                                        warped_dets,
                                        module_a=module_a_po,
                                        panel_id=panel_id_live,
                                        reliable=reliable,
                                        reproj_b=reproj,
                                        homography_inliers=hom_inliers,
                                        corner_source=corner_src,
                                        xy_backend=xy_back,
                                        angle=angle,
                                        category=category,
                                        live_report_lines=live_lines,
                                        latched_preds=latched_preds,
                                        latched_report_lines=latched_lines,
                                        latch_txt=latch_txt,
                                        latch_locked=latch_locked,
                                        snapshot_entries=snapshot_store.ranked_entries,
                                        snapshot_preds=last_snap_preds,
                                        consensus_preds=consensus_preds,
                                        opencv_grid_x=opencv_grid_x,
                                        opencv_grid_y=opencv_grid_y,
                                        hough_grid_x=hough_grid_x,
                                        hough_grid_y=hough_grid_y,
                                        grid_overlap_ratio=grid_overlap_ratio,
                                        grid_line_match_ratio=pan_meta.get('grid_line_match_ratio'),
                                        roi_coverage=corner_meta.get('roi_coverage'),
                                        warp_panel_coverage=pan_meta.get('warp_panel_coverage'),
                                        panel_present=panel_present,
                                        panel_presence_reason=panel_presence_reason,
                                    )
                                    snapshot_store.refresh_snapshot_artifacts(
                                        snap_path,
                                        dash,
                                        record_updates={
                                            'cxy_competition': {
                                                'n_accepted': len(comp.accepted),
                                                'n_rejected': len(comp.rejected),
                                                'predictions': consensus_preds,
                                            },
                                        },
                                    )
                                    _log(
                                        log_fh,
                                        f'[{fid}] konkurs CXY: {len(consensus_preds)} kart '
                                        f'(odrzucono {len(comp.rejected)} jednorazowych)',
                                    )
                                last_show = dash
                                _log(
                                    log_fh,
                                    f'[{fid}] migawka kolory: {len(live_preds)} kart',
                                )
                                for line in live_lines:
                                    _log(log_fh, f'[{fid}]   snap {line}')
                                for line in consensus_lines:
                                    _log(log_fh, f'[{fid}]   konkurs {line}')
                            if snap_path and args.dashboard:
                                if snap_path not in snapshot_paths:
                                    snapshot_paths.append(snap_path)
                                if browser_proc is None and not args.headless:
                                    try:
                                        import subprocess
                                        snap_dir = os.path.dirname(snap_path)
                                        browser_proc = subprocess.Popen(
                                            [
                                                sys.executable,
                                                '-m',
                                                'release.snapshot_browser',
                                                '--snapshots-dir',
                                                snap_dir,
                                            ],
                                            cwd=_ROOT,
                                        )
                                        _log(log_fh, f'[live_panel] przegladarka migawek: {snap_dir}')
                                    except OSError as exc:
                                        _log(log_fh, f'[live_panel] nie uruchomiono przegladarki migawek: {exc}')
                    if panel_review_ui is not None and mission_manager is not None:
                        panel_review_ui.bind_panel_id(panel_id_live)
                        panel_review_ui.set_status(
                            panel_id=panel_id_live,
                            snap_n=mission_manager.snapshot_count(),
                            snap_max=mission_manager.state.snapshots_per_panel,
                            mission_index=mission_manager.state.current_index,
                            mission_total=len(mission_manager.state.panel_ids),
                            panel_full=mission_manager.panel_is_full(),
                            mission_done=mission_manager.state.mission_done(),
                        )
                        if mission_manager.panel_is_full() and (consensus_lines or last_snap_lines):
                            panel_review_ui.set_report_lines(
                                consensus_lines or last_snap_lines,
                            )
                        panel_review_ui.pump()
                    if web_publisher is not None and last_show is not None:
                        _stream_dash[0] = last_show
                        web_publisher.set_stream_dashboard_frame(last_show)
                        from release.tracker_tuning import snapshot as tracker_snapshot
                        from release.web_dashboard import build_web_state, validate_report_payload

                        _ms = web_publisher.poll_mission_update()
                        if _ms:
                            _apply_mission_settings_patch(_ms)

                        maybe_seed_report_draft()
                        mission_info: dict = {}
                        report_ready = False
                        report_can_send = False
                        pause_rem = (
                            int(math.ceil(vision_pause_remaining()))
                            if snapshots_paused()
                            else 0
                        )
                        if mission_manager is not None:
                            panel_full = mission_manager.panel_is_full()
                            mission_info = {
                                'current_panel': mission_manager.current_panel,
                                'panel_full': panel_full,
                                'mission_done': mission_manager.state.mission_done(),
                                'panel_ids': list(mission_manager.state.panel_ids),
                                'text': mission_manager.mission_hud_text(
                                    pause_remaining=pause_rem,
                                ),
                            }
                            if args.report_mode == 'preset':
                                report_can_send = bool(
                                    not mission_manager.state.mission_done()
                                    and pause_rem <= 0
                                )
                                report_ready = True
                            else:
                                report_can_send = bool(
                                    not mission_manager.state.mission_done()
                                    and panel_full
                                    and pause_rem <= 0
                                )
                                report_ready = bool(panel_full and consensus_lines)
                        report_validation: dict = {}
                        if report_can_send and args.report_mode != 'preset':
                            draft = web_publisher.read_draft_lines()
                            check_lines = draft or list(consensus_lines)
                            ok_rep, rep_errors = validate_report_payload(
                                check_lines,
                                panel_id=panel_id_live,
                            )
                            report_validation = {'ok': ok_rep, 'errors': rep_errors}
                        web_state = build_web_state(
                            frame_id=fid,
                            panel_id=panel_id_live,
                            module_a_po=module_a_po,
                            reliable=reliable,
                            reliable_legacy=pan_meta.get('grid_xy_reliable_legacy'),
                            reproj_b=reproj,
                            homography_inliers=hom_inliers,
                            corner_source=corner_src,
                            xy_backend=xy_back,
                            angle=angle,
                            category=category,
                            grid_overlap_ratio=grid_overlap_ratio,
                            grid_line_match_ratio=pan_meta.get('grid_line_match_ratio'),
                            roi_coverage=corner_meta.get('roi_coverage'),
                            warp_panel_coverage=pan_meta.get('warp_panel_coverage'),
                            panel_present=panel_present,
                            panel_presence_reason=panel_presence_reason,
                            live_report_lines=live_lines,
                            latched_report_lines=latched_lines,
                            consensus_report_lines=consensus_lines,
                            latch_txt=latch_txt,
                            latch_locked=latch_locked,
                            snapshots=(
                                snapshot_store.ranked_entries if snapshot_store else []
                            ),
                            mission=mission_info,
                            report_ready=report_ready,
                            report_can_send=report_can_send,
                            report_pause_sec=pause_rem,
                            report_validation=report_validation,
                            report_mode=args.report_mode,
                            preset_panels={
                                pid: len(lines)
                                for pid, lines in preset_reports.items()
                            },
                            tracker=tracker_snapshot(),
                            flight=(
                                flight_controller.status_dict() if flight_controller else {}
                            ),
                            mission_settings=(
                                flight_controller.settings if flight_controller else {}
                            ),
                            analysis_active=not analysis_paused(),
                            overlay_corners_px=(
                                corners_px if _run_analysis else None
                            ),
                            overlay_frame_w=int(work_bgr.shape[1]),
                            overlay_frame_h=int(work_bgr.shape[0]),
                            overlay_tracker_hold=bool(tracker_hold),
                        )
                        web_publisher.publish_state(web_state)
                        web_publisher.publish(last_show, web_state)
                        web_cmd = web_publisher.poll_command()
                        if web_cmd and web_cmd.get('action') == 'send':
                            _cmd_src = str(web_cmd.get('source') or 'web')
                            apply_report_send(
                                str(web_cmd.get('panel_id', panel_id_live)),
                                list(web_cmd.get('lines') or []),
                                source=_cmd_src,
                            )
                    if ws_publisher is not None:
                        from release.integration_ws import build_speed_payload

                        _ws_speed = 1.0
                        if flight_controller is not None:
                            _ws_speed = float(flight_controller.speed)
                        ws_publisher.publish(build_speed_payload(_ws_speed))
                    if mjpeg_server is not None and last_show is not None:
                        mjpeg_server.push_frame(last_show)
                elif args.greenscreen and args.greenscreen_split and gs_bgr is not None:
                    from release.greenscreen_panel import compose_preview_with_greenscreen

                    main_vis = compose_preview_with_greenscreen(bgr, vis, target_w=args.preview_width)
                    last_show = _compose_preview(main_vis, thumb, target_w=args.preview_width)
                else:
                    last_show = _compose_preview(vis, thumb, target_w=args.preview_width)

                if args.out_video and last_show is not None:
                    out_path = args.out_video if os.path.isabs(args.out_video) else os.path.join(_ROOT, args.out_video)
                    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
                    if writer is None:
                        fh, fw = last_show.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        src_fps = float(getattr(src, 'fps', 30.0))
                        writer = cv2.VideoWriter(out_path, fourcc, float(src_fps), (fw, fh))
                        _log(log_fh, f'[live_panel] zapis video: {out_path} {fw}x{fh} @{src_fps:.1f}fps')
                    writer.write(last_show)

                key = -1
                if show_window and args.dashboard and last_show is not None:
                    show_dashboard_window(last_show, target_w=args.preview_width)
                    if video_realtime:
                        elapsed_ms = (time.monotonic() - loop_t0) * 1000.0
                        delay = max(1, frame_period_ms - int(elapsed_ms))
                        key = cv2.waitKeyEx(delay)
                    else:
                        key = cv2.waitKeyEx(1)
                elif show_window and cxy_latch is not None and cxy_latch.locked and latched_vis is not None and not args.dashboard:
                    _show_latch_window(latched_vis, args.preview_width)
                elif show_window and not args.dashboard:
                    _hide_latch_window()

                if show_window and not args.dashboard and video_realtime and last_show is not None:
                    cv2.imshow(live_window, last_show)
                    elapsed_ms = (time.monotonic() - loop_t0) * 1000.0
                    delay = max(1, frame_period_ms - int(elapsed_ms))
                    key = cv2.waitKeyEx(delay)
                elif show_window and not args.dashboard:
                    if last_show is not None:
                        cv2.imshow(live_window, last_show)
                    key = cv2.waitKeyEx(1)
                elif args.headless:
                    key = -1

                if key == ord('q'):
                    break
                if key == ord('w') and mission_manager is not None and panel_review_ui is not None:
                    panel_review_ui.send_for_panel(mission_manager.current_panel)
                    snapshot_store = mission_manager.current_store()
                if key == ord('r') and cxy_latch is not None:
                    cxy_latch.reset()
                    cxy_locked_announced = False
                    latched_vis = None
                    _hide_latch_window()
                    _log(log_fh, '[live_panel] CXY latch reset')
                if key == ord('s') and cxy_latch is not None and reliable:
                    manual_preds = list(last_snap_preds) if color_on_snapshot else list(live_preds)
                    manual_lines = list(last_snap_lines) if color_on_snapshot else list(live_lines)
                    manual_dets = list(warped_dets) if run_live_color else []
                    cxy_latch.force_snapshot(
                        frame_id=fid,
                        reproj_mean_px=reproj,
                        homography_inliers=hom_inliers,
                        predictions=manual_preds,
                        report_lines=manual_lines,
                        meta=pan_meta,
                    )
                    latched_preds = list(manual_preds)
                    latched_lines = list(manual_lines)
                    latch_txt = (
                        f'ZAMKNIĘTE @ {fid}  reproj={reproj:.1f}px  inl={hom_inliers}'
                    )
                    cxy_locked_announced = True
                    _log(log_fh, f'[{fid}] === CXY ręcznie (klawisz s) ===')
                    for line in manual_lines:
                        _log(log_fh, f'[{fid}]   {line}')
                    if corners_px is not None and warped is not None and not args.no_color_detect:
                        try:
                            paths = save_cxy_latch_artifacts(
                                cxy_latch_dir,
                                frame_id=fid,
                                work_bgr=work_bgr,
                                corners_tltrbrbl=corners_px,
                                warped_bgr=warped,
                                preds=manual_preds,
                                warped_dets=manual_dets,
                                reproj_px=float(reproj),
                                homography_inliers=int(hom_inliers),
                                report_lines=manual_lines,
                                meta=dict(pan_meta),
                            )
                            _log(log_fh, f'[{fid}] podglad latch: {paths.get("dashboard")}')
                            if not args.dashboard:
                                latched_vis = compose_latch_dashboard(
                                    work_bgr,
                                    corners_px,
                                    warped,
                                    manual_preds,
                                    warped_dets=manual_dets,
                                    frame_id=fid,
                                    reproj_px=float(reproj),
                                    homography_inliers=int(hom_inliers),
                                    report_lines=manual_lines,
                                )
                                if show_window:
                                    _show_latch_window(latched_vis, args.preview_width)
                                    _log(log_fh, f'[{fid}] okno podgladu: {_LATCH_WINDOW}')
                        except Exception as exc:
                            _log(log_fh, f'[{fid}] blad zapisu latch preview: {exc}')

                save_debug = (
                    debug_session is not None
                    and max(1, int(args.debug_every)) > 0
                    and (processed % max(1, int(args.debug_every)) == 0)
                )
                if save_debug:
                    board = draw_candidates_board(bgr, cand_rows, winner_label=corner_src) if cand_rows else vis.copy()
                    record = {
                        'frame_id': fid,
                        'corner_source': corner_src,
                        'reproj_mean_px': reproj,
                        'grid_structure_score': corner_meta.get('grid_structure_score'),
                        'panel_interior_score': corner_meta.get('panel_interior_score'),
                        'grid_xy_reliable': reliable,
                        'corner_mode': args.corner_mode,
                        'roi_source': corner_meta.get('roi_source', 'none'),
                        'xy_backend': xy_back,
                        'n_det': len(det),
                        'n_preds': len(live_preds),
                        'predictions': live_preds,
                        'candidates': [
                            {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in c.items()}
                            for c in cand_rows
                        ],
                        'pan_meta': {k: v for k, v in pan_meta.items() if k != 'homography'},
                    }
                    save_live_frame_bundle(
                        debug_session,
                        fid,
                        raw_bgr=bgr,
                        overlay_bgr=last_show,
                        warped_bgr=warped,
                        candidates_board=board,
                        record=record,
                    )
                    append_session_index(debug_session, {
                        'frame_id': fid,
                        'corner_source': corner_src,
                        'reproj_mean_px': reproj,
                        'grid_structure_score': corner_meta.get('grid_structure_score'),
                        'reliable': reliable,
                    })
                elif debug_session:
                    append_session_index(debug_session, {
                        'frame_id': fid,
                        'corner_source': corner_src,
                        'reproj_mean_px': reproj,
                        'grid_structure_score': corner_meta.get('grid_structure_score'),
                        'reliable': reliable,
                    })

                if args.save_dir and reliable and live_preds:
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
        if writer is not None:
            writer.release()
            _log(log_fh, f'[live_panel] zapisano video: {args.out_video}')
        if browser_proc is not None:
            try:
                browser_proc.terminate()
            except OSError:
                pass
        if web_publisher is not None:
            try:
                web_publisher.shutdown()
            except Exception:
                pass
        if web_httpd is not None:
            try:
                web_httpd.shutdown()
            except Exception:
                pass
        if web_control_httpd is not None:
            try:
                web_control_httpd.shutdown()
            except Exception:
                pass
        if ws_publisher is not None:
            try:
                ws_publisher.close()
            except Exception:
                pass
        if ws_subscriber is not None:
            try:
                ws_subscriber.close()
            except Exception:
                pass
        if analysis_worker is not None:
            try:
                analysis_worker.stop()
            except Exception:
                pass
        if vis_preview_worker is not None:
            try:
                vis_preview_worker.stop()
            except Exception:
                pass
        if log_fh is not None:
            log_fh.close()
        if panel_review_ui is not None:
            try:
                panel_review_ui.root.destroy()
            except Exception:
                pass
        cv2.destroyAllWindows()
        _log(None, f'[live_panel] stop processed={processed}')
        if mission_manager is not None:
            mission_manager.save_mission()
            for pid in mission_manager.state.panel_ids:
                st = mission_manager.store_for(pid)
                if st.entries:
                    st.write_session_index(
                        competition_min_votes=int(args.snapshot_competition_min_votes),
                        competition_min_ratio=float(args.snapshot_competition_min_ratio),
                    )
            _log(None, f'[live_panel] misja paneli: {dashboard_session}/mission.json')
        elif snapshot_store is not None:
            html_path = snapshot_store.write_session_index(
                competition_min_votes=int(args.snapshot_competition_min_votes),
                competition_min_ratio=float(args.snapshot_competition_min_ratio),
            )
            _log(None, f'[live_panel] galeria migawek: {html_path}')
            comp_path = os.path.join(snapshot_store.session_dir, 'cxy_competition_report.txt')
            if os.path.isfile(comp_path):
                _log(None, f'[live_panel] konkurs CXY: {comp_path}')
        if debug_session and args.debug_html:
            try:
                import subprocess
                root = args.debug_dir if os.path.isabs(args.debug_dir) else os.path.join(_ROOT, args.debug_dir)
                subprocess.run(
                    [sys.executable, '-m', 'pipelines.analyze_live_debug', '--root', root,
                     '--last-session-only', '--html'],
                    cwd=_ROOT,
                    check=False,
                )
            except OSError:
                pass


if __name__ == '__main__':
    main()
