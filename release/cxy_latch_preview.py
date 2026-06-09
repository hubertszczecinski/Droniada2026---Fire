"""Podgląd i zapis artefaktów po zatrzaśnięciu CXY (idealna siatka)."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import pipeline_competition as pc
from release.cv_text import put_text_utf8

_BGR_BY_COLOR = {
    'CZERWONA': (40, 40, 230),
    'ZIELONA': (50, 190, 50),
    'NIEBIESKA': (230, 120, 40),
    'ZOLTA': (20, 220, 240),
    'FIOLETOWA': (200, 60, 200),
    'POMARANCZOWA': (30, 130, 240),
    'UNKNOWN': (180, 180, 180),
}


def grid_row_to_warp_y(grid_row: int, panel_h: int) -> int:
    """Wiersz 1 = dół panelu na warpie, wiersz 10 = góra."""
    ch = panel_h / 10.0
    return int(round((10.0 - float(grid_row) + 0.5) * ch))


def grid_col_to_warp_x(grid_col: int, panel_w: int) -> int:
    cw = panel_w / 10.0
    return int(round((float(grid_col) - 0.5) * cw))


_CAPTION_H = 26
_CAPTION_BG = (42, 44, 48)


def draw_warped_panel_preview(
    warped_bgr: np.ndarray,
    preds: List[Dict[str, Any]],
    warped_dets: Optional[List[Dict[str, Any]]] = None,
    *,
    title: Optional[str] = None,
    title_on_panel: bool = False,
    opencv_grid_x: Optional[List[float]] = None,
    opencv_grid_y: Optional[List[float]] = None,
    hough_grid_x: Optional[List[float]] = None,
    hough_grid_y: Optional[List[float]] = None,
) -> np.ndarray:
    """Prostokątny panel z siatką 10×10 i etykietami CXY.

    Tytuł domyślnie pod obrazem (nie zasłania górnych wierszy siatki).
    ``title_on_panel=True`` — stary pasek na warpie (tylko debug).
    """
    thumb = warped_bgr.copy()
    h, w = thumb.shape[:2]
    for i in range(11):
        x = int(round(i * w / 10.0))
        y = int(round(i * h / 10.0))
        cv2.line(thumb, (x, 0), (x, h - 1), (180, 180, 180), 1)
        cv2.line(thumb, (0, y), (w - 1, y), (180, 180, 180), 1)
    if hough_grid_x is not None and hough_grid_y is not None:
        from module_panel.grid_overlap import draw_hough_grid_on_warped

        draw_hough_grid_on_warped(
            thumb, hough_grid_x, hough_grid_y, color=(0, 160, 255), thickness=1,
        )
    if opencv_grid_x is not None and opencv_grid_y is not None:
        from module_panel.grid_overlap import draw_opencv_grid_on_warped

        draw_opencv_grid_on_warped(
            thumb, opencv_grid_x, opencv_grid_y, color=(60, 220, 80), thickness=1,
        )
    for col in range(1, 11):
        cx = grid_col_to_warp_x(col, w)
        cv2.putText(
            thumb, str(col), (cx - 8, max(14, int(0.04 * h))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA,
        )
    for row in range(1, 11):
        cy = grid_row_to_warp_y(row, h)
        cv2.putText(
            thumb, str(row), (4, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA,
        )
    if warped_dets:
        for d in warped_dets:
            p = (int(round(d['warp_cx'])), int(round(d['warp_cy'])))
            bgr = _BGR_BY_COLOR.get(str(d.get('color', '')), _BGR_BY_COLOR['UNKNOWN'])
            cv2.circle(thumb, p, 9, bgr, -1, cv2.LINE_AA)
    for p in preds:
        gx = grid_col_to_warp_x(int(p['x']), w)
        gy = grid_row_to_warp_y(int(p['y']), h)
        color = str(p.get('color', 'UNKNOWN'))
        bgr = _BGR_BY_COLOR.get(color, _BGR_BY_COLOR['UNKNOWN'])
        cv2.circle(thumb, (gx, gy), 14, bgr, 2, cv2.LINE_AA)
        cv2.drawMarker(thumb, (gx, gy), (0, 0, 0), cv2.MARKER_CROSS, 14, 2)
        label = f"W{p['y']} K{p['x']}"
        cv2.putText(
            thumb, label, (gx + 12, gy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 2, cv2.LINE_AA,
        )
        cv2.putText(
            thumb, color[:3], (gx + 12, gy + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr, 1, cv2.LINE_AA,
        )
    if not title:
        return thumb
    if title_on_panel:
        cv2.rectangle(thumb, (0, 0), (w, 28), (240, 240, 240), -1)
        put_text_utf8(thumb, title, (8, 20), (20, 80, 20), scale=0.55, thickness=2)
        return thumb
    cap = np.full((_CAPTION_H, w, 3), _CAPTION_BG, dtype=np.uint8)
    put_text_utf8(cap, title, (8, 18), (210, 230, 210), scale=0.5, thickness=1)
    return np.vstack([thumb, cap])


def _draw_cxy_sidebar(
    panel_h: int,
    preds: List[Dict[str, Any]],
    *,
    frame_id: str,
    reproj_px: float,
    homography_inliers: int,
    report_lines: List[str],
) -> np.ndarray:
    lines = [
        'CXY ZATRZAŚNIĘTE',
        f'klatka: {frame_id}',
        f'reproj: {reproj_px:.1f} px',
        f'inliers: {homography_inliers}',
        '',
        'Wykryte kartki:',
    ]
    if preds:
        for p in sorted(preds, key=lambda d: (-int(d['y']), int(d['x']))):
            lines.append(f"  Wiersz {p['y']}, Kolumna {p['x']} — {p.get('color', '?')}")
    else:
        lines.append('  (brak)')
    lines.append('')
    lines.append('Raport:')
    if report_lines:
        lines.extend(f'  {ln}' for ln in report_lines)
    else:
        lines.append('  (brak)')
    line_h = 22
    pad = 14
    side_w = 340
    side = np.full((panel_h, side_w, 3), 248, dtype=np.uint8)
    y = pad + 16
    for text in lines:
        color = (0, 120, 0) if text == 'CXY ZATRZAŚNIĘTE' else (40, 40, 40)
        put_text_utf8(side, text, (pad, y), color, scale=0.48, thickness=1)
        y += line_h
    return side


def compose_latch_dashboard(
    work_bgr: np.ndarray,
    corners_tltrbrbl: np.ndarray,
    warped_bgr: np.ndarray,
    preds: List[Dict[str, Any]],
    *,
    warped_dets: Optional[List[Dict[str, Any]]] = None,
    frame_id: str = '',
    reproj_px: float = 0.0,
    homography_inliers: int = 0,
    report_lines: Optional[List[str]] = None,
) -> np.ndarray:
    """Kamera z ramką + panel prostokątny + lista CXY."""
    cam = work_bgr.copy()
    if corners_tltrbrbl is not None and corners_tltrbrbl.shape == (4, 2):
        q = pc.order_points(corners_tltrbrbl.astype(np.float32))
        pts = q.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(cam, [pts], True, (0, 255, 255), 3)
        dst = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.float32)
        try:
            h_inv = np.linalg.inv(cv2.getPerspectiveTransform(q, dst))
            for i in range(11):
                t = i / 10.0
                for seg in (
                    np.array([[[t * 1000.0, 0.0]], [[t * 1000.0, 1000.0]]], dtype=np.float32),
                    np.array([[[0.0, t * 1000.0]], [[1000.0, t * 1000.0]]], dtype=np.float32),
                ):
                    pts_ln = cv2.perspectiveTransform(seg, h_inv).reshape(-1, 2).astype(np.int32)
                    cv2.line(cam, tuple(pts_ln[0]), tuple(pts_ln[1]), (0, 220, 220), 1, cv2.LINE_AA)
        except cv2.error:
            pass
    title = f'Zatrzask @ {frame_id}  reproj={reproj_px:.1f}px' if frame_id else None
    panel_vis = draw_warped_panel_preview(
        warped_bgr, preds, warped_dets, title=title,
    )
    ph, pw = panel_vis.shape[:2]
    ch, cw = cam.shape[:2]
    if ph != ch:
        scale = ch / float(ph)
        panel_vis = cv2.resize(panel_vis, (max(1, int(round(pw * scale))), ch), interpolation=cv2.INTER_AREA)
        ph, pw = panel_vis.shape[:2]
    gap = np.full((ch, 10, 3), 40, dtype=np.uint8)
    mid = np.hstack([cam, gap, panel_vis])
    side = _draw_cxy_sidebar(
        ch, preds,
        frame_id=frame_id,
        reproj_px=reproj_px,
        homography_inliers=homography_inliers,
        report_lines=list(report_lines or []),
    )
    return np.hstack([mid, side])


def save_cxy_latch_artifacts(
    out_dir: str,
    *,
    frame_id: str,
    work_bgr: np.ndarray,
    corners_tltrbrbl: np.ndarray,
    warped_bgr: np.ndarray,
    preds: List[Dict[str, Any]],
    warped_dets: Optional[List[Dict[str, Any]]] = None,
    reproj_px: float = 0.0,
    homography_inliers: int = 0,
    report_lines: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Zapisz JPG + JSON podglądu zatrzaśniętej klatki."""
    os.makedirs(out_dir, exist_ok=True)
    stem = frame_id.replace('/', '_').replace(':', '_')
    paths: Dict[str, str] = {}

    panel_path = os.path.join(out_dir, f'{stem}_panel.jpg')
    cam_path = os.path.join(out_dir, f'{stem}_camera.jpg')
    dash_path = os.path.join(out_dir, f'{stem}_latch.jpg')
    json_path = os.path.join(out_dir, f'{stem}_cxy.json')
    txt_path = os.path.join(out_dir, f'{stem}_cxy.txt')

    panel_vis = draw_warped_panel_preview(
        warped_bgr, preds, warped_dets,
        title=f'Panel @ {frame_id}',
    )
    cv2.imwrite(panel_path, panel_vis)
    cv2.imwrite(cam_path, work_bgr)
    dashboard = compose_latch_dashboard(
        work_bgr, corners_tltrbrbl, warped_bgr, preds,
        warped_dets=warped_dets,
        frame_id=frame_id,
        reproj_px=reproj_px,
        homography_inliers=homography_inliers,
        report_lines=report_lines,
    )
    cv2.imwrite(dash_path, dashboard)

    payload = {
        'frame_id': frame_id,
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'reproj_mean_px': float(reproj_px),
        'homography_inliers': int(homography_inliers),
        'predictions': list(preds),
        'report_lines': list(report_lines or []),
        'meta': dict(meta or {}),
        'files': {},
    }
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    with open(txt_path, 'w', encoding='utf-8') as fh:
        fh.write(f'CXY zatrzask — {frame_id}\n')
        fh.write(f'reproj={reproj_px:.2f} px  inliers={homography_inliers}\n\n')
        for p in sorted(preds, key=lambda d: (-int(d['y']), int(d['x']))):
            fh.write(f"Wiersz {p['y']}, Kolumna {p['x']} — {p.get('color', '?')}\n")
        if report_lines:
            fh.write('\nRaport:\n')
            for ln in report_lines:
                fh.write(f'{ln}\n')

    paths['panel'] = panel_path
    paths['camera'] = cam_path
    paths['dashboard'] = dash_path
    paths['json'] = json_path
    paths['txt'] = txt_path
    payload['files'] = dict(paths)
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return paths
