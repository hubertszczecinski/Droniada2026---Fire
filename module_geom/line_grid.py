"""Line-first panel pipeline: multi-source corners + best XY backend selection."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pipeline_competition as pc

from module_geom.grid_homography import build_grid_homography
from module_geom.map_cards import map_yolo_to_cells_geom
from module_geom.lines_vp import detect_corners_geom_vp as _detect_corners_vp_lsd
from module_pose.api import (
    canonicalize_corners_by_white_anchor,
    detect_corners_black_panel,
    detect_corners_panel,
    _black_panel_interior_score,
)
from module_pose.grid_corners import detect_corners_white_grid
from module_pose.pnp_panel import solve_panel_pose
from module_pose.refine_corners import refine_panel_corners_uniform_grid
from module_panel.grid import map_yolo_to_cells_corners_homography, map_yolo_to_cells_warp_grid
from module_panel.warp import warp_panel_rect


def _quad_metrics(quad: np.ndarray, image_shape: Tuple[int, int, int]) -> Dict[str, float]:
    h, w = image_shape[:2]
    img_area = float(h * w)
    area = float(cv2.contourArea(quad.reshape(1, 4, 2)))
    x_span = float(np.ptp(quad[:, 0]))
    y_span = float(np.ptp(quad[:, 1]))
    asp = max(x_span, y_span) / max(1.0, min(x_span, y_span))
    return {
        'area_ratio': area / max(1.0, img_area),
        'aspect': asp,
    }


def _corners_from_warped_grid(
    image_bgr: np.ndarray,
    rough_corners: np.ndarray,
    *,
    passes: int = 2,
) -> Optional[np.ndarray]:
    current = rough_corners.astype(np.float32)
    refined: Optional[np.ndarray] = None
    for _ in range(max(1, passes)):
        refined = refine_panel_corners_uniform_grid(image_bgr, current)
        if refined is None:
            break
        current = refined.astype(np.float32)
    return current if refined is not None else None


def _gather_corner_candidates(
    image_bgr: np.ndarray,
    yolo_det: List[Tuple[int, float, float, float, float]],
    k: np.ndarray,
    dist: np.ndarray,
) -> List[Tuple[str, np.ndarray]]:
    """All corner hypotheses before scoring."""
    raw_seeds: List[Tuple[str, np.ndarray]] = []

    yolo = pc.detect_corners_yolo(yolo_det or [])
    if yolo is not None:
        raw_seeds.append(('yolo', yolo.astype(np.float32)))

    # Match grid_geom_white: panel quad ranking uses default intrinsics, not pose K.
    panel = detect_corners_panel(image_bgr)
    if panel is not None:
        raw_seeds.append(('img_panel', panel.astype(np.float32)))

    black = detect_corners_black_panel(image_bgr, k, dist)
    if black is not None:
        raw_seeds.append(('black_panel', black.astype(np.float32)))

    white = detect_corners_white_grid(image_bgr)
    if white is not None:
        raw_seeds.append(('white_grid_image', white.astype(np.float32)))

    warp_seeds = list(raw_seeds)
    for seed_label, rough in warp_seeds:
        for passes in (2, 3):
            refined = _corners_from_warped_grid(image_bgr, rough, passes=passes)
            if refined is not None:
                raw_seeds.append((f'warp{passes}_{seed_label}', refined.astype(np.float32)))

    candidates: List[Tuple[str, np.ndarray]] = []
    for label, raw in raw_seeds:
        c, _anc = canonicalize_corners_by_white_anchor(image_bgr, raw)
        candidates.append((label, c.astype(np.float32)))

    q_vp, _vp_meta = _detect_corners_vp_lsd(image_bgr)
    if q_vp is not None:
        c, _anc = canonicalize_corners_by_white_anchor(image_bgr, q_vp.astype(np.float32))
        candidates.append(('geom_vp_fallback', c.astype(np.float32)))

    dedup: List[Tuple[str, np.ndarray]] = []
    for label, q in candidates:
        if any(float(np.max(np.abs(q - d[1]))) < 10.0 for d in dedup):
            continue
        dedup.append((label, q))
    return dedup


def _score_line_grid_corners(
    image_bgr: np.ndarray,
    corners: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    *,
    min_inliers: int = 12,
    corner_label: str = '',
) -> Tuple[float, Dict[str, Any]]:
    m = _quad_metrics(corners, image_bgr.shape)
    if m['area_ratio'] > 0.52 or m['area_ratio'] < 0.025:
        return (-1e6, {**m, 'reject': 'area_ratio'})
    if m['aspect'] < 1.15 or m['aspect'] > 3.9:
        return (-1e6, {**m, 'reject': 'aspect'})

    _h, ok_h, hmeta = build_grid_homography(image_bgr, corners, min_inliers=min_inliers)
    inliers = int(hmeta.get('homography_inliers', 0)) if ok_h else 0
    ok_pnp, _rv, _tv, reproj = solve_panel_pose(corners, k, dist)
    reproj_v = float(reproj) if ok_pnp else 999.0
    interior = _black_panel_interior_score(image_bgr, corners)

    oversize = max(0.0, m['area_ratio'] - 0.38)
    score = inliers * 12.0 - reproj_v * 4.0 - oversize * 200.0 + interior * 220.0
    if not ok_h:
        score -= 60.0
    if not ok_pnp:
        score -= 120.0

    if corner_label == 'img_panel':
        score += 95.0
    elif corner_label == 'black_panel':
        score += 95.0
    elif corner_label.startswith('warp') and 'img_panel' in corner_label:
        score += 70.0
    elif corner_label.startswith('warp') and 'black_panel' in corner_label:
        score += 55.0
    elif corner_label.startswith('warp'):
        score += 35.0
    elif 'white_grid' in corner_label:
        score += 40.0
    if 'geom_vp' in corner_label:
        score -= 100.0

    return (
        score,
        {
            **m,
            'homography_inliers': inliers,
            'grid_homography_ok': bool(ok_h),
            'reproj_mean_px': reproj_v,
            'grid_line_ok': bool(hmeta.get('grid_line_ok', False)),
            'panel_interior_score': float(interior),
        },
    )


def detect_corners_line_grid(
    image_bgr: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    *,
    yolo_det: Optional[List[Tuple[int, float, float, float, float]]] = None,
    min_inliers: int = 12,
) -> Tuple[Optional[np.ndarray], str, Dict[str, Any]]:
    """
    line_grid v3: img_panel/black_panel + warp-refine seeds; VP last resort.
    """
    meta: Dict[str, Any] = {'method': 'line_grid_v3_multi'}
    candidates = _gather_corner_candidates(image_bgr, yolo_det or [], k, dist)

    best_score = -1e9
    best_quad: Optional[np.ndarray] = None
    best_label = 'none'
    best_detail: Dict[str, Any] = {}

    scored: List[Tuple[float, str, np.ndarray, Dict[str, Any]]] = []
    for label, raw in candidates:
        score, detail = _score_line_grid_corners(
            image_bgr, raw, k, dist, min_inliers=min_inliers, corner_label=label,
        )
        detail['corner_candidate'] = label
        scored.append((score, label, raw, detail))

    for score, label, raw, detail in scored:
        if score > best_score:
            best_score = score
            best_quad = raw
            best_label = label
            best_detail = detail

    # Oblique panels: black mask often beats color contour when interior is clearly stronger.
    by_label = {lbl: (sc, q, det) for sc, lbl, q, det in scored if sc > -1e5}
    ip = by_label.get('img_panel')
    bp = by_label.get('black_panel')
    if ip and bp and bp[2].get('panel_interior_score', 0) > ip[2].get('panel_interior_score', 0) + 0.18:
        best_score, best_label, best_quad, best_detail = bp[0], 'black_panel', bp[1], bp[2]
        best_detail['corner_override'] = 'black_interior'

    if best_quad is None or best_score < -1e5:
        meta['fail'] = 'no_line_corners'
        return None, 'none', meta

    meta.update(best_detail)
    meta['corner_score'] = float(best_score)
    return best_quad.astype(np.float32), best_label, meta


def _pred_agreement(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> int:
    if len(a) != len(b):
        return 0
    return sum(
        1 for pa, pb in zip(a, b)
        if pa.get('x') == pb.get('x') and pa.get('y') == pb.get('y')
    )


def _score_xy_backend(
    name: str,
    xy_meta: Dict[str, Any],
    *,
    corner_label: str,
    reproj_px: float,
    n_cards: int,
    agreement_with_corner: int = 0,
) -> float:
    """Runtime heuristic (no GT): frontal oracle favors corner_h + img_panel."""
    inliers = int(xy_meta.get('homography_inliers', 0))
    grid_ok = bool(xy_meta.get('grid_line_ok', False))
    hits = int((xy_meta.get('grid_x') or {}).get('hits', 0)) if isinstance(xy_meta.get('grid_x'), dict) else 0
    panel_corner = corner_label in ('img_panel', 'black_panel') or (
        'img_panel' in corner_label and corner_label.startswith('warp')
    )

    if name == 'corner_homography':
        score = 220.0 - reproj_px * 12.0
        if corner_label in ('img_panel', 'black_panel'):
            score += 120.0
        elif 'img_panel' in corner_label:
            score += 60.0
        elif 'black_panel' in corner_label:
            score += 50.0
    elif name == 'warp_grid_lines':
        score = (100.0 if grid_ok else 10.0) + hits * 8.0 - reproj_px * 3.0
        if corner_label.startswith('warp'):
            score += 30.0
        if corner_label == 'black_panel' and reproj_px <= 8.0:
            score -= 70.0
        if panel_corner and not grid_ok:
            score -= 80.0
    elif name == 'ransac_contact':
        score = inliers * 1.4 - reproj_px * 5.0
        if corner_label.startswith('warp2') and reproj_px <= 9.0:
            score += 55.0
        if corner_label == 'black_panel' and reproj_px <= 8.0:
            score += 35.0
        if panel_corner and reproj_px <= 8.0 and corner_label not in ('black_panel',) and not corner_label.startswith('warp'):
            score -= 90.0
        if agreement_with_corner < max(2, n_cards - 1):
            score -= 50.0
    else:  # ransac_center
        score = inliers * 0.7 - reproj_px * 6.0
        if panel_corner and reproj_px <= 8.0:
            score -= 70.0

    if name != 'corner_homography':
        if agreement_with_corner >= max(3, n_cards - 1):
            score += 40.0
        elif agreement_with_corner >= 2:
            score += 10.0
        else:
            score -= 30.0
    return score


def _select_xy_predictions(
    image_bgr: np.ndarray,
    yolo_det: List[Tuple[int, float, float, float, float]],
    corners_px: np.ndarray,
    *,
    src_wh: Tuple[int, int],
    corner_label: str,
    reproj_px: float,
    min_inliers: int = 12,
) -> Tuple[List[Dict[str, Any]], bool, Dict[str, Any]]:
    """Try corner / RANSAC / warp-grid XY; pick best-scoring backend."""
    w, h = src_wh
    n_cards = len(yolo_det)
    backends: List[Tuple[str, List[Dict[str, Any]], Dict[str, Any]]] = []

    preds_corner = map_yolo_to_cells_corners_homography(yolo_det, corners_px, w, h)
    backends.append(('corner_homography', preds_corner, {'xy_source': 'line_grid_corner_h'}))

    h_mat, ok_h, hmeta = build_grid_homography(image_bgr, corners_px, min_inliers=min_inliers)
    if ok_h and h_mat is not None:
        preds_rc, m_rc = map_yolo_to_cells_geom(
            yolo_det, h_mat, w, h, use_contact_point=False,
        )
        backends.append(('ransac_center', preds_rc, {**hmeta, **m_rc, 'xy_source': 'line_grid_ransac_center'}))
        preds_rt, m_rt = map_yolo_to_cells_geom(
            yolo_det, h_mat, w, h, use_contact_point=True,
        )
        backends.append(('ransac_contact', preds_rt, {**hmeta, **m_rt, 'xy_source': 'line_grid_ransac_contact'}))

    warped, h_warp = warp_panel_rect(image_bgr, corners_px)
    preds_wg, ok_wg, wg_meta = map_yolo_to_cells_warp_grid(
        yolo_det, warped, h_warp, w, h,
    )
    if ok_wg:
        backends.append(('warp_grid_lines', preds_wg, {**wg_meta, 'xy_source': 'line_grid_warp_grid'}))

    if not backends:
        return [], False, {'xy_source': 'line_grid_no_backend'}

    agree_base = backends[0][1]
    best_name = backends[0][0]
    best_preds = backends[0][1]
    best_meta = backends[0][2]
    best_score = -1e9

    for name, preds, meta in backends:
        agree = _pred_agreement(preds, agree_base) if name != 'corner_homography' else 0
        score = _score_xy_backend(
            name,
            meta,
            corner_label=corner_label,
            reproj_px=reproj_px,
            n_cards=n_cards,
            agreement_with_corner=agree if name != 'corner_homography' else 0,
        )
        if name != 'corner_homography':
            agree_c = _pred_agreement(preds, preds_corner)
            score += agree_c * 12.0
            if corner_label in ('img_panel', 'black_panel') and agree_c < 2:
                score -= 80.0
        if score > best_score:
            best_score = score
            best_name = name
            best_preds = preds
            best_meta = {**meta, 'xy_backend': name, 'xy_backend_score': float(score)}

    best_meta['xy_backend_selected'] = best_name
    return best_preds, True, best_meta


def analyze_cards_line_grid(
    image_bgr: np.ndarray,
    yolo_det: List[Tuple[int, float, float, float, float]],
    corners_px: np.ndarray,
    *,
    src_wh: Tuple[int, int],
    corner_label: str = '',
    reproj_px: float = 999.0,
    min_inliers: int = 12,
) -> Tuple[List[Dict[str, Any]], bool, Dict[str, Any]]:
    preds, ok, meta = _select_xy_predictions(
        image_bgr,
        yolo_det,
        corners_px,
        src_wh=src_wh,
        corner_label=corner_label,
        reproj_px=reproj_px,
        min_inliers=min_inliers,
    )
    if not ok:
        meta['corner_homography_for_xy'] = False
        return preds, False, meta
    meta['corner_homography_for_xy'] = meta.get('xy_backend') == 'corner_homography'
    return preds, True, meta
