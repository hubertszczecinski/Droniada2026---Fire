"""
Zewnętrzne rogi siatki 10×10 (TL, TR, BR, BL) — bez niebieskiego ROI HSV.

Źródła kandydatów: line_grid, border_scan, LSD grid_outer, alignment pipelines.
Wybór: struktura siatki + konsensus + perspektywa (skalowalne, bez kalibracji px).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pipeline_competition as pc

from module_pose.api import default_intrinsics
from release.safe_blue_roi import clip_quad_to_image

_MIN_GRID_PICK = 0.12
_ENSEMBLE_CLUSTER_PX = 72.0
_ENSEMBLE_MIN_MEMBERS = 2
_ENSEMBLE_ALLOWED = frozenset({
    'img_panel', 'border_scan', 'grid_outer_lsd', 'warp2_img_panel', 'warp',
})
# Słabe aligny — tylko gdy brak line_grid / border_scan z sensowną siatką.
_WEAK_ALIGN = frozenset({
    'white_grid', 'white_hlines', 'hsv_panel', 'morph_blob', 'dark_blob',
    'trapezoid', 'black_panel',
})
_TRUSTED_OUTER = frozenset({
    'img_panel', 'border_scan', 'grid_outer', 'grid_outer_lsd', 'lg_',
})


def enhance_for_corner_probe(image_bgr: np.ndarray) -> np.ndarray:
    """CLAHE na L — lepsze linie siatki przy cieniach (bez zmiany kolorów kartek mocno)."""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l2 = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l2, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def _is_trusted_outer(label: str, align: str) -> bool:
    cl = f'{label} {align}'.lower()
    return any(k in cl for k in _TRUSTED_OUTER)


def _has_trusted_grid_candidate(rows: List[Dict[str, Any]], min_gs: float = 0.28) -> bool:
    for r in rows:
        if r.get('corners') is None:
            continue
        if float(r.get('grid_structure_score', 0.0)) < min_gs:
            continue
        if _is_trusted_outer(str(r.get('label', '')), str(r.get('align_name', ''))):
            return True
    return False


def _filter_rows_for_pick(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Odrzuć white_grid/hsv gdy jest line_grid lub border_scan."""
    if not _has_trusted_grid_candidate(rows):
        return rows
    out: List[Dict[str, Any]] = []
    for r in rows:
        an = str(r.get('align_name', r.get('label', ''))).lower()
        cl = str(r.get('label', '')).lower()
        weak = any(w in an or w in cl for w in _WEAK_ALIGN)
        if weak:
            gs = float(r.get('grid_structure_score', 0.0))
            if gs < 0.55 and not _is_trusted_outer(cl, an):
                continue
        out.append(r)
    return out if out else rows


def _method_tier(lbl: str, align: str) -> int:
    name = f'{lbl} {align}'.lower()
    if 'lg_img_panel' in name or align == 'img_panel':
        return 4
    if 'warp' in name and 'img_panel' in name:
        return 4
    if 'border_scan' in name:
        return 2
    if 'grid_outer' in name or 'outer_grid' in name or align == 'grid_outer_lsd':
        return 0
    if 'white_hlines' in name or 'morph' in name or 'dark_blob' in name:
        return 0
    return 1


def _score_candidate_row(
    row: Dict[str, Any],
    img_w: int,
    *,
    ref_w_frac: Optional[float] = None,
    img_h: int = 0,
) -> float:
    from release.live_corners import (
        _is_near_axis_aligned_box,
        _quad_has_perspective_skew,
        _quad_width_frac_live,
    )

    reproj = float(row.get('reproj_mean_px', 999.0))
    lbl = str(row.get('label', ''))
    align = str(row.get('align_name', lbl))
    corners = row.get('corners')
    c_arr = np.asarray(corners, dtype=np.float32) if corners is not None else None
    grid_s = float(row.get('grid_structure_score', 0.0))
    interior_s = float(row.get('panel_interior_score', 0.0))

    trusted = _is_trusted_outer(lbl, align)
    reproj_eff = float(reproj)
    if trusted and grid_s >= 0.40:
        # line_grid: duży reproj PnP przy dobrych zewnętrznych rogach (pełny kadr / skew).
        reproj_eff = min(reproj_eff, 240.0)
    reproj_w = 0.15 * max(0.10, 1.0 - 0.82 * min(1.0, grid_s))
    if trusted:
        reproj_w *= 0.55
    score = reproj_eff * reproj_w
    score -= min(200.0, grid_s * 24.0)
    score -= min(95.0, interior_s * 35.0)

    if 'lg_img_panel' in lbl or 'img_panel' in lbl or align == 'img_panel':
        score -= 150.0
    if 'warp2_img_panel' in lbl:
        score -= 175.0
    elif 'warp' in lbl and 'img_panel' in lbl:
        score -= 140.0
    if 'border_scan' in lbl or align == 'border_scan':
        score -= 55.0
    if 'grid_outer' in lbl or align in ('grid_outer_lsd', 'outer_grid'):
        score += 130.0
    if 'white_grid' in lbl or align == 'white_grid':
        score += 320.0
    if 'white_hlines' in lbl or align == 'white_hlines':
        score += 200.0
    if 'hsv_panel' in lbl or align == 'hsv_panel':
        score += 280.0
    elif 'morph' in lbl or 'dark_blob' in lbl or align in ('morph_blob', 'dark_blob'):
        if grid_s < 0.30:
            score += 160.0
        elif grid_s >= 0.55:
            score -= 45.0

    if c_arr is not None:
        relax_geom = trusted and grid_s >= 0.45
        if _is_near_axis_aligned_box(c_arr, tol_deg=4.0):
            score += 40.0 if relax_geom else 220.0
        if not _quad_has_perspective_skew(c_arr):
            score += 20.0 if relax_geom else 110.0
        w_frac = _quad_width_frac_live(c_arr, img_w)
        if img_h > 0:
            area_frac = float(cv2.contourArea(c_arr.reshape(1, 4, 2).astype(np.float32))) / float(img_w * img_h)
            if area_frac < 0.10:
                score += 280.0
            elif area_frac > 0.62:
                score += 90.0 if relax_geom else 280.0
            elif 0.14 <= area_frac <= 0.48:
                score -= 35.0
        if w_frac > 0.90:
            score += 70.0 if relax_geom else 400.0
        elif w_frac < 0.38:
            score += 200.0
        elif 0.48 <= w_frac <= 0.88:
            score -= 25.0
        if ref_w_frac is not None and ref_w_frac > 0.42:
            if abs(w_frac - ref_w_frac) > 0.22:
                score += abs(w_frac - ref_w_frac) * 180.0

    return score


def _consensus_penalty(row: Dict[str, Any], peers: List[Dict[str, Any]]) -> float:
    c = row.get('corners')
    if c is None or len(peers) < 2:
        return 0.0
    q = np.asarray(c, dtype=np.float32)
    others: List[np.ndarray] = []
    for p in peers:
        if p is row:
            continue
        pc = p.get('corners')
        if pc is None:
            continue
        if float(p.get('grid_structure_score', 0.0)) < 0.35:
            continue
        others.append(np.asarray(pc, dtype=np.float32))
    if len(others) < 2:
        return 0.0
    med = np.median(np.stack(others, axis=0), axis=0)
    dist = float(np.mean(np.linalg.norm(q - med, axis=1)))
    if dist < 55.0:
        return 0.0
    return min(320.0, (dist - 55.0) * 2.2)


def _ref_outer_width(rows: List[Dict[str, Any]], img_w: int) -> Optional[float]:
    from release.live_corners import _quad_width_frac_live

    best = 0.0
    for r in rows:
        name = str(r.get('align_name', r.get('label', ''))).lower()
        if 'white_hlines' in name or 'white_grid' in name:
            continue
        c = r.get('corners')
        if c is None:
            continue
        wf = _quad_width_frac_live(np.asarray(c, dtype=np.float32), img_w)
        best = max(best, wf)
    return best if best > 0.42 else None


def _ensemble_from_pool(
    pool: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], str, Optional[List[int]]]:
    """
    Mediana rogów z klastra kandydatów zgodnych geometrycznie (≤_ENSEMBLE_CLUSTER_PX).
    """
    from release.live_corners import _quad_width_frac_live

    h, w = image_shape[:2]
    entries: List[Tuple[np.ndarray, float, float, str]] = []
    for r in pool:
        raw = r.get('corners')
        if raw is None:
            continue
        gs = float(r.get('grid_structure_score', 0.0))
        cl = str(r.get('label', '')).lower()
        an = str(r.get('align_name', '')).lower()
        if not any(k in cl or k in an for k in _ENSEMBLE_ALLOWED):
            if 'lg_' not in cl and 'img_panel' not in an and 'border_scan' not in an:
                continue
        if gs < 0.28:
            continue
        q = clip_quad_to_image(np.asarray(raw, dtype=np.float32), (h, w))
        wf = _quad_width_frac_live(q, w)
        if wf < 0.38 or wf > 0.92:
            continue
        tier = float(_method_tier(str(r.get('label', '')), str(r.get('align_name', ''))))
        wgt = max(0.15, gs) * (1.0 + 0.35 * tier)
        entries.append((q, wgt, gs, str(r.get('label', 'outer'))))

    if len(entries) < _ENSEMBLE_MIN_MEMBERS:
        return None, 'none', None

    quads = [e[0] for e in entries]
    weights = np.array([e[1] for e in entries], dtype=np.float64)
    weights /= weights.sum()

    # Najbardziej „centralny” kandydat (najmniejsza średnia odległość do innych)
    mean_d: List[float] = []
    for i, qi in enumerate(quads):
        ds = [float(np.mean(np.linalg.norm(qi - qj, axis=1))) for j, qj in enumerate(quads) if j != i]
        mean_d.append(float(np.mean(ds)) if ds else 9999.0)
    hub = int(np.argmin(mean_d))
    hub_q = quads[hub]

    cluster_idx = [
        i for i, q in enumerate(quads)
        if float(np.mean(np.linalg.norm(q - hub_q, axis=1))) <= _ENSEMBLE_CLUSTER_PX
    ]
    if len(cluster_idx) < _ENSEMBLE_MIN_MEMBERS:
        order = np.argsort(mean_d)
        cluster_idx = [hub]
        for idx in order:
            ii = int(idx)
            if ii == hub:
                continue
            if float(np.mean(np.linalg.norm(quads[ii] - hub_q, axis=1))) <= _ENSEMBLE_CLUSTER_PX:
                cluster_idx.append(ii)
            if len(cluster_idx) >= 3:
                break
        if len(cluster_idx) < _ENSEMBLE_MIN_MEMBERS:
            return None, 'none', None

    stack = np.stack([quads[i] for i in cluster_idx], axis=0)
    w_sub = weights[cluster_idx]
    w_sub = w_sub / max(1e-9, w_sub.sum())
    corners = np.average(stack, axis=0, weights=w_sub)
    corners = pc.order_points(corners.astype(np.float32))

    labels = sorted({entries[i][3] for i in cluster_idx})
    short = '+'.join(
        lbl.replace('align_', '')[:10] for lbl in labels[:3]
    )
    return corners, f'ensemble[{short}]', cluster_idx


def _best_lg_family_row(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_rp = 9999.0
    for r in rows:
        if r.get('corners') is None:
            continue
        cl = str(r.get('label', '')).lower()
        an = str(r.get('align_name', '')).lower()
        if 'lg_' not in cl and an != 'img_panel':
            continue
        rp = float(r.get('reproj_mean_px', 999.0))
        if rp < best_rp:
            best_rp = rp
            best = r
    return best


def _pick_alt_when_lg_border_far(
    rows: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Gdy line_grid i border_scan są daleko (≥300 px), lg często jest zły — wybierz morph/border
    z umiarkowanym reproj (bez dark_blob z reproj <20).
    """
    lg_row = _best_lg_family_row(rows)
    bs_row = None
    for r in rows:
        cl = str(r.get('label', '')).lower()
        an = str(r.get('align_name', '')).lower()
        if r.get('corners') is None:
            continue
        if 'border_scan' in cl or an == 'border_scan':
            bs_row = r
    if lg_row is None or bs_row is None:
        return None
    lg_rp = float(lg_row.get('reproj_mean_px', 999.0))
    bs_rp = float(bs_row.get('reproj_mean_px', 999.0))
    split = float(np.mean(np.linalg.norm(
        np.asarray(lg_row['corners'], dtype=np.float32)
        - np.asarray(bs_row['corners'], dtype=np.float32),
        axis=1,
    )))
    if split < 300.0:
        return None
    # Bardzo niski reproj lg przy dużym rozjechu z border → PnP mylący, szukaj alt.
    if lg_rp < min(55.0, bs_rp * 0.50) and lg_rp >= 22.0:
        return None
    alts: List[Dict[str, Any]] = []
    for r in rows:
        if r.get('corners') is None:
            continue
        cl = str(r.get('label', '')).lower()
        if not any(k in cl for k in ('border_scan', 'morph_blob')):
            continue
        rp = float(r.get('reproj_mean_px', 999.0))
        if 22.0 <= rp <= 145.0 and float(r.get('grid_structure_score', 0.0)) >= 0.35:
            alts.append(r)
    if not alts:
        return None
    return min(alts, key=lambda r: float(r.get('reproj_mean_px', 999.0)))


def _prefer_border_when_lg_diverges(
    rows: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Gdy line_grid i border_scan mocno się rozjeżdżają, border_scan bywa bliżej zewnętrznej siatki."""
    lg_row = _best_lg_family_row(rows)
    bs_row = None
    for r in rows:
        cl = str(r.get('label', '')).lower()
        an = str(r.get('align_name', '')).lower()
        if r.get('corners') is None:
            continue
        if 'border_scan' in cl or an == 'border_scan':
            bs_row = r
    if lg_row is None or bs_row is None:
        return None
    if float(bs_row.get('grid_structure_score', 0.0)) < 0.35:
        return None
    lg_rp = float(lg_row.get('reproj_mean_px', 999.0))
    bs_rp = float(bs_row.get('reproj_mean_px', 999.0))
    if lg_rp < min(55.0, bs_rp * 0.50):
        return None
    qc = np.asarray(lg_row['corners'], dtype=np.float32)
    bc = np.asarray(bs_row['corners'], dtype=np.float32)
    split = float(np.mean(np.linalg.norm(qc - bc, axis=1)))
    if split >= 380.0 and lg_rp >= bs_rp * 0.85:
        return bs_row
    if split >= 220.0 and lg_rp >= bs_rp * 0.98:
        return bs_row
    return None


def pick_outer_corners(
    rows: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], str]:
    if not rows:
        return None, 'none'
    rows = _filter_rows_for_pick(rows)
    h, w = image_shape[:2]
    ref_w = _ref_outer_width(rows, w)

    for r in rows:
        r['rank_score'] = _score_candidate_row(r, w, ref_w_frac=ref_w, img_h=h)

    ordered = sorted(rows, key=lambda r: float(r.get('rank_score', 999.0)))
    lg_rows = [
        r for r in ordered
        if r.get('corners') is not None and 'lg_' in str(r.get('label', '')).lower()
    ]
    if len(lg_rows) >= 2:
        best_lg = min(lg_rows, key=lambda r: float(r.get('reproj_mean_px', 999.0)))
        ordered = [best_lg] + [r for r in ordered if r is not best_lg]

    if len(ordered) >= 2:
        best_sc = float(ordered[0].get('rank_score', 999.0))
        pool = [
            r for r in ordered
            if r.get('corners') is not None
            and (
                float(r.get('rank_score', 999.0)) <= best_sc + 200.0
                or (
                    float(r.get('grid_structure_score', 0.0)) >= 0.45
                    and float(r.get('reproj_mean_px', 999.0)) <= 180.0
                )
            )
        ]
        if len(pool) >= 2:
            for r in pool:
                r['_consensus_penalty'] = _consensus_penalty(r, pool)

            def _key(r: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
                gs = float(r.get('grid_structure_score', 0.0))
                rp = float(r.get('reproj_mean_px', 999.0))
                tier = float(_method_tier(str(r.get('label', '')), str(r.get('align_name', ''))))
                cons = float(r.get('_consensus_penalty', 0.0))
                return (tier, gs, -cons, -min(rp, 140.0), -float(r.get('rank_score', 999.0)))

            pool.sort(key=_key, reverse=True)
            if pool[0] is not ordered[0]:
                ordered = [pool[0]] + [r for r in ordered if r is not pool[0]]

            alt_pick = _pick_alt_when_lg_border_far(rows)
            if alt_pick is not None:
                corners = clip_quad_to_image(
                    np.asarray(alt_pick['corners'], dtype=np.float32), (h, w),
                )
                alt_pick['corners'] = corners
                alt_pick['chosen'] = True
                for r in rows:
                    r['chosen'] = r is alt_pick
                rows.sort(key=lambda r: (0.0 if r.get('chosen') else 1.0, float(r.get('rank_score', 999.0))))
                return corners, str(alt_pick.get('label', 'outer'))

            border_pick = _prefer_border_when_lg_diverges(rows)
            if border_pick is not None:
                corners = clip_quad_to_image(
                    np.asarray(border_pick['corners'], dtype=np.float32), (h, w),
                )
                border_pick['corners'] = corners
                border_pick['chosen'] = True
                for r in rows:
                    r['chosen'] = r is border_pick
                rows.sort(key=lambda r: (0.0 if r.get('chosen') else 1.0, float(r.get('rank_score', 999.0))))
                return corners, str(border_pick.get('label', 'align_border_scan'))

            ens_corners, ens_label, ens_idx = _ensemble_from_pool(pool, (h, w))
            if ens_corners is not None and ens_idx is not None:
                for r in rows:
                    r['chosen'] = False
                pick_row = pool[ens_idx[0]]
                best_gs = float(pick_row.get('grid_structure_score', 0.0))
                for i in ens_idx:
                    gs_i = float(pool[i].get('grid_structure_score', 0.0))
                    if gs_i > best_gs:
                        best_gs = gs_i
                        pick_row = pool[i]
                pick_row['corners'] = ens_corners
                pick_row['label'] = ens_label
                pick_row['chosen'] = True
                rows.sort(key=lambda r: (0.0 if r.get('chosen') else 1.0, float(r.get('rank_score', 999.0))))
                return ens_corners, ens_label

    alt_pick = _pick_alt_when_lg_border_far(rows)
    if alt_pick is not None:
        h, w = image_shape[:2]
        corners = clip_quad_to_image(
            np.asarray(alt_pick['corners'], dtype=np.float32), (h, w),
        )
        alt_pick['corners'] = corners
        alt_pick['chosen'] = True
        for r in rows:
            r['chosen'] = r is alt_pick
        rows.sort(key=lambda r: (0.0 if r.get('chosen') else 1.0, float(r.get('rank_score', 999.0))))
        return corners, str(alt_pick.get('label', 'outer'))

    border_pick = _prefer_border_when_lg_diverges(rows)
    if border_pick is not None:
        h, w = image_shape[:2]
        corners = clip_quad_to_image(
            np.asarray(border_pick['corners'], dtype=np.float32), (h, w),
        )
        border_pick['corners'] = corners
        border_pick['label'] = str(border_pick.get('label', 'align_border_scan'))
        border_pick['chosen'] = True
        for r in rows:
            r['chosen'] = r is border_pick
        rows.sort(key=lambda r: (0.0 if r.get('chosen') else 1.0, float(r.get('rank_score', 999.0))))
        return corners, str(border_pick['label'])

    best: Optional[Dict[str, Any]] = None
    corners: Optional[np.ndarray] = None
    label = 'none'
    for cand in ordered:
        raw = cand.get('corners')
        if raw is None:
            continue
        gs = float(cand.get('grid_structure_score', 0.0))
        cl = str(cand.get('label', '')).lower()
        if gs < _MIN_GRID_PICK and 'lg_' not in cl and 'img_panel' not in cl:
            continue
        if gs < 0.05 and 'border_scan' not in cl:
            continue
        corners = clip_quad_to_image(np.asarray(raw, dtype=np.float32), (h, w))
        best = cand
        label = str(cand.get('label', 'outer'))
        break

    for r in rows:
        r['chosen'] = r is best

    if best is None or corners is None:
        return None, 'none'

    best['corners'] = corners
    best['label'] = label
    rows.sort(key=lambda r: (0.0 if r.get('chosen') else 1.0, float(r.get('rank_score', 999.0))))
    return corners, label


def _refine_outer_quad(
    image_bgr: np.ndarray,
    quad: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    from release.live_corners import _evaluate_panel_roi_candidate

    q0 = pc.order_points(quad.astype(np.float32))
    _, d0 = _evaluate_panel_roi_candidate(
        image_bgr, q0, k, dist, label='refine_check', preserve_geometry=True,
    )
    gs0 = float(d0.get('grid_structure_score', 0.0))
    rp0 = float(d0.get('reproj_mean_px', 999.0))
    try:
        from module_pose.refine_corners import refine_panel_corners_uniform_grid

        refined = refine_panel_corners_uniform_grid(image_bgr, q0)
        if refined is not None:
            q1 = pc.order_points(refined.astype(np.float32))
            _, d1 = _evaluate_panel_roi_candidate(
                image_bgr, q1, k, dist, label='refine_check', preserve_geometry=True,
            )
            gs1 = float(d1.get('grid_structure_score', 0.0))
            rp1 = float(d1.get('reproj_mean_px', 999.0))
            if gs1 >= gs0 - 0.02 and rp1 <= max(rp0 * 1.15, rp0 + 30.0):
                return q1
    except Exception:
        pass
    return q0


def probe_outer_corner_candidates(
    image_bgr: np.ndarray,
    k: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    *,
    fast: bool = True,
    probe_bgr: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Zbierz kandydatów zewnętrznych rogów — bez niebieskiego ROI."""
    det_bgr = probe_bgr if probe_bgr is not None else image_bgr
    from module_pose.api import _rough_quad_from_white_hlines, detect_corners_black_panel
    from release.alignment_pipelines import (
        _pad_bounds_asymmetric,
        pipeline_dark_blob,
        pipeline_hsv_panel,
        pipeline_morph_blob,
        pipeline_trapezoid,
        pipeline_white_grid,
    )
    from release.grid_outer_quad import detect_yellow_quad_border_scan, detect_yellow_quad_scalable
    from release.live_corners import (
        _prepare_detection_image,
        _quad_to_full_image,
        _try_add_align_corner_row,
        probe_line_grid_corner_candidates,
    )

    h, w = image_bgr.shape[:2]
    if k is None or dist is None:
        k, dist = default_intrinsics((h, w))

    work, scale, ox, oy = _prepare_detection_image(det_bgr)
    work_k, work_dist = default_intrinsics(work.shape[:2])

    rows: List[Dict[str, Any]] = []
    seen: List[np.ndarray] = []

    lg_rows = probe_line_grid_corner_candidates(det_bgr, k, dist)
    for r in lg_rows[: (8 if fast else 10)]:
        if r.get('corners') is None:
            continue
        _try_add_align_corner_row(
            rows, seen, image_bgr, k, dist, np.asarray(r['corners'], dtype=np.float32),
            f"align_{r['label']}",
            align_name=str(r['label']),
            align_conf=0.85,
            align_basis='line_grid outer',
        )

    scan_q = detect_yellow_quad_border_scan(det_bgr, None)
    if scan_q is not None:
        _try_add_align_corner_row(
            rows, seen, image_bgr, k, dist, scan_q,
            'align_border_scan',
            align_name='border_scan',
            align_conf=0.80,
            align_basis='grid line scan full frame',
        )

    og, _og_roi, og_src = detect_yellow_quad_scalable(det_bgr)
    if og is not None:
        _try_add_align_corner_row(
            rows, seen, image_bgr, k, dist, og,
            'align_grid_outer',
            align_name='grid_outer_lsd',
            align_conf=0.72,
            align_basis=f'outer grid LSD ({og_src})',
        )

    pipelines = (
        ('trapezoid', pipeline_trapezoid),
        ('hsv_panel', pipeline_hsv_panel),
        ('morph_blob', pipeline_morph_blob),
        ('dark_blob', pipeline_dark_blob),
    )
    if not fast:
        pipelines = pipelines + (('white_grid', pipeline_white_grid),)

    for name, fn in pipelines:
        res = fn(work)
        if not res.ok or res.quad is None:
            continue
        quad_full = _quad_to_full_image(res.quad.astype(np.float32), scale=scale, ox=ox, oy=oy)
        _try_add_align_corner_row(
            rows, seen, image_bgr, k, dist, quad_full,
            f'align_{name}',
            align_name=name,
            align_conf=float(res.confidence),
            align_basis=str(res.meta.get('basis', '')),
        )

    bp = detect_corners_black_panel(work, work_k, work_dist)
    if bp is not None and not fast:
        bp_full = _quad_to_full_image(bp.astype(np.float32), scale=scale, ox=ox, oy=oy)
        _try_add_align_corner_row(
            rows, seen, image_bgr, k, dist, bp_full,
            'align_black_panel',
            align_name='black_panel',
            align_conf=0.5,
            align_basis='black_panel',
        )

    wh = _rough_quad_from_white_hlines(work)
    if wh is not None and not fast:
        wh = _quad_to_full_image(wh.astype(np.float32), scale=scale, ox=ox, oy=oy)
        wh = _pad_bounds_asymmetric(
            wh, image_bgr.shape[:2], left=0.10, right=0.06, top=0.03, bottom=0.02,
        )
        _try_add_align_corner_row(
            rows, seen, image_bgr, k, dist, wh,
            'align_white_hlines',
            align_name='white_hlines',
            align_conf=0.45,
            align_basis='white_hlines',
        )

    return rows


def detect_outer_corners(
    image_bgr: np.ndarray,
    k: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    *,
    fast: bool = True,
    refine: bool = False,
    probe_bgr: Optional[np.ndarray] = None,
    enhance_probe: bool = True,
) -> Tuple[Optional[np.ndarray], str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Główny API: 4 zewnętrzne rogi siatki + metadane (bez niebieskiego ROI).
    """
    if k is None or dist is None:
        k, dist = default_intrinsics(image_bgr.shape[:2])

    meta: Dict[str, Any] = {'method': 'outer_corners', 'roi_source': 'none'}
    pimg = probe_bgr
    if pimg is None and enhance_probe:
        pimg = enhance_for_corner_probe(image_bgr)
        meta['probe_enhanced'] = 1.0

    rows = probe_outer_corner_candidates(
        image_bgr, k, dist, fast=fast, probe_bgr=pimg,
    )
    if not rows:
        meta['fail'] = 'no_candidates'
        return None, 'none', [], meta

    corners, label = pick_outer_corners(rows, image_bgr.shape[:2])
    if corners is None:
        meta['fail'] = 'no_pick'
        return None, 'none', rows, meta

    if refine:
        corners = _refine_outer_quad(image_bgr, corners, k, dist)
        if '+refined' not in label:
            label = f'{label}+refined'

    chosen = next((r for r in rows if r.get('chosen')), None)
    if chosen is not None:
        meta['grid_structure_score'] = chosen.get('grid_structure_score')
        meta['panel_interior_score'] = chosen.get('panel_interior_score')
        meta['reproj_mean_px'] = chosen.get('reproj_mean_px')
        meta['align_basis'] = chosen.get('align_basis', '')
        meta['align_confidence'] = chosen.get('align_confidence', 0.0)

    meta['corner_source'] = label
    return corners.astype(np.float32), label, rows, meta
