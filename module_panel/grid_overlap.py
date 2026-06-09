"""Zgodność trzech siatek na warpie: rogi (szara), CLAHE (pomarańcz), OpenCV (zielona)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def ideal_uniform_grid_lines(w: int, h: int, *, n: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.array([float(i) * float(w) / float(n) for i in range(n + 1)], dtype=np.float32)
    ys = np.array([float(j) * float(h) / float(n) for j in range(n + 1)], dtype=np.float32)
    xs[-1] = float(max(0, w - 1))
    ys[-1] = float(max(0, h - 1))
    return xs, ys


def _as_line_array(vals: Optional[List[float]], n: int = 10) -> Optional[np.ndarray]:
    if vals is None:
        return None
    arr = np.asarray(vals, dtype=np.float32).reshape(-1)
    if arr.shape[0] != n + 1:
        return None
    return arr


def _enhance_warped_gray(warped_bgr: np.ndarray) -> np.ndarray:
    """CLAHE — na ciemnym warpie linie siatki są ledwo widoczne dla zwykłej maski HSV."""
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _grid_line_mask(enhanced_gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """Jasne cienkie linie siatki na ciemnym tle → maska białych linii."""
    h, w = enhanced_gray.shape[:2]
    block = max(15, int(min(h, w) // 24) | 1)
    bw = cv2.adaptiveThreshold(
        enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -4,
    )
    white_frac = float(np.mean(bw > 0))
    if white_frac > 0.55:
        bw = 255 - bw
        white_frac = float(np.mean(bw > 0))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bw, white_frac


def _morph_profiles(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape[:2]
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(15, h // 12)))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 16), 3))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, v_kernel)
    hori = cv2.morphologyEx(mask, cv2.MORPH_OPEN, h_kernel)
    return np.mean(vert > 0, axis=0).astype(np.float32), np.mean(hori > 0, axis=1).astype(np.float32)


def _lines_from_profile_peaks(profile: np.ndarray, n: int = 10) -> np.ndarray:
    """11 linii: lokalne maksima profilu w oknach wokół równomiernego podziału."""
    size = int(profile.shape[0])
    if size <= 1:
        return np.linspace(0.0, 1.0, n + 1, dtype=np.float32)
    prof = profile.astype(np.float32)
    mx = float(np.max(prof))
    if mx > 1e-6:
        prof = prof / mx
    step = size / float(n)
    lines = np.zeros(n + 1, dtype=np.float32)
    for i in range(n + 1):
        center = i * step
        radius = max(5, int(round(step * 0.42)))
        lo = max(0, int(round(center - radius)))
        hi = min(size, int(round(center + radius + 1)))
        chunk = prof[lo:hi]
        if chunk.size == 0:
            lines[i] = float(center)
        else:
            lines[i] = float(lo + int(np.argmax(chunk)))
    lines = np.clip(lines, 0.0, float(size - 1))
    return np.maximum.accumulate(lines)


def _detect_lines_from_profile(
    profile: np.ndarray,
    n: int,
    *,
    min_hits: int = 3,
    min_strength: float = 0.08,
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    from module_panel.grid import _profile_line_model

    lines, ok, meta = _profile_line_model(profile, n, min_hits=min_hits, min_strength=min_strength)
    if ok:
        return lines.astype(np.float32), True, meta
    peaks = _lines_from_profile_peaks(profile, n)
    meta = dict(meta or {})
    meta['method'] = 'peaks'
    meta['polyfit_ok'] = False
    return peaks, bool(np.max(profile) > 1e-5), meta


def detect_enhanced_white_grid_lines(
    warped_bgr: np.ndarray,
    *,
    n: int = 10,
    relaxed: bool = False,
) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
    """Linie siatki z CLAHE + morfologia (działa na ciemnym warpie nag5)."""
    h, w = warped_bgr.shape[:2]
    eq = _enhance_warped_gray(warped_bgr)
    mask, white_frac = _grid_line_mask(eq)
    x_prof, y_prof = _morph_profiles(mask)
    min_hits = 2 if relaxed else 3
    min_str = 0.05 if relaxed else 0.08
    xs, ok_x, mx = _detect_lines_from_profile(x_prof, n, min_hits=min_hits, min_strength=min_str)
    ys, ok_y, my = _detect_lines_from_profile(y_prof, n, min_hits=min_hits, min_strength=min_str)
    signal = float(max(np.max(x_prof), np.max(y_prof)))
    ok = bool(ok_x and ok_y and signal > 1e-5)
    meta: Dict[str, Any] = {
        'enhanced_white_frac': white_frac,
        'profile_signal': signal,
        'grid_x': mx,
        'grid_y': my,
        'relaxed': relaxed,
    }
    return xs, ys, ok, meta


def detect_hough_white_grid_lines(
    warped_bgr: np.ndarray,
    *,
    n: int = 10,
) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
    """Hough/LSD na wzmocnionym warpie — tylko podgląd; metryka używa profilu CLAHE."""
    return detect_enhanced_white_grid_lines(warped_bgr, n=n, relaxed=True)


def _classify_warped_segment(x1: float, y1: float, x2: float, y2: float) -> Optional[str]:
    ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)
    if ang < 14.0 or ang > 166.0:
        return 'h'
    if 76.0 < ang < 104.0:
        return 'v'
    return None


def _hough_segments_enhanced(warped_bgr: np.ndarray) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float]]]:
    eq = _enhance_warped_gray(warped_bgr)
    mask, _ = _grid_line_mask(eq)
    edges = cv2.Canny(mask, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    h, w = eq.shape[:2]
    min_len = max(18, int(0.28 * float(min(h, w))))
    thresh = max(18, int(min(h, w) // 22))
    raw = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=thresh,
        minLineLength=min_len, maxLineGap=max(8, int(0.02 * min(h, w))),
    )
    h_segs: List[Tuple[float, float, float, float]] = []
    v_segs: List[Tuple[float, float, float, float]] = []
    if raw is not None:
        for ln in raw:
            x1, y1, x2, y2 = (float(ln[0][0]), float(ln[0][1]), float(ln[0][2]), float(ln[0][3]))
            cls = _classify_warped_segment(x1, y1, x2, y2)
            if cls == 'h':
                h_segs.append((x1, y1, x2, y2))
            elif cls == 'v':
                v_segs.append((x1, y1, x2, y2))
    return h_segs, v_segs


def _mean_cell_iou(
    ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray, *, n: int = 10,
) -> float:
    ious: List[float] = []
    for j in range(n):
        for i in range(n):
            ix0, ix1 = float(ax[i]), float(ax[i + 1])
            iy0, iy1 = float(ay[j]), float(ay[j + 1])
            dx0, dx1 = float(bx[i]), float(bx[i + 1])
            dy0, dy1 = float(by[j]), float(by[j + 1])
            inter_w = max(0.0, min(ix1, dx1) - max(ix0, dx0))
            inter_h = max(0.0, min(iy1, dy1) - max(iy0, dy0))
            inter = inter_w * inter_h
            area_a = max(1e-6, (ix1 - ix0) * (iy1 - iy0))
            area_b = max(1e-6, (dx1 - dx0) * (dy1 - dy0))
            union = area_a + area_b - inter
            ious.append(float(inter / max(1e-6, union)))
    return float(np.mean(ious)) if ious else 0.0


def _normalize_lines(xs: np.ndarray, ys: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.clip(xs.astype(np.float32), 0.0, float(w - 1))
    y = np.clip(ys.astype(np.float32), 0.0, float(h - 1))
    return np.maximum.accumulate(x), np.maximum.accumulate(y)


def _grid_spacing_score(lines: np.ndarray, size: int, n: int) -> float:
    """Kara za „pominiętą” kolumnę/wiersz — nierówne odstępy między liniami."""
    diffs = np.diff(lines.astype(np.float64))
    if diffs.size < 2:
        return 0.0
    expected = float(size) / float(n)
    med = float(np.median(diffs))
    if med < expected * 0.55:
        return 0.0
    bad = 0
    for d in diffs:
        if abs(d - med) > 0.30 * expected or abs(d - expected) > 0.38 * expected:
            bad += 1
    return float(1.0 - bad / float(diffs.size))


def _triple_line_consensus(
    corner_x: np.ndarray,
    corner_y: np.ndarray,
    white_x: np.ndarray,
    white_y: np.ndarray,
    op_x: np.ndarray,
    op_y: np.ndarray,
    w: int,
    h: int,
    *,
    n: int = 10,
    tol_frac: float = 0.12,
) -> Tuple[float, Dict[str, Any]]:
    """Ułamek linii, przy których wszystkie 3 siatki są w tej samej tolerancji."""
    cell_w = max(1e-6, float(w) / float(n))
    cell_h = max(1e-6, float(h) / float(n))
    tol_x = tol_frac * cell_w
    tol_y = tol_frac * cell_h
    matched = 0
    max_spread_x = 0.0
    max_spread_y = 0.0
    for i in range(n + 1):
        sx = float(max(corner_x[i], white_x[i], op_x[i]) - min(corner_x[i], white_x[i], op_x[i]))
        sy = float(max(corner_y[i], white_y[i], op_y[i]) - min(corner_y[i], white_y[i], op_y[i]))
        max_spread_x = max(max_spread_x, sx)
        max_spread_y = max(max_spread_y, sy)
        if sx <= tol_x:
            matched += 1
        if sy <= tol_y:
            matched += 1
    ratio = float(matched) / float(2 * (n + 1))
    detail = {
        'triple_matched_lines': int(matched),
        'triple_total_lines': int(2 * (n + 1)),
        'max_spread_x_px': max_spread_x,
        'max_spread_y_px': max_spread_y,
        'tol_x_px': float(tol_x),
        'tol_y_px': float(tol_y),
    }
    return ratio, detail


def _line_match_ratio(
    ref_x: np.ndarray, ref_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
    w: int, h: int, *, n: int = 10, tol_frac: float = 0.18,
) -> float:
    cell_w = max(1e-6, float(w) / float(n))
    cell_h = max(1e-6, float(h) / float(n))
    tol_x = tol_frac * cell_w
    tol_y = tol_frac * cell_h
    matched = 0
    for i in range(n + 1):
        if min(abs(float(test_x[i]) - float(rx)) for rx in ref_x) <= tol_x:
            matched += 1
        if min(abs(float(test_y[i]) - float(ry)) for ry in ref_y) <= tol_y:
            matched += 1
    return float(matched) / float(2 * (n + 1))


def warp_panel_coverage_score(warped_bgr: np.ndarray, *, n: int = 10) -> Tuple[float, Dict[str, Any]]:
    """
    Czy na warpie widać panel w kolumnach brzegowych (a nie tło / ucięty bok).

    Przy przesuniętych rogach YOLO lewa/prawa kolumna ma inny profil jasności niż środek.
    """
    h, w = warped_bgr.shape[:2]
    eq = _enhance_warped_gray(warped_bgr)
    y0 = int(0.12 * h)
    y1 = max(int(0.88 * h), y0 + 1)
    col_means: List[float] = []
    for i in range(n):
        x0 = int(i * w / float(n))
        x1 = int((i + 1) * w / float(n))
        if x1 <= x0:
            x1 = x0 + 1
        col_means.append(float(np.mean(eq[y0:y1, x0:x1])))
    inner = np.asarray(col_means[1:-1], dtype=np.float64)
    med = float(np.median(inner)) if inner.size else float(np.mean(col_means))
    mad = float(np.median(np.abs(inner - med))) + 2.0
    meta: Dict[str, Any] = {
        'col_means': [round(v, 1) for v in col_means],
        'col_median_inner': round(med, 1),
        'col_mad_inner': round(mad, 1),
    }
    if med < 12.0:
        meta['fail'] = 'warp_too_dark'
        return 0.0, meta
    left_dev = abs(float(col_means[0]) - med) / mad
    right_dev = abs(float(col_means[-1]) - med) / mad
    max_dev = max(left_dev, right_dev)
    meta['edge_left_dev_mad'] = round(left_dev, 2)
    meta['edge_right_dev_mad'] = round(right_dev, 2)
    # Tylko skrajne odchylenie brzegu (tło / ucięta kolumna); łagodniejsze niż MAD@2
    if max_dev >= 7.0:
        score = 0.0
    elif max_dev <= 3.5:
        score = 1.0
    else:
        score = float(max(0.0, 1.0 - (max_dev - 3.5) / 4.5))
    meta['warp_panel_coverage'] = score
    return score, meta


def _detect_opencv_grid_lines(
    warped_bgr: np.ndarray,
    *,
    grid_lines_x: Optional[List[float]] = None,
    grid_lines_y: Optional[List[float]] = None,
    n: int = 10,
) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
    op_x = _as_line_array(grid_lines_x, n)
    op_y = _as_line_array(grid_lines_y, n)
    if op_x is not None and op_y is not None:
        return op_x, op_y, True, {'source': 'meta'}

    from module_panel.grid import detect_grid_lines_warped

    eq_bgr = cv2.cvtColor(_enhance_warped_gray(warped_bgr), cv2.COLOR_GRAY2BGR)
    xs, ys, ok, meta = detect_grid_lines_warped(eq_bgr)
    meta = dict(meta)
    meta['source'] = 'detect_grid_lines_warped_enhanced'
    if ok:
        return xs, ys, True, meta

    xs, ys, ok2, meta2 = detect_enhanced_white_grid_lines(warped_bgr, n=n, relaxed=False)
    meta2['source'] = 'enhanced_strict'
    return xs, ys, ok2, meta2


def measure_three_grid_overlap(
    warped_bgr: np.ndarray,
    *,
    grid_lines_x: Optional[List[float]] = None,
    grid_lines_y: Optional[List[float]] = None,
    n: int = 10,
    line_tol_frac: float = 0.12,
) -> Tuple[float, Dict[str, Any]]:
    """
    Zgodność 3 siatek na warpie (jak na podglądzie po prawej):

    1. **corner** — szara, równy podział z 4 rogów YOLO,
    2. **white** — pomarańczowa, CLAHE + jasne linie na zdjęciu,
    3. **opencv** — zielona, profil OpenCV / meta modułu B.

    ``overlap_ratio`` = triple_consensus × min(spacing scores).
    Migawka dopiero gdy wszystkie 3 się zgadzają (nie pominięta kolumna).
    """
    h, w = warped_bgr.shape[:2]
    meta: Dict[str, Any] = {'w': int(w), 'h': int(h), 'overlap_method': 'triple_grid'}

    corner_x, corner_y = ideal_uniform_grid_lines(w, h, n=n)
    meta['corner_lines_x'] = [float(v) for v in corner_x]
    meta['corner_lines_y'] = [float(v) for v in corner_y]

    white_x, white_y, white_ok, w_meta = detect_enhanced_white_grid_lines(
        warped_bgr, n=n, relaxed=True,
    )
    meta.update(w_meta)
    meta['white_line_ok'] = white_ok
    meta['hough_lines_x'] = [float(v) for v in white_x]
    meta['hough_lines_y'] = [float(v) for v in white_y]

    h_segs, v_segs = _hough_segments_enhanced(warped_bgr)
    meta['hough_h_segments'] = len(h_segs)
    meta['hough_v_segments'] = len(v_segs)
    meta['hough_line_ok'] = bool(len(h_segs) + len(v_segs) >= 4)

    if not white_ok or float(w_meta.get('profile_signal', 0)) < 1e-5:
        meta['fail'] = 'no_grid_signal'
        meta['overlap_ratio'] = 0.0
        return 0.0, meta

    op_x, op_y, op_ok, op_meta = _detect_opencv_grid_lines(
        warped_bgr, grid_lines_x=grid_lines_x, grid_lines_y=grid_lines_y, n=n,
    )
    meta.update(op_meta)
    meta['grid_line_ok'] = bool(op_ok)

    corner_x, corner_y = _normalize_lines(corner_x, corner_y, w, h)
    white_x, white_y = _normalize_lines(white_x, white_y, w, h)
    op_x, op_y = _normalize_lines(op_x, op_y, w, h)

    triple_ratio, triple_detail = _triple_line_consensus(
        corner_x, corner_y, white_x, white_y, op_x, op_y, w, h, n=n, tol_frac=line_tol_frac,
    )
    meta.update(triple_detail)
    meta['triple_consensus'] = float(triple_ratio)

    sp_corner = _grid_spacing_score(corner_x, w, n) * _grid_spacing_score(corner_y, h, n)
    sp_white = _grid_spacing_score(white_x, w, n) * _grid_spacing_score(white_y, h, n)
    sp_opencv = _grid_spacing_score(op_x, w, n) * _grid_spacing_score(op_y, h, n)
    spacing_min = float(min(sp_corner, sp_white, sp_opencv))
    meta['spacing_corner'] = float(sp_corner)
    meta['spacing_white'] = float(sp_white)
    meta['spacing_opencv'] = float(sp_opencv)
    meta['spacing_min'] = spacing_min

    pair_cw = _line_match_ratio(corner_x, corner_y, white_x, white_y, w, h, n=n, tol_frac=line_tol_frac)
    pair_co = _line_match_ratio(corner_x, corner_y, op_x, op_y, w, h, n=n, tol_frac=line_tol_frac)
    pair_wo = _line_match_ratio(white_x, white_y, op_x, op_y, w, h, n=n, tol_frac=line_tol_frac)
    meta['pair_corner_white'] = float(pair_cw)
    meta['pair_corner_opencv'] = float(pair_co)
    meta['pair_white_opencv'] = float(pair_wo)

    overlap = float(triple_ratio * spacing_min)
    if spacing_min < 0.92:
        meta['spacing_warn'] = True

    warp_cov, warp_meta = warp_panel_coverage_score(warped_bgr, n=n)
    meta.update(warp_meta)

    meta['overlap_ratio'] = overlap
    meta['cell_iou'] = float(_mean_cell_iou(corner_x, corner_y, op_x, op_y, n=n))
    meta['line_match_ratio'] = float(pair_wo)
    meta['grid_lines_x'] = [float(v) for v in op_x]
    meta['grid_lines_y'] = [float(v) for v in op_y]
    return overlap, meta


def measure_opencv_hough_grid_overlap(
    warped_bgr: np.ndarray,
    *,
    grid_lines_x: Optional[List[float]] = None,
    grid_lines_y: Optional[List[float]] = None,
    n: int = 10,
    line_tol_frac: float = 0.12,
) -> Tuple[float, Dict[str, Any]]:
    return measure_three_grid_overlap(
        warped_bgr,
        grid_lines_x=grid_lines_x,
        grid_lines_y=grid_lines_y,
        n=n,
        line_tol_frac=line_tol_frac,
    )


def measure_panel_opencv_grid_overlap(
    warped_bgr: np.ndarray,
    *,
    grid_lines_x: Optional[List[float]] = None,
    grid_lines_y: Optional[List[float]] = None,
    n: int = 10,
    line_tol_frac: float = 0.12,
) -> Tuple[float, Dict[str, Any]]:
    return measure_three_grid_overlap(
        warped_bgr,
        grid_lines_x=grid_lines_x,
        grid_lines_y=grid_lines_y,
        n=n,
        line_tol_frac=line_tol_frac,
    )


def draw_opencv_grid_on_warped(
    warped_bgr: np.ndarray,
    grid_lines_x: List[float],
    grid_lines_y: List[float],
    *,
    color: Tuple[int, int, int] = (60, 220, 80),
    thickness: int = 1,
) -> np.ndarray:
    out = warped_bgr
    h, w = out.shape[:2]
    xs = _as_line_array(grid_lines_x)
    ys = _as_line_array(grid_lines_y)
    if xs is None or ys is None:
        return out
    for x in xs:
        xi = int(round(float(x)))
        cv2.line(out, (xi, 0), (xi, h - 1), color, thickness, cv2.LINE_AA)
    for y in ys:
        yi = int(round(float(y)))
        cv2.line(out, (0, yi), (w - 1, yi), color, thickness, cv2.LINE_AA)
    return out


def draw_hough_grid_on_warped(
    warped_bgr: np.ndarray,
    grid_lines_x: List[float],
    grid_lines_y: List[float],
    *,
    color: Tuple[int, int, int] = (0, 160, 255),
    thickness: int = 1,
) -> np.ndarray:
    return draw_opencv_grid_on_warped(
        warped_bgr, grid_lines_x, grid_lines_y, color=color, thickness=thickness,
    )


def draw_opencv_grid_on_camera(
    vis: np.ndarray,
    grid_lines_x: List[float],
    grid_lines_y: List[float],
    homography_img_to_warp: np.ndarray,
    *,
    warp_w: int,
    warp_h: int,
    color: Tuple[int, int, int] = (60, 220, 80),
    thickness: int = 1,
) -> None:
    xs = _as_line_array(grid_lines_x)
    ys = _as_line_array(grid_lines_y)
    if xs is None or ys is None:
        return
    try:
        h_w2i = np.linalg.inv(homography_img_to_warp.astype(np.float64))
    except np.linalg.LinAlgError:
        return
    wh = max(int(warp_h), 1)
    ww = max(int(warp_w), 1)

    def _draw_polyline(samples: np.ndarray) -> None:
        pts = cv2.perspectiveTransform(samples, h_w2i).reshape(-1, 2)
        pts_i = pts.astype(np.int32)
        for i in range(len(pts_i) - 1):
            p0 = (int(pts_i[i, 0]), int(pts_i[i, 1]))
            p1 = (int(pts_i[i + 1, 0]), int(pts_i[i + 1, 1]))
            cv2.line(vis, p0, p1, color, thickness, cv2.LINE_AA)

    n_seg = 8
    for x in xs:
        ts = np.linspace(0.0, float(wh - 1), n_seg + 1, dtype=np.float32)
        seg = np.array([[[float(x), float(t)]] for t in ts], dtype=np.float32)
        _draw_polyline(seg)
    for y in ys:
        ts = np.linspace(0.0, float(ww - 1), n_seg + 1, dtype=np.float32)
        seg = np.array([[[float(t), float(y)]] for t in ts], dtype=np.float32)
        _draw_polyline(seg)
