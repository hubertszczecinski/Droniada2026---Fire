"""Kolory kartek na migawkach — komórki GT + siatka z grid_lines (nie równomierna 10×10)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import pipeline_competition as pc
from release.card_color_profile import (
    CardColorProfile,
    CENTROID_MAX_DIST,
    classify_hsv_centroid,
    hsv_centroid_distance,
    load_active_profile,
    prune_centroids_by_hue,
    tight_range_from_centroid,
)
from release.live_card_detect import (
    _center_crop_patch,
    _color_core_frac,
    _enhance_patch_bgr_for_color,
    _patch_median_hsv,
    _white_grid_mask,
)

_ROOT = Path(__file__).resolve().parent.parent
_SCHEMA_LAYOUT = 'droniada_panel_color_layout_v1'

# Luźniejsze progi tylko na migawce (znane komórki z kartą).
_SNAP_MIN_SAT = 28.0
_SNAP_MIN_VAL = 22.0
_SNAP_MIN_S_MEAN = 24.0
_SNAP_MIN_COLOR_FRAC = 0.06
_SNAP_CENTROID_MAX = 4.2


def load_test_mov_gt(path: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    """Kompatybilność wsteczna — patrz load_panel_color_layout."""
    return load_panel_color_layout(path)


def load_panel_color_layout(path: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    """
    Układ kartek na panelu (nagranie kalibracyjne).

    • Jawna ścieżka (skrypty kalibracji, --layout) — ładuje plik.
    • Bez argumentu — tylko DRONIADA_PANEL_LAYOUT (opcjonalny tryb GT/debug).
    • Brak auto-ładowania config/panel_color_layout.json ani test_mov_gt.json —
      żeby layout nie wpływał na normalną pracę systemu.
    """
    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            p = _ROOT / p
        if p.is_file():
            with p.open('r', encoding='utf-8') as fh:
                return json.load(fh)
        return None
    env = os.environ.get('DRONIADA_PANEL_LAYOUT', '').strip()
    if not env:
        return None
    p = Path(env)
    if not p.is_absolute():
        p = _ROOT / env
    if p.is_file():
        with p.open('r', encoding='utf-8') as fh:
            return json.load(fh)
    return None


def uniform_grid_lines(shape: Tuple[int, int]) -> Tuple[List[float], List[float]]:
    h, w = int(shape[0]), int(shape[1])
    return (
        [float(i * w / 10.0) for i in range(11)],
        [float(i * h / 10.0) for i in range(11)],
    )


def grid_lines_for_warped(
    warped_bgr: np.ndarray,
) -> Tuple[List[float], List[float], str]:
    """Siatka z detekcji linii na warpie; fallback: równomierna 10×10."""
    from module_panel.grid import detect_grid_lines_warped

    xs, ys, ok, _meta = detect_grid_lines_warped(warped_bgr)
    if ok and len(xs) >= 11 and len(ys) >= 11:
        return [float(x) for x in xs], [float(y) for y in ys], 'detected'
    glx, gly = uniform_grid_lines(warped_bgr.shape[:2])
    return glx, gly, 'uniform'


def sample_cell_median_hsv(
    warped_bgr: np.ndarray,
    grid_lines_x: Sequence[float],
    grid_lines_y: Sequence[float],
    row: int,
    col: int,
) -> Optional[Tuple[float, float, float]]:
    core = extract_cell_core_bgr(warped_bgr, grid_lines_x, grid_lines_y, row, col)
    if core.size == 0:
        return None
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    return _patch_median_hsv(hsv)


def aggregate_hsv_samples(
    samples: Sequence[Tuple[float, float, float]],
) -> Optional[Tuple[float, float, float]]:
    if not samples:
        return None
    arr = np.array(samples, dtype=np.float64)
    return (
        float(np.median(arr[:, 0])),
        float(np.median(arr[:, 1])),
        float(np.median(arr[:, 2])),
    )


def gt_cells_from_config(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(cfg.get('cells') or [])


def cell_rect_from_grid_lines(
    grid_lines_x: Sequence[float],
    grid_lines_y: Sequence[float],
    row: int,
    col: int,
) -> Tuple[int, int, int, int]:
    """Wiersz 1 = dół panelu (ostatni przedział grid_lines_y)."""
    ri = 10 - int(row)
    ci = int(col) - 1
    if ri < 0 or ri > 9 or ci < 0 or ci > 9:
        raise ValueError(f'Poza siatką: row={row} col={col}')
    x0 = int(round(float(grid_lines_x[ci])))
    x1 = int(round(float(grid_lines_x[ci + 1])))
    y0 = int(round(float(grid_lines_y[ri])))
    y1 = int(round(float(grid_lines_y[ri + 1])))
    return x0, y0, x1, y1


def extract_cell_core_bgr(
    warped_bgr: np.ndarray,
    grid_lines_x: Sequence[float],
    grid_lines_y: Sequence[float],
    row: int,
    col: int,
) -> np.ndarray:
    x0, y0, x1, y1 = cell_rect_from_grid_lines(grid_lines_x, grid_lines_y, row, col)
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 0, 3), dtype=np.uint8)
    patch = warped_bgr[y0:y1, x0:x1]
    patch = _enhance_patch_bgr_for_color(patch)
    return _center_crop_patch(patch, _color_core_frac(row, col))


def _color_fraction_on_core(hsv: np.ndarray, cls_id: int, mh: float, ms: float, mv: float) -> float:
    lo, hi = tight_range_from_centroid(int(cls_id), mh, ms, mv)
    white = _white_grid_mask(hsv)
    mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0
    mask = mask & ~white
    return float(np.count_nonzero(mask)) / float(max(1, mask.size))


def classify_cell_relaxed(
    core_bgr: np.ndarray,
    *,
    row: int,
    col: int,
    expected_color: Optional[str] = None,
    profile: Optional[CardColorProfile] = None,
) -> Optional[Tuple[int, float, float, Tuple[float, float, float]]]:
    """Klasyfikacja środka komórki — progi na migawki (ciemne / brzeg panelu)."""
    if core_bgr.size == 0:
        return None
    prof = profile or load_active_profile()
    hsv = cv2.cvtColor(core_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    sat_mean = float(np.mean(sat))
    val_mean = float(np.mean(val))
    if sat_mean < _SNAP_MIN_S_MEAN or val_mean < _SNAP_MIN_VAL:
        return None
    med = _patch_median_hsv(hsv)
    if med is None:
        return None
    mh, ms, mv = med
    if ms < _SNAP_MIN_SAT or mv < _SNAP_MIN_VAL:
        return None

    if not prof.has_centroids():
        return None
    cls_id, dist, _ = classify_hsv_centroid(mh, ms, mv, prof.centroids_by_cls)
    cap = _SNAP_CENTROID_MAX
    if expected_color:
        exp_id = int(pc.COLOR_TO_CLASS.get(str(expected_color).upper(), -1))
        if exp_id >= 0:
            exp_name = str(expected_color).upper()
            # Hue z migawki Test.mov — żółta H~25–45, czerwona H~175+, niebieska H~100–125.
            hue_ok = (
                (exp_name == 'ZOLTA' and 18.0 <= mh <= 52.0 and ms >= 80.0)
                or (exp_name == 'CZERWONA' and (mh <= 15.0 or mh >= 165.0) and ms >= 80.0)
                or (exp_name == 'NIEBIESKA' and 95.0 <= mh <= 130.0 and ms >= 70.0)
                or (exp_name == 'ZIELONA' and 45.0 <= mh <= 100.0 and ms >= 70.0)
            )
            best_d = float('inf')
            for cent in prof.centroids_by_cls.get(exp_id, []):
                best_d = min(best_d, hsv_centroid_distance(mh, ms, mv, cent))
            if best_d <= cap or hue_ok:
                cls_id = exp_id
                dist = min(best_d, 0.5 if hue_ok else best_d)
    if cls_id is None or dist > cap:
        return None

    frac = _color_fraction_on_core(hsv, int(cls_id), mh, ms, mv)
    if frac < _SNAP_MIN_COLOR_FRAC and dist > CENTROID_MAX_DIST:
        return None
    return (int(cls_id), float(frac), float(sat_mean), (mh, ms, mv))


def detect_gt_cells_on_warped(
    warped_bgr: np.ndarray,
    grid_lines_x: Sequence[float],
    grid_lines_y: Sequence[float],
    cells: Sequence[Dict[str, Any]],
    *,
    profile: Optional[CardColorProfile] = None,
) -> List[Dict[str, Any]]:
    """Czytaj kolory tylko w znanych komórkach Test.mov (migawki)."""
    out: List[Dict[str, Any]] = []
    prof = profile or load_active_profile()
    for spec in cells:
        color = str(spec['color']).upper()
        row = int(spec['row'])
        col = int(spec['col'])
        core = extract_cell_core_bgr(warped_bgr, grid_lines_x, grid_lines_y, row, col)
        meta = classify_cell_relaxed(
            core, row=row, col=col, expected_color=color, profile=prof,
        )
        if meta is None:
            continue
        cls_id, frac, sat_mean, med = meta
        if pc.CLASS_TO_COLOR.get(int(cls_id)) != color:
            continue
        out.append({
            'x': col,
            'y': row,
            'color': color,
            'cls_id': int(cls_id),
            'color_frac': float(frac),
            'sat_mean': float(sat_mean),
            'median_hsv': {'h': med[0], 's': med[1], 'v': med[2]},
            'source': 'gt_cell',
        })
    return out


def merge_snapshot_centroids(
    profile: CardColorProfile,
    samples_by_color: Dict[str, List[Tuple[float, float, float]]],
) -> CardColorProfile:
    """Dopisz centroidy z migawek (np. ciemna żółć / turkus na zielonym polu testowym)."""
    cents = {int(k): list(v) for k, v in profile.centroids_by_cls.items()}
    for name, pts in samples_by_color.items():
        cls_id = int(pc.COLOR_TO_CLASS.get(name.upper(), -1))
        if cls_id < 0 or not pts:
            continue
        existing = cents.get(cls_id, [])
        for h, s, v in pts:
            if any(hsv_centroid_distance(h, s, v, c) < 0.35 for c in existing):
                continue
            existing.append((float(h), float(s), float(v)))
        cents[cls_id] = existing
    profile.centroids_by_cls = prune_centroids_by_hue(cents)
    return profile


def frame_index_from_id(frame_id: str) -> int:
    stem = str(frame_id).replace('vid_', '').split('_')[0]
    try:
        return int(stem)
    except ValueError:
        return -1


def collect_gt_samples_from_session(
    session_dir: str | Path,
    video_path: str | Path,
    cfg: Dict[str, Any],
    *,
    max_reproj_b: float = 28.0,
    min_frame: int = 250,
    rotate_deg: int = 0,
) -> Tuple[Dict[str, List[Tuple[float, float, float]]], List[Dict[str, Any]]]:
    """Zbierz mediany HSV z komórek GT na migawkach (dobre klatki)."""
    import os

    os.environ.setdefault(
        'DRONIADA_YOLO_POSE_WEIGHTS',
        str(_ROOT / 'runs/pose/droniada_real_finetune/weights/best.pt'),
    )
    from release.yolo_pose_live import detect_corners_yolo_pose
    from module_panel.warp import warp_panel_rect

    session_dir = Path(session_dir)
    snap_dir = session_dir / 'snapshots'
    cells = gt_cells_from_config(cfg)
    by_color: Dict[str, List[Tuple[float, float, float]]] = {c['color']: [] for c in cells}
    reports: List[Dict[str, Any]] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Nie otwarto wideo: {video_path}')

    for json_path in sorted(snap_dir.glob('*_snapshot.json')):
        data = json.loads(json_path.read_text(encoding='utf-8'))
        fid = str(data.get('frame_id', ''))
        fi = frame_index_from_id(fid)
        reproj_b = float(data.get('reproj_b', 999.0))
        pan = data.get('pan_meta') or {}
        glx = pan.get('grid_lines_x')
        gly = pan.get('grid_lines_y')
        if not glx or not gly or len(glx) < 11 or len(gly) < 11:
            reports.append({'frame_id': fid, 'skip': 'no_grid_lines'})
            continue
        if reproj_b > max_reproj_b or fi < min_frame:
            reports.append({'frame_id': fid, 'skip': f'reproj={reproj_b:.1f} fi={fi}'})
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fi))
        ok, frame = cap.read()
        if not ok:
            reports.append({'frame_id': fid, 'skip': 'no_frame'})
            continue
        if int(rotate_deg) == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        corners, _, _ = detect_corners_yolo_pose(frame)
        if corners is None:
            reports.append({'frame_id': fid, 'skip': 'no_corners'})
            continue
        warped, _ = warp_panel_rect(frame, corners)

        row_report: Dict[str, Any] = {'frame_id': fid, 'reproj_b': reproj_b, 'cells': {}}
        for spec in cells:
            color = str(spec['color']).upper()
            row, col = int(spec['row']), int(spec['col'])
            core = extract_cell_core_bgr(warped, glx, gly, row, col)
            meta = classify_cell_relaxed(core, row=row, col=col, expected_color=color)
            hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV) if core.size else None
            med = _patch_median_hsv(hsv) if hsv is not None and hsv.size else None
            if med and med[1] >= _SNAP_MIN_SAT and med[2] >= _SNAP_MIN_VAL:
                by_color[color].append(med)
            got = pc.CLASS_TO_COLOR.get(meta[0]) if meta else None
            row_report['cells'][spec.get('user_label', f'{row},{col}')] = {
                'expected': color,
                'got': got,
                'median_hsv': med,
                'ok': got == color,
            }
        reports.append(row_report)

    cap.release()
    return by_color, reports


def collect_samples_from_video(
    video_path: str | Path,
    cfg: Dict[str, Any],
    *,
    frame_step: int = 15,
    max_reproj_px: float = 35.0,
    min_frame: int = 0,
    max_frames: int = 0,
    rotate_deg: int = 0,
    min_samples_per_color: int = 3,
) -> Tuple[Dict[str, List[Tuple[float, float, float]]], List[Dict[str, Any]]]:
    """
    Przeskanuj nagranie panelu — mediany HSV ze środka znanych komórek (layout JSON).

    Nie wymaga migawek: wystarczy wideo + układ kart w ``cells``.
    """
    os.environ.setdefault(
        'DRONIADA_YOLO_POSE_WEIGHTS',
        str(_ROOT / 'runs/pose/droniada_real_finetune/weights/best.pt'),
    )
    from release.yolo_pose_live import detect_corners_yolo_pose
    from module_panel.warp import warp_panel_rect

    cells = gt_cells_from_config(cfg)
    by_color: Dict[str, List[Tuple[float, float, float]]] = {
        str(c['color']).upper(): [] for c in cells
    }
    reports: List[Dict[str, Any]] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Nie otwarto wideo: {video_path}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames > 0:
        total = min(total, int(max_frames))

    step = max(1, int(frame_step))
    for fi in range(int(min_frame), max(total, 0), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        if int(rotate_deg) == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        corners, _, meta = detect_corners_yolo_pose(frame)
        if corners is None:
            reports.append({'frame': fi, 'skip': 'no_corners'})
            continue
        reproj = float(meta.get('reproj_mean_px', 999.0))
        if reproj > float(max_reproj_px):
            reports.append({'frame': fi, 'skip': f'reproj={reproj:.1f}'})
            continue
        warped, _ = warp_panel_rect(frame, corners)
        glx, gly, grid_src = grid_lines_for_warped(warped)

        row_report: Dict[str, Any] = {
            'frame': fi,
            'reproj_px': reproj,
            'grid': grid_src,
            'cells': {},
        }
        n_added = 0
        for spec in cells:
            color = str(spec['color']).upper()
            row, col = int(spec['row']), int(spec['col'])
            med = sample_cell_median_hsv(warped, glx, gly, row, col)
            label = str(spec.get('user_label') or spec.get('label') or f'{row},{col}')
            cell_ok = med is not None and med[1] >= _SNAP_MIN_SAT and med[2] >= _SNAP_MIN_VAL
            row_report['cells'][label] = {
                'color': color,
                'row': row,
                'col': col,
                'median_hsv': {'h': med[0], 's': med[1], 'v': med[2]} if med else None,
                'ok': cell_ok,
            }
            if cell_ok and med is not None:
                by_color[color].append(med)
                n_added += 1
        if n_added > 0:
            reports.append(row_report)
        else:
            reports.append({'frame': fi, 'skip': 'no_valid_cells', 'reproj_px': reproj})

    cap.release()
    return by_color, reports


def build_centroids_from_samples(
    samples_by_color: Dict[str, List[Tuple[float, float, float]]],
) -> Dict[str, Tuple[float, float, float]]:
    """Mediana HSV per kolor z wielu klatek nagrania."""
    out: Dict[str, Tuple[float, float, float]] = {}
    for name, pts in samples_by_color.items():
        med = aggregate_hsv_samples(pts)
        if med is not None:
            out[str(name).upper()] = med
    return out
