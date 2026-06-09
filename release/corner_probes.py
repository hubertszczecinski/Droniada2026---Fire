from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pipeline_competition as pc
from module_pose.api import canonicalize_corners_by_white_anchor, detect_corners_black_panel, detect_corners_panel
from module_pose.pnp_panel import solve_panel_pose
from module_panel.analyze import detect_panel_corners_for_module_b

ProbeFn = Callable[[np.ndarray], Optional[np.ndarray]]

@dataclass
class ProbeResult:
    name: str
    found: bool
    reproj_px: float = float('inf')
    pnp_ok: bool = False
    anchor: str = ''
    white_t: str = ''
    area_ratio: float = 0.0
    aspect: float = 0.0
    err: str = ''

@dataclass
class ProbeStats:
    frames: int = 0
    found: int = 0
    pnp_ok: int = 0
    reproj_sum: float = 0.0
    reproj_best_sum: float = 0.0
    wins: int = 0

    def record(self, r: ProbeResult, is_winner: bool) -> None:
        self.frames += 1
        if r.found:
            self.found += 1
        if r.pnp_ok:
            self.pnp_ok += 1
            self.reproj_sum += r.reproj_px
        if is_winner and r.pnp_ok:
            self.wins += 1
            self.reproj_best_sum += r.reproj_px

def _quad_from_contours(mask: np.ndarray, min_area_frac: float) -> Optional[np.ndarray]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    h, w = mask.shape[:2]
    area_img = float(h * w)
    best_quad = None
    best_area = 0.0
    best_rect = None
    best_rect_area = 0.0
    for c in cnts:
        a = float(cv2.contourArea(c))
        if a < min_area_frac * area_img:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and a > best_area:
            best_area = a
            best_quad = approx.reshape(-1, 2).astype(np.float32)
        rect = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
        if a > best_rect_area:
            best_rect_area = a
            best_rect = rect
    if best_quad is not None:
        return pc.order_points(best_quad)
    if best_rect is not None:
        return pc.order_points(best_rect)
    return None

def _mask_thresh(bgr: np.ndarray, thresh: int, inv: bool = True) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    flag = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blur, thresh, 255, flag)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return _quad_from_contours(mask, 0.08)

def _mask_otsu(bgr: np.ndarray, inv: bool = True) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    flag = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blur, 0, 255, flag + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return _quad_from_contours(mask, 0.06)

def _mask_adaptive(bgr: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return _quad_from_contours(mask, 0.06)

def _canny_probe(bgr: np.ndarray, lo: int, hi: int, min_frac: float) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(gray, lo, hi)
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return _quad_from_contours(edge, min_frac)

def _largest_minarearect(bgr: np.ndarray, min_frac: float) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 85, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    area_img = float(bgr.shape[0] * bgr.shape[1])
    best = None
    best_a = 0.0
    for c in cnts:
        a = float(cv2.contourArea(c))
        if a < min_frac * area_img or a <= best_a:
            continue
        best_a = a
        best = cv2.boxPoints(cv2.minAreaRect(c))
    if best is None:
        return None
    return pc.order_points(best.astype(np.float32))

def _panel_chain(bgr: np.ndarray) -> Optional[np.ndarray]:
    c, _src = detect_panel_corners_for_module_b(bgr, [])
    return c

def build_probes(mode: str = 'live') -> List[Tuple[str, ProbeFn]]:
    if mode == 'full':
        return [
            ('img_black_panel', detect_corners_black_panel),
            ('img_panel', detect_corners_panel),
            ('img_mask85_inv', lambda im: _mask_thresh(im, 85, True)),
            ('img_mask60_inv', lambda im: _mask_thresh(im, 60, True)),
            ('img_otsu_inv', _mask_otsu),
            ('img_adaptive_inv', _mask_adaptive),
            ('canny_40_130', lambda im: _canny_probe(im, 40, 130, 0.03)),
            ('pc_detect_corners_img', pc.detect_corners_img),
            ('panel_corner_chain', _panel_chain),
        ]
    return [('img_panel', detect_corners_panel)]

def score_corners(bgr: np.ndarray, corners: Optional[np.ndarray], k: np.ndarray, dist: np.ndarray) -> Tuple[float, bool, str, str, float, float]:
    if corners is None:
        return (float('inf'), False, '', '', 0.0, 0.0)
    h, w = bgr.shape[:2]
    c_can, anc = canonicalize_corners_by_white_anchor(bgr, corners)
    dst = np.array([[0, 0], [pc.RECT_W - 1, 0], [pc.RECT_W - 1, pc.RECT_H - 1], [0, pc.RECT_H - 1]], dtype=np.float32)
    h_mat = cv2.getPerspectiveTransform(c_can.astype(np.float32), dst)
    warped = cv2.warpPerspective(bgr, h_mat, (pc.RECT_W, pc.RECT_H))
    white_t = pc.detect_white_corner_transform(warped)
    ok_pnp, _rv, _tv, reproj = solve_panel_pose(c_can, k, dist, refine_lm=False)
    area_ratio = float(cv2.contourArea(c_can.astype(np.float32))) / float(max(1, h * w))
    side1 = float(np.linalg.norm(c_can[0] - c_can[1]))
    side2 = float(np.linalg.norm(c_can[1] - c_can[2]))
    asp = max(side1, side2) / max(1e-06, min(side1, side2))
    return (float(reproj), bool(ok_pnp), anc, white_t, area_ratio, asp)

def run_all_probes(bgr: np.ndarray, k: np.ndarray, dist: np.ndarray, probes: List[Tuple[str, ProbeFn]]) -> List[ProbeResult]:
    out: List[ProbeResult] = []
    for name, fn in probes:
        try:
            corners = fn(bgr)
            reproj, pnp_ok, anc, wt, area, asp = score_corners(bgr, corners, k, dist)
            out.append(ProbeResult(name=name, found=corners is not None, reproj_px=reproj, pnp_ok=pnp_ok, anchor=anc, white_t=wt, area_ratio=area, aspect=asp))
        except Exception as e:
            out.append(ProbeResult(name=name, found=False, err=str(e)))
    return out

def _quality_score(r: ProbeResult) -> float:
    if not r.found:
        return float('inf')
    score = r.reproj_px
    if r.area_ratio > 0.48:
        score += 400.0
    elif r.area_ratio < 0.05:
        score += 150.0
    elif 0.07 <= r.area_ratio <= 0.42:
        score -= 25.0
    if r.aspect > 3.5 or r.aspect < 1.2:
        score += 80.0
    if not r.pnp_ok:
        score += 1000.0
    return score

def pick_winner(results: List[ProbeResult]) -> Optional[str]:
    found = [r for r in results if r.found]
    if not found:
        return None
    return min(found, key=_quality_score).name

def format_probe_line(r: ProbeResult) -> str:
    if not r.found:
        return f'  {r.name:22s} MISS {r.err}'
    plausible = r.found and r.area_ratio < 0.55 and 1.2 <= r.aspect <= 3.5
    tag = 'OK' if r.pnp_ok and r.reproj_px < 12.0 and plausible else 'WEAK' if r.found else 'MISS'
    return f'  {r.name:22s} {tag:4s} reproj={r.reproj_px:6.2f} anchor={r.anchor:2s} white={r.white_t:4s} area={r.area_ratio:.3f} asp={r.aspect:.2f}'

def summary_table(stats: Dict[str, ProbeStats]) -> str:
    lines = ['', '=== PODSUMOWANIE PROB ===']
    rows = []
    for name, s in stats.items():
        if s.frames == 0:
            continue
        mean_r = s.reproj_sum / max(1, s.pnp_ok)
        rows.append((s.wins, s.pnp_ok, mean_r, name, s.found, s.frames))
    rows.sort(reverse=True)
    for wins, pnp_ok, mean_r, name, found, frames in rows:
        lines.append(f'  {name:22s} wins={wins:4d} pnp_ok={pnp_ok:4d}/{frames} found={found:4d} mean_reproj={mean_r:.2f}')
    return '\n'.join(lines)
