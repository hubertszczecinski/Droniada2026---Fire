"""When card X,Y from the grid is trustworthy enough to report."""
from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

DEFAULT_MAX_REPROJ_PX = 8.0
DEFAULT_MIN_HOMOGRAPHY_INLIERS = 12
# Blender (legacy 360° orbit): step 0 = frontal; 11 = drugi łagodny kąt.
DEFAULT_ALLOWED_ORBIT_STEPS: Set[int] = {0, 11}
DEFAULT_MAX_ORBIT_AZIMUTH_OFFSET_DEG = 60.0
# Sentinel: --orbit-steps frontal_bank / frontal na nowym datasecie (|azimuth|≤60°).
FRONTAL_ORBIT_BANK: Set[int] = set()


def orbit_azimuth_offset_deg_from_pose(pose_json: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(pose_json, dict):
        return None
    cam = pose_json.get('camera') or {}
    off = cam.get('orbit_azimuth_offset_deg')
    if off is None:
        return None
    try:
        return float(off)
    except (TypeError, ValueError):
        return None


def pose_is_frontal_orbit_bank(
    pose_json: Optional[Dict[str, Any]],
    max_abs_azimuth_deg: float = DEFAULT_MAX_ORBIT_AZIMUTH_OFFSET_DEG,
) -> bool:
    """Nowy dataset Blender: offset względem frontu; stary: step 0 / 11."""
    off = orbit_azimuth_offset_deg_from_pose(pose_json)
    if off is not None:
        return abs(off) <= float(max_abs_azimuth_deg) + 1e-06
    step = orbit_step_from_pose(pose_json)
    if step is None:
        return True
    return int(step) in DEFAULT_ALLOWED_ORBIT_STEPS


def orbit_step_from_pose(pose_json: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(pose_json, dict):
        return None
    cam = pose_json.get('camera') or {}
    step = cam.get('orbit_step_index')
    if step is None:
        return None
    try:
        return int(step)
    except (TypeError, ValueError):
        return None


def probe_grid_homography_quality(
    image_bgr: np.ndarray,
    corners_px: np.ndarray,
    *,
    min_inliers: int = DEFAULT_MIN_HOMOGRAPHY_INLIERS,
) -> Dict[str, Any]:
    """Run grid-line RANSAC homography probe (does not change predictions)."""
    from module_geom.grid_homography import build_grid_homography

    _h, ok, meta = build_grid_homography(image_bgr, corners_px, min_inliers=min_inliers)
    out = {
        'grid_homography_ok': bool(ok),
        'homography_inliers': int(meta.get('homography_inliers', 0)),
        'homography_total_pts': int(meta.get('homography_total_pts', 0)),
        'grid_line_ok': bool(meta.get('grid_line_ok', False)),
    }
    if meta.get('fail'):
        out['grid_homography_fail'] = str(meta['fail'])
    return out


def assess_grid_xy_reliable(
    *,
    pnp_ok: bool,
    reproj_mean_px: float,
    meta: Dict[str, Any],
    pose_json: Optional[Dict[str, Any]] = None,
    max_reproj_px: float = DEFAULT_MAX_REPROJ_PX,
    min_homography_inliers: int = DEFAULT_MIN_HOMOGRAPHY_INLIERS,
    allowed_orbit_steps: Optional[Set[int]] = DEFAULT_ALLOWED_ORBIT_STEPS,
    require_homography_inliers: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Stricter than reproj-only:
    - PnP OK and reproj <= max_reproj_px
    - grid homography RANSAC inliers >= min (or geom path already validated)
    - optional orbit_step in allowed_orbit_steps (None = skip orbit gate)
    """
    reasons: Dict[str, bool] = {}
    orbit_step = orbit_step_from_pose(pose_json)
    inliers = int(meta.get('homography_inliers', 0))
    grid_line_ok = bool(meta.get('grid_line_ok', False))
    geom_ok = bool(meta.get('grid_homography_ok', False)) or str(meta.get('xy_source', '')).startswith('geom')

    reasons['pnp_ok'] = bool(pnp_ok)
    reproj_limit = float(max_reproj_px)
    if str(meta.get('xy_mode', '')) == 'line_grid' and int(meta.get('homography_inliers', 0)) >= 50:
        reproj_limit = max(reproj_limit, 10.5)
    reasons['reproj_ok'] = bool(pnp_ok and np.isfinite(reproj_mean_px) and reproj_mean_px <= reproj_limit)

    if require_homography_inliers:
        reasons['grid_inliers_ok'] = inliers >= min_homography_inliers
    else:
        reasons['grid_inliers_ok'] = bool(grid_line_ok or geom_ok or inliers >= min_homography_inliers)

    # Live + YOLO: homografia z rogów (corner_homography) bywa OK przy 0 inlierach RANSAC na ciemnym warpie.
    if not reasons['grid_inliers_ok'] and reasons.get('reproj_ok') and bool(
        meta.get('corner_homography_for_xy'),
    ):
        reasons['grid_inliers_ok'] = True
        reasons['grid_inliers_corner_h'] = True

    if allowed_orbit_steps is None:
        reasons['orbit_ok'] = True
    elif orbit_step is None:
        # Live / brak metadanych — nie blokuj raportu wyłącznie z powodu orbity.
        reasons['orbit_ok'] = True
    else:
        if allowed_orbit_steps == FRONTAL_ORBIT_BANK:
            reasons['orbit_ok'] = pose_is_frontal_orbit_bank(pose_json)
        else:
            reasons['orbit_ok'] = int(orbit_step) in allowed_orbit_steps

    ok = all(reasons.values())
    detail = {
        'grid_xy_reliable': ok,
        'grid_xy_reliable_v2': ok,
        'grid_xy_reliable_reasons': reasons,
        'orbit_step_index': orbit_step,
        'homography_inliers': inliers,
        'reproj_mean_px': float(reproj_mean_px) if np.isfinite(reproj_mean_px) else None,
        'allowed_orbit_steps': sorted(allowed_orbit_steps) if allowed_orbit_steps is not None else None,
    }
    return ok, detail
