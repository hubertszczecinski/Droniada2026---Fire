"""Ustawienia runtime z panelu sterowania WWW (próg migawek)."""
from __future__ import annotations

from typing import Any, Dict

DEFAULT_RUNTIME_SETTINGS: Dict[str, float | int | bool] = {
    'snapshot_min_grid_overlap': 0.45,
    'snapshot_min_grid_overlap_unreliable': 0.45,
    'snapshot_min_stable': 1,
    'snapshot_max_reproj': 25.0,
    'snapshot_min_warp_coverage': 0.15,
}


def validate_runtime_settings(raw: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if 'snapshot_min_grid_overlap' in raw:
        out['snapshot_min_grid_overlap'] = max(0.0, min(1.0, float(raw['snapshot_min_grid_overlap'])))
    if 'snapshot_min_grid_overlap_unreliable' in raw:
        v = float(raw['snapshot_min_grid_overlap_unreliable'])
        out['snapshot_min_grid_overlap_unreliable'] = max(0.0, min(1.0, v))
    if 'snapshot_min_stable' in raw:
        out['snapshot_min_stable'] = max(1, min(10, int(raw['snapshot_min_stable'])))
    if 'snapshot_max_reproj' in raw:
        out['snapshot_max_reproj'] = max(1.0, min(80.0, float(raw['snapshot_max_reproj'])))
    if 'snapshot_min_warp_coverage' in raw:
        out['snapshot_min_warp_coverage'] = max(0.0, min(1.0, float(raw['snapshot_min_warp_coverage'])))
    return out


def merge_runtime_settings(
    base: Dict[str, Any],
    patch: Dict[str, Any],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(DEFAULT_RUNTIME_SETTINGS)
    merged.update(base)
    merged.update(validate_runtime_settings(patch))
    return merged


def runtime_snapshot(settings: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return merge_runtime_settings({}, settings or {})
