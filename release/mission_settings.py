"""Ustawienia misji drona (GUI WWW) — zasięg 5–12 m, pauzy, migawki."""
from __future__ import annotations

from typing import Any, Dict

DEFAULT_MISSION_SETTINGS: Dict[str, float] = {
    'dist_min_m': 5.0,
    'dist_max_m': 15.0,
    'dist_hold_tier1_m': 11.0,
    'dist_hold_tier2_m': 9.0,
    'dist_hold_tier3_m': 7.0,
    'hold_stabilize_s': 15.0,
    'creep_speed': 0.9,
    'creep_duration_s': 2.0,
    'cruise_speed': 0.35,
    'report_send_pause_s': 5.0,
    'snapshots_per_panel': 5.0,
}


def validate_mission_settings(raw: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if 'dist_min_m' in raw:
        out['dist_min_m'] = max(1.0, min(50.0, float(raw['dist_min_m'])))
    if 'dist_max_m' in raw:
        out['dist_max_m'] = max(2.0, min(60.0, float(raw['dist_max_m'])))
    for key in ('dist_hold_tier1_m', 'dist_hold_tier2_m', 'dist_hold_tier3_m'):
        if key in raw:
            out[key] = max(1.0, min(50.0, float(raw[key])))
    # starsze GUI: jeden cel → najbliższy tier
    if 'dist_hold_target_m' in raw and 'dist_hold_tier3_m' not in raw:
        out['dist_hold_tier3_m'] = max(1.0, min(35.0, float(raw['dist_hold_target_m'])))
    if 'hold_stabilize_s' in raw:
        out['hold_stabilize_s'] = max(3.0, min(300.0, float(raw['hold_stabilize_s'])))
    if 'creep_speed' in raw:
        out['creep_speed'] = max(0.5, min(0.99, float(raw['creep_speed'])))
    if 'creep_duration_s' in raw:
        out['creep_duration_s'] = max(0.5, min(60.0, float(raw['creep_duration_s'])))
    if 'cruise_speed' in raw:
        out['cruise_speed'] = max(0.05, min(0.95, float(raw['cruise_speed'])))
    if 'report_send_pause_s' in raw:
        out['report_send_pause_s'] = max(0.0, min(300.0, float(raw['report_send_pause_s'])))
    if 'snapshots_per_panel' in raw:
        out['snapshots_per_panel'] = max(1.0, min(20.0, float(raw['snapshots_per_panel'])))
    if 'dist_min_m' in out and 'dist_max_m' in out and out['dist_min_m'] >= out['dist_max_m']:
        out['dist_max_m'] = out['dist_min_m'] + 1.0
    if 'dist_min_m' in out and 'dist_max_m' in out:
        lo, hi = out['dist_min_m'], out['dist_max_m']
        for key in ('dist_hold_tier1_m', 'dist_hold_tier2_m', 'dist_hold_tier3_m'):
            if key in out:
                out[key] = max(lo, min(hi, out[key]))
        t1 = out.get('dist_hold_tier1_m', DEFAULT_MISSION_SETTINGS['dist_hold_tier1_m'])
        t2 = out.get('dist_hold_tier2_m', DEFAULT_MISSION_SETTINGS['dist_hold_tier2_m'])
        t3 = out.get('dist_hold_tier3_m', DEFAULT_MISSION_SETTINGS['dist_hold_tier3_m'])
        tiers = sorted([t1, t2, t3], reverse=True)
        out['dist_hold_tier1_m'], out['dist_hold_tier2_m'], out['dist_hold_tier3_m'] = tiers
    return out


def merge_mission_settings(base: Dict[str, float], patch: Dict[str, Any]) -> Dict[str, float]:
    merged = dict(DEFAULT_MISSION_SETTINGS)
    merged.update(base)
    for key, val in patch.items():
        if val is not None and key in DEFAULT_MISSION_SETTINGS:
            merged[key] = float(val)
    merged.update(validate_mission_settings(merged))
    return merged


def snapshot(settings: Dict[str, float] | None = None) -> Dict[str, float]:
    out = dict(DEFAULT_MISSION_SETTINGS)
    if settings:
        out.update(settings)
    return merge_mission_settings({}, out)
