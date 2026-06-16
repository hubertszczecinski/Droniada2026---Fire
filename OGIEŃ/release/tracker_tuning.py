"""Runtime tuning żółtej ramki (tracker + interval YOLO)."""
from __future__ import annotations

from typing import Callable, Dict, Optional

from release.live_corners import _TRACKER

_runtime_interval_ms = 350
LogFn = Optional[Callable[[object, str], None]]


def snapshot() -> Dict[str, float]:
    return {
        'smooth_alpha': float(_TRACKER.alpha),
        'hold_frames': float(_TRACKER.hold_frames),
        'tracker_good_reproj': float(_TRACKER.good_reproj),
        'interval_ms': float(_runtime_interval_ms),
    }


def get_interval_ms() -> int:
    return int(_runtime_interval_ms)


def init_from_args(
    *,
    smooth_alpha: float,
    hold_frames: int,
    tracker_good_reproj: float,
    interval_ms: int,
) -> Dict[str, float]:
    global _runtime_interval_ms
    _runtime_interval_ms = int(interval_ms)
    return apply(
        {
            'smooth_alpha': smooth_alpha,
            'hold_frames': hold_frames,
            'tracker_good_reproj': tracker_good_reproj,
            'interval_ms': interval_ms,
        },
    )


def apply(settings: Dict[str, object], *, log_fn: LogFn = None) -> Dict[str, float]:
    global _runtime_interval_ms
    changed: list[str] = []
    if 'smooth_alpha' in settings and settings['smooth_alpha'] is not None:
        v = float(settings['smooth_alpha'])
        v = max(0.05, min(1.0, v))
        if abs(v - _TRACKER.alpha) > 1e-6:
            _TRACKER.alpha = v
            changed.append(f'alpha={v:.2f}')
    if 'hold_frames' in settings and settings['hold_frames'] is not None:
        v = max(0, int(settings['hold_frames']))
        if v != _TRACKER.hold_frames:
            _TRACKER.hold_frames = v
            changed.append(f'hold={v}')
    if 'tracker_good_reproj' in settings and settings['tracker_good_reproj'] is not None:
        v = max(8.0, min(80.0, float(settings['tracker_good_reproj'])))
        if abs(v - _TRACKER.good_reproj) > 1e-6:
            _TRACKER.good_reproj = v
            changed.append(f'good_reproj={v:.0f}px')
    if 'interval_ms' in settings and settings['interval_ms'] is not None:
        v = max(80, min(2000, int(settings['interval_ms'])))
        if v != _runtime_interval_ms:
            _runtime_interval_ms = v
            changed.append(f'interval={v}ms')
    if changed and log_fn is not None:
        log_fn(f'[live_panel] tracker WWW: {", ".join(changed)}')
    return snapshot()


def validate_settings(raw: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if 'smooth_alpha' in raw:
        out['smooth_alpha'] = max(0.05, min(1.0, float(raw['smooth_alpha'])))
    if 'hold_frames' in raw:
        out['hold_frames'] = float(max(0, min(60, int(raw['hold_frames']))))
    if 'tracker_good_reproj' in raw:
        out['tracker_good_reproj'] = max(8.0, min(80.0, float(raw['tracker_good_reproj'])))
    if 'interval_ms' in raw:
        out['interval_ms'] = float(max(80, min(2000, int(raw['interval_ms']))))
    return out
