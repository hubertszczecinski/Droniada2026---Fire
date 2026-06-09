"""
Automatyka lotu względem panelu — jeden sygnał WebSocket ``speed`` ∈ [0, 1].

1 = hover (stój), 0 = pełna prędkość do przodu.

Fazy:
  depart   — po Wyślij: lot do następnego panelu, wizja wyłączona
  approach — zbliżanie; cele zatrzymania tier1→tier2→tier3 (np. 11→9→7 m)
  hold     — stój (speed=1), stabilizacja, migawki
  creep    — podjazd do kolejnego tieru lub krótki ruch przy ostatnim tierze
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from release.mission_settings import DEFAULT_MISSION_SETTINGS, snapshot as mission_snapshot


class FlightPhase(str, enum.Enum):
    DEPART = 'depart'
    APPROACH = 'approach'
    HOLD = 'hold'
    CREEP = 'creep'
    DONE = 'done'


@dataclass
class FlightMissionController:
    settings: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MISSION_SETTINGS))
    phase: FlightPhase = FlightPhase.APPROACH
    speed: float = 1.0
    vision_active: bool = True
    _phase_entered: float = 0.0
    _depart_until: float = 0.0
    _distance_m: Optional[float] = None
    _hold_tier_idx: int = 0
    _creep_inter_tier: bool = False
    _time_fn: Any = field(default=time.monotonic, repr=False)

    def __post_init__(self) -> None:
        from release.mission_settings import merge_mission_settings
        self.settings = merge_mission_settings({}, self.settings)
        if self._phase_entered <= 0.0:
            self._phase_entered = self._now()

    def _now(self) -> float:
        return float(self._time_fn())

    def depart_remaining_s(self) -> float:
        if self.phase != FlightPhase.DEPART:
            return 0.0
        return max(0.0, float(self._depart_until) - self._now())

    def apply_settings(self, patch: Dict[str, float]) -> None:
        self.settings = mission_snapshot({**self.settings, **patch})

    def _begin_depart(self) -> None:
        pause = float(self.settings.get('report_send_pause_s', 5.0))
        self._enter(FlightPhase.DEPART)
        self._depart_until = self._now() + max(0.0, pause)
        self.speed = float(self.settings.get('cruise_speed', 0.35))
        self.vision_active = False

    def on_report_sent(self) -> None:
        self._begin_depart()

    def on_snapshots_exhausted(self) -> None:
        """Po 3 postojach bez pełnej galerii — lot dalej jak po Wyślij."""
        self._begin_depart()

    def on_mission_done(self) -> None:
        self._enter(FlightPhase.DONE)
        self.speed = 1.0
        self.vision_active = False

    def reset_for_panel(self) -> None:
        if self.phase == FlightPhase.DONE:
            return
        self._hold_tier_idx = 0
        self._creep_inter_tier = False
        self._enter(FlightPhase.APPROACH)
        self.speed = float(self.settings.get('cruise_speed', 0.35))
        self.vision_active = True

    def update(
        self,
        *,
        distance_m: Optional[float],
        panel_full: bool,
        snap_count: int,
        snap_max: int,
        mission_done: bool,
        panel_reliable: bool = True,
        external_pause_remaining: float = 0.0,
    ) -> float:
        self._distance_m = distance_m
        now = self._now()

        if external_pause_remaining > 0.0 and self.phase != FlightPhase.DEPART:
            self._enter(FlightPhase.DEPART)
            self._depart_until = now + external_pause_remaining

        if self.phase == FlightPhase.DEPART:
            if now < self._depart_until:
                self.speed = float(self.settings.get('cruise_speed', 0.35))
                self.vision_active = False
            elif mission_done:
                self.on_mission_done()
            else:
                self.reset_for_panel()
            return self.speed

        if mission_done:
            self.on_mission_done()
            return self.speed

        dist_min = float(self.settings['dist_min_m'])
        dist_max = float(self.settings['dist_max_m'])
        hold_s = float(self.settings['hold_stabilize_s'])
        creep_speed = float(self.settings['creep_speed'])
        creep_s = float(self.settings['creep_duration_s'])
        cruise = float(self.settings['cruise_speed'])
        hold_targets = self._hold_targets()
        dist_target = hold_targets[min(self._hold_tier_idx, len(hold_targets) - 1)]

        d = distance_m
        if d is None or not (0.5 < float(d) < 80.0):
            if self.phase in (FlightPhase.APPROACH, FlightPhase.DEPART):
                # Stała trasa między panelami — jedź dalej, nie stój bez PnP.
                self.speed = cruise
                self.vision_active = self.phase != FlightPhase.DEPART
            else:
                # Przy panelu (hold/creep) — ostrożny postój przy chwilowej utracie odległości.
                self.speed = 1.0
                self.vision_active = True
            return self.speed

        in_band = dist_min <= float(d) <= dist_max
        at_hold = in_band and float(d) <= dist_target + 0.8

        if self.phase == FlightPhase.APPROACH:
            self.vision_active = True
            if not panel_reliable:
                self.speed = cruise
                return self.speed
            if at_hold:
                self._enter(FlightPhase.HOLD)
                self.speed = 1.0
            elif float(d) > dist_max:
                self.speed = cruise
            elif float(d) < dist_min:
                self.speed = 1.0
            else:
                gap = max(0.0, float(d) - dist_target)
                span = max(0.5, dist_max - dist_target)
                self.speed = min(0.98, cruise + (gap / span) * (1.0 - cruise))
            return self.speed

        if self.phase == FlightPhase.HOLD:
            self.vision_active = True
            self.speed = 1.0
            elapsed = now - self._phase_entered
            need_snaps = snap_count < snap_max
            if need_snaps and elapsed >= hold_s and not panel_full:
                if self._hold_tier_idx < len(hold_targets) - 1:
                    self._hold_tier_idx += 1
                    self._creep_inter_tier = True
                    self._enter(FlightPhase.CREEP)
                    self.speed = creep_speed
                else:
                    self.on_snapshots_exhausted()
            return self.speed

        if self.phase == FlightPhase.CREEP:
            self.vision_active = True
            self.speed = creep_speed
            tier_target = hold_targets[min(self._hold_tier_idx, len(hold_targets) - 1)]
            creep_done = (
                float(d) <= tier_target + 0.8
                or (now - self._phase_entered) >= creep_s
            )
            if creep_done:
                self._creep_inter_tier = False
                self._enter(FlightPhase.HOLD)
                self.speed = 1.0
            return self.speed

        self.speed = 1.0
        return self.speed

    def status_dict(self) -> Dict[str, Any]:
        targets = self._hold_targets()
        tier_m = targets[min(self._hold_tier_idx, len(targets) - 1)] if targets else None
        return {
            'phase': self.phase.value,
            'speed': round(float(self.speed), 3),
            'vision_active': bool(self.vision_active),
            'distance_m': self._distance_m,
            'hold_tier_idx': self._hold_tier_idx,
            'hold_tier_m': tier_m,
            'hold_targets_m': targets,
            'settings': dict(self.settings),
        }

    def _hold_targets(self) -> List[float]:
        s = self.settings
        return sorted(
            [
                float(s.get('dist_hold_tier1_m', 11.0)),
                float(s.get('dist_hold_tier2_m', 9.0)),
                float(s.get('dist_hold_tier3_m', 7.0)),
            ],
            reverse=True,
        )

    def _enter(self, phase: FlightPhase) -> None:
        if self.phase != phase:
            self.phase = phase
            self._phase_entered = self._now()
