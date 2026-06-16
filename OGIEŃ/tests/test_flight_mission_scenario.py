"""
Scenariusz misji drona (opis operatora) — hamowanie, tiery, pauza po Wyślij.

speed: 0 = pełna prędkość (cruise), 1 = stój.
"""
from __future__ import annotations

import math

from release.flight_mission import FlightMissionController, FlightPhase


class _FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _fc(**settings) -> FlightMissionController:
    clock = _FakeClock()
    defaults = {
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
    }
    defaults.update(settings)
    return FlightMissionController(settings=defaults, _time_fn=clock)


def _upd(
    fc: FlightMissionController,
    *,
    d: float | None = None,
    reliable: bool = True,
    snaps: int = 0,
    snap_max: int = 5,
    full: bool = False,
    mission_done: bool = False,
) -> float:
    return fc.update(
        distance_m=d,
        panel_full=full,
        snap_count=snaps,
        snap_max=snap_max,
        mission_done=mission_done,
        panel_reliable=reliable,
    )


def test_cruise_until_reliable_then_brake_gradually():
  fc = _fc(cruise_speed=0.35)
  _upd(fc, d=None, reliable=False)
  assert fc.phase == FlightPhase.APPROACH
  assert fc.speed == 0.35

  _upd(fc, d=14.0, reliable=True)
  assert fc.phase == FlightPhase.APPROACH
  assert 0.35 < fc.speed < 0.98

  _upd(fc, d=11.0, reliable=True)
  assert fc.phase == FlightPhase.HOLD
  assert fc.speed == 1.0


def test_approach_braking_formula_at_mid_range():
  fc = _fc(cruise_speed=0.35, dist_max_m=15.0, dist_hold_tier1_m=11.0)
  _upd(fc, d=13.0, reliable=True)
  gap = 13.0 - 11.0
  span = 15.0 - 11.0
  expected = min(0.98, 0.35 + (gap / span) * (1.0 - 0.35))
  assert abs(fc.speed - expected) < 1e-6


def test_hold_waits_for_snapshots_then_report_depart_pause_exact():
  fc = _fc(report_send_pause_s=5.0, hold_stabilize_s=15.0)
  clock = fc._time_fn
  assert isinstance(clock, _FakeClock)

  _upd(fc, d=11.0, reliable=True)
  assert fc.phase == FlightPhase.HOLD

  clock.advance(16.0)
  _upd(fc, d=11.0, reliable=True, snaps=5, full=True)
  assert fc.phase == FlightPhase.HOLD
  assert fc.speed == 1.0

  fc.on_report_sent()
  assert fc.phase == FlightPhase.DEPART
  assert fc.vision_active is False
  assert abs(fc.depart_remaining_s() - 5.0) < 1e-6

  clock.advance(4.9)
  _upd(fc, d=None, reliable=False)
  assert fc.phase == FlightPhase.DEPART
  assert fc.vision_active is False
  assert fc.depart_remaining_s() > 0.0

  clock.advance(0.2)
  _upd(fc, d=None, reliable=False)
  assert fc.phase == FlightPhase.APPROACH
  assert fc.vision_active is True


def test_tier_creep_when_not_enough_snapshots():
  fc = _fc(hold_stabilize_s=10.0)
  clock = fc._time_fn

  _upd(fc, d=11.0, reliable=True)
  assert fc._hold_tier_idx == 0

  clock.advance(11.0)
  _upd(fc, d=11.0, reliable=True, snaps=2)
  assert fc.phase == FlightPhase.CREEP
  assert fc._hold_tier_idx == 1

  _upd(fc, d=9.0, reliable=True, snaps=2)
  assert fc.phase == FlightPhase.HOLD
  assert fc._hold_tier_idx == 1

  clock.advance(11.0)
  _upd(fc, d=9.0, reliable=True, snaps=3)
  assert fc.phase == FlightPhase.CREEP
  assert fc._hold_tier_idx == 2

  _upd(fc, d=7.0, reliable=True, snaps=3)
  assert fc.phase == FlightPhase.HOLD
  assert fc._hold_tier_idx == 2


def test_after_third_tier_without_snaps_depart_like_send():
  fc = _fc(hold_stabilize_s=5.0, report_send_pause_s=7.0)
  clock = fc._time_fn

  _upd(fc, d=11.0, reliable=True)
  clock.advance(6.0)
  _upd(fc, d=11.0, reliable=True, snaps=1)
  assert fc.phase == FlightPhase.CREEP
  _upd(fc, d=9.0, reliable=True, snaps=1)
  assert fc.phase == FlightPhase.HOLD

  clock.advance(6.0)
  _upd(fc, d=9.0, reliable=True, snaps=1)
  assert fc.phase == FlightPhase.CREEP
  _upd(fc, d=7.0, reliable=True, snaps=1)
  assert fc.phase == FlightPhase.HOLD
  assert fc._hold_tier_idx == 2

  clock.advance(6.0)
  _upd(fc, d=7.0, reliable=True, snaps=2)
  assert fc.phase == FlightPhase.DEPART
  assert fc.vision_active is False
  assert math.isclose(fc.depart_remaining_s(), 7.0, abs_tol=0.01)


def test_after_depart_waits_for_reliable_before_braking():
  fc = _fc(report_send_pause_s=3.0)
  clock = fc._time_fn

  fc.on_report_sent()
  clock.advance(3.1)
  _upd(fc, d=12.0, reliable=False)
  assert fc.phase == FlightPhase.APPROACH
  assert fc.speed == fc.settings['cruise_speed']

  _upd(fc, d=12.0, reliable=True)
  assert fc.speed > fc.settings['cruise_speed']
