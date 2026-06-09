"""Testy automatyki lotu — speed WebSocket."""
from release.flight_mission import FlightMissionController, FlightPhase


def test_hold_at_first_tier():
    fc = FlightMissionController()
    fc.update(distance_m=13.0, panel_full=False, snap_count=0, snap_max=5, mission_done=False)
    assert fc.phase == FlightPhase.APPROACH
    assert fc.speed < 1.0
    fc.update(distance_m=11.0, panel_full=False, snap_count=0, snap_max=5, mission_done=False)
    assert fc.phase == FlightPhase.HOLD
    assert fc.speed == 1.0
    assert fc._hold_tier_idx == 0
    assert fc.vision_active is True


def test_depart_after_report():
    fc = FlightMissionController()
    fc.on_report_sent()
    assert fc.phase == FlightPhase.DEPART
    assert fc.vision_active is False
    assert fc.speed < 1.0


def test_depart_without_distance_keeps_cruise():
    fc = FlightMissionController()
    fc.on_report_sent()
    fc.update(distance_m=None, panel_full=False, snap_count=0, snap_max=5, mission_done=False)
    assert fc.phase == FlightPhase.DEPART
    assert fc.speed == fc.settings['cruise_speed']


def test_approach_without_distance_keeps_cruise():
    fc = FlightMissionController()
    fc.update(distance_m=None, panel_full=False, snap_count=0, snap_max=5, mission_done=False)
    assert fc.phase == FlightPhase.APPROACH
    assert fc.speed == fc.settings['cruise_speed']


def test_hold_without_distance_stops():
    fc = FlightMissionController()
    fc.phase = FlightPhase.HOLD
    fc.update(distance_m=None, panel_full=False, snap_count=0, snap_max=5, mission_done=False)
    assert fc.speed == 1.0


def test_creep_to_next_tier():
    fc = FlightMissionController(settings={'hold_stabilize_s': 0.0, 'creep_duration_s': 0.5})
    fc.phase = FlightPhase.HOLD
    fc._phase_entered = 0.0
    fc._hold_tier_idx = 0
    fc.update(distance_m=11.0, panel_full=False, snap_count=0, snap_max=5, mission_done=False)
    assert fc.phase == FlightPhase.CREEP
    assert fc._creep_inter_tier is True
    assert fc._hold_tier_idx == 1
    fc.update(distance_m=9.0, panel_full=False, snap_count=0, snap_max=5, mission_done=False)
    assert fc.phase == FlightPhase.HOLD
    assert fc._hold_tier_idx == 1


def test_depart_completes_before_mission_done():
    fc = FlightMissionController(settings={'report_send_pause_s': 1.0})
    fc.on_report_sent()
    fc._depart_until = 0.0
    fc.update(distance_m=None, panel_full=False, snap_count=0, snap_max=5, mission_done=True)
    assert fc.phase == FlightPhase.DONE
    assert fc.vision_active is False


def test_depart_then_approach_next_panel():
    fc = FlightMissionController(settings={'report_send_pause_s': 1.0})
    fc.on_report_sent()
    fc._depart_until = 0.0
    fc.update(distance_m=None, panel_full=False, snap_count=0, snap_max=5, mission_done=False)
    assert fc.phase == FlightPhase.APPROACH
    assert fc.vision_active is True


def test_last_tier_without_snaps_starts_depart():
    fc = FlightMissionController(settings={'hold_stabilize_s': 0.0, 'report_send_pause_s': 4.0})
    fc.phase = FlightPhase.HOLD
    fc._phase_entered = 0.0
    fc._hold_tier_idx = 2
    fc.update(distance_m=7.0, panel_full=False, snap_count=1, snap_max=5, mission_done=False)
    assert fc.phase == FlightPhase.DEPART
    assert fc.vision_active is False
