"""Integracja: suwaki WWW → pliki sesji → poll → logika lotu/migawek."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from release.flight_mission import FlightMissionController
from release.mission_settings import merge_mission_settings
from release.tracker_tuning import apply, snapshot as tracker_snapshot, validate_settings
from release.web_dashboard import LiveWebPublisher
from release.web_runtime_settings import merge_runtime_settings, runtime_snapshot


class TestWebSettingsWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.pub = LiveWebPublisher(self.tmp)

    def test_mission_roundtrip_file_is_source_of_truth(self) -> None:
        self.pub.write_mission_settings({
            'dist_min_m': 6.0,
            'dist_max_m': 18.0,
            'dist_hold_tier1_m': 12.0,
            'dist_hold_tier2_m': 10.0,
            'dist_hold_tier3_m': 8.0,
            'hold_stabilize_s': 20.0,
            'creep_speed': 0.85,
            'creep_duration_s': 3.0,
            'cruise_speed': 0.42,
            'report_send_pause_s': 8.0,
            'snapshots_per_panel': 6.0,
        })
        patch = self.pub.poll_mission_update()
        self.assertIsNotNone(patch)
        fc = FlightMissionController(settings=merge_mission_settings({}, patch or {}))
        self.assertAlmostEqual(fc.settings['cruise_speed'], 0.42)
        self.assertAlmostEqual(fc.settings['report_send_pause_s'], 8.0)
        self.assertAlmostEqual(fc.settings['creep_duration_s'], 3.0)

        payload = self.pub._build_payload({})
        self.assertAlmostEqual(payload['mission_settings']['cruise_speed'], 0.42)

    def test_runtime_poll_updates_gate_fields(self) -> None:
        self.pub.write_runtime_settings({
            'snapshot_min_grid_overlap': 0.82,
            'snapshot_min_stable': 4,
            'snapshot_max_reproj': 12.0,
        })
        upd = self.pub.poll_runtime_update()
        self.assertIsNotNone(upd)
        gate = runtime_snapshot({})
        gate.update(upd or {})
        self.assertAlmostEqual(gate['snapshot_min_grid_overlap'], 0.82)
        self.assertEqual(gate['snapshot_min_stable'], 4)
        self.assertAlmostEqual(gate['snapshot_max_reproj'], 12.0)

    def test_tracker_write_merges_partial_post(self) -> None:
        self.pub.write_tracker_settings({
            'smooth_alpha': 0.4,
            'hold_frames': 15,
            'interval_ms': 280,
            'tracker_good_reproj': 25.0,
        })
        self.pub.write_tracker_settings({'smooth_alpha': 0.55})
        data = self.pub.read_tracker_settings()
        self.assertAlmostEqual(data['smooth_alpha'], 0.55)
        self.assertEqual(int(data['hold_frames']), 15)
        self.assertEqual(int(data['interval_ms']), 280)

    def test_tracker_poll_applies_to_runtime(self) -> None:
        self.pub.write_tracker_settings(validate_settings({
            'smooth_alpha': 0.33,
            'hold_frames': 9,
            'interval_ms': 400,
            'tracker_good_reproj': 30.0,
        }))
        patch = self.pub.poll_tracker_update()
        self.assertIsNotNone(patch)
        apply(patch or {})
        snap = tracker_snapshot()
        self.assertAlmostEqual(snap['smooth_alpha'], 0.33)
        self.assertEqual(int(snap['hold_frames']), 9)
        self.assertEqual(int(snap['interval_ms']), 400)

    def test_build_payload_runtime_from_file_not_ram(self) -> None:
        self.pub.write_runtime_settings({'snapshot_max_reproj': 11.0})
        payload = self.pub._build_payload({
            'params': {'module_b': {'reproj_mean_px': 99.0}},
        })
        self.assertAlmostEqual(payload['runtime_settings']['snapshot_max_reproj'], 11.0)


if __name__ == '__main__':
    unittest.main()
