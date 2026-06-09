"""Misja wielopanelowa — limit migawek i reset po wysyłce."""
from __future__ import annotations

import os
import tempfile
import unittest

from release.live_dashboard import LiveSnapshotStore
from release.mission_panels import MultiPanelMissionManager


class TestMissionPanels(unittest.TestCase):
    def test_submit_clears_snapshots_and_advances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            m = MultiPanelMissionManager(
                tmp, ['A', 'B'], snapshots_per_panel=2,
                store_kwargs={'max_reproj': 99.0, 'min_stable_frames': 1},
            )
            store = m.store_for('A')
            store.entries.append({
                'frame_id': 't1',
                'reproj_b': 5.0,
                'dashboard_path': os.path.join(store.session_dir, 'snapshots', 't1_dashboard.png'),
                'json_path': os.path.join(store.session_dir, 'snapshots', 't1_snapshot.json'),
                'thumb_path': os.path.join(store.session_dir, 'snapshots', 't1_thumb.jpg'),
            })
            open(store.entries[0]['dashboard_path'], 'wb').close()
            self.assertEqual(m.snapshot_count('A'), 1)
            nxt, n_cleared = m.submit_panel('A', report_lines=['line'], predictions=[])
            self.assertEqual(m.snapshot_count('A'), 0)
            self.assertGreaterEqual(n_cleared, 0)
            self.assertEqual(nxt, 'B')
            self.assertTrue(os.path.isfile(os.path.join(tmp, 'panels', 'A', 'submitted_draft.json')))

    def test_mission_hud_text_after_submit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            m = MultiPanelMissionManager(tmp, ['A', 'B', 'C'], snapshots_per_panel=5, store_kwargs={})
            m.submit_panel('A', report_lines=['x'], predictions=[])
            self.assertEqual(m.current_panel, 'B')
            self.assertIn('Lot do panelu B', m.mission_hud_text(pause_remaining=3))
            self.assertIn('(2/3)', m.mission_hud_text(pause_remaining=3))

    def test_mission_hud_text_when_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            m = MultiPanelMissionManager(tmp, ['A'], snapshots_per_panel=5, store_kwargs={})
            m.submit_panel('A', report_lines=['x'], predictions=[])
            self.assertIn('Misja zakończona', m.mission_hud_text())
            self.assertIn('1/1', m.mission_hud_text())

    def test_panel_full_blocks_quota(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            m = MultiPanelMissionManager(tmp, ['A'], snapshots_per_panel=2, store_kwargs={})
            st = m.current_store()
            st.entries = [{'frame_id': 'a', 'reproj_b': 1}, {'frame_id': 'b', 'reproj_b': 2}]
            self.assertTrue(m.panel_is_full())
            self.assertFalse(m.can_accept_snapshots())


if __name__ == '__main__':
    unittest.main()
