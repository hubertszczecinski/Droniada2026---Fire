"""Konkurs CXY z migawek — odfiltrowanie jednorazowych detekcji."""
from __future__ import annotations

import unittest

from release.snapshot_cxy_competition import (
    SnapshotObservation,
    run_snapshot_cxy_competition,
)


def _obs(fid: str, preds: list[tuple[int, int, str]], reproj: float = 10.0) -> SnapshotObservation:
    return SnapshotObservation(
        frame_id=fid,
        reproj_b=reproj,
        grid_overlap_ratio=0.9,
        predictions=[{'x': x, 'y': y, 'color': c} for y, x, c in preds],
        report_lines=[],
        panel_id='A',
        angle_deg=90,
    )


class TestSnapshotCxyCompetition(unittest.TestCase):
    def test_drops_one_off_detection(self) -> None:
        stable = [(3, 3, 'POMARANCZOWA'), (3, 8, 'ZIELONA')]
        obs = [
            _obs('a', stable),
            _obs('b', stable + [(9, 2, 'ZOLTA')]),
            _obs('c', stable),
        ]
        res = run_snapshot_cxy_competition(obs, min_votes=2)
        keys = {(c.y, c.x, c.color) for c in res.accepted}
        self.assertIn((3, 3, 'POMARANCZOWA'), keys)
        self.assertIn((3, 8, 'ZIELONA'), keys)
        self.assertNotIn((9, 2, 'ZOLTA'), keys)
        rejected = {(c.y, c.x, c.color) for c in res.rejected}
        self.assertIn((9, 2, 'ZOLTA'), rejected)

    def test_single_snapshot_needs_min_votes_one(self) -> None:
        obs = [_obs('only', [(8, 7, 'POMARANCZOWA')])]
        res = run_snapshot_cxy_competition(obs, min_votes=1)
        self.assertEqual(len(res.accepted), 1)

    def test_min_votes_two_blocks_singleton(self) -> None:
        obs = [_obs('only', [(8, 7, 'POMARANCZOWA')])]
        res = run_snapshot_cxy_competition(obs, min_votes=2)
        self.assertEqual(len(res.accepted), 0)
        self.assertEqual(len(res.rejected), 1)

    def test_same_cell_keeps_best_color(self) -> None:
        obs = [
            _obs('a', [(3, 3, 'POMARANCZOWA'), (3, 3, 'ZOLTA')]),
            _obs('b', [(3, 3, 'POMARANCZOWA')]),
            _obs('c', [(3, 3, 'POMARANCZOWA')]),
        ]
        res = run_snapshot_cxy_competition(obs, min_votes=2)
        self.assertEqual(len(res.accepted), 1)
        self.assertEqual(res.accepted[0].y, 3)
        self.assertEqual(res.accepted[0].x, 3)
        self.assertEqual(res.accepted[0].color, 'POMARANCZOWA')

    def test_caps_at_four_cards(self) -> None:
        cards = [
            (1, 1, 'CZERWONA'),
            (1, 2, 'ZIELONA'),
            (1, 3, 'NIEBIESKA'),
            (1, 4, 'ZOLTA'),
            (1, 5, 'FIOLETOWA'),
        ]
        obs = [_obs('a', cards), _obs('b', cards), _obs('c', cards)]
        res = run_snapshot_cxy_competition(obs, min_votes=2, max_cards=4)
        self.assertEqual(len(res.accepted), 4)
        self.assertGreaterEqual(len(res.rejected), 1)


if __name__ == '__main__':
    unittest.main()
