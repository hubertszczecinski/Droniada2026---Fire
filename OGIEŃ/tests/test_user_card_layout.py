"""Dowolny układ kart na syntetycznym panelu — bez stałych współrzędnych."""
from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.test_live_card_detect_colors import (
    _random_placements,
    _synthetic_panel_with_cards,
)
from release.live_card_detect import detect_cards_on_warped


class TestArbitraryUserLayout(unittest.TestCase):
    def test_detections_match_placed_cells(self):
        placements = _random_placements(42)
        warped = _synthetic_panel_with_cards(placements)
        dets = detect_cards_on_warped(warped, max_cells=6)
        found = {(d['grid_row'], d['grid_col'], d['color']) for d in dets}
        expected = {(r, c, col) for r, c, col in placements}
        self.assertTrue(
            expected <= found,
            f'brakuje {expected - found}, jest {found}',
        )


if __name__ == '__main__':
    unittest.main()
