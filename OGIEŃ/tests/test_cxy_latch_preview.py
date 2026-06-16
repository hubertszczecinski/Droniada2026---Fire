"""Testy podglądu latch — poprawne mapowanie wierszy na warpie."""
from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from release.cxy_latch_preview import grid_col_to_warp_x, grid_row_to_warp_y


class TestCxyLatchPreview(unittest.TestCase):
    def test_grid_row_1_is_bottom(self):
        h = 500
        y1 = grid_row_to_warp_y(1, h)
        y10 = grid_row_to_warp_y(10, h)
        self.assertGreater(y1, y10)
        self.assertAlmostEqual(y1, 0.95 * h, delta=0.06 * h)

    def test_grid_col_1_is_left(self):
        w = 1000
        x1 = grid_col_to_warp_x(1, w)
        x10 = grid_col_to_warp_x(10, w)
        self.assertLess(x1, x10)


if __name__ == '__main__':
    unittest.main()
