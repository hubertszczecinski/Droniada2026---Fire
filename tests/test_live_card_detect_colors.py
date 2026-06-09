"""Regresja detekcji kolorów — bez hardcodowanego układu kart."""
from __future__ import annotations

import os
import random
import sys
import unittest

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from release.live_card_detect import _classify_patch_bgr, detect_cards_on_warped

_CARD_BGR = {
    'POMARANCZOWA': (30, 130, 240),
    'ZOLTA': (20, 220, 240),
    'FIOLETOWA': (200, 60, 200),
    'ZIELONA': (50, 190, 50),
    'CZERWONA': (40, 40, 220),
    'NIEBIESKA': (220, 120, 40),
}


def _synthetic_panel_with_cards(
    placements: list[tuple[int, int, str]],
) -> np.ndarray:
    """Czarny panel 10×10 z kartkami (wiersz 1 = dół siatki)."""
    h, w = 640, 640
    warped = np.full((h, w, 3), 25, dtype=np.uint8)
    cw, ch = w / 10.0, h / 10.0
    for comp_row, comp_col, color_name in placements:
        img_row = 10 - comp_row
        col0 = comp_col - 1
        x0, y0 = int(col0 * cw + 8), int(img_row * ch + 8)
        x1, y1 = int((col0 + 1) * cw - 8), int((img_row + 1) * ch - 8)
        warped[y0:y1, x0:x1] = _CARD_BGR[color_name]
    for i in range(11):
        x, y = int(round(i * cw)), int(round(i * ch))
        if 0 <= x < w:
            warped[:, max(0, x - 1) : min(w, x + 2)] = (235, 235, 235)
        if 0 <= y < h:
            warped[max(0, y - 1) : min(h, y + 2), :] = (235, 235, 235)
    return warped


def _random_placements(seed: int, n: int = 4) -> list[tuple[int, int, str]]:
    rng = random.Random(seed)
    colors = ['POMARANCZOWA', 'ZOLTA', 'FIOLETOWA', 'ZIELONA']
    cells: set[tuple[int, int]] = set()
    out: list[tuple[int, int, str]] = []
    while len(out) < n:
        r, c = rng.randint(3, 8), rng.randint(3, 8)
        if (r, c) in cells:
            continue
        cells.add((r, c))
        out.append((r, c, colors[len(out)]))
    return out


from release.live_card_detect import _classify_patch_bgr, detect_cards_on_warped
from release.card_color_profile import load_active_profile

_NO_PROFILE = '/tmp/droniada_test_no_card_colors.json'


class _DefaultProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._prev = os.environ.get('DRONIADA_CARD_COLORS')
        os.environ['DRONIADA_CARD_COLORS'] = _NO_PROFILE
        load_active_profile(force=True)

    def tearDown(self) -> None:
        if self._prev is None:
            os.environ.pop('DRONIADA_CARD_COLORS', None)
        else:
            os.environ['DRONIADA_CARD_COLORS'] = self._prev
        load_active_profile(force=True)


class TestColorClassification(_DefaultProfileTests):
    def test_white_patch_not_green(self):
        patch = np.full((48, 48, 3), 245, dtype=np.uint8)
        meta = _classify_patch_bgr(patch, grid_row=5, grid_col=5)
        self.assertIsNone(meta)

    def test_synthetic_purple_patch(self):
        patch = np.zeros((64, 64, 3), dtype=np.uint8)
        patch[:, :] = (30, 30, 30)
        patch[12:52, 12:52] = _CARD_BGR['FIOLETOWA']
        meta = _classify_patch_bgr(patch, grid_row=5, grid_col=5)
        self.assertIsNotNone(meta)
        self.assertEqual(meta[0], 4)

    def test_dim_purple_not_rejected_as_blue(self):
        """Słabe światło: niski V, H często wpadnie w niebieski — i tak fiolet."""
        patch = np.zeros((64, 64, 3), dtype=np.uint8)
        patch[:, :] = (25, 25, 25)
        dim = (np.full((40, 40, 3), _CARD_BGR['FIOLETOWA'], dtype=np.uint8).astype(np.float32) * 0.35)
        patch[12:52, 12:52] = np.clip(dim, 0, 255).astype(np.uint8)
        meta = _classify_patch_bgr(patch, grid_row=5, grid_col=5)
        self.assertIsNotNone(meta, 'przyciemniony fiolet nie powinien odpaść jako niebieski (V)')
        self.assertEqual(meta[0], 4)


class TestArbitraryLayouts(_DefaultProfileTests):
    def test_random_four_colors_detected(self):
        for seed in range(8):
            placements = _random_placements(seed)
            warped = _synthetic_panel_with_cards(placements)
            dets = detect_cards_on_warped(warped, max_cells=6)
            found = {(d['grid_row'], d['grid_col'], d['color']) for d in dets}
            expected = {(r, c, col) for r, c, col in placements}
            self.assertTrue(
                expected <= found,
                f'seed={seed}: brakuje {expected - found}, jest {found}',
            )


if __name__ == '__main__':
    unittest.main()
