"""Test kalibracji kolorów kartek."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from release.card_color_profile import (
    CardColorProfile,
    build_profile_from_folder,
    load_active_profile,
    sample_median_hsv,
    _normalize_stem,
)
from release.live_card_detect import _classify_patch_bgr

_CARD_BGR = {
    'CZERWONA': (40, 40, 220),
    'ZIELONA': (50, 190, 50),
    'NIEBIESKA': (220, 120, 40),
    'ZOLTA': (20, 220, 240),
    'FIOLETOWA': (200, 60, 200),
    'POMARANCZOWA': (30, 130, 240),
}


def _solid_card(name: str, size: int = 200) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = _CARD_BGR[name]
    return img


class TestCardColorProfile(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop('DRONIADA_CARD_COLORS', None)
        load_active_profile(force=True)

    def test_normalize_stem_with_digit_suffix(self):
        self.assertEqual(_normalize_stem('CZERWONA2.png'), 'CZERWONA')
        self.assertEqual(_normalize_stem('zolta2.jpg'), 'ZOLTA')

    def test_build_uses_numbered_and_base_variants(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cv2.imwrite(str(td_path / 'ZOLTA.png'), _solid_card('POMARANCZOWA'))
            cv2.imwrite(str(td_path / 'ZOLTA2.png'), _solid_card('ZOLTA'))
            cv2.imwrite(str(td_path / 'ZIELONA.png'), _solid_card('NIEBIESKA'))
            cv2.imwrite(str(td_path / 'ZIELONA2.png'), _solid_card('ZIELONA'))
            for name in _CARD_BGR:
                if name in ('ZOLTA', 'ZIELONA'):
                    continue
                cv2.imwrite(str(td_path / f'{name}2.png'), _solid_card(name))
            profile, samples = build_profile_from_folder(td_path)
            self.assertEqual(profile.meta['source_files']['ZOLTA'], 'ZOLTA2.png')
            self.assertAlmostEqual(samples['ZOLTA']['h'], 30.0, delta=8.0)
            self.assertEqual(len(profile.centroids_by_cls[3]), 2)
            self.assertEqual(len(profile.centroids_by_cls[1]), 1)

    def test_build_from_six_images(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for name in _CARD_BGR:
                cv2.imwrite(str(td_path / f'{name}.jpg'), _solid_card(name))
            profile, samples = build_profile_from_folder(td_path)
            self.assertEqual(len(samples), 6)
            self.assertEqual(len(profile.ranges_by_cls), 6)

    def test_calibrated_classifies_purple_not_blue(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for name in _CARD_BGR:
                cv2.imwrite(str(td_path / f'{name}.jpg'), _solid_card(name))
            out = td_path / 'profile.json'
            profile, _ = build_profile_from_folder(td_path)
            profile.save(out, source=str(td_path))
            os.environ['DRONIADA_CARD_COLORS'] = str(out)
            load_active_profile(force=True)
            patch = _solid_card('FIOLETOWA')
            meta = _classify_patch_bgr(patch, grid_row=5, grid_col=5)
            self.assertIsNotNone(meta)
            self.assertEqual(meta[0], 4)

    def test_sample_median_hsv(self):
        med = sample_median_hsv(_solid_card('NIEBIESKA'))
        self.assertIsNotNone(med)
        h, s, v = med
        self.assertGreater(s, 40.0)
        self.assertGreater(v, 40.0)


if __name__ == '__main__':
    unittest.main()
