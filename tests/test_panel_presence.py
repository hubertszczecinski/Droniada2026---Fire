import unittest

import numpy as np

from release.panel_presence import resolve_panel_presence


class TestPanelPresence(unittest.TestCase):
    def test_reliable_b_overrides_corner_high_reproj(self) -> None:
        corners = np.array([[100, 100], [900, 100], [900, 900], [100, 900]], dtype=np.float64)
        meta = {'reproj_mean_px': 120.0, 'method': 'yolo_pose', 'det_conf': 0.9, 'kpt_conf_min': 0.8}
        present, reason, _ = resolve_panel_presence(
            corners,
            meta,
            'yolo_pose',
            image_shape=(1080, 1920),
            reliable_b=True,
            reproj_b=1.2,
            max_reproj_b=18.0,
        )
        self.assertTrue(present)
        self.assertEqual(reason, 'reliable_b')

    def test_tracker_hold_stays_absent(self) -> None:
        corners = np.array([[100, 100], [900, 100], [900, 900], [100, 900]], dtype=np.float64)
        meta = {'tracker_held': True, 'reproj_mean_px': 1.0}
        present, reason, _ = resolve_panel_presence(
            corners,
            meta,
            'yolo_pose+hold',
            image_shape=(1080, 1920),
            reliable_b=True,
            reproj_b=1.0,
        )
        self.assertFalse(present)
        self.assertEqual(reason, 'tracker_hold')


if __name__ == '__main__':
    unittest.main()
