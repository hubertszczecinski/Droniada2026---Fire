"""Test zgodności 3 siatek na warpie."""
import unittest

import cv2
import numpy as np

from module_panel.grid_overlap import (
    detect_enhanced_white_grid_lines,
    ideal_uniform_grid_lines,
    measure_three_grid_overlap,
    warp_panel_coverage_score,
)
from release.live_dashboard import snapshot_frame_eligible, snapshot_scene_eligible


def _synthetic_warped(w: int = 500, h: int = 400) -> np.ndarray:
    img = np.full((h, w, 3), 48, dtype=np.uint8)
    for i in range(11):
        x = int(round(i * w / 10.0))
        y = int(round(i * h / 10.0))
        cv2.line(img, (x, 0), (x, h - 1), (230, 230, 230), 2, cv2.LINE_AA)
        cv2.line(img, (0, y), (w - 1, y), (230, 230, 230), 2, cv2.LINE_AA)
    return img


class TestGridOverlap(unittest.TestCase):
    def test_enhanced_on_synthetic(self) -> None:
        warped = _synthetic_warped()
        _xs, _ys, ok, meta = detect_enhanced_white_grid_lines(warped, relaxed=True)
        self.assertTrue(ok, meta)

    def test_three_grids_aligned_synthetic(self) -> None:
        warped = _synthetic_warped()
        ratio, meta = measure_three_grid_overlap(warped)
        self.assertNotEqual(meta.get('fail'), 'no_grid_signal', meta)
        self.assertGreaterEqual(float(meta.get('triple_consensus', 0)), 0.85, meta)
        self.assertGreaterEqual(ratio, 0.75, meta)

    def test_skipped_column_low_overlap(self) -> None:
        w, h = 500, 400
        warped = _synthetic_warped(w, h)
        xs, ys = ideal_uniform_grid_lines(w, h)
        bad_x = list(xs.astype(float))
        bad_x[5] = bad_x[4] + 0.55 * (w / 10.0)
        for i in range(6, 11):
            bad_x[i] = bad_x[i - 1] + (w / 10.0)
        ratio, meta = measure_three_grid_overlap(
            warped,
            grid_lines_x=bad_x,
            grid_lines_y=[float(v) for v in ys],
        )
        self.assertLess(float(meta.get('spacing_opencv', 1.0)), 0.95, meta)
        self.assertLess(ratio, 0.88, meta)

    def test_snapshot_scene_blocks_cv_fallback(self) -> None:
        ok, reason = snapshot_scene_eligible(
            panel_present=True,
            corner_source='cv_fallback+module_a',
            require_yolo_corners=True,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, 'cv_fallback')

    def test_snapshot_blocks_high_reproj_a(self) -> None:
        ok, reason = snapshot_frame_eligible(
            reproj_b=5.0, max_reproj_px=18.0, grid_overlap_ratio=0.95,
            min_grid_overlap=0.88, require_module_a=True, module_a_ok=True,
            reproj_a=20.0, max_reproj_a_px=15.0,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, 'reproj_a')

    def test_snapshot_ignores_broken_pose_reproj_a(self) -> None:
        ok, reason = snapshot_frame_eligible(
            reproj_b=10.0, max_reproj_px=18.0, grid_overlap_ratio=0.90,
            min_grid_overlap=0.75, require_module_a=True, module_a_ok=True,
            reproj_a=536.0, max_reproj_a_px=18.0,
            warp_panel_coverage=1.0, min_warp_panel_coverage=0.20,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, 'grid_overlap')

    def test_warp_coverage_detects_bright_edge(self) -> None:
        h, w, n = 500, 1000, 10
        img = np.full((h, w, 3), 30, dtype=np.uint8)
        for i in range(1, n):
            x = int(i * w / n)
            cv2.line(img, (x, 0), (x, h - 1), (50, 50, 50), 1)
        img[:, 0 : w // n] = 180
        score, meta = warp_panel_coverage_score(img, n=n)
        self.assertLess(score, 0.5, meta)

    def test_enhanced_on_dark_warped_nag5(self) -> None:
        path = 'dataset/debug_grid_overlap/nag5_f60_warped.jpg'
        warped = cv2.imread(path)
        if warped is None:
            self.skipTest('brak nag5 debug warp')
        ratio, meta = measure_three_grid_overlap(warped)
        self.assertGreater(float(meta.get('profile_signal', 0)), 0.01, meta)
        self.assertLess(ratio, 0.88, meta)


if __name__ == '__main__':
    unittest.main()
