import unittest

from release.web_runtime_settings import merge_runtime_settings, validate_runtime_settings


class TestWebRuntimeSettings(unittest.TestCase):
    def test_validate_clamps_overlap(self):
        out = validate_runtime_settings({'snapshot_min_grid_overlap': 1.5})
        self.assertAlmostEqual(out['snapshot_min_grid_overlap'], 1.0)

    def test_merge_applies_snapshot_thresholds(self):
        m = merge_runtime_settings({}, {'snapshot_min_grid_overlap': 0.8, 'snapshot_min_stable': 3})
        self.assertAlmostEqual(m['snapshot_min_grid_overlap'], 0.8)
        self.assertEqual(m['snapshot_min_stable'], 3)


if __name__ == '__main__':
    unittest.main()
