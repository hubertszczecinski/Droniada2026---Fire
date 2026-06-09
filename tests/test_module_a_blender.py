"""Moduł A — ustawienie panelu + odległość na dataset Blender."""
from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from module_pose.api import load_pose_gt_json, pose_from_paths
from module_pose.panel_stand import load_stand_calibration, STAND_CATEGORIES
from pipelines.eval_module_a_blender import eval_module_a_blender

_DATASET = os.path.join(_ROOT, 'dataset')
_POSE_DIR = os.path.join(_DATASET, 'labels_pose')
_CALIB = os.path.join(_ROOT, 'module_pose', 'data', 'panel_stand_linear.json')

_MIN_POSE_OK_PCT = 95.0
_MIN_STAND_ACC_PCT = 78.0
_MIN_VERTICAL_ACC_PCT = 90.0
_MAX_DIST_ERR_MEDIAN_M = 2.0


@unittest.skipUnless(os.path.isdir(_POSE_DIR), 'brak dataset/labels_pose (Blender)')
@unittest.skipUnless(os.path.isfile(_CALIB), 'brak module_pose/data/panel_stand_linear.json — uruchom calibrate_panel_stand')
class TestModuleABlender(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = eval_module_a_blender(_DATASET)

    def test_pose_ok_rate(self):
        self.assertGreaterEqual(
            float(self.report['pose_ok_pct']),
            _MIN_POSE_OK_PCT,
            self.report,
        )

    def test_stand_category_accuracy(self):
        self.assertGreaterEqual(
            float(self.report['stand_category_acc_pct']),
            _MIN_STAND_ACC_PCT,
            self.report['by_category'],
        )

    def test_vertical_stand_accuracy(self):
        vert = (self.report.get('by_category') or {}).get('vertical') or {}
        self.assertGreaterEqual(
            float(vert.get('category_acc_pct', 0.0)),
            _MIN_VERTICAL_ACC_PCT,
            vert,
        )

    def test_distance_median_error(self):
        med = self.report.get('distance_err_median_m')
        self.assertIsNotNone(med)
        self.assertLessEqual(float(med), _MAX_DIST_ERR_MEDIAN_M)

    def test_integration_dict_shape(self):
        """Dict pod późniejsze shmsrc — kąty, odległość, ustawienie panelu."""
        found = False
        for name in sorted(os.listdir(_POSE_DIR))[:20]:
            if not name.endswith('.json'):
                continue
            stem = name.replace('.json', '')
            img_p = os.path.join(_DATASET, 'images', f'{stem}.png')
            pj = os.path.join(_POSE_DIR, name)
            yp = os.path.join(_DATASET, 'labels_yolo', f'{stem}.txt')
            if not os.path.isfile(img_p):
                continue
            pose = pose_from_paths(
                img_p,
                yolo_path=yp if os.path.isfile(yp) else None,
                pose_gt_json_path=pj,
            )
            if not pose.ok:
                continue
            gt = load_pose_gt_json(pj)
            d = pose.to_integration_dict(panel_id=str((gt.get('panel') or {}).get('id', 'A')))
            for key in (
                'ok', 'distance_camera_to_panel_center_m',
                'roll_deg', 'pitch_deg', 'yaw_deg',
                'report_angle_deg', 'panel_angle_category', 'panel_stand_label_pl',
                'stand_confidence', 'reproj_mean_px', 'method',
            ):
                self.assertIn(key, d)
            self.assertIn(d['panel_angle_category'], STAND_CATEGORIES)
            found = True
            break
        self.assertTrue(found, 'brak udanej próbki pose na Blenderze')

    def test_calibration_loaded(self):
        cal = load_stand_calibration(_CALIB)
        self.assertIsNotNone(cal)
        self.assertIn('W', cal)
        self.assertGreaterEqual(int(cal.get('n_samples', 0)), 50)


class TestModuleAStandFeatures(unittest.TestCase):
    def test_calibration_json_parseable(self):
        with open(_CALIB, 'r', encoding='utf-8') as fh:
            cal = json.load(fh)
        self.assertEqual(len(cal['W']), 3)


if __name__ == '__main__':
    unittest.main()
