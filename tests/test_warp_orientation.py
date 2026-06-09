import unittest

import cv2
import numpy as np
import pipeline_competition as pc

from module_panel.warp import orient_warped_panel_by_white_anchor, warp_panel_rect


class TestWarpOrientation(unittest.TestCase):
    def test_orient_fx_puts_white_bottom_left(self):
        h, w = pc.RECT_H, pc.RECT_W
        warped = np.zeros((h, w, 3), dtype=np.uint8)
        ch, cw = h // 10, w // 10
        warped[h - ch:h, w - cw:w] = 255  # white at BR → needs fx

        out, _h, t = orient_warped_panel_by_white_anchor(warped)
        self.assertEqual(t, 'fx')
        patch = out[h - ch:h, 0:cw]
        self.assertGreater(float(np.mean(patch)), 200.0)

    def test_warp_panel_rect_oriented(self):
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        pts = np.array([[100, 80], [700, 90], [680, 520], [120, 510]], dtype=np.float32)
        ch, cw = pc.RECT_H // 10, pc.RECT_W // 10
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts.astype(np.int32), 255)
        img[mask > 0] = (20, 20, 20)
        bl = pts[3].astype(int)
        x0, y0 = max(0, bl[0] - 20), max(0, bl[1] - 20)
        img[y0:y0 + ch, x0:x0 + cw] = 255

        warped, _ = warp_panel_rect(img, pts)
        t = pc.detect_white_corner_transform(warped)
        self.assertIn(t, ('id', 'fx', 'fy', 'fxy'))
        self.assertEqual(t, 'id')


if __name__ == '__main__':
    unittest.main()
