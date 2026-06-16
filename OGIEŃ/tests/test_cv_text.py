import unittest

import cv2
import numpy as np

from release.cv_text import put_text_utf8, text_width_utf8


class TestCvText(unittest.TestCase):
    def test_polish_chars_render_without_error(self) -> None:
        img = np.full((80, 640, 3), 245, dtype=np.uint8)
        line = 'Odległość: 13.10 m  kąt=90°  MODUŁ A — podejście'
        put_text_utf8(img, line, (10, 40), (25, 25, 25), scale=0.88, thickness=2)
        w = text_width_utf8(line, 0.88, 2)
        self.assertGreater(w, 100)
        roi = img[10:50, 10:10 + w]
        self.assertGreater(int(np.std(roi)), 1.0)


if __name__ == '__main__':
    unittest.main()
