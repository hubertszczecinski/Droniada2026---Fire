"""Regresja: nag3 klatki 3–4 — tylko pomarańcz na (8,7), bez fałszywej zieleni z brzegu."""
from __future__ import annotations

import os
import sys
import unittest

import cv2

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from release.transform import apply_rotate
from release.live_corners import detect_corners_live
from release.live_card_detect import detect_cards_on_warped
from release.panel_black import calibrate_panel_black_from_corners
from module_geom.camera import resolve_intrinsics
from module_panel.warp import warp_panel_rect

_NAG3_VIDEO = os.path.join(_ROOT, 'dataset', 'my_capture', 'Droniada_nag3.mov')
_YOLO_WEIGHTS = os.path.join(_ROOT, 'runs', 'pose', 'droniada_real_finetune', 'weights', 'best.pt')
_EXPECTED_FRAMES = {
    3: {(8, 7, 'POMARANCZOWA')},
    4: {(8, 7, 'POMARANCZOWA')},
}
_FALSE_EDGE = {
    (10, 7, 'ZIELONA'),
    (1, 1, 'ZIELONA'),
    (9, 1, 'ZIELONA'),
    (8, 1, 'ZIELONA'),
    (10, 1, 'ZIELONA'),
    (9, 2, 'ZIELONA'),
}


def _read_frame(video_path: str, index: int, rotate: int = 180):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        return None
    return apply_rotate(bgr, rotate)


def _detect_on_frame(bgr) -> list[tuple[int, int, str]]:
    h, w = bgr.shape[:2]
    if os.path.isfile(_YOLO_WEIGHTS):
        os.environ['DRONIADA_YOLO_POSE_WEIGHTS'] = _YOLO_WEIGHTS
    k, dist, _ = resolve_intrinsics((h, w), profile='tarot_t10x_2a:wide')
    corners, _, _ = detect_corners_live(
        bgr, k, dist, corner_mode='yolo_pose', use_tracker=False,
    )
    warped, hmat = warp_panel_rect(bgr, corners)
    bt = calibrate_panel_black_from_corners(bgr, corners)
    dets = detect_cards_on_warped(
        warped,
        black_thresholds=bt,
        corners_tltrbrbl=corners,
        homography_img_to_warp=hmat,
        image_shape=(h, w),
    )
    return [(int(d['grid_row']), int(d['grid_col']), str(d['color'])) for d in dets]


@unittest.skipUnless(os.path.isfile(_NAG3_VIDEO), 'brak dataset/my_capture/Droniada_nag3.mov')
@unittest.skipUnless(os.path.isfile(_YOLO_WEIGHTS), 'brak wag YOLO droniada_real_finetune')
class TestLiveCardDetectNag3(unittest.TestCase):
    def test_frames_3_and_4_only_orange_8_7(self):
        for frame_idx, expected in _EXPECTED_FRAMES.items():
            bgr = _read_frame(_NAG3_VIDEO, frame_idx)
            self.assertIsNotNone(bgr, f'nie wczytano klatki {frame_idx}')
            found = set(_detect_on_frame(bgr))
            self.assertTrue(
                expected <= found,
                f'klatka {frame_idx}: brak oczekiwanych {expected - found}, jest {found}',
            )
            self.assertFalse(
                found & _FALSE_EDGE,
                f'klatka {frame_idx}: fałszywe detekcje z brzegu/siatki: {found & _FALSE_EDGE}',
            )
            bad_green = {
                (r, c)
                for r, c, color in found
                if color == 'ZIELONA' and (c >= 9 or r >= 9 or (r >= 8 and c >= 6))
            }
            self.assertFalse(
                bad_green,
                f'klatka {frame_idx}: zieleń z brzegu warpu: {bad_green}',
            )


if __name__ == '__main__':
    unittest.main()
