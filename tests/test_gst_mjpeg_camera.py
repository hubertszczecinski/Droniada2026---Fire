"""Testy GStreamer MJPEG (pipeline string, env)."""
from __future__ import annotations

import os
import unittest
from unittest import mock

from release.gst_mjpeg_camera import (
    GstMjpegCameraFeed,
    should_use_gst_capture,
    use_hw_jpeg,
    wants_stream_passthrough,
)


class TestGstMjpegEnv(unittest.TestCase):
    def test_passthrough_when_gst_on(self) -> None:
        with mock.patch('release.gst_mjpeg_camera._gst_available', return_value=True):
            with mock.patch.dict(os.environ, {
                'DRONIADA_USE_GST_CAPTURE': '1',
                'DRONIADA_CAMERA_FOURCC': 'MJPG',
                'DRONIADA_STREAM_PASSTHROUGH': '1',
            }, clear=False):
                self.assertTrue(should_use_gst_capture())
                self.assertTrue(wants_stream_passthrough())

    def test_passthrough_off(self) -> None:
        with mock.patch('release.gst_mjpeg_camera._gst_available', return_value=True):
            with mock.patch.dict(os.environ, {
                'DRONIADA_STREAM_PASSTHROUGH': '0',
            }, clear=False):
                self.assertFalse(wants_stream_passthrough())

    def test_pipeline_contains_tee(self) -> None:
        feed = GstMjpegCameraFeed('/dev/video0', width=640, height=480, fps=30)
        pipe = feed._build_pipeline()
        self.assertIn('tee name=t', pipe)
        self.assertIn('appsink name=mjpeg', pipe)
        self.assertIn('appsink name=bgr', pipe)

    def test_pipeline_rotate_uses_flip(self) -> None:
        feed = GstMjpegCameraFeed('/dev/video0', width=640, height=480, rotate=180)
        pipe = feed._build_pipeline()
        self.assertIn('videoflip method=rotate-180', pipe)

    def test_hw_jpeg_env_off(self) -> None:
        with mock.patch.dict(os.environ, {'DRONIADA_GST_HW_JPEG': '0'}, clear=False):
            self.assertFalse(use_hw_jpeg())


if __name__ == '__main__':
    unittest.main()
