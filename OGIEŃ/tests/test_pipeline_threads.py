"""Testy wątków pipeline."""
from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from release.pipeline_threads import (
    DropQueueWorker,
    LatestFrameBuffer,
    OverlayCache,
    VisPreviewWorker,
)


class TestPipelineThreads(unittest.TestCase):
    def test_latest_frame_buffer(self) -> None:
        buf = LatestFrameBuffer()
        bgr = np.zeros((48, 64, 3), dtype=np.uint8)
        buf.publish(True, bgr, 'live_000001')
        ok, out, fid, seq = buf.read_copy()
        self.assertTrue(ok)
        assert out is not None
        self.assertEqual(fid, 'live_000001')
        self.assertEqual(seq, 1)

    def test_overlay_cache_version(self) -> None:
        oc = OverlayCache()
        st, ver0 = oc.snapshot()
        self.assertIsNone(st)
        oc.update({'reliable': True})
        st, ver1 = oc.snapshot()
        assert st is not None
        self.assertEqual(ver1, 1)
        self.assertTrue(st['reliable'])

    def test_drop_queue_worker(self) -> None:
        done = threading.Event()

        def _job(x: int) -> int:
            time.sleep(0.05)
            done.set()
            return x * 2

        w = DropQueueWorker(_job, name='test-worker')
        w.start()
        w.submit(21)
        deadline = time.monotonic() + 2.0
        out = None
        while time.monotonic() < deadline:
            out = w.poll_result()
            if out is not None:
                break
            time.sleep(0.01)
        w.stop()
        self.assertTrue(done.is_set())
        self.assertEqual(out, 42)

    def test_vis_preview_worker_runs(self) -> None:
        buf = LatestFrameBuffer()
        oc = OverlayCache()
        pushed: list[int] = []

        def _push(bgr, state) -> None:
            pushed.append(int(state.get('n', 0)))

        w = VisPreviewWorker(buf, oc, _push, interval_s=0.03, stream_enabled=True)
        oc.update({'n': 7})
        bgr = np.zeros((24, 32, 3), dtype=np.uint8)
        buf.publish(True, bgr, 'live_000002')
        w.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not pushed:
            time.sleep(0.02)
        w.stop()
        self.assertTrue(pushed)


if __name__ == '__main__':
    unittest.main()
