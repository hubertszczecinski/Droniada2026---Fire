"""OpenCV VideoCapture z opcjonalnym pipeline GStreamer (CAP_GSTREAMER)."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2


def open_video_capture(
    *,
    gstreamer_pipeline: Optional[str] = None,
    device=None,
    use_v4l2: bool = True,
) -> cv2.VideoCapture:
    """
    Gdy ``gstreamer_pipeline`` jest ustawiony → cv2.VideoCapture(pipeline, CAP_GSTREAMER).
    W przeciwnym razie zwykłe V4L2 / indeks / ścieżka pliku.
    """
    if gstreamer_pipeline and str(gstreamer_pipeline).strip():
        pipeline = str(gstreamer_pipeline).strip()
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError(f'gstreamer_open_failed pipeline={pipeline[:120]!r}...')
        return cap
    api = cv2.CAP_V4L2 if use_v4l2 else cv2.CAP_ANY
    cap: Optional[cv2.VideoCapture] = None
    if isinstance(device, str) and device.startswith('/dev/video'):
        try:
            idx = int(device.replace('/dev/video', ''))
            cap = cv2.VideoCapture(idx, api)
        except ValueError:
            cap = None
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(device, api)
    else:
        cap = cv2.VideoCapture(device, api)
    if not cap.isOpened():
        raise RuntimeError(f'capture_open_failed device={device!r} api={api}')
    return cap
