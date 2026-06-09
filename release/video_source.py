from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
from release.transform import apply_rotate

@dataclass
class VideoConfig:
    path: str
    rotate_deg: int = 0
    loop: bool = True

class VideoSource:
    __slots__ = ('cfg', '_cap', '_frame_idx', '_ended')

    def __init__(self, cfg: VideoConfig) -> None:
        self.cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx = 0
        self._ended = False

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 30.0
        v = float(self._cap.get(cv2.CAP_PROP_FPS))
        return v if v > 1.0 else 30.0

    def open(self) -> None:
        path = self.cfg.path
        # W kontenerze Jetson GStreamer dla plików .mov bywa wolny / wiesza się — FFmpeg jest stabilniejszy.
        self._cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f'video_open_failed path={path}')
        self._ended = False
        self._frame_idx = 0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> Tuple[bool, np.ndarray, str]:
        if self._cap is None:
            raise RuntimeError('video_not_open')
        ok, frame = self._cap.read()
        if not ok or frame is None:
            if self.cfg.loop and not self._ended:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    self._ended = True
                    return (False, np.zeros((480, 640, 3), dtype=np.uint8), f'vid_{self._frame_idx:06d}')
            else:
                self._ended = True
                return (False, np.zeros((480, 640, 3), dtype=np.uint8), f'vid_{self._frame_idx:06d}')
        self._frame_idx += 1
        if self.cfg.rotate_deg:
            frame = apply_rotate(frame, self.cfg.rotate_deg)
        return (True, frame, f'vid_{self._frame_idx:06d}')

    def __enter__(self) -> 'VideoSource':
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
