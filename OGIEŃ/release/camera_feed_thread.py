"""Wątek kamery — ciągły odczyt + podgląd WWW niezależny od YOLO."""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np

from release.camera_source import CameraConfig, CameraSource

FrameCallback = Callable[[bool, Optional[np.ndarray], str], None]


class SharedCameraFeed:
    """Jeden wątek czyta V4L2; analiza bierze ostatnią klatkę bez blokowania podglądu."""

    __slots__ = (
        '_cfg', '_src', '_lock', '_latest', '_stop', '_thread', '_on_frame',
    )

    def __init__(
        self,
        cfg: CameraConfig,
        *,
        on_frame: Optional[FrameCallback] = None,
    ) -> None:
        self._cfg = cfg
        self._src = CameraSource(cfg)
        self._lock = threading.Lock()
        self._latest: Tuple[bool, Optional[np.ndarray], str] = (
            False, None, 'live_000000',
        )
        self._stop = threading.Event()
        self._on_frame = on_frame
        self._thread = threading.Thread(target=self._run, name='droniada-cam', daemon=True)

    @property
    def open_meta(self) -> dict:
        return self._src.open_meta

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._src.close()

    def get_latest(self) -> Tuple[bool, Optional[np.ndarray], str]:
        with self._lock:
            ok, bgr, fid = self._latest
            if ok and bgr is not None:
                return True, bgr.copy(), fid
            return ok, bgr, fid

    def _run(self) -> None:
        try:
            self._src.open()
        except Exception:
            pass
        while not self._stop.is_set():
            try:
                ok, bgr, fid = self._src.read()
            except Exception:
                ok, bgr, fid = False, None, f'live_{self._src._frame_idx:06d}'
                time.sleep(0.2)
                continue
            with self._lock:
                self._latest = (ok, bgr if ok else None, fid)
            if self._on_frame is not None:
                try:
                    self._on_frame(ok, bgr, fid)
                except Exception:
                    pass
            if not ok:
                time.sleep(0.05)

    def __enter__(self) -> 'SharedCameraFeed':
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
