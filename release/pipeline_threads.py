"""Wątki pipeline live — rozłożenie obciążenia CPU między rdzenie."""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

FrameTriple = Tuple[bool, Optional[np.ndarray], str]
OverlayState = Dict[str, Any]
DrawVisFn = Callable[[np.ndarray, OverlayState], None]
PushVisFn = Callable[[np.ndarray], None]


class OverlayCache:
    """Thread-safe cache overlay YOLO dla podglądu między inferencjami."""

    __slots__ = ('_lock', '_state', '_version')

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Optional[OverlayState] = None
        self._version = 0

    def update(self, state: Optional[OverlayState]) -> None:
        with self._lock:
            self._state = dict(state) if state is not None else None
            self._version += 1

    def snapshot(self) -> Tuple[Optional[OverlayState], int]:
        with self._lock:
            if self._state is None:
                return None, self._version
            return dict(self._state), self._version


class LatestFrameBuffer:
    """Ostatnia klatka z wątku kamery (bez blokowania analizy)."""

    __slots__ = ('_lock', '_ok', '_bgr', '_fid', '_seq')

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ok = False
        self._bgr: Optional[np.ndarray] = None
        self._fid = 'live_000000'
        self._seq = 0

    def publish(self, ok: bool, bgr: Optional[np.ndarray], fid: str) -> None:
        if not ok or bgr is None:
            return
        with self._lock:
            if self._bgr is None or self._bgr.shape != bgr.shape:
                self._bgr = bgr.copy()
            else:
                np.copyto(self._bgr, bgr)
            self._ok = True
            self._fid = str(fid)
            self._seq += 1

    def read_copy(self) -> Tuple[bool, Optional[np.ndarray], str, int]:
        with self._lock:
            if not self._ok or self._bgr is None:
                return False, None, self._fid, self._seq
            return True, self._bgr.copy(), self._fid, self._seq


class VisPreviewWorker:
    """
    Osobny rdzeń: kamera + overlay → stream vis (MJPEG).
    Nie czeka na YOLO — bierze świeżą klatkę i ostatni znany overlay.
    """

    __slots__ = (
        '_frame_buf', '_overlay', '_draw_push', '_interval_s', '_stop',
        '_thread', '_last_seq', '_last_overlay_ver', '_stream_enabled',
    )

    def __init__(
        self,
        frame_buf: LatestFrameBuffer,
        overlay: OverlayCache,
        draw_push: Callable[[np.ndarray, OverlayState], None],
        *,
        interval_s: float = 0.033,
        stream_enabled: bool = True,
    ) -> None:
        self._frame_buf = frame_buf
        self._overlay = overlay
        self._draw_push = draw_push
        self._interval_s = max(0.02, float(interval_s))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name='droniada-vis-preview',
            daemon=True,
        )
        self._last_seq = -1
        self._last_overlay_ver = -1
        self._stream_enabled = bool(stream_enabled)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            if not self._stream_enabled:
                self._stop.wait(0.1)
                continue
            ok, bgr, _fid, seq = self._frame_buf.read_copy()
            state, overlay_ver = self._overlay.snapshot()
            stale = (
                seq == self._last_seq
                and overlay_ver == self._last_overlay_ver
            )
            if ok and bgr is not None and state is not None and not stale:
                try:
                    self._draw_push(bgr, state)
                    self._last_seq = seq
                    self._last_overlay_ver = overlay_ver
                except Exception:
                    pass
            delay = self._interval_s - (time.monotonic() - t0)
            if delay > 0:
                self._stop.wait(delay)


class DropQueueWorker:
    """
    Jeden wątek roboczy z kolejką depth=1 — zawsze najnowsza klatka (YOLO / dashboard).
  Odrzuca zaległe zadania gdy inferencja nie nadąża.
    """

    __slots__ = ('_fn', '_in_q', '_out_q', '_stop', '_thread', '_err')

    def __init__(self, fn: Callable[..., Any], *, name: str = 'droniada-worker') -> None:
        self._fn = fn
        self._in_q: queue.Queue = queue.Queue(maxsize=1)
        self._out_q: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._err: Optional[BaseException] = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._in_q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    @property
    def busy(self) -> bool:
        return self._in_q.unfinished_tasks > 0 or not self._out_q.empty()

    def submit(self, *args: Any, **kwargs: Any) -> bool:
        """Zastąp zaległe zadanie najnowszym. Zwraca False gdy worker zatrzymany."""
        if self._stop.is_set():
            return False
        try:
            while True:
                self._in_q.get_nowait()
                self._in_q.task_done()
        except queue.Empty:
            pass
        try:
            self._in_q.put_nowait((args, kwargs))
            return True
        except queue.Full:
            return False

    def poll_result(self) -> Any:
        try:
            return self._out_q.get_nowait()
        except queue.Empty:
            return None

    def poll_error(self) -> Optional[BaseException]:
        err = self._err
        self._err = None
        return err

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._in_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                self._in_q.task_done()
                break
            args, kwargs = item
            try:
                result = self._fn(*args, **kwargs)
            except Exception as exc:
                self._err = exc
                self._in_q.task_done()
                continue
            try:
                while True:
                    self._out_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._out_q.put_nowait(result)
            except queue.Full:
                pass
            self._in_q.task_done()


def limit_blas_threads() -> None:
    """Jeden wątek BLAS/OpenMP na worker — unika walki wielu rdzeni o ten sam job."""
    import os

    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(key, '1')
