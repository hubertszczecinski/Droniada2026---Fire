"""GStreamer: jeden v4l2src → tee(MJPEG na stream | jpegdec→BGR dla YOLO).

Na Jetsonie (L4T) preferuj ``nvjpegdec`` / ``nvjpegenc`` zamiast CPU ``jpegdec``/``cv2.imencode``.
"""
from __future__ import annotations

import os
import platform
import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np

MjpegCallback = Callable[[bytes], None]
BgrCallback = Callable[[np.ndarray], None]

_FLIP_METHOD = {
    90: 'clockwise',
    180: 'rotate-180',
    270: 'counterclockwise',
}

_GST_PLUGINS_PROBE: Optional[dict] = None


def _gst_available() -> bool:
    try:
        import gi  # noqa: F401
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst  # noqa: F401
        return True
    except Exception:
        return False


def _probe_gst_plugins() -> dict:
    global _GST_PLUGINS_PROBE
    if _GST_PLUGINS_PROBE is not None:
        return _GST_PLUGINS_PROBE
    found: dict = {'nvjpegdec': False, 'nvjpegenc': False, 'nvvidconv': False, 'jpegdec': False}
    if not _gst_available():
        _GST_PLUGINS_PROBE = found
        return found
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst

        Gst.init(None)
        for name in list(found.keys()):
            found[name] = Gst.ElementFactory.find(name) is not None
    except Exception:
        pass
    _GST_PLUGINS_PROBE = found
    return found


def use_hw_jpeg() -> bool:
    """Sprzętowy MJPEG Jetson (nvjpeg*). Wyłącz: DRONIADA_GST_HW_JPEG=0."""
    raw = os.environ.get('DRONIADA_GST_HW_JPEG', '').strip().lower()
    if raw in ('0', 'false', 'no', 'off'):
        return False
    if raw in ('1', 'true', 'yes', 'on'):
        return True
    if platform.machine().lower() not in ('aarch64', 'arm64'):
        return False
    plugs = _probe_gst_plugins()
    return bool(plugs.get('nvjpegdec'))


def _jpeg_decode_to_bgr_tail(*, flip: Optional[str], hw: bool) -> str:
    plugs = _probe_gst_plugins()
    if hw and plugs.get('nvjpegdec'):
        dec = 'nvjpegdec'
        mid = 'nvvidconv' if plugs.get('nvvidconv') else 'videoconvert'
        if flip:
            return (
                f'{dec} ! {mid} ! videoflip method={flip} ! '
                'videoconvert ! video/x-raw,format=BGR ! '
            )
        return f'{dec} ! {mid} ! videoconvert ! video/x-raw,format=BGR ! '
    if flip:
        return (
            f'jpegdec ! videoflip method={flip} ! '
            'videoconvert ! video/x-raw,format=BGR ! '
        )
    return 'jpegdec ! videoconvert ! video/x-raw,format=BGR ! '


def _mjpeg_reencode_tail(*, flip: str, hw: bool, quality: int = 85) -> str:
    """Obrót w strumieniu MJPEG — dekod→flip→enc (nvjpeg* na Jetsonie)."""
    plugs = _probe_gst_plugins()
    if hw and plugs.get('nvjpegdec') and plugs.get('nvjpegenc'):
        mid = 'nvvidconv' if plugs.get('nvvidconv') else 'videoconvert'
        return (
            f'nvjpegdec ! {mid} ! videoflip method={flip} ! nvjpegenc ! image/jpeg ! '
            'appsink name=mjpeg emit-signals=true max-buffers=1 drop=true'
        )
    return (
        f'jpegdec ! videoflip method={flip} ! jpegenc quality={int(quality)} ! image/jpeg ! '
        'appsink name=mjpeg emit-signals=true max-buffers=1 drop=true'
    )


class GstMjpegCameraFeed:
    """
    Jeden deskryptor V4L2:
    - gałąź MJPEG → surowe JPEG (passthrough) lub HW/SW re-enc przy obrocie
    - gałąź BGR → analiza YOLO (nvjpegdec na Jetsonie)
    """

    __slots__ = (
        '_device', '_width', '_height', '_fps', '_rotate', '_on_mjpeg', '_on_bgr',
        '_lock', '_latest', '_stop', '_thread', '_open_meta', '_pipeline',
        '_loop', '_hw_jpeg',
    )

    def __init__(
        self,
        device: str,
        *,
        width: int = 0,
        height: int = 0,
        fps: int = 30,
        rotate: int = 0,
        on_mjpeg: Optional[MjpegCallback] = None,
        on_bgr: Optional[BgrCallback] = None,
    ) -> None:
        self._device = device if str(device).startswith('/dev') else f'/dev/video{int(device)}'
        self._width = int(width)
        self._height = int(height)
        self._fps = max(1, int(fps))
        self._rotate = int(rotate or 0)
        self._on_mjpeg = on_mjpeg
        self._on_bgr = on_bgr
        self._hw_jpeg = use_hw_jpeg()
        self._lock = threading.Lock()
        self._latest: Tuple[bool, Optional[np.ndarray], str] = (
            False, None, 'live_000000',
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='droniada-gst-cam', daemon=True)
        self._pipeline = None
        self._loop = None
        flip = _FLIP_METHOD.get(self._rotate)
        self._open_meta = {
            'device': self._device,
            'fourcc': 'MJPG',
            'width': self._width,
            'height': self._height,
            'fps': self._fps,
            'rotate': self._rotate,
            'gstreamer_passthrough': flip is None,
            'gstreamer_hw_jpeg': self._hw_jpeg,
            'gstreamer_plugins': _probe_gst_plugins(),
            'backend': 'gstreamer_tee',
        }

    @property
    def open_meta(self) -> dict:
        return dict(self._open_meta)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        loop = self._loop
        if loop is not None:
            try:
                import gi
                gi.require_version('GLib', '2.0')
                from gi.repository import GLib
                GLib.idle_add(loop.quit)
            except Exception:
                pass
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def get_latest(self) -> Tuple[bool, Optional[np.ndarray], str]:
        with self._lock:
            ok, bgr, fid = self._latest
            if ok and bgr is not None:
                return True, bgr.copy(), fid
            return ok, bgr, fid

    def _build_pipeline(self) -> str:
        caps = 'image/jpeg'
        if self._width > 0 and self._height > 0:
            caps = (
                f'image/jpeg,width={self._width},height={self._height},'
                f'framerate={self._fps}/1'
            )
        flip = _FLIP_METHOD.get(self._rotate)
        hw = self._hw_jpeg
        if flip:
            mjpeg_tail = _mjpeg_reencode_tail(flip=flip, hw=hw)
        else:
            mjpeg_tail = (
                'jpegparse ! appsink name=mjpeg emit-signals=true '
                'max-buffers=1 drop=true'
            )
        bgr_decode = _jpeg_decode_to_bgr_tail(flip=flip, hw=hw)
        bgr_tail = (
            f'{bgr_decode}'
            'appsink name=bgr emit-signals=true max-buffers=1 drop=true'
        )
        return (
            f'v4l2src device={self._device} io-mode=2 ! {caps} ! tee name=t '
            f't. ! queue max-size-buffers=1 leaky=downstream ! {mjpeg_tail} '
            f't. ! queue max-size-buffers=1 leaky=downstream ! {bgr_tail}'
        )

    def _run(self) -> None:
        if not _gst_available():
            return
        import gi
        gi.require_version('Gst', '1.0')
        gi.require_version('GLib', '2.0')
        from gi.repository import Gst, GLib

        Gst.init(None)
        pipe = self._build_pipeline()
        self._open_meta['gstreamer_pipeline'] = pipe
        try:
            self._pipeline = Gst.parse_launch(pipe)
        except Exception as exc:
            self._open_meta['error'] = str(exc)
            return
        mjpeg_sink = self._pipeline.get_by_name('mjpeg')
        bgr_sink = self._pipeline.get_by_name('bgr')
        if mjpeg_sink is None or bgr_sink is None:
            self._open_meta['error'] = 'missing_appsink'
            return
        frame_idx = 0

        def on_mjpeg_sample(_sink) -> Gst.FlowReturn:
            if self._stop.is_set():
                return Gst.FlowReturn.EOS
            sample = _sink.emit('pull-sample')
            if sample is None:
                return Gst.FlowReturn.ERROR
            buf = sample.get_buffer()
            ok, info = buf.map(Gst.MapFlags.READ)
            if not ok:
                return Gst.FlowReturn.ERROR
            try:
                data = bytes(info.data)
            finally:
                buf.unmap(info)
            if self._on_mjpeg is not None and data:
                try:
                    self._on_mjpeg(data)
                except Exception:
                    pass
            return Gst.FlowReturn.OK

        def on_bgr_sample(_sink) -> Gst.FlowReturn:
            nonlocal frame_idx
            if self._stop.is_set():
                return Gst.FlowReturn.EOS
            sample = _sink.emit('pull-sample')
            if sample is None:
                return Gst.FlowReturn.ERROR
            buf = sample.get_buffer()
            caps_s = sample.get_caps()
            struct = caps_s.get_structure(0)
            w = int(struct.get_value('width'))
            h = int(struct.get_value('height'))
            ok, info = buf.map(Gst.MapFlags.READ)
            if not ok:
                return Gst.FlowReturn.ERROR
            try:
                arr = np.frombuffer(info.data, dtype=np.uint8)
                bgr = arr.reshape((h, w, 3)).copy()
            finally:
                buf.unmap(info)
            frame_idx += 1
            fid = f'live_{frame_idx:06d}'
            with self._lock:
                self._latest = (True, bgr, fid)
            if self._on_bgr is not None:
                try:
                    self._on_bgr(bgr)
                except Exception:
                    pass
            return Gst.FlowReturn.OK

        mjpeg_sink.connect('new-sample', on_mjpeg_sample)
        bgr_sink.connect('new-sample', on_bgr_sample)

        def on_bus_message(_bus, msg) -> bool:
            if msg.type == Gst.MessageType.ERROR:
                err, _ = msg.parse_error()
                self._open_meta['error'] = str(err)
                if self._loop is not None:
                    self._loop.quit()
            elif msg.type == Gst.MessageType.EOS and self._loop is not None:
                self._loop.quit()
            return True

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', on_bus_message)

        self._loop = GLib.MainLoop()
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._open_meta['error'] = 'state_playing_failed'
            return
        self._open_meta['running'] = True
        self._loop.run()
        self._pipeline.set_state(Gst.State.NULL)

    def __enter__(self) -> 'GstMjpegCameraFeed':
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


def should_use_gst_capture() -> bool:
    """GStreamer tee: jeden V4L2 → BGR (HW decode) + MJPEG passthrough."""
    raw = os.environ.get('DRONIADA_USE_GST_CAPTURE', '1').strip().lower()
    if raw in ('0', 'false', 'no', 'off'):
        return False
    fourcc = os.environ.get('DRONIADA_CAMERA_FOURCC', 'MJPG').strip().upper()
    if fourcc not in ('MJPG', 'MJPEG'):
        return False
    return _gst_available()


def wants_stream_passthrough() -> bool:
    """Surowy MJPEG z kamery → HTTP (bez cv2.imencode), gdy GStreamer capture włączony."""
    if os.environ.get('DRONIADA_STREAM_PASSTHROUGH', '1').strip().lower() in ('0', 'false', 'no'):
        return False
    return should_use_gst_capture()


def should_use_gst_passthrough() -> bool:
    """Alias — zachowanie wsteczne."""
    return wants_stream_passthrough()
