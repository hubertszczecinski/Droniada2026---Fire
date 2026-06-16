"""Źródło kamery V4L2 (Mac / Jetson USB)."""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

DeviceRef = Union[int, str]


def _device_to_v4l2_path(device: DeviceRef) -> Optional[str]:
    if isinstance(device, str) and device.startswith('/dev/video'):
        return device
    if isinstance(device, int):
        return f'/dev/video{int(device)}'
    return None


def apply_v4l2_controls(
    device: DeviceRef,
    *,
    brightness: Optional[int] = None,
    extra: Optional[List[str]] = None,
) -> bool:
    """
    UVC na Jetsonie: fabryczne brightness=-11 daje mean≈0 (czarny podgląd).
    Ustaw jasność przed capture (v4l2-ctl).
    """
    path = _device_to_v4l2_path(device)
    if path is None or not os.path.exists(path):
        return False
    applied: List[str] = []
    if brightness is not None:
        subprocess.run(
            ['v4l2-ctl', '-d', path, f'--set-ctrl=brightness={int(brightness)}'],
            check=False,
            capture_output=True,
        )
        applied.append(f'brightness={int(brightness)}')
    for token in extra or []:
        if '=' not in token:
            continue
        subprocess.run(
            ['v4l2-ctl', '-d', path, f'--set-ctrl={token}'],
            check=False,
            capture_output=True,
        )
        applied.append(token)
    return bool(applied)


@dataclass
class CameraConfig:
    device: DeviceRef = 1
    width: int = 0
    height: int = 0
    fourcc: Optional[str] = None
    warmup_frames: int = 15
    use_v4l2: bool = True
    v4l2_brightness: Optional[int] = None
    gstreamer_pipeline: Optional[str] = None

    @property
    def device_index(self) -> int:
        if isinstance(self.device, int):
            return self.device
        if isinstance(self.device, str) and self.device.startswith('/dev/video'):
            try:
                return int(self.device.replace('/dev/video', ''))
            except ValueError:
                pass
        return 0


def camera_config_from_env(
    *,
    default_device: DeviceRef = 1,
    default_width: int = 0,
    default_height: int = 0,
) -> CameraConfig:
    """Konfiguracja z env (Jetson Docker): DRONIADA_CAMERA_DEVICE, FOURCC, WIDTH, HEIGHT."""
    dev_raw = os.environ.get('DRONIADA_CAMERA_DEVICE', '').strip()
    if dev_raw:
        device: DeviceRef = dev_raw if dev_raw.startswith('/dev') else int(dev_raw)
    elif os.environ.get('DRONIADA_CAMERA', '').strip() != '':
        device = int(os.environ['DRONIADA_CAMERA'])
    else:
        device = default_device

    fourcc = os.environ.get('DRONIADA_CAMERA_FOURCC', '').strip() or None
    w = int(os.environ.get('DRONIADA_CAMERA_WIDTH', '0') or 0) or default_width
    h = int(os.environ.get('DRONIADA_CAMERA_HEIGHT', '0') or 0) or default_height
    warmup = int(os.environ.get('DRONIADA_CAMERA_WARMUP', '15') or 15)
    br = os.environ.get('DRONIADA_CAMERA_BRIGHTNESS', '').strip()
    v4l2_brightness = int(br) if br != '' else None
    if v4l2_brightness is None and sys.platform.startswith('linux'):
        # USB UVC na Jetsonie: domyślnie -11 → prawie czarny obraz przy MJPEG.
        v4l2_brightness = 50
    gst = os.environ.get('DRONIADA_CAP_PIPELINE', '').strip() or None
    return CameraConfig(
        device=device,
        width=w,
        height=h,
        fourcc=fourcc,
        warmup_frames=warmup,
        use_v4l2=sys.platform.startswith('linux'),
        v4l2_brightness=v4l2_brightness,
        gstreamer_pipeline=gst,
    )


class CameraSource:
    __slots__ = ('cfg', '_cap', '_frame_idx', '_last_bgr', '_open_meta', '_read_fails')

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx = 0
        self._last_bgr: Optional[np.ndarray] = None
        self._open_meta: dict = {}
        self._read_fails = 0

    @property
    def open_meta(self) -> dict:
        return dict(self._open_meta)

    def open(self) -> None:
        from release.opencv_gst import open_video_capture

        if self.cfg.gstreamer_pipeline:
            self._cap = open_video_capture(gstreamer_pipeline=self.cfg.gstreamer_pipeline)
        else:
            if self.cfg.v4l2_brightness is not None:
                apply_v4l2_controls(self.cfg.device, brightness=self.cfg.v4l2_brightness)
            self._cap = open_video_capture(
                device=self.cfg.device,
                use_v4l2=self.cfg.use_v4l2,
            )

        if not self.cfg.gstreamer_pipeline:
            if self.cfg.fourcc:
                fc = self.cfg.fourcc.strip().upper()[:4]
                self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fc))
            if self.cfg.width > 0:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.cfg.width))
            if self.cfg.height > 0:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.cfg.height))
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        aw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        last_mean = -1.0
        last_std = -1.0
        for _ in range(max(1, self.cfg.warmup_frames)):
            ok, frame = self._cap.read()
            if ok and frame is not None:
                self._last_bgr = frame
                last_mean = float(frame.mean())
                last_std = float(frame.std())

        self._open_meta = {
            'device': self.cfg.device,
            'gstreamer_pipeline': self.cfg.gstreamer_pipeline,
            'fourcc': self.cfg.fourcc,
            'width': aw,
            'height': ah,
            'v4l2_brightness': self.cfg.v4l2_brightness,
            'first_frame_mean': last_mean,
            'first_frame_std': last_std,
        }
        if last_mean >= 0 and last_mean < 15.0:
            import warnings
            warnings.warn(
                f'Kamera {self.cfg.device!r}: ciemny obraz (mean={last_mean:.1f}, std={last_std:.1f}). '
                f'Podnieś DRONIADA_CAMERA_BRIGHTNESS (np. 60–100) lub uruchom: '
                f'python3 -m release.camera_probe --suggest-brightness',
                stacklevel=2,
            )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _reopen_capture(self) -> bool:
        """Jetson UVC: po close() V4L2 potrzebuje chwili zanim znów przyjmie REQBUFS."""
        import time

        self.close()
        for attempt in range(5):
            time.sleep(0.15 * (attempt + 1))
            try:
                self.open()
                return self._cap is not None
            except Exception:
                self.close()
        return False

    def read(self) -> Tuple[bool, np.ndarray, str]:
        if self._cap is None:
            if not self._reopen_capture():
                fid = f'live_{self._frame_idx + 1:06d}'
                if self._last_bgr is not None:
                    return (False, self._last_bgr, fid)
                return (False, np.zeros((480, 640, 3), dtype=np.uint8), fid)
        ok, frame = self._cap.read()
        self._frame_idx += 1
        fid = f'live_{self._frame_idx:06d}'
        if not ok or frame is None:
            self._read_fails += 1
            if self._read_fails >= 8:
                if self._reopen_capture():
                    self._read_fails = 0
                    ok, frame = self._cap.read()
                else:
                    self._read_fails = 0
            if not ok or frame is None:
                if self._last_bgr is not None:
                    return (False, self._last_bgr, fid)
                return (False, np.zeros((480, 640, 3), dtype=np.uint8), fid)
        self._read_fails = 0
        self._last_bgr = frame
        return (True, frame, fid)

    def __enter__(self) -> 'CameraSource':
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
