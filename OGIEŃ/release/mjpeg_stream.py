"""Prosty serwer HTTP z podglądem MJPEG (multipart/x-mixed-replace) — bez X11."""
from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

import cv2
import numpy as np

_BOUNDARY = b'--droniadaframe'
_INDEX_HTML = """<!DOCTYPE html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Droniada live</title>
  <style>
    body { margin: 0; background: #111; color: #ccc; font-family: system-ui, sans-serif; }
    header { padding: 0.5rem 1rem; font-size: 0.9rem; }
    img { display: block; max-width: 100%; height: auto; margin: 0 auto; }
  </style>
</head>
<body>
  <header>Droniada &mdash; podglad na zywo (<code>/stream.mjpg</code>)</header>
  <img src="/stream.mjpg" alt="live stream">
</body>
</html>
""".encode('utf-8')


class MjpegStreamServer:
    """Wątek w tle: GET / → strona, GET /stream.mjpg → strumień JPEG."""

    def __init__(self, host: str = '0.0.0.0', port: int = 8088) -> None:
        self.host = host
        self.port = int(port)
        self._jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._running = True
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def push_jpeg_bytes(self, data: bytes) -> None:
        """Surowy JPEG z kamery (GStreamer passthrough) — bez cv2.imencode."""
        if not data:
            return
        with self._cv:
            self._jpeg = data
            self._cv.notify_all()

    def push_frame(self, bgr: np.ndarray, *, quality: int = 85) -> None:
        if bgr is None or bgr.size == 0:
            return
        ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return
        self.push_jpeg_bytes(buf.tobytes())

    def _wait_frame(self, timeout: float) -> Optional[bytes]:
        with self._cv:
            if self._jpeg is None:
                self._cv.wait(timeout=timeout)
            return self._jpeg

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                pass

            def do_GET(self) -> None:
                if self.path in ('/', '/index.html'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(_INDEX_HTML)))
                    self.end_headers()
                    self.wfile.write(_INDEX_HTML)
                    return

                if self.path.startswith('/stream.mjpg'):
                    self.send_response(200)
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Connection', 'close')
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=droniadaframe')
                    self.end_headers()
                    try:
                        while server._running:
                            jpeg = server._wait_frame(0.05)
                            if jpeg is None:
                                continue
                            header = (
                                _BOUNDARY + b'\r\n'
                                b'Content-Type: image/jpeg\r\n'
                                + f'Content-Length: {len(jpeg)}\r\n\r\n'.encode('ascii')
                            )
                            self.wfile.write(header)
                            self.wfile.write(jpeg)
                            self.wfile.write(b'\r\n')
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return

                self.send_response(404)
                self.end_headers()

        return Handler

    def start_background(self) -> None:
        if self._thread is not None:
            return
        handler = self._make_handler()
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, name='mjpeg-http', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        with self._cv:
            self._cv.notify_all()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def url(self) -> str:
        host = self.host if self.host not in ('0.0.0.0', '::') else 'localhost'
        return f'http://{host}:{self.port}/'


def start_mjpeg_stream(host: str, port: int, log: Optional[Callable[[str], None]] = None) -> MjpegStreamServer:
    srv = MjpegStreamServer(host=host, port=port)
    srv.start_background()
    if log is not None:
        log(f'[mjpeg] podgląd: {srv.url()} (stream: {srv.url()}stream.mjpg)')
    return srv
