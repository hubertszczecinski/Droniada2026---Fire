#!/usr/bin/env python3
"""Podgląd kamery (MJPEG :8087) + migawki — ten sam tor co panel :8088 (OpenCV V4L2).

GStreamer+obrót w Dockerze headless daje szary obraz (brak EGL) — patrz jetson_competition_start.sh.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import cv2
import numpy as np

from release.camera_feed_thread import SharedCameraFeed
from release.camera_source import CameraConfig, apply_v4l2_controls
from release.mjpeg_stream import MjpegStreamServer, _BOUNDARY
from release.transform import apply_rotate

_DEFAULT_SNAPSHOT_EVENTS = frozenset({
    'hold_started',
    'hold_stopped',
    'snapshot',
    'trigger',
    'migawka',
})

_INDEX_HTML = """<!DOCTYPE html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Migawka woda — podgląd</title>
  <style>
    body { margin: 0; background: #0f1419; color: #c8d0d8; font-family: system-ui, sans-serif; }
    header { padding: 0.75rem 1rem; border-bottom: 1px solid #243040; font-size: 0.95rem; }
    header code { color: #7eb8ff; }
    .live { padding: 0.5rem 1rem 0; }
    .live img { display: block; max-width: 100%; height: auto; border-radius: 8px; background: #000; }
    .gallery { padding: 1rem; }
    .gallery h2 { font-size: 1rem; font-weight: 600; margin: 0 0 0.75rem; color: #e8eef4; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 0.75rem; }
    .card { background: #1a2330; border: 1px solid #2a3848; border-radius: 8px; overflow: hidden; }
    .card img { width: 100%; height: 120px; object-fit: cover; display: block; }
    .card .meta { padding: 0.4rem 0.5rem; font-size: 0.7rem; color: #8a9aaa; word-break: break-all; }
    .empty { color: #6a7a8a; font-size: 0.9rem; padding: 1rem 0; }
    .status { font-size: 0.8rem; color: #6a8a6a; margin-top: 0.25rem; }
  </style>
</head>
<body>
  <header>
  Migawka woda &mdash; podgląd na żywo (<code>/stream.mjpg</code>)
  <div class="status" id="status">Ładowanie migawek…</div>
  </header>
  <div class="live">
    <img src="/stream.mjpg" alt="live">
  </div>
  <div class="gallery">
    <h2>Zapisane migawki</h2>
    <div id="grid" class="grid"></div>
    <p id="empty" class="empty" style="display:none">Brak migawek — czekam na trigger WebSocket :8765</p>
  </div>
  <script>
    async function refresh() {
      try {
        const r = await fetch('/api/snapshots');
        const data = await r.json();
        const grid = document.getElementById('grid');
        const empty = document.getElementById('empty');
        const status = document.getElementById('status');
        grid.innerHTML = '';
        const items = data.snapshots || [];
        status.textContent = items.length
          ? `${items.length} migawka/migawek w /migawka-woda`
          : 'Brak migawek — trigger z WS :8765';
        if (!items.length) {
          empty.style.display = 'block';
          return;
        }
        empty.style.display = 'none';
        for (const s of items.slice().reverse()) {
          const card = document.createElement('a');
          card.className = 'card';
          card.href = '/snapshots/' + encodeURIComponent(s.name);
          card.target = '_blank';
          card.innerHTML = `<img src="/snapshots/${encodeURIComponent(s.name)}" alt=""><div class="meta">${s.name}</div>`;
          grid.appendChild(card);
        }
      } catch (e) {
        document.getElementById('status').textContent = 'Błąd ładowania listy migawek';
      }
    }
    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>
""".encode('utf-8')


def _setup_v4l2_format(device: str, width: int, height: int, fourcc: str) -> None:
    if not device.startswith('/dev/video'):
        return
    fmt = fourcc.strip().upper()[:4] or 'MJPG'
    subprocess.run(
        [
            'v4l2-ctl', '-d', device,
            f'--set-fmt-video=width={width},height={height},pixelformat={fmt}',
        ],
        check=False,
        capture_output=True,
    )


def _encode_stream_jpeg(bgr: np.ndarray, *, target_w: int = 960, quality: int = 78) -> Optional[bytes]:
    """Jak web_dashboard._encode_stream_jpeg — skala + JPEG dla /stream.mjpg."""
    if bgr is None or bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
    tw = max(480, int(target_w))
    if w > tw:
        scale = tw / float(w)
        img = cv2.resize(
            bgr,
            (tw, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        img = bgr
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


class WodaCameraPanel:
    def __init__(
        self,
        *,
        camera_device: str,
        output_dir: Path,
        http_host: str,
        http_port: int,
        ws_url: str,
        snapshot_events: frozenset[str],
        jpeg_quality: int,
        stream_width: int,
        width: int,
        height: int,
        rotate: int,
        fourcc: Optional[str],
        brightness: Optional[int],
        warmup_frames: int,
    ) -> None:
        self.camera_device = camera_device
        self.output_dir = output_dir
        self.http_host = http_host
        self.http_port = int(http_port)
        self.ws_url = ws_url.strip()
        self.snapshot_events = snapshot_events
        self.snapshot_all = snapshot_events == frozenset({'__all__'})
        self.jpeg_quality = int(jpeg_quality)
        self.stream_width = int(stream_width)
        self.width = int(width)
        self.height = int(height)
        self.rotate = int(rotate)
        self.fourcc = (fourcc or 'MJPG').strip().upper()[:4]
        self.brightness = brightness
        _setup_v4l2_format(camera_device, self.width, self.height, self.fourcc)
        if brightness is not None:
            apply_v4l2_controls(camera_device, brightness=brightness)
        self._camera_cfg = CameraConfig(
            device=camera_device,
            width=width,
            height=height,
            fourcc=self.fourcc,
            warmup_frames=max(1, int(warmup_frames)),
            v4l2_brightness=brightness,
        )
        self._feed: Optional[SharedCameraFeed] = None
        self._frame_lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._stop = threading.Event()
        self._mjpeg = MjpegStreamServer(host=http_host, port=http_port)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        self._snapshot_lock = threading.Lock()

    def _on_frame(self, ok: bool, bgr: Optional[np.ndarray], _fid: str) -> None:
        if not ok or bgr is None:
            return
        frame = apply_rotate(bgr, self.rotate) if self.rotate else bgr
        with self._frame_lock:
            self._latest = frame.copy()
        data = _encode_stream_jpeg(frame, target_w=self.stream_width, quality=self.jpeg_quality)
        if data:
            self._mjpeg.push_jpeg_bytes(data)

    def _start_camera(self) -> bool:
        if self._feed is not None:
            return True
        self._feed = SharedCameraFeed(self._camera_cfg, on_frame=self._on_frame)
        self._feed.start()
        for _ in range(80):
            ok, bgr, _ = self._feed.get_latest()
            if ok and bgr is not None and float(bgr.std()) > 8.0:
                meta = self._feed.open_meta
                print(
                    f'[woda] OpenCV V4L2: {meta.get("device")} '
                    f'{meta.get("width")}x{meta.get("height")} fourcc={meta.get("fourcc")} '
                    f'mean={meta.get("first_frame_mean"):.1f} std={meta.get("first_frame_std"):.1f}',
                    flush=True,
                )
                return True
            time.sleep(0.05)
        meta = self._feed.open_meta if self._feed else {}
        print(
            f'[woda] UWAGA: słaby sygnał kamery (std={meta.get("first_frame_std", 0):.1f}) — '
            'sprawdź HDMI / brightness',
            flush=True,
        )
        return True

    def _list_snapshots(self) -> list[dict]:
        if not self.output_dir.is_dir():
            return []
        files = sorted(self.output_dir.glob('snapshot_*.jpg'), key=lambda p: p.stat().st_mtime)
        return [{'name': p.name, 'mtime': p.stat().st_mtime} for p in files]

    def _save_snapshot(self, trigger: str) -> Optional[Path]:
        with self._frame_lock:
            if self._latest is None:
                print('[woda] migawka pominięta — brak klatki', flush=True)
                return None
            frame = self._latest.copy()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).astimezone()
        stamp = now.strftime('%Y%m%d_%H%M%S')
        ms = int(now.microsecond / 1000)
        with self._snapshot_lock:
            name = f'snapshot_{stamp}_{ms:03d}.jpg'
            path = self.output_dir / name
        ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return None
        print(
            f'[woda] migawka ({trigger}): {path} mean={float(frame.mean()):.1f} std={float(frame.std()):.1f}',
            flush=True,
        )
        return path

    def _should_snapshot(self, raw: str) -> tuple[bool, str]:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return self.snapshot_all, 'raw'
        if not isinstance(data, dict):
            return self.snapshot_all, 'raw'
        event = str(data.get('event', '')).strip().lower()
        if not event:
            return self.snapshot_all, 'raw'
        if self.snapshot_all or event in self.snapshot_events:
            return True, event
        return False, event

    def _ws_loop(self) -> None:
        if not self.ws_url:
            return
        try:
            import websocket
        except ImportError:
            return
        backoff = 2.0
        while not self._stop.is_set():
            ws = None
            try:
                ws = websocket.create_connection(self.ws_url, timeout=10)
                ws.settimeout(1.0)
                backoff = 2.0
                while not self._stop.is_set():
                    try:
                        raw = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not raw:
                        continue
                    text = raw.decode('utf-8') if isinstance(raw, bytes) else str(raw)
                    should, label = self._should_snapshot(text)
                    if should:
                        self._save_snapshot(label)
            except Exception as exc:
                print(f'[woda] WebSocket: {exc} — retry {backoff:.0f}s', flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 30.0)
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                pass

            def _send_json(self, code: int, payload: dict) -> None:
                body = json.dumps(payload).encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                path = self.path.split('?', 1)[0]
                if path in ('/', '/index.html'):
                    self.send_response(HTTPStatus.OK)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(_INDEX_HTML)))
                    self.end_headers()
                    self.wfile.write(_INDEX_HTML)
                    return
                if path == '/api/snapshots':
                    self._send_json(HTTPStatus.OK, {'snapshots': panel._list_snapshots()})
                    return
                if path.startswith('/snapshots/'):
                    name = unquote(path[len('/snapshots/'):])
                    if '..' in name or '/' in name:
                        self.send_error(HTTPStatus.BAD_REQUEST)
                        return
                    file_path = panel.output_dir / name
                    if not file_path.is_file():
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    data = file_path.read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if path.startswith('/stream.mjpg'):
                    self.send_response(HTTPStatus.OK)
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Connection', 'close')
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=droniadaframe')
                    self.end_headers()
                    try:
                        while panel._mjpeg._running and not panel._stop.is_set():
                            jpeg = panel._mjpeg._wait_frame(0.05)
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
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                if self.path.split('?', 1)[0] != '/api/snapshot':
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                path = panel._save_snapshot('http')
                self._send_json(
                    HTTPStatus.OK,
                    {'success': path is not None, 'path': str(path) if path else None},
                )

        return Handler

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        handler = self._make_handler()
        self._httpd = ThreadingHTTPServer((self.http_host, self.http_port), handler)
        self._http_thread = threading.Thread(target=self._httpd.serve_forever, name='woda-http', daemon=True)
        self._http_thread.start()
        print(f'[woda] podgląd: http://{self.http_host}:{self.http_port}/', flush=True)
        print(f'[woda] zapis: {self.output_dir}', flush=True)
        if not self._start_camera():
            print('[woda] kamera niegotowa', flush=True)
        if self.ws_url:
            print(f'[woda] WebSocket: {self.ws_url}', flush=True)
            threading.Thread(target=self._ws_loop, name='woda-ws', daemon=True).start()

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        self._mjpeg.stop()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._feed is not None:
            self._feed.stop()
            self._feed = None


def main() -> None:
    ap = argparse.ArgumentParser(description='Podgląd kamery woda + migawki WS')
    ap.add_argument('--camera', default=os.environ.get('WODA_CAMERA_DEVICE', '/dev/video0'))
    ap.add_argument('--output-dir', default=os.environ.get('WODA_OUTPUT_DIR', '/migawka-woda'))
    ap.add_argument('--host', default=os.environ.get('WODA_HTTP_BIND', '0.0.0.0'))
    ap.add_argument('--port', type=int, default=int(os.environ.get('WODA_HTTP_PORT', '8087')))
    ap.add_argument('--ws-url', default=os.environ.get('WODA_WS_URL', 'ws://127.0.0.1:8765'))
    ap.add_argument(
        '--snapshot-on',
        default=os.environ.get('WODA_SNAPSHOT_ON', 'hold_started,hold_stopped,snapshot,trigger,migawka'),
    )
    ap.add_argument('--width', type=int, default=int(os.environ.get('WODA_CAMERA_WIDTH', '1920')))
    ap.add_argument('--height', type=int, default=int(os.environ.get('WODA_CAMERA_HEIGHT', '1080')))
    ap.add_argument('--rotate', type=int, default=int(os.environ.get('WODA_ROTATE', '180')))
    ap.add_argument('--fourcc', default=os.environ.get('WODA_CAMERA_FOURCC', 'MJPG'))
    ap.add_argument('--brightness', type=int, default=int(os.environ.get('WODA_CAMERA_BRIGHTNESS', '60')))
    ap.add_argument('--warmup-frames', type=int, default=int(os.environ.get('WODA_CAMERA_WARMUP', '15')))
    ap.add_argument('--jpeg-quality', type=int, default=int(os.environ.get('WODA_JPEG_QUALITY', '78')))
    ap.add_argument('--stream-width', type=int, default=int(os.environ.get('WODA_STREAM_WIDTH', '960')))
    args = ap.parse_args()

    events = {e.strip().lower() for e in args.snapshot_on.split(',') if e.strip()}
    if 'all' in events:
        snapshot_events: frozenset[str] = frozenset({'__all__'})
    else:
        snapshot_events = frozenset(events) if events else _DEFAULT_SNAPSHOT_EVENTS

    WodaCameraPanel(
        camera_device=args.camera,
        output_dir=Path(args.output_dir),
        http_host=args.host,
        http_port=args.port,
        ws_url=args.ws_url,
        snapshot_events=snapshot_events,
        jpeg_quality=args.jpeg_quality,
        stream_width=args.stream_width,
        width=args.width,
        height=args.height,
        rotate=args.rotate,
        fourcc=args.fourcc.strip() or None,
        brightness=args.brightness,
        warmup_frames=args.warmup_frames,
    ).run_forever()


if __name__ == '__main__':
    main()
