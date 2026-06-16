"""
WebSocket — integracja z orchestratorem autonomii.

- ``DRONIADA_WS_URL`` — wysyłka ``speed`` do orchestratora (outbound).
- ``DRONIADA_AUTONOMY_WS_URL`` — odbiór zdarzeń ``hold_started`` / ``hold_stopped``.

Host autonomii jest *edge-triggered*: jedna wiadomość na start/stop holdu, bez
strumienia odliczania. Klient utrzymuje lokalny timer (``timeout`` z
``hold_started``) i anuluje go przy ``hold_stopped`` lub preempcji.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

AUTONOMY_EVENTS = frozenset({'hold_started', 'hold_stopped', 'hold_expired'})


def hold_duration_s(timeout_from_host: Optional[float] = None) -> float:
    """Czas trwania holdu z hosta (lokalny timer śledzący hold_started)."""
    if timeout_from_host is not None:
        try:
            return max(0.0, float(timeout_from_host))
        except (TypeError, ValueError):
            pass
    return 8.0


def autonomy_analysis_pause_s() -> float:
    """Pauza analizy po ``hold_stopped`` (env ``DRONIADA_AUTONOMY_HOLD_PAUSE_S``, domyślnie 8 s)."""
    raw = os.environ.get('DRONIADA_AUTONOMY_HOLD_PAUSE_S', '').strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return 8.0


def autonomy_hold_pause_s(timeout_from_host: Optional[float] = None) -> float:
    """Alias wsteczny — patrz ``autonomy_analysis_pause_s`` / ``hold_duration_s``."""
    if timeout_from_host is not None:
        return hold_duration_s(timeout_from_host)
    return autonomy_analysis_pause_s()


def parse_autonomy_event(raw: str) -> Optional[Dict[str, Any]]:
    """Parsuj wiadomość JSON z hosta autonomii (hold_started / hold_stopped)."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    event = str(data.get('event', '')).strip()
    if event not in {'hold_started', 'hold_stopped'}:
        return None
    out: Dict[str, Any] = {'event': event}
    if event == 'hold_started':
        try:
            out['timeout'] = float(data.get('timeout', hold_duration_s()))
        except (TypeError, ValueError):
            out['timeout'] = hold_duration_s()
    else:
        try:
            out['timeout'] = float(data.get('timeout', 0.0))
        except (TypeError, ValueError):
            out['timeout'] = 0.0
    return out


def _corners_screen_size(
    corners_px: Optional[np.ndarray],
    image_size: tuple[int, int],
) -> Optional[Dict[str, float]]:
    if corners_px is None:
        return None
    h, w = int(image_size[0]), int(image_size[1])
    if w <= 0 or h <= 0:
        return None
    pts = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 4:
        return None
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return {
        'x': float(max(0.0, min(1.0, (x1 - x0) / w))),
        'y': float(max(0.0, min(1.0, (y1 - y0) / h))),
    }


def _predictions_to_processed(preds: Optional[List[dict]]) -> Optional[List[Dict[str, Any]]]:
    if not preds:
        return None
    out: List[Dict[str, Any]] = []
    for p in preds:
        out.append({
            'grid_row': int(p.get('grid_row', 0)),
            'grid_col': int(p.get('grid_col', 0)),
            'color': str(p.get('color', '')),
        })
    return out


def build_speed_payload(speed: float) -> Dict[str, float]:
    """Jedyny parametr dla orchestratora lotu: 1 = stój, 0 = pełna prędkość."""
    s = float(speed)
    s = max(0.0, min(1.0, s))
    return {'speed': round(s, 3)}


def build_frame_payload(
    *,
    frame_id: str,
    panel_id: str,
    image_size: tuple[int, int],
    module_a_po: Any = None,
    corners_px: Optional[np.ndarray] = None,
    live_preds: Optional[List[dict]] = None,
    latched_preds: Optional[List[dict]] = None,
    latch_locked: bool = False,
    reliable: bool = False,
    track_id: Optional[str] = None,
) -> Dict[str, Any]:
    roll = pitch = yaw = 0.0
    dist = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    if module_a_po is not None and getattr(module_a_po, 'ok', False):
        roll = float(getattr(module_a_po, 'roll_deg', 0.0))
        pitch = float(getattr(module_a_po, 'pitch_deg', 0.0))
        yaw = float(getattr(module_a_po, 'yaw_deg', 0.0))
        d_m = float(getattr(module_a_po, 'distance_m', 0.0))
        dist = {'x': 0.0, 'y': 0.0, 'z': d_m}
    size = _corners_screen_size(corners_px, image_size)
    preds_use = latched_preds if latch_locked and latched_preds else live_preds
    processed = _predictions_to_processed(preds_use) if (reliable or latch_locked) else None
    tid = track_id or f'panel_{panel_id}'
    return {
        'frame_id': str(frame_id),
        'timestamp_ms': int(time.time() * 1000),
        'panels': [
            {
                'track_id': tid,
                'panel_id': str(panel_id),
                'orientation': {'roll': roll, 'pitch': pitch, 'yaw': yaw},
                'distance': dist,
                'size': size,
                'processed': processed,
                'meta': {
                    'reliable': bool(reliable),
                    'latch_locked': bool(latch_locked),
                },
            },
        ],
    }


class IntegrationWsPublisher:
    def __init__(self, url: str) -> None:
        self.url = url.strip()
        self._q: queue.Queue = queue.Queue(maxsize=32)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='droniada-ws', daemon=True)
        self._thread.start()

    @staticmethod
    def from_env() -> Optional['IntegrationWsPublisher']:
        url = os.environ.get('DRONIADA_WS_URL', '').strip()
        if not url:
            return None
        return IntegrationWsPublisher(url)

    def publish(self, payload: Dict[str, Any]) -> None:
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(payload)
            except queue.Full:
                pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        try:
            import websocket
        except ImportError:
            return
        backoff = 2.0
        while not self._stop.is_set():
            ws = None
            try:
                ws = websocket.create_connection(self.url, timeout=10)
                backoff = 2.0
                while not self._stop.is_set():
                    try:
                        item = self._q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    ws.send(json.dumps(item, ensure_ascii=False))
            except Exception:
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 30.0)
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass


class IntegrationWsSubscriber:
    """Klient WS — edge-triggered hold_started / hold_stopped + lokalny timer."""

    def __init__(self, url: str) -> None:
        self.url = url.strip()
        self._q: queue.Queue = queue.Queue(maxsize=16)
        self._stop = threading.Event()
        self._timer: Optional[threading.Timer] = None
        self._timer_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name='droniada-autonomy-ws',
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def from_env() -> Optional['IntegrationWsSubscriber']:
        url = os.environ.get('DRONIADA_AUTONOMY_WS_URL', '').strip()
        if not url:
            return None
        return IntegrationWsSubscriber(url)

    def poll(self) -> Optional[Dict[str, Any]]:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        self._cancel_local_timer()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def _enqueue(self, event: Dict[str, Any]) -> None:
        try:
            self._q.put_nowait(event)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(event)
            except queue.Full:
                pass

    def _cancel_local_timer(self) -> None:
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def _start_local_timer(self, timeout_s: float) -> None:
        self._cancel_local_timer()
        delay = max(0.0, float(timeout_s))

        def _on_expire() -> None:
            with self._timer_lock:
                self._timer = None
            self._enqueue({'event': 'hold_expired', 'timeout': 0.0})

        with self._timer_lock:
            self._timer = threading.Timer(delay, _on_expire)
            self._timer.daemon = True
            self._timer.start()

    def _handle_hold_started(self, timeout_s: float) -> None:
        duration_s = hold_duration_s(timeout_s)
        self._cancel_local_timer()
        self._enqueue({'event': 'hold_started', 'timeout': duration_s})
        self._start_local_timer(duration_s)

    def _handle_hold_stopped(self, *, reason: str = 'server') -> None:
        had_timer = False
        with self._timer_lock:
            had_timer = self._timer is not None
        self._cancel_local_timer()
        if reason == 'disconnect' and not had_timer:
            return
        self._enqueue({
            'event': 'hold_stopped',
            'timeout': 0.0,
            'reason': reason,
        })

    def _run(self) -> None:
        try:
            import websocket
        except ImportError:
            return
        backoff = 2.0
        while not self._stop.is_set():
            ws = None
            try:
                ws = websocket.create_connection(self.url, timeout=10)
                ws.settimeout(1.0)
                backoff = 2.0
                while not self._stop.is_set():
                    try:
                        raw = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not raw:
                        continue
                    event = parse_autonomy_event(raw)
                    if event is None:
                        continue
                    name = str(event.get('event', ''))
                    if name == 'hold_started':
                        self._handle_hold_started(float(event.get('timeout', 8.0)))
                    elif name == 'hold_stopped':
                        self._handle_hold_stopped(reason='server')
            except Exception:
                self._handle_hold_stopped(reason='disconnect')
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 30.0)
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
