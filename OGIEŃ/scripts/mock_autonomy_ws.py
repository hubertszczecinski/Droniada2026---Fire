#!/usr/bin/env python3
"""Mock hosta autonomii — wysyła hold_started / hold_stopped do podłączonych klientów.

Użycie:
  python3 scripts/mock_autonomy_ws.py --port 8765
  # w drugim terminalu:
  DRONIADA_AUTONOMY_WS_URL=ws://127.0.0.1:8765 ./scripts/run_local_video_test.sh ...

W konsoli mocka: Enter = hold_started (zbieraj migawki), s = hold_stopped (raport+pauza), q = quit
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def main() -> None:
    ap = argparse.ArgumentParser(description='Mock WebSocket autonomii (hold events)')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--timeout', type=float, default=8.0, help='timeout w hold_started')
    args = ap.parse_args()

    clients: set = set()

    async def handler(ws):  # type: ignore[no-untyped-def]
        clients.add(ws)
        try:
            async for _ in ws:
                pass
        finally:
            clients.discard(ws)

    try:
        import websockets
    except ImportError:
        print('Zainstaluj: pip install websockets', file=sys.stderr)
        raise SystemExit(1)

    async with websockets.serve(handler, args.host, args.port):
        print(f'Mock autonomii: ws://{args.host}:{args.port}')
        print('Enter=hold_started  s=hold_stopped  q=quit')
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            cmd = line.strip().lower()
            if cmd in ('q', 'quit'):
                break
            if cmd in ('s', 'stop'):
                payload = {'event': 'hold_stopped', 'timeout': 0.0}
            else:
                payload = {'event': 'hold_started', 'timeout': float(args.timeout)}
            msg = json.dumps(payload)
            dead = []
            for ws in list(clients):
                try:
                    await ws.send(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                clients.discard(ws)
            print(f'→ {msg}  (klientów: {len(clients)})')


if __name__ == '__main__':
    asyncio.run(main())
