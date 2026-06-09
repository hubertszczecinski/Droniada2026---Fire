"""Wybór urządzenia inferencji (CPU/CUDA) — Jetson / desktop."""
from __future__ import annotations

import os


def resolve_yolo_device() -> str | int:
    """Zwraca device dla ultralytics: 0 / 'cpu' / wartość z DRONIADA_DEVICE."""
    raw = os.environ.get('DRONIADA_DEVICE', '').strip()
    if raw:
        if raw.lower() in ('cpu', 'cuda'):
            return raw.lower()
        try:
            return int(raw)
        except ValueError:
            return raw
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except ImportError:
        pass
    return 'cpu'


def device_label(device: str | int) -> str:
    if device == 'cpu':
        return 'cpu'
    try:
        import torch

        if torch.cuda.is_available():
            idx = int(device) if str(device).isdigit() else 0
            return f'cuda:{idx} ({torch.cuda.get_device_name(idx)})'
    except Exception:
        pass
    return str(device)
