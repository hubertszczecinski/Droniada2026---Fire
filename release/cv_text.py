"""Tekst UTF-8 na obrazach OpenCV (cv2.putText nie obsługuje polskich znaków)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_FONT_CANDIDATES = (
    str(_ROOT / 'config' / 'fonts' / 'Arial.ttf'),
    str(_ROOT / 'config' / 'fonts' / 'DejaVuSans.ttf'),
    '/System/Library/Fonts/Supplemental/Arial.ttf',
    '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
    '/Library/Fonts/Arial.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/TTF/DejaVuSans.ttf',
    '/usr/share/fonts/dejavu/DejaVuSans.ttf',
)


def _scale_to_px(scale: float, thickness: int = 2) -> int:
    base = max(11, int(round(float(scale) * 24.0)))
    if int(thickness) >= 3:
        base += 2
    elif int(thickness) >= 2:
        base += 1
    return base


@lru_cache(maxsize=16)
def _load_font(size: int):
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, int(size))
            except OSError:
                continue
    raise RuntimeError(
        'Brak czcionki UTF-8 (Arial/DejaVu). Dodaj config/fonts/Arial.ttf do projektu.',
    )


def text_width_utf8(text: str, scale: float, thickness: int = 2) -> int:
    font = _load_font(_scale_to_px(scale, thickness))
    bbox = font.getbbox(str(text))
    return max(0, int(bbox[2] - bbox[0]))


def put_text_utf8(
    bgr: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int],
    *,
    scale: float = 0.55,
    thickness: int = 2,
) -> None:
    """Rysuje tekst na BGR (org = lewy-dolny punkt jak w cv2.putText)."""
    from PIL import Image, ImageDraw

    line = str(text)
    if not line or bgr is None or bgr.size == 0:
        return

    x, y = int(org[0]), int(org[1])
    font = _load_font(_scale_to_px(scale, thickness))
    ascent, descent = font.getmetrics()
    bbox = font.getbbox(line)
    text_w = max(1, int(bbox[2] - bbox[0]))
    text_h = max(1, int(bbox[3] - bbox[1]))
    pad = 3

    x1 = max(0, x + int(bbox[0]) - pad)
    y1 = max(0, y - ascent - pad)
    x2 = min(bgr.shape[1], x + int(bbox[2]) + pad)
    y2 = min(bgr.shape[0], y + int(descent) + pad)
    if x2 <= x1 or y2 <= y1:
        return

    roi = np.ascontiguousarray(bgr[y1:y2, x1:x2])
    pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    rgb = (int(color[2]), int(color[1]), int(color[0]))
    # Bez stroke_width — Pillow psuje polskie ogonki (ś, ć, ż…) czarnymi plamami / ?.
    draw.text((x - x1 + int(bbox[0]), y - y1 - ascent), line, font=font, fill=rgb)
    bgr[y1:y2, x1:x2] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
