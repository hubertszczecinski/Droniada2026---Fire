"""Wczytanie raportów paneli A/B/C z pliku JSON (tryb preset na zawody)."""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple

from module_panel.competition_report import validate_competition_report_lines

_PANEL_ORDER = ('A', 'B', 'C')
_NOW_PLACEHOLDER_RE = re.compile(r'\{\{NOW[^}]*\}\}')


def _lines_for_validation(lines: List[str]) -> List[str]:
    """Podstaw dummy timestamp, żeby walidacja przeszła szablony z {{NOW}} / {{NOW+1s+523ms}}."""
    return [_NOW_PLACEHOLDER_RE.sub('[00:00:00.000]', str(ln)) for ln in lines]


def resolve_preset_reports_path(path: str, *, root: str | None = None) -> str:
    p = os.path.abspath(path)
    if os.path.isfile(p):
        return p
    if root:
        alt = os.path.join(os.path.abspath(root), path)
        if os.path.isfile(alt):
            return alt
    return p


def load_preset_reports_file(
    path: str,
    *,
    root: str | None = None,
    min_cards: int = 3,
    max_cards: int = 4,
) -> Dict[str, List[str]]:
    """
    Wczytaj ``config/preset_reports.json`` i zwaliduj każdy panel.

    Zwraca ``{'A': [linie...], 'B': [...], ...}``.
    """
    full = resolve_preset_reports_path(path, root=root)
    if not os.path.isfile(full):
        raise FileNotFoundError(f'Brak pliku raportów preset: {full}')
    with open(full, encoding='utf-8') as fh:
        raw = json.load(fh)
    ok, errors, panels = parse_preset_reports_data(
        raw,
        min_cards=min_cards,
        max_cards=max_cards,
    )
    if not ok:
        raise ValueError('preset_reports_invalid: ' + '; '.join(errors))
    return panels


def parse_preset_reports_data(
    raw: object,
    *,
    min_cards: int = 3,
    max_cards: int = 4,
) -> Tuple[bool, List[str], Dict[str, List[str]]]:
    errors: List[str] = []
    panels: Dict[str, List[str]] = {}
    if not isinstance(raw, dict):
        return False, ['Oczekiwano obiektu JSON z kluczem "panels"'], {}
    block = raw.get('panels')
    if not isinstance(block, dict) or not block:
        return False, ['Brak sekcji "panels" w pliku preset'], {}
    for key, lines in block.items():
        pid = str(key).strip().upper()[:1]
        if pid not in _PANEL_ORDER:
            errors.append(f'Nieznany panel {key!r} — dozwolone A, B, C')
            continue
        if not isinstance(lines, list):
            errors.append(f'Panel {pid}: oczekiwano listy linii raportu')
            continue
        norm = [str(ln).strip() for ln in lines if str(ln).strip()]
        ok, verr = validate_competition_report_lines(
            _lines_for_validation(norm),
            min_cards=min_cards,
            max_cards=max_cards,
            expected_panel=pid,
            allow_empty=False,
        )
        if not ok:
            for e in verr:
                errors.append(f'Panel {pid}: {e}')
            continue
        panels[pid] = norm
    for pid in _PANEL_ORDER:
        if pid not in panels:
            errors.append(f'Brak raportu dla panelu {pid}')
    return len(errors) == 0, errors, panels
