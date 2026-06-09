"""
Raportowanie zgodne z regulaminem Droniada Challenge 2026 — „Ogień i woda” (basic).

Źródło: Regulamin_konkursu_Droniada_Challenge_2026 v3.0, sekcja ETAP BASIC - OGIEŃ
oraz „Zasady adresowania” (X/Y 1–10, Wiersz = Y, Kolumna = X).

Komunikaty online (KSID / strona sędziów):
- wykrycie panelu + kąt ustawienia (0°, 45°, 90°),
- każda kartka: panel, kąt, wiersz, kolumna, kolor (osobna linia na kartę).
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Regulamin 2026 — dozwolone kolory kartek (bez powtórzeń w obrębie panelu).
REGULAMENT_COLORS: Tuple[str, ...] = (
    'CZERWONA',
    'ZIELONA',
    'NIEBIESKA',
    'ZOLTA',
    'FIOLETOWA',
    'POMARANCZOWA',
)

# Kanoniczna forma w polu Kolor: małe litery, ASCII (zgodne z KSID / logami).
REGULAMENT_COLOR_REPORT: Dict[str, str] = {
    'CZERWONA': 'czerwona',
    'ZIELONA': 'zielona',
    'NIEBIESKA': 'niebieska',
    'ZOLTA': 'zolta',
    'FIOLETOWA': 'fioletowa',
    'POMARANCZOWA': 'pomaranczowa',
}

PANEL_ANGLES_DEG: Tuple[int, ...] = (0, 45, 90)

_COLOR_ALIASES: Dict[str, str] = {
    'CZERWONY': 'CZERWONA',
    'CZERWONA': 'CZERWONA',
    'RED': 'CZERWONA',
    'ZIELONY': 'ZIELONA',
    'ZIELONA': 'ZIELONA',
    'GREEN': 'ZIELONA',
    'NIEBIESKI': 'NIEBIESKA',
    'NIEBIESKA': 'NIEBIESKA',
    'BLUE': 'NIEBIESKA',
    'ZOLTY': 'ZOLTA',
    'ZOLTA': 'ZOLTA',
    'ZÓŁTA': 'ZOLTA',
    'YELLOW': 'ZOLTA',
    'FIOLETOWY': 'FIOLETOWA',
    'FIOLETOWA': 'FIOLETOWA',
    'PURPLE': 'FIOLETOWA',
    'POMARANCZOWY': 'POMARANCZOWA',
    'POMARANCZOWA': 'POMARANCZOWA',
    'ORANGE': 'POMARANCZOWA',
}

_CARD_LINE_RE = re.compile(
    r'^\[([^\]]+)\]\s*'
    r'WYKRYTO\s+ZMIAN[EĘ]\s*->\s*'
    r'Panel:\s*([A-Z])\s*'
    r'\((\d+)°[^)]*\)\s*'
    r'\|\s*Pozycja:\s*Wiersz\s*(\d+)\s*,\s*Kolumna\s*(\d+)\s*'
    r'\|\s*Kolor:\s*([A-Za-zĄąĆćĘęŁłŃńÓóŚśŹźŻż]+)\s*$',
    re.IGNORECASE,
)

_PANEL_LINE_RE = re.compile(
    r'^\[([^\]]+)\]\s*'
    r'PANEL\s*->\s*'
    r'Panel:\s*([A-Z])\s*'
    r'\|\s*(?:Polozenie|Położenie|Ustawienie):\s*(\d+)°\s*$',
    re.IGNORECASE,
)

# Ścisły format wysyłki do sędziów (regulamin 2026 — składnia, interpunkcja, ASCII).
_STRICT_TIMESTAMP_RE = re.compile(
    r'^(?:\d{2}:\d{2}:\d{2}\.\d{3}|HH:MM:SS\.mmm)$',
)
_STRICT_CARD_LINE_RE = re.compile(
    r'^\['
    r'(?:\d{2}:\d{2}:\d{2}\.\d{3}|HH:MM:SS\.mmm)'
    r'\]\s+'
    r'WYKRYTO ZMIANE\s*->\s*'
    r'Panel:\s*([A-Z])\s+'
    r'\((0|45|90)°\)\s*'
    r'\|\s*Pozycja:\s*Wiersz\s*(10|[1-9])\s*,\s*Kolumna\s*(10|[1-9])\s*'
    r'\|\s*Kolor:\s*(czerwona|zielona|niebieska|zolta|fioletowa|pomaranczowa)\s*$',
    re.IGNORECASE,
)
REGULAMENT_COLOR_VALUES: Tuple[str, ...] = tuple(REGULAMENT_COLOR_REPORT.values())


def snap_panel_angle_deg(angle_deg: Union[int, float]) -> int:
    """Regulamin: panel może być ustawiony 0°, 45° lub 90°."""
    a = float(angle_deg)
    best = min(PANEL_ANGLES_DEG, key=lambda x: abs(a - float(x)))
    return int(best)


def normalize_color_name(color: Any) -> Optional[str]:
    """Zwraca kanoniczny klucz REGULAMENT_COLORS lub None."""
    if color is None:
        return None
    key = str(color).strip().upper()
    key = key.replace('Ł', 'L').replace('Ó', 'O').replace('Ą', 'A').replace('Ę', 'E')
    key = key.replace('Ż', 'Z').replace('Ś', 'S').replace('Ć', 'C').replace('Ń', 'N')
    key = _COLOR_ALIASES.get(key, key)
    if key in REGULAMENT_COLORS:
        return key
    return None


def color_for_report(color: Any) -> str:
    """Tekst pola Kolor w raporcie (małe litery, ASCII)."""
    canon = normalize_color_name(color)
    if canon is None:
        return 'nieznany'
    return REGULAMENT_COLOR_REPORT[canon]


def _format_timestamp(
    when: Optional[datetime] = None,
    *,
    literal: Optional[str] = None,
) -> str:
    if literal is not None:
        return str(literal)
    t = when or datetime.now()
    return t.strftime('%H:%M:%S.') + f'{t.microsecond // 1000:03d}'


def format_panel_detected_line(
    panel_id: str,
    angle_deg: Union[int, float],
    *,
    when: Optional[datetime] = None,
    timestamp_literal: Optional[str] = None,
) -> str:
    """Komunikat o wykryciu panelu i kącie (regulamin: detekcja paneli)."""
    pid = str(panel_id).strip().upper()[:1]
    ang = snap_panel_angle_deg(angle_deg)
    return (
        f'[{_format_timestamp(when, literal=timestamp_literal)}] PANEL -> Panel: {pid} | '
        f'Polozenie: {ang}°'
    )


def format_card_detected_line(
    panel_id: str,
    angle_deg: Union[int, float],
    grid_row: int,
    grid_col: int,
    color: Any,
    *,
    when: Optional[datetime] = None,
    timestamp_literal: Optional[str] = None,
) -> str:
    """Jedna kartka — regulamin: kolor + współrzędne X,Y online."""
    pid = str(panel_id).strip().upper()[:1]
    ang = snap_panel_angle_deg(angle_deg)
    row = int(grid_row)
    col = int(grid_col)
    if not (1 <= row <= 10 and 1 <= col <= 10):
        raise ValueError(f'wiersz/kolumna poza siatką 1–10: ({row}, {col})')
    return (
        f'[{_format_timestamp(when, literal=timestamp_literal)}] WYKRYTO ZMIANE -> Panel: {pid} ({ang}°) | '
        f'Pozycja: Wiersz {row}, Kolumna {col} | Kolor: {color_for_report(color)}'
    )


def prediction_to_report_line(
    panel_id: str,
    angle_deg: Union[int, float],
    prediction: Dict[str, Any],
    *,
    when: Optional[datetime] = None,
    timestamp_literal: Optional[str] = None,
) -> str:
    return format_card_detected_line(
        panel_id,
        angle_deg,
        int(prediction['y']),
        int(prediction['x']),
        prediction.get('color', 'UNKNOWN'),
        when=when,
        timestamp_literal=timestamp_literal,
    )


def predictions_to_report_lines(
    panel_id: str,
    angle_deg: Union[int, float],
    preds: List[Dict[str, Any]],
    *,
    when: Optional[datetime] = None,
    include_panel_line: bool = False,
    timestamp_literal: Optional[str] = None,
) -> List[str]:
    """Linie raportu dla listy detekcji (posortowane: wiersz, kolumna)."""
    ordered = sorted(preds, key=lambda p: (int(p.get('y', 0)), int(p.get('x', 0))))
    lines: List[str] = []
    if include_panel_line and ordered:
        lines.append(
            format_panel_detected_line(
                panel_id, angle_deg, when=when, timestamp_literal=timestamp_literal,
            ),
        )
    for p in ordered:
        lines.append(
            prediction_to_report_line(
                panel_id, angle_deg, p, when=when, timestamp_literal=timestamp_literal,
            ),
        )
    return lines


def parse_competition_report_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parsuje linię raportu regulaminowego.

    Zwraca dict z kluczem ``event`` ∈ {``card_detected``, ``panel_detected``}.
    """
    line = line.strip()
    if not line:
        return None
    m = _PANEL_LINE_RE.match(line)
    if m:
        return {
            'event': 'panel_detected',
            'timestamp': m.group(1),
            'panel': m.group(2).upper(),
            'angle_deg': snap_panel_angle_deg(int(m.group(3))),
        }
    m = _CARD_LINE_RE.match(line)
    if m:
        color_key = normalize_color_name(m.group(6))
        return {
            'event': 'card_detected',
            'timestamp': m.group(1),
            'panel': m.group(2).upper(),
            'angle_deg': snap_panel_angle_deg(int(m.group(3))),
            'y': int(m.group(4)),
            'x': int(m.group(5)),
            'color': REGULAMENT_COLOR_REPORT.get(color_key or '', 'nieznany'),
            'color_key': color_key,
        }
    return None


def report_lines_to_predictions(lines: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in lines:
        rec = parse_competition_report_line(line)
        if rec is None or rec.get('event') != 'card_detected':
            continue
        if rec.get('color_key') is None:
            continue
        out.append({
            'x': int(rec['x']),
            'y': int(rec['y']),
            'color': str(rec['color_key']),
        })
    return out


def _strict_line_issues(line: str, line_no: int) -> List[str]:
    """Błędy składni jednej linii kartki (format wysyłki do sędziów)."""
    raw = line.strip()
    if not raw:
        return []
    issues: List[str] = []
    prefix = f'Linia {line_no}: '
    if 'ZMIANĘ' in raw or 'ZMIANĘ' in raw:
        issues.append(
            prefix + 'użyj „WYKRYTO ZMIANE” (ASCII, bez „ę”), zgodnie z regulaminem',
        )
    if 'PANEL ->' in raw.upper() and 'WYKRYTO' not in raw.upper():
        issues.append(
            prefix + 'linia PANEL nie należy do raportu kart — zostaw tylko linie WYKRYTO ZMIANE',
        )
        return issues
    m_ts = re.match(r'^\[([^\]]+)\]', raw)
    if m_ts is None:
        issues.append(prefix + 'brak znacznika czasu [HH:MM:SS.mmm] na początku linii')
    elif not _STRICT_TIMESTAMP_RE.match(m_ts.group(1)):
        issues.append(
            prefix + 'znacznik czasu musi być HH:MM:SS.mmm (np. 12:34:56.789)',
        )
    if '| poziomy' in raw.lower() or '| pionowy' in raw.lower():
        issues.append(
            prefix + 'kąt panelu: tylko (0°), (45°) lub (90°) — bez dopisków w nawiasie',
        )
    m_color = re.search(r'\|\s*Kolor:\s*(.+?)\s*$', raw, re.IGNORECASE)
    if m_color:
        color_txt = m_color.group(1).strip()
        if color_txt != color_txt.lower():
            issues.append(
                prefix + f'pole Kolor: małe litery ({color_txt!r} → {color_txt.lower()!r})',
            )
        if color_txt.lower() not in REGULAMENT_COLOR_VALUES:
            allowed = ', '.join(REGULAMENT_COLOR_VALUES)
            issues.append(
                prefix + f'niedozwolony kolor {color_txt!r} — dozwolone: {allowed}',
            )
    if not _STRICT_CARD_LINE_RE.match(raw):
        if not issues:
            issues.append(
                prefix
                + 'niezgodna składnia — oczekiwano: '
                + '[HH:MM:SS.mmm] WYKRYTO ZMIANE -> Panel: A (90°) | '
                + 'Pozycja: Wiersz 8, Kolumna 7 | Kolor: pomaranczowa',
            )
    return issues


def validate_competition_report_lines(
    lines: List[str],
    *,
    min_cards: int = 3,
    max_cards: int = 4,
    expected_panel: Optional[str] = None,
    allow_empty: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Walidacja raportu przed wysyłką do sędziów.

    ``min_cards`` / ``max_cards`` — liczba linii kart (regulamin zwykle 3–4).
    ``allow_empty=True`` — pusta wysyłka dozwolona (reset migawek bez linii).
    """
    errors: List[str] = []
    card_lines: List[str] = []
    for idx, line in enumerate(lines, start=1):
        stripped = str(line).strip()
        if not stripped:
            continue
        if re.search(r'PANEL\s*->', stripped, re.IGNORECASE) and 'WYKRYTO' not in stripped.upper():
            errors.extend(_strict_line_issues(stripped, idx))
            continue
        card_lines.append(stripped)
        errors.extend(_strict_line_issues(stripped, idx))

    n = len(card_lines)
    if n == 0 and allow_empty:
        return len(errors) == 0, errors
    if n < int(min_cards):
        errors.append(
            f'Za mało kart: {n} — wymagane {min_cards}–{max_cards} kartki',
        )
    elif n > int(max_cards):
        errors.append(
            f'Za dużo kart: {n} — maksymalnie {max_cards} kartki',
        )

    parsed: List[Dict[str, Any]] = []
    for line in card_lines:
        m = _STRICT_CARD_LINE_RE.match(line)
        if m is None:
            continue
        parsed.append({
            'panel': m.group(1).upper(),
            'angle_deg': int(m.group(2)),
            'y': int(m.group(3)),
            'x': int(m.group(4)),
            'color': m.group(5).lower(),
        })

    if len(parsed) != n:
        return False, errors

    panels = {p['panel'] for p in parsed}
    if len(panels) > 1:
        errors.append(f'Niespójny panel w liniach: {", ".join(sorted(panels))}')
    exp = str(expected_panel).strip().upper()[:1] if expected_panel else None
    if exp and panels and exp not in panels:
        errors.append(f'Oczekiwano panelu {exp}, w raporcie: {", ".join(sorted(panels))}')

    angles = {p['angle_deg'] for p in parsed}
    if len(angles) > 1:
        errors.append(f'Niespójny kąt panelu: {", ".join(str(a) for a in sorted(angles))}°')

    positions = [(p['y'], p['x']) for p in parsed]
    if len(positions) != len(set(positions)):
        errors.append('Powtórzona pozycja (wiersz, kolumna) — każda kartka ma unikalne współrzędne')

    colors = [p['color'] for p in parsed]
    if len(colors) != len(set(colors)):
        errors.append('Powtórzony kolor — na panelu bez powtórzeń (regulamin)')

    return len(errors) == 0, errors


def card_to_structured_dict(
    panel_id: str,
    angle_deg: Union[int, float],
    grid_row: int,
    grid_col: int,
    color: Any,
) -> Dict[str, Any]:
    """JSON do KSID / API (obok linii tekstowej)."""
    canon = normalize_color_name(color)
    return {
        'panel_id': str(panel_id).strip().upper()[:1],
        'panel_angle_deg': snap_panel_angle_deg(angle_deg),
        'grid_row': int(grid_row),
        'grid_col': int(grid_col),
        'color': REGULAMENT_COLOR_REPORT.get(canon or '', 'nieznany'),
        'color_key': canon,
    }
