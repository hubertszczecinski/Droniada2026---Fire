# Online reporting - Droniada Challenge 2026 (Fire & Water / OGIEŃ)

Source: competition rules v3.0, **BASIC stage - OGIEŃ** section.

## Rules summary

| Element | Description |
|---------|-------------|
| Panels | 3 panels (A, B, C) - separate pose detection and photo per panel |
| Grid | 10×10; **X** = column 1–10 (left→right), **Y** = row 1–10 (bottom→top) |
| Panel angle | Only **0°, 45°, or 90°** - online message + photo (3 pts/panel) |
| Cards | Colour + position (X,Y) - online report; 10 cards per mission |
| Colours | red, green, blue, yellow, violet, orange (no duplicates on one panel) |

## Format in this project

Implementation: `module_panel/competition_report.py`.

### Panel detected

```
[HH:MM:SS.mmm] PANEL -> Panel: A | Polozenie: 90°
```

### Card detected (grid change)

```
[HH:MM:SS.mmm] WYKRYTO ZMIANE -> Panel: A (90°) | Pozycja: Wiersz 8, Kolumna 7 | Kolor: pomaranczowa
```

- Timestamp: live system clock `HH:MM:SS.mmm`; Blender dataset uses placeholder `HH:MM:SS.mmm`.
- **WYKRYTO ZMIANE** - ASCII form (no Polish diacritics); parser also accepts legacy **ZMIANĘ**.
- **Kolor** field: lowercase ASCII (`zolta`, `pomaranczowa`).
- Angle in parentheses after panel ID is rounded to 0 / 45 / 90.

### JSON (mission / scoring system)

On **Submit** in the web UI, `submitted_draft.json` gets `report_structured` - parsed events (`card_detected`, `panel_detected`).

## Code usage

```python
from module_panel.report import predictions_to_report_lines, format_panel_detected_line

lines = predictions_to_report_lines('A', 90, predictions)
panel_line = format_panel_detected_line('A', 90)  # optional before cards
```

GT parser / evaluation: `pipeline_competition.parse_report_line` (backward compatible with dataset logs).
