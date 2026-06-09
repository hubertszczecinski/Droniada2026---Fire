# Raportowanie online — Droniada Challenge 2026 (Ogień i woda)

Źródło: `Regulamin_konkursu_Droniada_Challenge_2026 (3).pdf`, wersja 3.0, sekcja **ETAP BASIC - OGIEŃ**.

## Wymagania regulaminu

| Element | Opis |
|--------|------|
| Panele | 3 panele (A, B, C) — osobna detekcja położenia i zdjęcie |
| Siatka | 10×10; **X** = kolumna 1–10 (lewo→prawo), **Y** = wiersz 1–10 (dół→góra) |
| Kąt panelu | Tylko **0°, 45° lub 90°** — komunikat online + zdjęcie (3 pkt/panel) |
| Kartki | Kolor + pozycja (X,Y) — raport online; łącznie 10 kart na misję |
| Kolory | czerwona, zielona, niebieska, żółta, fioletowa, pomarańczowa (bez powtórzeń na panelu) |

## Format w projekcie

Implementacja: `module_panel/competition_report.py`.

### Wykrycie panelu

```
[HH:MM:SS.mmm] PANEL -> Panel: A | Polozenie: 90°
```

### Wykrycie kartki (zmiana na siatce)

```
[HH:MM:SS.mmm] WYKRYTO ZMIANE -> Panel: A (90°) | Pozycja: Wiersz 8, Kolumna 7 | Kolor: pomaranczowa
```

- Znacznik czasu: `HH:MM:SS.mmm` na żywo z zegara systemowego; w datasetcie Blender placeholder `HH:MM:SS.mmm`.
- **WYKRYTO ZMIANE** — forma ASCII (bez „ę”); parser akceptuje też starsze **ZMIANĘ**.
- Pole **Kolor**: małe litery, ASCII (`zolta`, `pomaranczowa`).
- Kąt w nawiasie po ID panelu jest zaokrąglany do 0 / 45 / 90.

### JSON (misja / KSID)

Przy **Wyślij** w `submitted_draft.json` dodawane jest `report_structured` — lista zdarzeń z parsera (`card_detected`, `panel_detected`).

## Użycie w kodzie

```python
from module_panel.report import predictions_to_report_lines, format_panel_detected_line

lines = predictions_to_report_lines('A', 90, predictions)
panel_line = format_panel_detected_line('A', 90)  # opcjonalnie przed kartami
```

Parser GT / ewaluacja: `pipeline_competition.parse_report_line` (kompatybilność wsteczna z logami datasetu).
