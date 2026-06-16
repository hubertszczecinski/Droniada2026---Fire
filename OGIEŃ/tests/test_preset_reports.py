"""Testy wczytywania raportów preset (tryb operatorski A/B/C)."""
import json
import os
import tempfile

from release.preset_reports import load_preset_reports_file, parse_preset_reports_data


def _valid_panels():
    return {
        'panels': {
            'A': [
                '[12:00:00.000] WYKRYTO ZMIANE -> Panel: A (90°) | Pozycja: Wiersz 3, Kolumna 4 | Kolor: czerwona',
                '[12:00:01.000] WYKRYTO ZMIANE -> Panel: A (90°) | Pozycja: Wiersz 5, Kolumna 6 | Kolor: zielona',
                '[12:00:02.000] WYKRYTO ZMIANE -> Panel: A (90°) | Pozycja: Wiersz 7, Kolumna 8 | Kolor: niebieska',
            ],
            'B': [
                '[12:10:00.000] WYKRYTO ZMIANE -> Panel: B (45°) | Pozycja: Wiersz 2, Kolumna 3 | Kolor: zolta',
                '[12:10:01.000] WYKRYTO ZMIANE -> Panel: B (45°) | Pozycja: Wiersz 4, Kolumna 5 | Kolor: pomaranczowa',
                '[12:10:02.000] WYKRYTO ZMIANE -> Panel: B (45°) | Pozycja: Wiersz 6, Kolumna 7 | Kolor: fioletowa',
            ],
            'C': [
                '[12:20:00.000] WYKRYTO ZMIANE -> Panel: C (0°) | Pozycja: Wiersz 1, Kolumna 2 | Kolor: czerwona',
                '[12:20:01.000] WYKRYTO ZMIANE -> Panel: C (0°) | Pozycja: Wiersz 3, Kolumna 4 | Kolor: zielona',
                '[12:20:02.000] WYKRYTO ZMIANE -> Panel: C (0°) | Pozycja: Wiersz 5, Kolumna 6 | Kolor: zolta',
            ],
        },
    }


def test_parse_preset_reports_ok():
    ok, errors, panels = parse_preset_reports_data(_valid_panels())
    assert ok, errors
    assert set(panels) == {'A', 'B', 'C'}
    assert len(panels['A']) == 3


def test_load_preset_reports_file_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'preset_reports.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(_valid_panels(), fh)
        loaded = load_preset_reports_file(path)
        assert loaded['B'][0].startswith('[12:10:00.000]')


def test_parse_preset_missing_panel():
    data = dict(_valid_panels())
    del data['panels']['C']
    ok, errors, _ = parse_preset_reports_data(data)
    assert not ok
    assert any('panelu C' in e for e in errors)
