"""Testy formatu raportu regulamin Droniada 2026 — Ogień i woda."""
import pipeline_competition as pc
from module_panel.competition_report import (
    REGULAMENT_COLORS,
    color_for_report,
    format_card_detected_line,
    format_panel_detected_line,
    parse_competition_report_line,
    predictions_to_report_lines,
    report_lines_to_predictions,
    snap_panel_angle_deg,
)


def test_snap_panel_angle():
    assert snap_panel_angle_deg(2) == 0
    assert snap_panel_angle_deg(44) == 45
    assert snap_panel_angle_deg(88) == 90


def test_card_line_roundtrip():
    line = format_card_detected_line('A', 90, 8, 7, 'POMARANCZOWA', timestamp_literal='12:00:00.000')
    assert 'WYKRYTO ZMIANE' in line
    assert 'pomaranczowa' in line
    rec = parse_competition_report_line(line)
    assert rec['event'] == 'card_detected'
    assert rec['panel'] == 'A'
    assert rec['angle_deg'] == 90
    assert rec['y'] == 8 and rec['x'] == 7
    assert rec['color_key'] == 'POMARANCZOWA'


def test_legacy_dataset_line_still_parses():
    legacy = (
        '[HH:MM:SS.mmm] WYKRYTO ZMIANĘ -> Panel: A (90°) | '
        'Pozycja: Wiersz 8, Kolumna 7 | Kolor: pomaranczowa'
    )
    rec = pc.parse_report_line(legacy)
    assert rec is not None
    assert rec['x'] == 7 and rec['y'] == 8
    assert rec['color'] == 'POMARANCZOWA'


def test_blender_angle_suffix_in_parens():
    legacy = (
        '[HH:MM:SS.mmm] WYKRYTO ZMIANE -> Panel: B (45° | poziomy) | '
        'Pozycja: Wiersz 1, Kolumna 10 | Kolor: zolta'
    )
    rec = parse_competition_report_line(legacy)
    assert rec is not None
    assert rec['color_key'] == 'ZOLTA'


def test_panel_detected_line():
    line = format_panel_detected_line('C', 45, timestamp_literal='09:00:00.000')
    rec = parse_competition_report_line(line)
    assert rec['event'] == 'panel_detected'
    assert rec['panel'] == 'C'
    assert rec['angle_deg'] == 45


def test_predictions_to_report_lines_sorted():
    preds = [
        {'x': 10, 'y': 1, 'color': 'CZERWONA'},
        {'x': 1, 'y': 1, 'color': 'ZIELONA'},
    ]
    lines = predictions_to_report_lines('A', 0, preds, timestamp_literal='00:00:00.000')
    assert len(lines) == 2
    assert 'Wiersz 1, Kolumna 1' in lines[0]
    assert 'Wiersz 1, Kolumna 10' in lines[1]


def test_report_lines_to_predictions():
    lines = predictions_to_report_lines(
        'A', 90, [{'x': 3, 'y': 5, 'color': 'NIEBIESKA'}],
        timestamp_literal='t',
    )
    back = report_lines_to_predictions(lines)
    assert back == [{'x': 3, 'y': 5, 'color': 'NIEBIESKA'}]


def test_all_regulament_colors_in_report():
    for c in REGULAMENT_COLORS:
        assert color_for_report(c) == color_for_report(c).lower()


def _sample_lines(n: int = 3, panel: str = 'A') -> list:
    colors = ['czerwona', 'zielona', 'niebieska', 'zolta']
    lines = []
    for i in range(n):
        lines.append(
            format_card_detected_line(
                panel, 90, i + 1, i + 2, colors[i],
                timestamp_literal='12:00:00.000',
            ),
        )
    return lines


def test_validate_report_accepts_three_cards():
    from module_panel.competition_report import validate_competition_report_lines

    ok, errors = validate_competition_report_lines(_sample_lines(3))
    assert ok, errors
    assert errors == []


def test_validate_report_accepts_four_cards():
    from module_panel.competition_report import validate_competition_report_lines

    ok, errors = validate_competition_report_lines(_sample_lines(4))
    assert ok, errors


def test_validate_report_rejects_two_cards():
    from module_panel.competition_report import validate_competition_report_lines

    ok, errors = validate_competition_report_lines(_sample_lines(2), min_cards=3)
    assert not ok
    assert any('Za mało' in e for e in errors)


def test_validate_report_accepts_two_cards_when_min_zero():
    from module_panel.competition_report import validate_competition_report_lines

    ok, errors = validate_competition_report_lines(_sample_lines(2), min_cards=0)
    assert ok, errors


def test_validate_report_rejects_five_cards():
    from module_panel.competition_report import validate_competition_report_lines

    lines = _sample_lines(3)
    lines.append(
        format_card_detected_line('A', 90, 5, 6, 'FIOLETOWA', timestamp_literal='12:00:01.000'),
    )
    lines.append(
        format_card_detected_line('A', 90, 7, 8, 'POMARANCZOWA', timestamp_literal='12:00:02.000'),
    )
    ok, errors = validate_competition_report_lines(lines)
    assert not ok
    assert any('Za dużo' in e for e in errors)


def test_validate_report_rejects_zmiane_with_diacritic():
    from module_panel.competition_report import validate_competition_report_lines

    bad = _sample_lines(3)
    bad[0] = bad[0].replace('WYKRYTO ZMIANE', 'WYKRYTO ZMIANĘ')
    ok, errors = validate_competition_report_lines(bad)
    assert not ok
    assert any('WYKRYTO ZMIANE' in e for e in errors)


def test_validate_report_rejects_duplicate_color():
    from module_panel.competition_report import validate_competition_report_lines

    lines = [
        format_card_detected_line('A', 90, 1, 1, 'CZERWONA', timestamp_literal='12:00:00.000'),
        format_card_detected_line('A', 90, 2, 2, 'CZERWONA', timestamp_literal='12:00:01.000'),
        format_card_detected_line('A', 90, 3, 3, 'ZIELONA', timestamp_literal='12:00:02.000'),
    ]
    ok, errors = validate_competition_report_lines(lines)
    assert not ok
    assert any('Powtórzony kolor' in e for e in errors)
