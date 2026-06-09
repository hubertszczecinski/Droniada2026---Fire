"""Raport konkursowy — re-eksport z ``competition_report`` (regulamin 2026)."""
from module_panel.competition_report import (
    REGULAMENT_COLORS,
    card_to_structured_dict,
    color_for_report,
    format_card_detected_line,
    format_panel_detected_line,
    parse_competition_report_line,
    predictions_to_report_lines,
    report_lines_to_predictions,
    snap_panel_angle_deg,
    validate_competition_report_lines,
)

__all__ = [
    'REGULAMENT_COLORS',
    'card_to_structured_dict',
    'color_for_report',
    'format_card_detected_line',
    'format_panel_detected_line',
    'parse_competition_report_line',
    'predictions_to_report_lines',
    'report_lines_to_predictions',
    'snap_panel_angle_deg',
    'validate_competition_report_lines',
]
