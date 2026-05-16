from typing import Any, Dict, List

def predictions_to_report_lines(panel_id: str, angle_deg: int, preds: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for p in preds:
        color = p.get('color', 'UNKNOWN')
        if isinstance(color, str):
            color_lower = color.lower()
        else:
            color_lower = str(color).lower()
        lines.append(f"[HH:MM:SS.mmm] WYKRYTO ZMIANĘ -> Panel: {panel_id} ({angle_deg}°) | Pozycja: Wiersz {p['y']}, Kolumna {p['x']} | Kolor: {color_lower}")
    return lines
