"""Jeden panel live: moduł A + B, pełne dane, galeria migawek (niski reproj)."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import cv2
import numpy as np

from release.cv_text import put_text_utf8, text_width_utf8
from release.cxy_latch_preview import draw_warped_panel_preview
from release.pose_runtime import PoseFrameOutput

_DASHBOARD_WINDOW = 'droniada_dashboard'
_SNAPSHOT_WINDOW = 'droniada_snapshots'
_PARAMS_H_MIN = 500
_PARAMS_TITLE_H = 58
_STRIP_H = 130
_GAP = 10
_BG = (32, 32, 32)
_SIDE_BG = (245, 245, 245)
_SECTION_A = (220, 238, 220)
_SECTION_B = (220, 230, 248)
_SECTION_LAT = (255, 236, 210)
_SECTION_SNAP = (235, 235, 235)
_TEXT = (25, 25, 25)
_MUTED = (80, 80, 80)
_OK = (0, 120, 0)
_WARN = (0, 90, 200)
_FAIL = (0, 0, 200)
_WHITE = (255, 255, 255)
_LINE_H = 52
_PAD = 22
_FONT_MAIN = 1.04
_FONT_MUTED = 0.88
_FONT_HEADER = 1.12
_FONT_TITLE = 1.32
_SNAP_CLICK_DELTA = 0
_SNAP_LEFT_RECT = (0, 0, 0, 0)
_SNAP_RIGHT_RECT = (0, 0, 0, 0)


class _ParamsLayout(TypedDict):
    line_h: int
    pad: int
    title_h: int
    min_h: int
    font_main: float
    font_muted: float
    font_header: float
    font_title: float
    sec_step: int


def _params_layout(width: int) -> _ParamsLayout:
    """Większa czcionka na szerszym podglądzie — czytelne na drugim monitorze."""
    sm = max(1.55, min(2.15, float(width) / 620.0))
    return _ParamsLayout(
        line_h=int(round(_LINE_H * sm)),
        pad=_PAD,
        title_h=int(round(_PARAMS_TITLE_H * sm)),
        min_h=int(round(_PARAMS_H_MIN * sm * 0.9)),
        font_main=_FONT_MAIN * sm,
        font_muted=_FONT_MUTED * sm,
        font_header=_FONT_HEADER * sm,
        font_title=_FONT_TITLE * sm,
        sec_step=int(round(48 * sm)),
    )


def new_dashboard_session(root: str) -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(root, f'session_{ts}')
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, 'snapshots'), exist_ok=True)
    return path


def _fit_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(max(1, h))
    nw = max(1, int(round(w * scale)))
    return cv2.resize(img, (nw, target_h), interpolation=cv2.INTER_AREA)


def _fit_width(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / float(max(1, w))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img, (target_w, nh), interpolation=cv2.INTER_AREA)


def _put_line(
    canvas: np.ndarray,
    y: int,
    text: str,
    *,
    color: Tuple[int, int, int] = _TEXT,
    scale: float = _FONT_MAIN,
    thickness: int = 2,
    x: int = _PAD,
    line_h: Optional[int] = None,
) -> int:
    put_text_utf8(canvas, text, (x, y), color, scale=scale, thickness=thickness)
    return y + (line_h if line_h is not None else _LINE_H)


def _section_header(
    canvas: np.ndarray,
    y: int,
    title: str,
    bg: Tuple[int, int, int],
    *,
    x0: int = 0,
    x1: Optional[int] = None,
    layout: Optional[_ParamsLayout] = None,
) -> int:
    if x1 is None:
        x1 = canvas.shape[1]
    pad = layout['pad'] if layout else _PAD
    font_header = layout['font_header'] if layout else _FONT_HEADER
    sec_step = layout['sec_step'] if layout else 48
    bar_top = int(round(30 * (sec_step / 48.0)))
    bar_bot = int(round(14 * (sec_step / 48.0)))
    cv2.rectangle(canvas, (x0, y - bar_top), (x1, y + bar_bot), bg, -1)
    put_text_utf8(
        canvas, title, (x0 + pad, y + int(round(8 * (sec_step / 48.0)))),
        _TEXT, scale=font_header, thickness=2,
    )
    return y + sec_step


def _text_width(text: str, scale: float, thickness: int = 2) -> int:
    return text_width_utf8(text, scale, thickness)


def _wrap_text(text: str, max_width: int, scale: float, thickness: int = 2) -> List[str]:
    if max_width <= 0 or not text:
        return [text or '']
    words = text.split(' ')
    lines: List[str] = []
    current = ''
    for word in words:
        trial = word if not current else f'{current} {word}'
        if _text_width(trial, scale, thickness) <= max_width:
            current = trial
            continue
        if current:
            lines.append(current)
            current = ''
        if _text_width(word, scale, thickness) <= max_width:
            current = word
            continue
        chunk = ''
        for ch in word:
            trial_ch = chunk + ch
            if _text_width(trial_ch, scale, thickness) <= max_width:
                chunk = trial_ch
            else:
                if chunk:
                    lines.append(chunk)
                chunk = ch
        current = chunk
    if current:
        lines.append(current)
    return lines if lines else ['']


def _cxy_list_height(
    preds: List[Dict[str, Any]],
    *,
    limit: int = 8,
    line_h: Optional[int] = None,
) -> int:
    lh = line_h if line_h is not None else _LINE_H
    if not preds:
        return lh
    shown = min(len(preds), limit)
    extra = lh if len(preds) > limit else 0
    return shown * lh + extra


def _report_block_height(
    lines: List[str],
    width: int,
    *,
    include_header: bool = True,
    layout: Optional[_ParamsLayout] = None,
) -> int:
    if not lines:
        return 0
    pad = layout['pad'] if layout else _PAD
    line_h = layout['line_h'] if layout else _LINE_H
    font_muted = layout['font_muted'] if layout else _FONT_MUTED
    sec_step = layout['sec_step'] if layout else 48
    max_w = max(40, width - 2 * pad)
    h = sec_step if include_header else 0
    for ln in lines:
        h += len(_wrap_text(ln, max_w, font_muted)) * line_h
    return h + 8


def _put_wrapped_lines(
    canvas: np.ndarray,
    y: int,
    text: str,
    *,
    x: int,
    max_x: int,
    color: Tuple[int, int, int] = _MUTED,
    scale: float = _FONT_MUTED,
    thickness: int = 2,
    line_h: Optional[int] = None,
) -> int:
    max_w = max(40, max_x - x)
    for ln in _wrap_text(text, max_w, scale, thickness):
        y = _put_line(
            canvas, y, ln, color=color, scale=scale, thickness=thickness, x=x, line_h=line_h,
        )
    return y


def _draw_report_block(
    canvas: np.ndarray,
    y: int,
    title: str,
    lines: List[str],
    bg: Tuple[int, int, int],
    *,
    layout: Optional[_ParamsLayout] = None,
) -> int:
    if not lines:
        return y
    width = canvas.shape[1]
    pad = layout['pad'] if layout else _PAD
    font_muted = layout['font_muted'] if layout else _FONT_MUTED
    line_h = layout['line_h'] if layout else _LINE_H
    y = _section_header(canvas, y, title, bg, x0=0, x1=width, layout=layout)
    for ln in lines:
        y = _put_wrapped_lines(
            canvas, y, ln, x=pad, max_x=width - pad,
            scale=font_muted, line_h=line_h,
        )
    return y + 4


def _draw_cxy_list(
    canvas: np.ndarray,
    y: int,
    preds: List[Dict[str, Any]],
    *,
    prefix: str = '',
    x: int = _PAD,
    limit: int = 8,
    layout: Optional[_ParamsLayout] = None,
) -> int:
    font_muted = layout['font_muted'] if layout else _FONT_MUTED
    line_h = layout['line_h'] if layout else _LINE_H
    if not preds:
        return _put_line(
            canvas, y, f'{prefix}(brak w tej klatce)', color=_MUTED, scale=font_muted, x=x,
            line_h=line_h,
        )
    for p in sorted(preds, key=lambda d: (-int(d['y']), int(d['x'])))[:limit]:
        color_name = str(p.get('color', '?'))
        y = _put_line(
            canvas, y,
            f'{prefix}W{p["y"]} K{p["x"]} — {color_name}',
            scale=font_muted, x=x, line_h=line_h,
        )
    if len(preds) > limit:
        y = _put_line(
            canvas, y, f'{prefix}… +{len(preds) - limit}', color=_MUTED, scale=font_muted, x=x,
            line_h=line_h,
        )
    return y


def _draw_bottom_params_panel(
    width: int,
    *,
    module_a: Optional[PoseFrameOutput],
    panel_id: str,
    reliable: bool,
    reproj_b: float,
    homography_inliers: int,
    corner_source: str,
    xy_backend: str,
    angle: int,
    category: str,
    live_preds: List[Dict[str, Any]],
    snapshot_preds: Optional[List[Dict[str, Any]]],
    consensus_preds: Optional[List[Dict[str, Any]]],
    live_report_lines: List[str],
    latched_preds: Optional[List[Dict[str, Any]]],
    latched_report_lines: Optional[List[str]],
    latch_txt: Optional[str],
    latch_locked: bool,
    snapshot_entries: List[Dict[str, Any]],
    panel_present: bool = True,
    panel_presence_reason: str = 'ok',
    grid_overlap_ratio: Optional[float] = None,
    grid_line_match_ratio: Optional[float] = None,
    roi_coverage: Optional[float] = None,
    warp_panel_coverage: Optional[float] = None,
) -> np.ndarray:
    from module_pose.panel_stand import STAND_LABEL_PL

    live_report_lines = list(live_report_lines or [])
    latched_report_lines = list(latched_report_lines or []) if latched_report_lines else []

    lay = _params_layout(width)
    lh = lay['line_h']
    pad = lay['pad']
    title_h = lay['title_h']
    fm, fmu, ft = lay['font_main'], lay['font_muted'], lay['font_title']

    y0 = title_h + int(round(18 * (lh / _LINE_H)))
    col_h_a = y0 + lay['sec_step'] + (
        lh if module_a is None
        else 6 * lh if module_a.ok
        else 2 * lh
    )
    snap_preds = list(snapshot_preds or [])
    cons_preds = list(consensus_preds or [])
    col_h_b = y0 + lay['sec_step'] + 8 * lh + _cxy_list_height(live_preds, limit=6, line_h=lh)
    if snap_preds:
        col_h_b += 2 * lh + _cxy_list_height(snap_preds, limit=6, line_h=lh)
    if cons_preds:
        col_h_b += 2 * lh + _cxy_list_height(cons_preds, limit=6, line_h=lh)
    col_h_c = y0 + lay['sec_step']
    if latch_locked and (latched_preds or latch_txt):
        col_h_c += lh * (1 if latch_txt else 0)
        col_h_c += _cxy_list_height(latched_preds or [], limit=5, line_h=lh)
    elif latch_txt and not latch_locked:
        col_h_c += lay['sec_step'] + lh
    col_h_c += 8 + lay['sec_step'] + (min(len(snapshot_entries), 4) if snapshot_entries else 1) * lh

    cols_bottom = max(col_h_a, col_h_b, col_h_c) + 10
    report_h = (
        _report_block_height(live_report_lines, width, layout=lay)
        + _report_block_height(
            latched_report_lines, width, include_header=bool(latched_report_lines), layout=lay,
        )
    )
    panel_h = max(lay['min_h'], cols_bottom + report_h + pad)
    panel = np.full((panel_h, width, 3), _SIDE_BG, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width, title_h), (210, 210, 210), -1)
    put_text_utf8(
        panel, 'DRONIADA LIVE', (pad, int(title_h * 0.72)), _TEXT, scale=ft, thickness=3,
    )

    col_w = max(1, width // 3)
    col_bounds = [(0, col_w), (col_w, col_w * 2), (col_w * 2, width)]
    for x_div in (col_w, col_w * 2):
        cv2.line(panel, (x_div, title_h), (x_div, cols_bottom), (190, 190, 190), 2)

    y0 = title_h + int(round(18 * (lh / _LINE_H)))
    x_a = col_bounds[0][0] + pad
    y_a = y0
    y_a = _put_line(
        panel, y_a,
        f'Panel w kadrze: {"TAK" if panel_present else "NIE"}'
        + (f' ({panel_presence_reason})' if not panel_present else ''),
        color=_OK if panel_present else _FAIL,
        x=col_bounds[0][0] + pad, scale=fm, line_h=lh,
    )
    y_a = _section_header(
        panel, y_a, 'MODUŁ A — podejście', _SECTION_A,
        x0=col_bounds[0][0], x1=col_bounds[0][1], layout=lay,
    )
    if module_a is None:
        y_a = _put_line(panel, y_a, 'Wyłączony', color=_MUTED, scale=fmu, x=x_a, line_h=lh)
    elif module_a.ok:
        pl = STAND_LABEL_PL.get(module_a.panel_angle_category, module_a.panel_angle_category)
        y_a = _put_line(panel, y_a, 'Status: OK', color=_OK, x=x_a, scale=fm, line_h=lh)
        y_a = _put_line(panel, y_a, f'Odległość: {module_a.distance_m:.2f} m', x=x_a, scale=fm, line_h=lh)
        y_a = _put_line(panel, y_a, f'Stojak: {pl} ({module_a.report_angle_deg}°)', x=x_a, scale=fm, line_h=lh)
        y_a = _put_line(
            panel, y_a, f'Pewność stojaka: {module_a.stand_confidence:.0%}',
            color=_MUTED, scale=fmu, x=x_a, line_h=lh,
        )
        y_a = _put_line(
            panel, y_a,
            f'Roll {module_a.roll_deg:.0f}°  Pitch {module_a.pitch_deg:.0f}°  Yaw {module_a.yaw_deg:.0f}°',
            scale=fmu, x=x_a, line_h=lh,
        )
        y_a = _put_line(
            panel, y_a, f'Reproj A: {module_a.reproj_mean_px:.1f} px', scale=fmu, x=x_a, line_h=lh,
        )
    else:
        reason = str(module_a.meta.get('reason', module_a.meta.get('fail', '?')))
        y_a = _put_line(panel, y_a, 'Status: FAIL', color=_FAIL, x=x_a, scale=fm, line_h=lh)
        y_a = _put_line(panel, y_a, reason, color=_MUTED, scale=fmu, x=x_a, line_h=lh)

    x_b = col_bounds[1][0] + pad
    y_b = y0
    y_b = _section_header(
        panel, y_b, 'MODUŁ B — skan (LIVE)', _SECTION_B,
        x0=col_bounds[1][0], x1=col_bounds[1][1], layout=lay,
    )
    y_b = _put_line(
        panel, y_b,
        f'Reliable: {"TAK" if reliable else "NIE"}',
        color=_OK if reliable else _FAIL,
        x=x_b, scale=fm, line_h=lh,
    )
    y_b = _put_line(
        panel, y_b, f'Reproj B: {reproj_b:.1f} px   Inliers: {homography_inliers}',
        x=x_b, scale=fm, line_h=lh,
    )
    if grid_overlap_ratio is not None:
        y_b = _put_line(
            panel, y_b,
            f'3 siatki: {100.0 * float(grid_overlap_ratio):.0f}%',
            color=_MUTED, scale=fmu, x=x_b, line_h=lh,
        )
    if grid_line_match_ratio is not None:
        y_b = _put_line(
            panel, y_b,
            f'Pokrycie linii: {100.0 * float(grid_line_match_ratio):.0f}%',
            color=_MUTED, scale=fmu, x=x_b, line_h=lh,
        )
    if roi_coverage is not None:
        y_b = _put_line(
            panel, y_b,
            f'ROI rogów: {100.0 * float(roi_coverage):.0f}%',
            color=_MUTED, scale=fmu, x=x_b, line_h=lh,
        )
    if warp_panel_coverage is not None:
        y_b = _put_line(
            panel, y_b,
            f'Pokrycie panelu: {100.0 * float(warp_panel_coverage):.0f}%',
            color=_MUTED, scale=fmu, x=x_b, line_h=lh,
        )
    y_b = _put_line(panel, y_b, f'Rogi: {corner_source}', color=_MUTED, scale=fmu, x=x_b, line_h=lh)
    y_b = _put_line(panel, y_b, f'Backend: {xy_backend}', color=_MUTED, scale=fmu, x=x_b, line_h=lh)
    y_b = _put_line(
        panel, y_b, f'Panel {panel_id}  kąt={angle}°  {category}', scale=fmu, x=x_b, line_h=lh,
    )
    y_b = _put_line(panel, y_b, f'Kartki live: {len(live_preds)}', x=x_b, scale=fm, line_h=lh)
    y_b = _draw_cxy_list(panel, y_b, live_preds, x=x_b, limit=6, layout=lay)
    if snap_preds:
        y_b = _put_line(
            panel, y_b, f'Kolory (migawka): {len(snap_preds)}', color=_OK, x=x_b, scale=fm, line_h=lh,
        )
        y_b = _draw_cxy_list(panel, y_b, snap_preds, prefix='• ', x=x_b, limit=6, layout=lay)
    if cons_preds:
        y_b = _put_line(
            panel, y_b, f'Konkurs CXY ({len(cons_preds)} kart)', color=_OK, x=x_b, scale=fm, line_h=lh,
        )
        y_b = _draw_cxy_list(panel, y_b, cons_preds, prefix='★ ', x=x_b, limit=6, layout=lay)

    x_c = col_bounds[2][0] + pad
    y_c = y0
    if latch_locked and (latched_preds or latch_txt):
        y_c = _section_header(
            panel, y_c, 'ZATRASK CXY (zamrożone)', _SECTION_LAT,
            x0=col_bounds[2][0], x1=col_bounds[2][1], layout=lay,
        )
        if latch_txt:
            y_c = _put_line(panel, y_c, latch_txt, color=_OK, scale=fmu, x=x_c, line_h=lh)
        if latched_preds:
            y_c = _draw_cxy_list(panel, y_c, latched_preds, prefix='• ', x=x_c, limit=5, layout=lay)
    elif latch_txt and not latch_locked:
        y_c = _section_header(
            panel, y_c, 'ZATRASK CXY', _SECTION_LAT,
            x0=col_bounds[2][0], x1=col_bounds[2][1], layout=lay,
        )
        y_c = _put_line(panel, y_c, latch_txt, color=_WARN, scale=fmu, x=x_c, line_h=lh)

    y_c = max(y_c, y0) + 8
    y_c = _section_header(
        panel, y_c, 'MIGAWKI (niski reproj)', _SECTION_SNAP,
        x0=col_bounds[2][0], x1=col_bounds[2][1], layout=lay,
    )
    if snapshot_entries:
        for ent in snapshot_entries[:4]:
            extra = ''
            if ent.get('module_a_ok'):
                extra = f'  d={ent.get("distance_m", 0):.1f}m'
            y_c = _put_line(
                panel, y_c,
                f'#{ent.get("rank", "?")} {ent["frame_id"]}  reproj={ent["reproj_b"]:.1f}px{extra}',
                color=_OK, scale=fmu, x=x_c, line_h=lh,
            )
    else:
        y_c = _put_line(
            panel, y_c, 'Brak — reliable + reproj ≤ próg', color=_MUTED, scale=fmu, x=x_c, line_h=lh,
        )

    cv2.line(panel, (0, cols_bottom), (width, cols_bottom), (190, 190, 190), 2)
    y_report = cols_bottom + 12
    y_report = _draw_report_block(panel, y_report, 'RAPORT LIVE', live_report_lines, _SECTION_B, layout=lay)
    if latched_report_lines:
        y_report = _draw_report_block(
            panel, y_report, 'RAPORT ZATRASK', latched_report_lines, _SECTION_LAT, layout=lay,
        )

    return panel


def _draw_snapshot_strip(
    width: int,
    entries: List[Dict[str, Any]],
) -> np.ndarray:
    strip = np.full((_STRIP_H, width, 3), _BG, dtype=np.uint8)
    if not entries:
        put_text_utf8(
            strip,
            'Galeria migawek — zapis przy reproj B <= próg i reliable=TAK',
            (12, _STRIP_H // 2),
            (190, 190, 190),
            scale=0.52,
            thickness=2,
        )
        return strip

    x = 10
    thumb_h = _STRIP_H - 36
    for ent in entries[:8]:
        thumb_path = ent.get('thumb_path')
        if thumb_path and os.path.isfile(thumb_path):
            thumb = cv2.imread(thumb_path)
            if thumb is not None:
                thumb = _fit_height(thumb, thumb_h)
                tw = thumb.shape[1]
                if x + tw + 8 > width:
                    break
                strip[8:8 + thumb_h, x:x + tw] = thumb
                label = f"#{ent.get('rank', '?')} {ent['reproj_b']:.1f}px"
                put_text_utf8(
                    strip, label, (x, _STRIP_H - 8), (200, 230, 200), scale=0.48, thickness=2,
                )
                x += tw + 12
    return strip


def draw_mission_hud_overlay(
    bgr: np.ndarray,
    *,
    panel_id: str,
    snap_n: int,
    snap_max: int,
    mission_index: int,
    mission_total: int,
    consensus_lines: Optional[List[str]] = None,
    panel_full: bool = False,
    editable_hint: bool = True,
) -> np.ndarray:
    """Prawy górny róg podglądu kamery — status misji + skrót konkursu."""
    out = bgr.copy()
    h, w = out.shape[:2]
    lines: List[str] = [
        f'PANEL {panel_id}  ({mission_index + 1}/{mission_total})',
        f'Migawki {snap_n}/{snap_max}' + ('  PELNY' if panel_full else ''),
    ]
    if consensus_lines:
        lines.append('Konkurs CXY:')
        for ln in consensus_lines[:5]:
            short = ln
            if 'Pozycja:' in ln:
                short = ln.split('Pozycja:')[-1].strip()
                if 'Kolor:' in ln:
                    short = 'Poz: ' + ln.split('Pozycja:')[-1].strip()
            lines.append(' ' + short[:56])
    elif snap_n == 0:
        lines.append('(brak konkursu)')
    if editable_hint:
        lines.append('Edycja: okno Tk')
        lines.append('[W] wyslij')
    scale = 0.52
    th = 2
    lh = 22
    pad = 10
    max_tw = 0
    for ln in lines:
        max_tw = max(max_tw, text_width_utf8(ln, scale, th))
    box_w = min(w - 20, max_tw + 2 * pad)
    box_h = min(h - 20, len(lines) * lh + 2 * pad)
    x0 = w - box_w - 12
    y0 = 12
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (28, 32, 38), -1)
    cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (80, 200, 120), 2, cv2.LINE_AA)
    y = y0 + pad + 16
    for i, ln in enumerate(lines):
        col = (220, 240, 220) if i == 0 else (200, 200, 200)
        if 'PELNY' in ln:
            col = (80, 200, 255)
        put_text_utf8(out, ln, (x0 + pad, y), col, scale=scale, thickness=th)
        y += lh
    return out


def compose_unified_dashboard(
    camera_vis: np.ndarray,
    warped_bgr: np.ndarray,
    live_preds: List[Dict[str, Any]],
    warped_dets: Optional[List[Dict[str, Any]]],
    *,
    module_a: Optional[PoseFrameOutput] = None,
    panel_id: str = 'A',
    reliable: bool = False,
    reproj_b: float = 999.0,
    homography_inliers: int = 0,
    corner_source: str = 'none',
    xy_backend: str = '-',
    angle: int = 0,
    category: str = 'horizontal',
    live_report_lines: Optional[List[str]] = None,
    latched_preds: Optional[List[Dict[str, Any]]] = None,
    latched_report_lines: Optional[List[str]] = None,
    latch_txt: Optional[str] = None,
    latch_locked: bool = False,
    snapshot_entries: Optional[List[Dict[str, Any]]] = None,
    snapshot_preds: Optional[List[Dict[str, Any]]] = None,
    consensus_preds: Optional[List[Dict[str, Any]]] = None,
    target_w: int = 1600,
    opencv_grid_x: Optional[List[float]] = None,
    opencv_grid_y: Optional[List[float]] = None,
    hough_grid_x: Optional[List[float]] = None,
    hough_grid_y: Optional[List[float]] = None,
    grid_overlap_ratio: Optional[float] = None,
    grid_line_match_ratio: Optional[float] = None,
    roi_coverage: Optional[float] = None,
    warp_panel_coverage: Optional[float] = None,
    panel_present: bool = True,
    panel_presence_reason: str = 'ok',
) -> np.ndarray:
    """Kamera | panel warp (LIVE) u góry; parametry u dołu + pasek migawek."""
    entries = list(snapshot_entries or [])
    panel_title = 'Moduł B — LIVE (kolory tylko na migawce)'
    if not live_preds:
        panel_title += ' — bez kartek w podglądzie'
    if grid_overlap_ratio is not None:
        panel_title += f'  3siatki={100.0 * float(grid_overlap_ratio):.0f}%'
    panel_vis = draw_warped_panel_preview(
        warped_bgr, live_preds,
        warped_dets if not live_preds else None,
        title=panel_title,
        opencv_grid_x=opencv_grid_x,
        opencv_grid_y=opencv_grid_y,
        hough_grid_x=hough_grid_x,
        hough_grid_y=hough_grid_y,
    )

    main_h = max(camera_vis.shape[0], panel_vis.shape[0], 560)
    cam = _fit_height(camera_vis, main_h)
    pan = _fit_height(panel_vis, main_h)
    gap = np.full((main_h, _GAP, 3), _BG, dtype=np.uint8)
    top = np.hstack([cam, gap, pan])

    params = _draw_bottom_params_panel(
        top.shape[1],
        module_a=module_a,
        panel_id=panel_id,
        reliable=reliable,
        reproj_b=reproj_b,
        homography_inliers=homography_inliers,
        corner_source=corner_source,
        xy_backend=xy_backend,
        angle=angle,
        category=category,
        live_preds=live_preds,
        snapshot_preds=list(snapshot_preds) if snapshot_preds else None,
        consensus_preds=list(consensus_preds) if consensus_preds else None,
        live_report_lines=list(live_report_lines or []),
        latched_preds=list(latched_preds) if latched_preds else None,
        latched_report_lines=list(latched_report_lines) if latched_report_lines else None,
        latch_txt=latch_txt,
        latch_locked=latch_locked,
        snapshot_entries=entries,
        panel_present=panel_present,
        panel_presence_reason=panel_presence_reason,
        grid_overlap_ratio=grid_overlap_ratio,
        grid_line_match_ratio=grid_line_match_ratio,
        roi_coverage=roi_coverage,
        warp_panel_coverage=warp_panel_coverage,
    )

    strip = _draw_snapshot_strip(top.shape[1], entries)
    vgap = np.full((_GAP, top.shape[1], 3), _BG, dtype=np.uint8)
    out = np.vstack([top, vgap, params, vgap, strip])

    return _fit_width(out, max(1200, int(target_w)))


def snapshot_frame_eligible(
    *,
    reproj_b: float,
    max_reproj_px: float,
    grid_overlap_ratio: float,
    min_grid_overlap: float = 0.95,
    pnp_ok: bool = True,
    require_module_a: bool = False,
    module_a_ok: bool = True,
    reproj_a: float = 999.0,
    max_reproj_a_px: float = 10.0,
    reliable: bool = False,
    require_reliable: bool = False,
    homography_inliers: int = 0,
    min_homography_inliers: int = 0,
    block_unreliable_zero_inl: bool = True,
    min_grid_overlap_if_unreliable: float = 0.84,
    warp_panel_coverage: float = 0.0,
    min_warp_panel_coverage: float = 0.25,
) -> Tuple[bool, str]:
    """
    Migawka: 3 siatki + niski reproj A/B + pokrycie panelu na warpie.

    Gdy ``reliable=False`` i ``homography_inliers==0`` (typowy nag5): ostrzejsze progi
    (reproj A, warp, wyższe 3siatki) zamiast globalnego ``require_reliable``.
    """
    if reproj_b > max_reproj_px:
        return False, 'reproj_b'
    if require_module_a:
        if not module_a_ok:
            return False, 'module_a'
        ra = float(reproj_a)
        cap_a = float(max_reproj_a_px)
        if cap_a > 0 and ra > cap_a:
            # PnP modułu A bywa zepsuty (nag5), a reproj B + 3 siatki są OK.
            pose_unusable = ra > max(80.0, float(reproj_b) * 5.0 + 15.0)
            if not pose_unusable:
                return False, 'reproj_a'
    if not pnp_ok:
        return False, 'pnp'
    if require_reliable and not reliable:
        return False, 'not_reliable'
    if int(min_homography_inliers) > 0 and int(homography_inliers) < int(min_homography_inliers):
        return False, 'inliers'
    if float(warp_panel_coverage) < float(min_warp_panel_coverage):
        return False, 'warp_coverage'
    overlap_need = float(min_grid_overlap)
    if block_unreliable_zero_inl and not reliable and int(homography_inliers) <= 0:
        overlap_need = max(overlap_need, float(min_grid_overlap_if_unreliable))
    if float(grid_overlap_ratio) < overlap_need:
        return False, 'grid_overlap'
    return True, 'grid_overlap'


class LiveSnapshotStore:
    """Automatyczne migawki gdy siatka panelu i OpenCV pokrywają się (IoU) + niski reproj."""

    __slots__ = (
        'session_dir', 'max_snapshots', 'max_reproj', 'min_stable_frames',
        'require_module_a', 'replace_margin',
        'entries', '_stable_run',
    )

    def __init__(
        self,
        session_dir: str,
        *,
        max_snapshots: int = 8,
        max_reproj: float = 15.0,
        min_stable_frames: int = 2,
        require_module_a: bool = False,
        replace_margin: float = 0.05,
    ) -> None:
        self.session_dir = session_dir
        self.max_snapshots = max(1, int(max_snapshots))
        self.max_reproj = float(max_reproj)
        self.min_stable_frames = max(1, int(min_stable_frames))
        self.require_module_a = bool(require_module_a)
        self.replace_margin = float(max(0.0, replace_margin))
        self.entries: List[Dict[str, Any]] = []
        self._stable_run = 0
        os.makedirs(os.path.join(session_dir, 'snapshots'), exist_ok=True)

    @property
    def ranked_entries(self) -> List[Dict[str, Any]]:
        out = sorted(self.entries, key=lambda e: float(e['reproj_b']))
        for i, ent in enumerate(out, start=1):
            ent['rank'] = i
        return out

    def _min_stable_needed(self, grid_overlap_ratio: float) -> int:
        if float(grid_overlap_ratio) >= 0.98:
            return 1
        return self.min_stable_frames

    def _should_replace(self, reproj_b: float) -> bool:
        if len(self.entries) < self.max_snapshots:
            return True
        worst = max(float(e['reproj_b']) for e in self.entries)
        return reproj_b < worst - self.replace_margin

    def maybe_save(
        self,
        *,
        frame_id: str,
        dashboard_bgr: np.ndarray,
        reproj_b: float,
        snapshot_ok: bool,
        reliable: bool,
        record: Dict[str, Any],
        module_a_ok: bool = False,
        snapshot_reason: str = '',
        grid_overlap_ratio: float = 0.0,
    ) -> Optional[str]:
        if not snapshot_ok or reproj_b > self.max_reproj:
            self._stable_run = 0
            return None
        if self.require_module_a and not module_a_ok:
            self._stable_run = 0
            return None
        self._stable_run += 1
        need_stable = self._min_stable_needed(grid_overlap_ratio)
        if self._stable_run < need_stable:
            return None
        if not self._should_replace(reproj_b):
            return None

        stem = frame_id.replace('/', '_').replace(':', '_')
        snap_dir = os.path.join(self.session_dir, 'snapshots')
        png_path = os.path.join(snap_dir, f'{stem}_dashboard.png')
        json_path = os.path.join(snap_dir, f'{stem}_snapshot.json')
        thumb_path = os.path.join(snap_dir, f'{stem}_thumb.jpg')

        cv2.imwrite(png_path, dashboard_bgr)
        thumb = _fit_height(dashboard_bgr, 280)
        cv2.imwrite(thumb_path, thumb)
        payload = {
            'frame_id': frame_id,
            'saved_at': datetime.now().isoformat(timespec='seconds'),
            'reproj_b': float(reproj_b),
            'reliable': bool(reliable),
            'grid_overlap_ratio': float(grid_overlap_ratio),
            'snapshot_reason': str(snapshot_reason),
            'module_a_ok': bool(module_a_ok),
            **record,
        }
        with open(json_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        ent = {
            'frame_id': frame_id,
            'reproj_b': float(reproj_b),
            'module_a_ok': bool(module_a_ok),
            'distance_m': float(record.get('module_a', {}).get('distance_m', 0) or 0),
            'dashboard_path': png_path,
            'json_path': json_path,
            'thumb_path': thumb_path,
        }
        self.entries = [e for e in self.entries if e['frame_id'] != frame_id]
        self.entries.append(ent)
        if len(self.entries) > self.max_snapshots:
            ranked = sorted(self.entries, key=lambda e: float(e['reproj_b']), reverse=True)
            drop = ranked[0]
            self.entries = [e for e in self.entries if e['frame_id'] != drop['frame_id']]
        return png_path

    def clear_all(self) -> int:
        """Usuń wszystkie migawki z dysku i z pamięci (po wysyłce / resecie panelu)."""
        snap_dir = os.path.join(self.session_dir, 'snapshots')
        n = 0
        for ent in list(self.entries):
            for key in ('dashboard_path', 'json_path', 'thumb_path'):
                path = ent.get(key)
                if path and os.path.isfile(path):
                    try:
                        os.remove(path)
                        n += 1
                    except OSError:
                        pass
        self.entries = []
        self._stable_run = 0
        if os.path.isdir(snap_dir):
            for name in os.listdir(snap_dir):
                path = os.path.join(snap_dir, name)
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        return n

    def refresh_snapshot_artifacts(
        self,
        png_path: str,
        dashboard_bgr: np.ndarray,
        *,
        record_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Po detekcji kolorów: nadpisz PNG migawki, miniaturę i pola w JSON."""
        thumb_path = png_path.replace('_dashboard.png', '_thumb.jpg')
        json_path = png_path.replace('_dashboard.png', '_snapshot.json')
        cv2.imwrite(png_path, dashboard_bgr)
        cv2.imwrite(thumb_path, _fit_height(dashboard_bgr, 94))
        if not os.path.isfile(json_path):
            return
        with open(json_path, encoding='utf-8') as fh:
            payload = json.load(fh)
        if record_updates:
            for key, val in record_updates.items():
                if (
                    key in payload
                    and isinstance(payload[key], dict)
                    and isinstance(val, dict)
                ):
                    payload[key].update(val)
                else:
                    payload[key] = val
        with open(json_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def write_session_index(
        self,
        *,
        competition_min_votes: int = 2,
        competition_min_ratio: float = 0.0,
    ) -> str:
        ranked = self.ranked_entries
        competition_summary: Optional[Dict[str, Any]] = None
        if ranked:
            from release.snapshot_cxy_competition import update_session_competition

            comp = update_session_competition(
                self.session_dir,
                min_votes=int(competition_min_votes),
                min_support_ratio=float(competition_min_ratio),
            )
            if comp is not None:
                competition_summary = {
                    'n_accepted': len(comp.accepted),
                    'n_rejected': len(comp.rejected),
                    'min_votes': comp.min_votes,
                    'predictions': comp.predictions,
                }
        index = {
            'session_dir': self.session_dir,
            'updated_at': datetime.now().isoformat(timespec='seconds'),
            'max_reproj': self.max_reproj,
            'snapshots': ranked,
            'cxy_competition': competition_summary,
        }
        index_path = os.path.join(self.session_dir, 'index.json')
        with open(index_path, 'w', encoding='utf-8') as fh:
            json.dump(index, fh, ensure_ascii=False, indent=2)

        html_path = os.path.join(self.session_dir, 'index.html')
        rows = []
        for ent in ranked:
            dash = ent.get('dashboard_path', '')
            rel = os.path.relpath(dash, self.session_dir) if dash else ''
            rows.append(
                f'<tr><td>#{ent.get("rank", "?")}</td>'
                f'<td>{ent["frame_id"]}</td>'
                f'<td>{ent["reproj_b"]:.2f} px</td>'
                f'<td>{"OK" if ent.get("module_a_ok") else "—"}</td>'
                f'<td>{ent.get("distance_m", 0):.2f} m</td>'
                f'<td><a href="{rel}"><img src="{rel}" width="480"/></a></td></tr>'
            )
        html = f"""<!DOCTYPE html>
<html lang="pl"><head><meta charset="utf-8"/>
<title>Droniada — migawki live</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #1a1a1a; color: #eee; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #444; padding: 8px; vertical-align: top; }}
th {{ background: #333; }}
a {{ color: #8cf; }}
</style></head><body>
<h1>Migawki sesji live (niski reproj)</h1>
<p>Katalog: {self.session_dir}</p>
<p>Próg reproj B: ≤ {self.max_reproj} px · liczba migawek: {len(ranked)}</p>
<table>
<tr><th>#</th><th>Klatka</th><th>reproj B</th><th>Moduł A</th><th>Odległość</th><th>Podgląd</th></tr>
{''.join(rows) if rows else '<tr><td colspan="6">Brak migawek w tej sesji.</td></tr>'}
</table>
</body></html>"""
        with open(html_path, 'w', encoding='utf-8') as fh:
            fh.write(html)
        return html_path


def show_dashboard_window(img: np.ndarray, target_w: int = 1600) -> None:
    show = _fit_width(img, max(1200, int(target_w)))
    cv2.namedWindow(_DASHBOARD_WINDOW, cv2.WINDOW_NORMAL)
    cv2.imshow(_DASHBOARD_WINDOW, show)
    cv2.resizeWindow(_DASHBOARD_WINDOW, show.shape[1], min(show.shape[0], 960))


def _on_snapshot_mouse(event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
    global _SNAP_CLICK_DELTA
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    lx, ly, lw, lh = _SNAP_LEFT_RECT
    rx, ry, rw, rh = _SNAP_RIGHT_RECT
    if lx <= x <= lx + lw and ly <= y <= ly + lh:
        _SNAP_CLICK_DELTA = -1
    elif rx <= x <= rx + rw and ry <= y <= ry + rh:
        _SNAP_CLICK_DELTA = +1


def advance_snapshot_index(current_idx: int, key_code: int, total: int) -> int:
    if total <= 0:
        return -1
    idx = max(0, min(current_idx, total - 1))
    if key_code in (81, 2424832, ord('a'), ord('A')):
        idx = (idx - 1) % total
    elif key_code in (83, 2555904, ord('d'), ord('D')):
        idx = (idx + 1) % total
    return idx


def show_snapshot_browser(
    snapshot_paths: List[str],
    current_idx: int,
    *,
    target_w: int = 1400,
) -> int:
    global _SNAP_CLICK_DELTA, _SNAP_LEFT_RECT, _SNAP_RIGHT_RECT
    if not snapshot_paths:
        return -1

    total = len(snapshot_paths)
    idx = max(0, min(current_idx, total - 1))
    img = cv2.imread(snapshot_paths[idx])
    if img is None:
        return idx

    max_w = max(900, int(target_w * 0.82))
    if img.shape[1] > max_w:
        img = _fit_width(img, max_w)

    top_h = 70
    canvas = np.full((img.shape[0] + top_h, img.shape[1], 3), (28, 28, 28), dtype=np.uint8)
    canvas[top_h:, :] = img

    # Klikalne strzalki
    btn_w, btn_h = 56, 40
    y0 = 14
    lx = 14
    rx = canvas.shape[1] - btn_w - 14
    _SNAP_LEFT_RECT = (lx, y0, btn_w, btn_h)
    _SNAP_RIGHT_RECT = (rx, y0, btn_w, btn_h)
    cv2.rectangle(canvas, (lx, y0), (lx + btn_w, y0 + btn_h), (220, 220, 220), -1)
    cv2.rectangle(canvas, (rx, y0), (rx + btn_w, y0 + btn_h), (220, 220, 220), -1)
    cv2.putText(canvas, '<', (lx + 20, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, '>', (rx + 20, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)

    name = os.path.basename(snapshot_paths[idx])
    title = f'MIGAWKI  {idx + 1}/{total}   {name}'
    put_text_utf8(canvas, title, (88, 40), _WHITE, scale=0.68, thickness=2)
    put_text_utf8(
        canvas, 'kliknij strzałki lub A/D, lewo/prawo', (88, 62),
        (180, 180, 180), scale=0.48, thickness=1,
    )

    cv2.namedWindow(_SNAPSHOT_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(_SNAPSHOT_WINDOW, _on_snapshot_mouse)
    cv2.imshow(_SNAPSHOT_WINDOW, canvas)
    cv2.resizeWindow(_SNAPSHOT_WINDOW, canvas.shape[1], min(canvas.shape[0], 900))

    if _SNAP_CLICK_DELTA != 0:
        idx = (idx + _SNAP_CLICK_DELTA) % total
        _SNAP_CLICK_DELTA = 0
    return idx
