#!/usr/bin/env python3
"""Analiza zapisanych sesji live_debug — raport + index.html."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load_sessions(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    sessions = [p for p in root.iterdir() if p.is_dir()]
    sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return sessions


def _load_frames(session: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for jp in sorted(session.glob('*.json')):
        if jp.name == 'summary.json':
            continue
        try:
            with open(jp, encoding='utf-8') as f:
                rec = json.load(f)
            rec['_json_path'] = str(jp)
            rec['_session'] = session.name
            rows.append(rec)
        except (json.JSONDecodeError, OSError):
            continue
    return rows


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _diagnose(rec: Dict[str, Any]) -> List[str]:
    tips: List[str] = []
    reproj = _safe_float(rec.get('reproj_mean_px'), 999.0)
    grid = _safe_float(rec.get('grid_structure_score'), 0.0)
    interior = _safe_float(rec.get('panel_interior_score'), 0.0)
    reliable = bool(rec.get('grid_xy_reliable', False))
    corner = str(rec.get('corner_source', ''))
    if reproj > 25:
        tips.append('reproj_wysoki: panel za daleko / zły czworokąt / OSD w kadrze')
    if grid < 0.2:
        tips.append('slaba_siatka: wyprostowany panel nie ma widocznej siatki 10x10')
    if interior < 0.35:
        tips.append('slabe_tlo: maska nie widzi czarnego panelu (oświetlenie / tło)')
    if corner in ('hsv_panel', 'white_grid') and reproj > 15:
        tips.append('hsv_za_maly: profil HSV może obcinać panel — preferuj trapezoid')
    if corner == 'dark_blob' and reproj > 20:
        tips.append('blob_za_duzy: ciemny blob łapie tło')
    if not reliable and reproj < 12 and grid > 0.3:
        tips.append('reliable_off: homografia/inliers — sprawdź grid_xy_reliable')
    if not tips:
        tips.append('ok')
    return tips


def analyze(root: Path, *, last_session_only: bool = False) -> Dict[str, Any]:
    sessions = _load_sessions(root)
    if last_session_only and sessions:
        sessions = sessions[:1]
    all_frames: List[Dict[str, Any]] = []
    for sess in sessions:
        all_frames.extend(_load_frames(sess))

    by_corner = Counter(str(f.get('corner_source', '?')) for f in all_frames)
    reprojs = [float(f.get('reproj_mean_px', 999)) for f in all_frames if f.get('reproj_mean_px') is not None]
    grids = [float(f.get('grid_structure_score', 0)) for f in all_frames if f.get('grid_structure_score') is not None]
    reliable_n = sum(1 for f in all_frames if f.get('grid_xy_reliable'))

    problems: List[Dict[str, Any]] = []
    for f in all_frames:
        tips = _diagnose(f)
        if tips != ['ok']:
            problems.append({
                'frame_id': f.get('frame_id'),
                'session': f.get('_session'),
                'corner_source': f.get('corner_source'),
                'reproj_mean_px': f.get('reproj_mean_px'),
                'grid_structure_score': f.get('grid_structure_score'),
                'tips': tips,
            })

    # Czy wybrany kandydat był najlepszy wg rank_score?
    rank_issues = 0
    for f in all_frames:
        cands = f.get('candidates') or []
        if len(cands) < 2:
            continue
        chosen = [c for c in cands if c.get('chosen')]
        if not chosen:
            continue
        best_rank = min(float(c.get('rank_score', 1e9)) for c in cands)
        ch_rank = float(chosen[0].get('rank_score', 1e9))
        if ch_rank > best_rank + 1.0:
            rank_issues += 1

    summary = {
        'root': str(root),
        'sessions': len(sessions),
        'frames': len(all_frames),
        'reliable_frames': reliable_n,
        'corner_source_counts': dict(by_corner),
        'reproj_mean': sum(reprojs) / len(reprojs) if reprojs else None,
        'reproj_max': max(reprojs) if reprojs else None,
        'grid_mean': sum(grids) / len(grids) if grids else None,
        'problem_frames': len(problems),
        'rank_score_mismatch': rank_issues,
        'problems': problems[:40],
    }
    return summary


def _write_html(root: Path, sessions: List[Path]) -> str:
    parts = [
        '<!doctype html><html><head><meta charset="utf-8">',
        '<title>Live debug — Droniada</title>',
        '<style>body{font-family:system-ui;margin:20px;background:#f4f4f4}',
        '.card{background:#fff;border:1px solid #ccc;margin:12px 0;padding:12px}',
        'img{max-width:100%;height:auto}',
        '.bad{color:#b00020}.ok{color:#166534}pre{background:#eee;padding:8px;overflow:auto}</style></head><body>',
        '<h1>Live debug — sesje rogów panelu</h1>',
        '<p>Każda klatka: raw | overlay | warped | <b>candidates</b> (kolory = metody).</p>',
    ]
    for sess in sessions[:8]:
        parts.append(f'<h2>Sesja {sess.name}</h2>')
        for jp in sorted(sess.glob('*.json'))[:80]:
            if jp.name in ('summary.json',):
                continue
            stem = jp.stem
            try:
                rec = json.loads(jp.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError):
                continue
            tips = _diagnose(rec)
            cls = 'ok' if tips == ['ok'] else 'bad'
            parts.append('<div class="card">')
            parts.append(f'<h3 class="{cls}">{stem} — {rec.get("corner_source")} reproj={rec.get("reproj_mean_px")}</h3>')
            parts.append(f'<pre>{json.dumps({k: rec[k] for k in rec if k != "candidates"}, indent=2, ensure_ascii=False)}</pre>')
            for suffix in ('_candidates.jpg', '_overlay.jpg', '_warped.jpg'):
                img = sess / (stem + suffix)
                if img.is_file():
                    rel = os.path.relpath(img, root)
                    parts.append(f'<div><img src="{rel}" alt="{suffix}"/></div>')
            parts.append('</div>')
    parts.append('</body></html>')
    out = root / 'index.html'
    out.write_text('\n'.join(parts), encoding='utf-8')
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join(_ROOT, 'live_debug'))
    ap.add_argument('--last-session-only', action='store_true')
    ap.add_argument('--html', action='store_true', help='generuj live_debug/index.html')
    ap.add_argument('--out', default=None, help='zapisz summary.json')
    args = ap.parse_args()
    root = Path(args.root)
    summary = analyze(root, last_session_only=args.last_session_only)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.html:
        sessions = _load_sessions(root)
        path = _write_html(root, sessions)
        print(f'HTML: {path}', flush=True)


if __name__ == '__main__':
    main()
