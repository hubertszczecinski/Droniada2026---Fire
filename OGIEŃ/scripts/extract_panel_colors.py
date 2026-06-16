#!/usr/bin/env python3
"""
Ekstrakcja kolorów kartek z nagrania panelu (powtarzalna kalibracja).

Ten sam mechanizm co na migawkach Test.mov:
  • znany układ komórek (layout JSON),
  • siatka grid_lines (detekcja lub równomierna),
  • próbka HSV ze środka komórki,
  • centroidy w config/card_colors.json.

Użycie (nagranie panelu z kartami w znanych miejscach):

  cp config/panel_color_layout.example.json config/panel_color_layout.json
  # edytuj row/col/color w layout

  ./scripts/extract_panel_colors.sh \\
    --video dataset/my_capture/panel_calib.mov \\
    --layout config/panel_color_layout.json

Na zawodach: usuń config/panel_color_layout.json (detekcja pełnego panelu, bez GT).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pipeline_competition as pc

from release.card_color_profile import (
    _DEFAULT_PATH,
    _primary_competition_median,
    build_profile_from_folder,
    prune_centroids_by_hue,
    ranges_from_medians,
)
from release.snapshot_cell_color import (
    aggregate_hsv_samples,
    build_centroids_from_samples,
    collect_gt_samples_from_session,
    collect_samples_from_video,
    gt_cells_from_config,
    load_panel_color_layout,
    merge_snapshot_centroids,
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else _ROOT / path


def _print_sample_summary(samples: dict[str, list]) -> None:
    print('\nZebrane próbki HSV (nagranie):')
    for name in sorted(samples):
        pts = samples[name]
        med = aggregate_hsv_samples(pts)
        if med is None:
            print(f'  {name:14s}  brak')
            continue
        print(f'  {name:14s}  n={len(pts):3d}  H={med[0]:.0f}  S={med[1]:.0f}  V={med[2]:.0f}')


def _apply_panel_to_profile(
    profile,
    panel_centroids: dict[str, tuple[float, float, float]],
    panel_raw: dict[str, list],
    *,
    competition_photos: bool,
):
    """Centroidy + zakresy inRange z median nagrania (+ opcjonalnie *2)."""
    merge_map = {k: list(v) for k, v in panel_raw.items() if v}
    profile = merge_snapshot_centroids(profile, merge_map)

    for name, med in panel_centroids.items():
        cls_id = int(pc.COLOR_TO_CLASS.get(name.upper(), -1))
        if cls_id < 0:
            continue
        cents = profile.centroids_by_cls.get(cls_id, [])
        if competition_photos:
            primary = _primary_competition_median(cents) if cents else med
            profile.ranges_by_cls[cls_id] = ranges_from_medians(cls_id, [primary])
        else:
            profile.centroids_by_cls[cls_id] = [med]
            profile.ranges_by_cls[cls_id] = ranges_from_medians(cls_id, [med])
    return profile


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Ekstrakcja kolorów z nagrania panelu → config/card_colors.json',
    )
    ap.add_argument('--video', '-v', type=Path, required=True, help='Nagranie panelu z kartami')
    ap.add_argument(
        '--layout', '-l',
        type=Path,
        default=_ROOT / 'config/panel_color_layout.json',
        help='Układ kart (row/col/color); domyślnie config/panel_color_layout.json',
    )
    ap.add_argument(
        '--cards-folder', '-f',
        type=Path,
        default=_ROOT / 'config/competition_cards',
        help='Zdjęcia *2 (baza zawodowa); puste = tylko nagranie',
    )
    ap.add_argument('--out', '-o', type=Path, default=_DEFAULT_PATH)
    ap.add_argument(
        '--report',
        type=Path,
        default=None,
        help='Raport JSON (domyślnie dataset/results/panel_colors_*.json)',
    )
    ap.add_argument('--session', action='append', default=[], help='Opcjonalna sesja migawek do dopisania')
    ap.add_argument('--frame-step', type=int, default=15)
    ap.add_argument('--max-reproj', type=float, default=35.0)
    ap.add_argument('--min-frame', type=int, default=0)
    ap.add_argument('--rotate', type=int, default=None, help='Domyślnie z layout JSON')
    ap.add_argument(
        '--panel-only',
        action='store_true',
        help='Profil wyłącznie z nagrania (bez zdjęć *2)',
    )
    ap.add_argument(
        '--no-cards',
        action='store_true',
        help='Synonim --panel-only',
    )
    ap.add_argument('--notes', default='', help='Notatka w profilu JSON')
    args = ap.parse_args()

    layout_path = _resolve(args.layout)
    cfg = load_panel_color_layout(layout_path)
    if not cfg:
        raise SystemExit(
            f'Brak layoutu: {layout_path}\n'
            f'Skopiuj config/panel_color_layout.example.json → config/panel_color_layout.json'
        )

    video = _resolve(args.video)
    if not video.is_file():
        raise SystemExit(f'Brak wideo: {video}')

    rotate = int(args.rotate if args.rotate is not None else cfg.get('rotate_deg', 0))
    panel_only = bool(args.panel_only or args.no_cards)

    weights = _ROOT / 'runs/pose/droniada_real_finetune/weights/best.pt'
    if weights.is_file():
        os.environ['DRONIADA_YOLO_POSE_WEIGHTS'] = str(weights)

    print('=== Ekstrakcja kolorów z nagrania panelu ===')
    print(f'Wideo:   {video}')
    print(f'Layout:  {layout_path}')
    print(f'Komórki: {len(gt_cells_from_config(cfg))}')
    print(f'Klatki:  co {args.frame_step}, reproj ≤ {args.max_reproj}px')
    print('')

    by_color, frame_reports = collect_samples_from_video(
        video,
        cfg,
        frame_step=int(args.frame_step),
        max_reproj_px=float(args.max_reproj),
        min_frame=int(args.min_frame),
        rotate_deg=rotate,
    )

    for sess in args.session:
        sess_path = _resolve(Path(sess))
        if not sess_path.is_dir():
            print(f'Pominięto sesję (brak): {sess_path}')
            continue
        extra, _ = collect_gt_samples_from_session(
            sess_path,
            video,
            cfg,
            max_reproj_b=float(args.max_reproj),
            min_frame=int(args.min_frame),
            rotate_deg=rotate,
        )
        for name, pts in extra.items():
            by_color.setdefault(name, []).extend(pts)
        print(f'Dopisano migawki: {sess_path.name}')

    _print_sample_summary(by_color)

    missing = [c['color'] for c in gt_cells_from_config(cfg) if len(by_color.get(c['color'], [])) < 3]
    if missing:
        print(f'\nUWAGA: mało próbek dla: {", ".join(missing)} (sprawdź layout / jakość nagrania)')

    panel_centroids = build_centroids_from_samples(by_color)
    if not panel_centroids:
        raise SystemExit('Nie zebrano żadnych próbek HSV — przerwij i popraw nagranie lub layout.')

    if panel_only:
        from release.card_color_profile import CardColorProfile

        ranges = {}
        cents = {}
        calib = {}
        for name, med in panel_centroids.items():
            cls_id = int(pc.COLOR_TO_CLASS[name.upper()])
            ranges[cls_id] = ranges_from_medians(cls_id, [med])
            cents[cls_id] = [med]
            h, s, v = med
            calib[name] = [{'file': 'panel_recording', 'h': h, 's': s, 'v': v}]
        profile = CardColorProfile(
            ranges,
            source=str(video),
            meta={'calibrated': True, 'source_files': {k: 'panel_recording' for k in panel_centroids}},
            centroids_by_cls=cents,
        )
        profile.meta['calibration_samples'] = calib
        notes = args.notes or f'Kalibracja z nagrania panelu ({layout_path.name})'
    else:
        cards = _resolve(args.cards_folder)
        profile, _samples = build_profile_from_folder(cards)
        profile = _apply_panel_to_profile(
            profile,
            panel_centroids,
            by_color,
            competition_photos=True,
        )
        profile.centroids_by_cls = prune_centroids_by_hue(profile.centroids_by_cls)
        notes = args.notes or f'*2 + nagranie panelu ({layout_path.name})'

    calib_out = profile.meta.get('calibration_samples') or {}
    for name, med in panel_centroids.items():
        h, s, v = med
        entries = list(calib_out.get(name) or [])
        entries.append({
            'file': 'panel_recording_median',
            'h': round(h, 1),
            's': round(s, 1),
            'v': round(v, 1),
        })
        calib_out[name] = entries
    profile.meta['calibration_samples'] = calib_out

    out_path = _resolve(args.out)
    profile.save(
        out_path,
        source=str(video),
        notes=notes,
        calibration_samples=calib_out,
    )
    print(f'\nZapisano profil: {out_path}')

    report_path = args.report
    if report_path is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = _ROOT / 'dataset/results' / f'panel_colors_{stamp}.json'
    else:
        report_path = _resolve(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        'created': datetime.now(timezone.utc).isoformat(),
        'video': str(video),
        'layout': str(layout_path),
        'layout_name': cfg.get('name', ''),
        'cells': gt_cells_from_config(cfg),
        'panel_centroids': {
            k: {'h': v[0], 's': v[1], 'v': v[2]} for k, v in panel_centroids.items()
        },
        'sample_counts': {k: len(v) for k, v in by_color.items()},
        'frames_used': len([r for r in frame_reports if not r.get('skip')]),
        'frames_scanned': len(frame_reports),
        'profile_out': str(out_path),
        'panel_only': panel_only,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Raport:          {report_path}')
    print('\nNa zawodach usuń config/panel_color_layout.json (lub ustaw DRONIADA_PANEL_LAYOUT=).')
    print('Restart panelu / Jetson po zmianie card_colors.json.')


if __name__ == '__main__':
    main()
