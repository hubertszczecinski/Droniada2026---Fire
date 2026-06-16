#!/usr/bin/env python3
"""
Profil uniwersalny: zdjęcia *2 + wiele nagrań panelu (layout JSON per wideo).

Centroidy z każdego nagrania; zakresy inRange tylko z *2 (zawody).
Klasyfikacja po najbliższym centroidzie + walidacja względem niego.

Przykład:
  ./scripts/build_universal_profile.sh
  python3 scripts/build_universal_profile.py \\
    --recording dataset/my_capture/Droniada_nag5.mov config/nag5_panel_layout.json \\
    --recording dataset/my_capture/Test.mov config/test_mov_gt.json
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
    collect_samples_from_video,
    gt_cells_from_config,
    load_panel_color_layout,
    merge_snapshot_centroids,
)


def _print_sample_summary(samples: dict[str, list]) -> None:
    print('\nŁącznie (wszystkie nagrania):')
    for name in sorted(samples):
        pts = samples[name]
        med = aggregate_hsv_samples(pts)
        if med is None:
            print(f'  {name:14s}  brak')
            continue
        print(f'  {name:14s}  n={len(pts):3d}  H={med[0]:.0f}  S={med[1]:.0f}  V={med[2]:.0f}')


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else _ROOT / path


def main() -> None:
    ap = argparse.ArgumentParser(description='Profil *2 + wiele nagrań panelu')
    ap.add_argument(
        '--recording', '-r',
        action='append',
        nargs=2,
        metavar=('VIDEO', 'LAYOUT'),
        default=[],
        help='Para: ścieżka wideo + layout JSON (można wielokrotnie)',
    )
    ap.add_argument('--cards-folder', type=Path, default=_ROOT / 'config/competition_cards')
    ap.add_argument('--out', '-o', type=Path, default=_DEFAULT_PATH)
    ap.add_argument('--frame-step', type=int, default=30)
    ap.add_argument('--max-reproj', type=float, default=15.0)
    ap.add_argument('--min-frame', type=int, default=400)
    ap.add_argument('--notes', default='*2 + uniwersalna kalibracja z nagrań panelu')
    args = ap.parse_args()

    recordings = list(args.recording)
    if not recordings:
        recordings = [
            ('dataset/my_capture/Droniada_nag5.mov', 'config/nag5_panel_layout.json'),
            ('dataset/my_capture/Test.mov', 'config/test_mov_gt.json'),
        ]

    weights = _ROOT / 'runs/pose/droniada_real_finetune/weights/best.pt'
    if weights.is_file():
        os.environ['DRONIADA_YOLO_POSE_WEIGHTS'] = str(weights)

    cards = _resolve(args.cards_folder)
    profile, _ = build_profile_from_folder(cards)
    recording_medians: dict[str, list[tuple[float, float, float]]] = {}
    sources: list[dict] = []

    print('=== Profil uniwersalny ( *2 + nagrania ) ===\n')

    for video_rel, layout_rel in recordings:
        video = _resolve(Path(video_rel))
        layout = _resolve(Path(layout_rel))
        cfg = load_panel_color_layout(layout)
        if not cfg:
            raise SystemExit(f'Brak layoutu: {layout}')
        if not video.is_file():
            raise SystemExit(f'Brak wideo: {video}')

        rotate = int(cfg.get('rotate_deg', 180))
        print(f'→ {video.name}  layout={layout.name}  komórki={len(gt_cells_from_config(cfg))}')

        by_color, _reports = collect_samples_from_video(
            video,
            cfg,
            frame_step=int(args.frame_step),
            max_reproj_px=float(args.max_reproj),
            min_frame=int(args.min_frame),
            rotate_deg=rotate,
        )
        medians = build_centroids_from_samples(by_color)
        src_entry: dict = {'video': str(video), 'layout': str(layout), 'medians': {}}
        for name, med in medians.items():
            h, s, v = med
            print(f'    {name:14s} n={len(by_color.get(name, [])):3d}  H={h:.0f} S={s:.0f} V={v:.0f}')
            recording_medians.setdefault(name, []).append(med)
            src_entry['medians'][name] = {'h': h, 's': s, 'v': v}
        sources.append(src_entry)

    _print_sample_summary({k: list(v) for k, v in recording_medians.items()})

    profile = merge_snapshot_centroids(
        profile,
        {k: list(v) for k, v in recording_medians.items()},
    )
    profile.centroids_by_cls = prune_centroids_by_hue(profile.centroids_by_cls, max_hue_spread=42.0)

    for cls_id in sorted(profile.centroids_by_cls):
        name = pc.CLASS_TO_COLOR[int(cls_id)]
        primary = _primary_competition_median(profile.centroids_by_cls[int(cls_id)])
        profile.ranges_by_cls[int(cls_id)] = ranges_from_medians(int(cls_id), [primary])

    calib_out = dict(profile.meta.get('calibration_samples') or {})
    for src in sources:
        tag = Path(src['video']).stem
        for name, med in src['medians'].items():
            entries = list(calib_out.get(name) or [])
            entries.append({
                'file': f'panel_{tag}_median',
                'h': med['h'],
                's': med['s'],
                'v': med['v'],
            })
            calib_out[name] = entries
    profile.meta['calibration_samples'] = calib_out
    profile.meta['recording_sources'] = sources

    out_path = _resolve(args.out)
    profile.save(out_path, source=str(cards), notes=args.notes, calibration_samples=calib_out)

    print(f'\nZapisano: {out_path}')
    print('Centroidy per kolor:')
    for cls_id in sorted(profile.centroids_by_cls):
        name = pc.CLASS_TO_COLOR[int(cls_id)]
        cents = profile.centroids_by_cls[int(cls_id)]
        parts = ', '.join(f'H={c[0]:.0f}' for c in cents)
        print(f'  {name:14s}  [{parts}]')


if __name__ == '__main__':
    main()
