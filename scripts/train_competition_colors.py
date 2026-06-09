#!/usr/bin/env python3
"""
Kalibracja kolorów przed zawodami: zdjęcia *2 od organizatora + nagrania tego samego panelu.

Domyślnie: config/competition_cards/*2.png + Test.mov / Test2.mov / Test3.mov
z układem config/test_mov_gt.json (te same kartki, ten sam panel).

Próbki z wideo są filtrowane (jasność, odległość H od *2) — odrzuca ciemny szum siatki,
który psuł fałszywą zieleń.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pipeline_competition as pc

from release.card_color_profile import (
    _DEFAULT_PATH,
    _collect_calibration_files,
    _pick_calibration_paths,
    _primary_competition_median,
    build_profile_from_folder,
    hue_distance,
    prune_centroids_by_hue,
    ranges_from_medians,
    sample_median_hsv,
    CardColorProfile,
    color_names,
)
from release.snapshot_cell_color import (
    aggregate_hsv_samples,
    build_centroids_from_samples,
    collect_samples_from_video,
    gt_cells_from_config,
    load_panel_color_layout,
    merge_snapshot_centroids,
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else _ROOT / path


def _photo_medians(folder: Path) -> dict[str, tuple[float, float, float]]:
    by_color = _collect_calibration_files(folder)
    out: dict[str, tuple[float, float, float]] = {}
    for name in color_names():
        paths = _pick_calibration_paths(by_color.get(name) or [])
        if not paths:
            continue
        meds: list[tuple[float, float, float]] = []
        for path in paths:
            import cv2

            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            med = sample_median_hsv(bgr)
            if med is not None:
                meds.append(med)
        if meds:
            out[name] = _primary_competition_median(meds)
    return out


def _build_cards_profile(
    folder: Path,
    *,
    fallback: Path | None,
) -> CardColorProfile:
    by_color = _collect_calibration_files(folder)
    missing = [n for n in color_names() if not by_color[n]]
    if not missing:
        profile, _ = build_profile_from_folder(folder)
        return profile

    fallback_path = _resolve(fallback) if fallback else None
    if fallback_path is None or not fallback_path.is_file():
        raise SystemExit(
            f'Brakuje zdjęć *2 dla: {", ".join(missing)}. '
            f'Dodaj do {folder} albo podaj --fallback-profile.',
        )
    old = CardColorProfile.from_json(fallback_path)
    import cv2

    ranges = {int(k): list(v) for k, v in old.ranges_by_cls.items()}
    cents: dict[int, list[tuple[float, float, float]]] = {
        int(k): list(v) for k, v in old.centroids_by_cls.items()
    }
    calib = dict(old.meta.get('calibration_samples') or {})
    source_files = dict(old.meta.get('source_files') or {})

    for name in color_names():
        if name in missing:
            print(f'  (fallback profilu: {name})')
            continue
        cls_id = int(pc.COLOR_TO_CLASS[name])
        use_paths = _pick_calibration_paths(by_color[name])
        medians: list[tuple[float, float, float]] = []
        entries: list[dict] = []
        for path in use_paths:
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f'Nie wczytano: {path}')
            med = sample_median_hsv(bgr)
            if med is None:
                raise ValueError(f'Za mało koloru: {path}')
            medians.append(med)
            h, s, v = med
            entries.append({'file': path.name, 'h': round(h, 1), 's': round(s, 1), 'v': round(v, 1)})
        primary = _primary_competition_median(medians)
        ranges[cls_id] = ranges_from_medians(cls_id, [primary])
        cents[cls_id] = list(medians)
        calib[name] = entries
        numbered = [p for p in use_paths if p.stem[-1].isdigit()]
        source_files[name] = use_paths[-1].name

    profile = CardColorProfile(
        ranges,
        source=str(folder),
        meta={'calibrated': True, 'source_files': source_files, 'calibration_samples': calib},
        centroids_by_cls=cents,
    )
    return profile


def _filter_video_samples(
    color: str,
    pts: list[tuple[float, float, float]],
    photo: tuple[float, float, float] | None,
) -> list[tuple[float, float, float]]:
    if not pts:
        return []
    name = str(color).upper()
    kept: list[tuple[float, float, float]] = []
    ph, ps, pv = photo if photo else (None, None, None)
    for h, s, v in pts:
        if float(v) < 55.0 or float(s) < 35.0:
            continue
        if name == 'ZIELONA':
            if float(v) < 88.0:
                continue
            if float(s) < 72.0:
                continue
        if name == 'ZOLTA' and float(v) < 58.0:
            continue
        if photo is not None:
            if hue_distance(float(h), float(ph)) > 26.0:
                continue
            if float(v) < max(85.0, float(pv) * 0.72):
                continue
        kept.append((float(h), float(s), float(v)))
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description='Trening kolorów: *2 + nagrania Test panelu')
    ap.add_argument(
        '--video', '-v',
        action='append',
        default=[],
        help='Nagranie panelu (można wielokrotnie); domyślnie Test.mov Test2 Test3',
    )
    ap.add_argument(
        '--layout', '-l',
        type=Path,
        default=_ROOT / 'config/test_mov_gt.json',
        help='Układ kart na panelu (JSON)',
    )
    ap.add_argument('--cards-folder', type=Path, default=_ROOT / 'config/competition_cards')
    ap.add_argument('--fallback-profile', type=Path, default=_DEFAULT_PATH)
    ap.add_argument('--out', '-o', type=Path, default=_DEFAULT_PATH)
    ap.add_argument('--frame-step', type=int, default=20)
    ap.add_argument('--max-reproj', type=float, default=15.0)
    ap.add_argument('--min-frame', type=int, default=350)
    ap.add_argument('--notes', default='*2 organizator + nagrania Test panelu (pre-zawody)')
    args = ap.parse_args()

    videos = [str(v) for v in args.video] if args.video else [
        'dataset/my_capture/Test.mov',
        'dataset/my_capture/Test2.mov',
        'dataset/my_capture/Test3.mov',
    ]
    layout = _resolve(args.layout)
    cards = _resolve(args.cards_folder)
    cfg = load_panel_color_layout(layout)
    if not cfg:
        raise SystemExit(f'Brak layoutu: {layout}')

    weights = _ROOT / 'runs/pose/droniada_real_finetune/weights/best.pt'
    if weights.is_file():
        os.environ['DRONIADA_YOLO_POSE_WEIGHTS'] = str(weights)

    print('=== Trening kolorów (zawody) ===')
    print(f'Karty *2: {cards}')
    print(f'Layout:   {layout.name}  komórki={len(gt_cells_from_config(cfg))}')
    print(f'Wideo:    {", ".join(Path(v).name for v in videos)}\n')

    profile = _build_cards_profile(cards, fallback=_resolve(args.fallback_profile))
    photo_medians = _photo_medians(cards)
    recording_medians: dict[str, list[tuple[float, float, float]]] = {}
    sources: list[dict] = []

    for video_rel in videos:
        video = _resolve(Path(video_rel))
        if not video.is_file():
            print(f'POMINIĘTO (brak pliku): {video}')
            continue
        rotate = int(cfg.get('rotate_deg', 180))
        print(f'→ {video.name}')

        raw, _reports = collect_samples_from_video(
            video,
            cfg,
            frame_step=int(args.frame_step),
            max_reproj_px=float(args.max_reproj),
            min_frame=int(args.min_frame),
            rotate_deg=rotate,
        )
        filtered_map: dict[str, list[tuple[float, float, float]]] = {}
        src_entry: dict = {'video': str(video), 'medians': {}, 'raw_n': {}, 'filtered_n': {}}
        for name in sorted(raw):
            photo = photo_medians.get(name)
            filt = _filter_video_samples(name, list(raw[name]), photo)
            filtered_map[name] = filt
            src_entry['raw_n'][name] = len(raw[name])
            src_entry['filtered_n'][name] = len(filt)
            med = aggregate_hsv_samples(filt)
            if med is None:
                print(f'    {name:14s}  surowe={len(raw[name]):3d}  po filtrze=0  (pominięto)')
                continue
            h, s, v = med
            print(
                f'    {name:14s}  surowe={len(raw[name]):3d}  '
                f'filt={len(filt):3d}  H={h:.0f} S={s:.0f} V={v:.0f}',
            )
            recording_medians.setdefault(name, []).append(med)
            src_entry['medians'][name] = {'h': h, 's': s, 'v': v}
        sources.append(src_entry)

    profile = merge_snapshot_centroids(
        profile,
        {k: list(v) for k, v in recording_medians.items()},
    )
    profile.centroids_by_cls = prune_centroids_by_hue(profile.centroids_by_cls, max_hue_spread=32.0)

    for cls_id in sorted(profile.centroids_by_cls):
        name = pc.CLASS_TO_COLOR[int(cls_id)]
        cents = profile.centroids_by_cls[int(cls_id)]
        photo = photo_medians.get(name)
        if photo is not None:
            primary = photo
        else:
            primary = _primary_competition_median(cents)
        profile.centroids_by_cls[int(cls_id)] = [
            c for c in cents
            if hue_distance(c[0], primary[0]) <= 32.0 and float(c[2]) >= max(85.0, float(primary[2]) * 0.68)
        ] or [primary]
        profile.ranges_by_cls[int(cls_id)] = ranges_from_medians(int(cls_id), [primary])

    calib_out = dict(profile.meta.get('calibration_samples') or {})
    for src in sources:
        tag = Path(src['video']).stem
        for name, med in src.get('medians', {}).items():
            entries = list(calib_out.get(name) or [])
            entries.append({
                'file': f'panel_{tag}_median',
                'h': med['h'],
                's': med['s'],
                'v': med['v'],
            })
            calib_out[name] = entries
    profile.meta['calibration_samples'] = calib_out
    profile.meta['training_sources'] = sources
    profile.meta['photo_medians'] = {
        k: {'h': v[0], 's': v[1], 'v': v[2]} for k, v in photo_medians.items()
    }

    out_path = _resolve(args.out)
    profile.save(out_path, source=str(cards), notes=args.notes, calibration_samples=calib_out)

    print(f'\nZapisano: {out_path}')
    print('Centroidy (klasyfikacja live):')
    for cls_id in sorted(profile.centroids_by_cls):
        name = pc.CLASS_TO_COLOR[int(cls_id)]
        parts = ', '.join(f'H={c[0]:.0f} V={c[2]:.0f}' for c in profile.centroids_by_cls[int(cls_id)])
        print(f'  {name:14s}  [{parts}]')


if __name__ == '__main__':
    main()
