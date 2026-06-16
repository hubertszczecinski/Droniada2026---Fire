#!/usr/bin/env python3
"""Uczenie kolorów z migawek Test.mov + weryfikacja GT (tylko migawki)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from release.card_color_profile import (
    _DEFAULT_PATH,
    build_profile_from_folder,
    load_active_profile,
)
from release.snapshot_cell_color import (
    collect_gt_samples_from_session,
    load_test_mov_gt,
    merge_snapshot_centroids,
)


def _session_dirs(root: Path, explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(p) if Path(p).is_absolute() else root / p for p in explicit]
    out: list[Path] = []
    base = root / 'dataset' / 'live_dashboard'
    if not base.is_dir():
        return out
    for p in sorted(base.glob('session_*'), reverse=True):
        if (p / 'snapshots').is_dir() and any((p / 'snapshots').glob('*_snapshot.json')):
            out.append(p)
    return out


def _print_report(reports: list[dict], cells_cfg: list[dict]) -> tuple[int, int]:
    labels = [str(c.get('user_label', f"{c['row']},{c['col']}")) for c in cells_cfg]
    total = 0
    ok = 0
    print('\n=== Weryfikacja migawek (GT) ===')
    for rep in reports:
        if rep.get('skip'):
            print(f"  SKIP {rep['frame_id']}: {rep['skip']}")
            continue
        hits = []
        for lab in labels:
            cell = (rep.get('cells') or {}).get(lab, {})
            if not cell:
                continue
            total += 1
            mark = 'OK' if cell.get('ok') else 'FAIL'
            if cell.get('ok'):
                ok += 1
            got = cell.get('got') or '—'
            exp = cell.get('expected', '?')
            med = cell.get('median_hsv')
            hsv_s = ''
            if med:
                hsv_s = f" H={med[0]:.0f} S={med[1]:.0f} V={med[2]:.0f}"
            hits.append(f'{lab}:{mark}({got}){hsv_s}')
        print(f"  {rep['frame_id']} reprojB={rep.get('reproj_b', 0):.1f}  " + ' | '.join(hits))
    print(f'\nTrafienia GT: {ok}/{total}')
    return ok, total


def main() -> None:
    ap = argparse.ArgumentParser(description='Ucz kolory z migawek Test.mov i zweryfikuj GT.')
    ap.add_argument(
        '--cards-folder', '-f',
        type=Path,
        default=_ROOT / 'config' / 'competition_cards',
    )
    ap.add_argument(
        '--gt-config',
        type=Path,
        default=_ROOT / 'config' / 'test_mov_gt.json',
    )
    ap.add_argument(
        '--out', '-o',
        type=Path,
        default=_DEFAULT_PATH,
    )
    ap.add_argument(
        '--session',
        action='append',
        default=[],
        help='Katalog sesji (można wielokrotnie). Domyślnie: wszystkie session_* z migawkami.',
    )
    ap.add_argument('--min-frame', type=int, default=250, help='Pomiń wczesne klatki (zły warp).')
    ap.add_argument('--max-reproj', type=float, default=28.0)
    ap.add_argument('--verify-only', action='store_true', help='Tylko weryfikacja (profil z --out).')
    args = ap.parse_args()

    cfg = load_test_mov_gt(args.gt_config)
    if not cfg:
        raise SystemExit(f'Brak GT: {args.gt_config}')
    video = _ROOT / str(cfg.get('video', 'dataset/my_capture/Test.mov'))
    rotate = int(cfg.get('rotate_deg', 180))
    cells = cfg.get('cells') or []

    if not args.verify_only:
        profile, samples = build_profile_from_folder(args.cards_folder)
        all_samples: dict[str, list] = {c['color']: [] for c in cells}
        sessions = _session_dirs(_ROOT, args.session)
        if not sessions:
            raise SystemExit('Brak sesji z migawkami w dataset/live_dashboard/session_*')
        print(f'Uczenie z {len(sessions)} sesji, wideo: {video.name}')
        for sess in sessions:
            print(f'  → {sess.name}')
            by_color, _ = collect_gt_samples_from_session(
                sess, video, cfg,
                max_reproj_b=float(args.max_reproj),
                min_frame=int(args.min_frame),
                rotate_deg=rotate,
            )
            for name, pts in by_color.items():
                all_samples[name].extend(pts)
        for name, pts in all_samples.items():
            print(f'  próbki z migawek {name}: {len(pts)}')
        profile = merge_snapshot_centroids(profile, all_samples)
        calib = profile.meta.get('calibration_samples') or {}
        for name, pts in all_samples.items():
            if not pts:
                continue
            import numpy as np
            arr = np.array(pts)
            med = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])), float(np.median(arr[:, 2])))
            entries = list(calib.get(name) or [])
            entries.append({
                'file': 'snapshot_median',
                'h': round(med[0], 1),
                's': round(med[1], 1),
                'v': round(med[2], 1),
            })
            calib[name] = entries
        profile.meta['calibration_samples'] = calib
        profile.meta['notes'] = 'Kalibracja *2 + mediany z migawek Test.mov'
        profile.save(
            args.out,
            source=str(args.cards_folder.resolve()),
            notes=profile.meta['notes'],
            samples=samples,
            calibration_samples=calib,
        )
        print(f'Zapisano: {args.out}')

    os.environ['DRONIADA_CARD_COLORS'] = str(args.out.resolve())
    load_active_profile(force=True)

    reports: list[dict] = []
    for sess in _session_dirs(_ROOT, args.session):
        _, rep = collect_gt_samples_from_session(
            sess, video, cfg,
            max_reproj_b=float(args.max_reproj),
            min_frame=int(args.min_frame),
            rotate_deg=rotate,
        )
        reports.extend(rep)

    ok, total = _print_report(reports, cells)
    raise SystemExit(0 if total > 0 and ok == total else 1)


if __name__ == '__main__':
    main()
