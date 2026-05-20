from __future__ import annotations
import argparse
import json
import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from release.frames import iter_dataset_frames
from release.panel_runtime import PanelConfig, PanelRuntime

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--panel-id', default='A')
    ap.add_argument('--angle-source', default='rmat_linear', choices=['json', 'rmat_linear', 'rmat_theta', 'geom', 'pnp'])
    ap.add_argument('--xy-mode', default='line_grid', choices=['grid_geom', 'grid_geom_white', 'warp_grid', 'geom_grid', 'line_grid'])
    ap.add_argument('--calibration', default=None)
    ap.add_argument('--require-reliable-report', action='store_true', help='nie generuj raportu kartek, gdy reprojection/siatka sa niewiarygodne')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    frames = iter_dataset_frames(args.dataset, max_frames=args.max_frames)
    if not frames:
        print(json.dumps({'ok': False, 'reason': 'no_frames', 'dataset': args.dataset}))
        return
    rt = PanelRuntime(PanelConfig(panel_id=args.panel_id, xy_mode=args.xy_mode, angle_source=args.angle_source, angle_calibration_path=args.calibration, require_reliable_report=args.require_reliable_report))
    results = rt.run_loop(frames)
    ok_n = sum(1 for r in results if r.ok)
    cards = sum(len(r.predictions) for r in results)
    summary = {'frames': len(results), 'panel_ok': ok_n, 'cards_total': cards}
    rows = [{'frame_id': r.frame_id, 'ok': r.ok, 'panel_id': r.panel_id, 'report_angle_deg': r.report_angle_deg, 'panel_angle_category': r.panel_angle_category, 'n_cards': len(r.predictions), 'grid_xy_reliable': r.grid_xy_reliable, 'report_lines': r.report_lines} for r in results]
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({**summary, 'out': args.out}, ensure_ascii=False))
    else:
        sample = [{k: v for k, v in row.items() if k != 'report_lines'} for row in rows[:3]]
        print(json.dumps({'summary': summary, 'sample': sample}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
