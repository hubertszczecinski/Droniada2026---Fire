from __future__ import annotations
import argparse
import json
import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from release.frames import iter_dataset_frames
from release.pose_runtime import PoseConfig, PoseRuntime

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--intrinsics-json', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    frames = iter_dataset_frames(args.dataset, max_frames=args.max_frames)
    if not frames:
        print(json.dumps({'ok': False, 'reason': 'no_frames', 'dataset': args.dataset}))
        return
    rt = PoseRuntime(PoseConfig(use_pose_json_intrinsics=args.intrinsics_json))
    results = rt.run_loop(frames)
    ok_n = sum(1 for r in results if r.ok)
    summary = {'frames': len(results), 'pose_ok': ok_n, 'pose_ok_pct': 100.0 * ok_n / len(results)}
    rows = [{'frame_id': r.frame_id, 'ok': r.ok, 'roll_deg': r.roll_deg, 'pitch_deg': r.pitch_deg, 'yaw_deg': r.yaw_deg, 'distance_m': r.distance_m, 'confidence': r.confidence, 'method': r.method, 'reproj_mean_px': r.reproj_mean_px} for r in results]
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({**summary, 'out': args.out}, ensure_ascii=False))
    else:
        print(json.dumps({'summary': summary, 'frames': rows[:5], 'truncated': len(rows) > 5}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
