from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pipeline_competition as pc
from module_panel.analyze import analyze_panel_image, detect_panel_corners_for_module_b
from module_panel.reliability import orbit_step_from_pose
from module_pose.api import intrinsics_from_pose_json, load_pose_gt_json


def _parse_orbit_steps(raw: str):
    if raw.strip().lower() in ('', 'any', 'all'):
        return None
    if raw.strip().lower() == 'frontal':
        return {0}
    return {int(x.strip()) for x in raw.split(',') if x.strip()}

_BGR_BY_COLOR: Dict[str, Tuple[int, int, int]] = {
    'CZERWONA': (40, 40, 230),
    'ZIELONA': (50, 190, 50),
    'NIEBIESKA': (230, 120, 40),
    'ZOLTA': (20, 220, 240),
    'FIOLETOWA': (200, 60, 200),
    'POMARANCZOWA': (30, 130, 240),
    'UNKNOWN': (220, 220, 220),
}


def _short_color(name: str) -> str:
    return {
        'CZERWONA': 'RED',
        'ZIELONA': 'GREEN',
        'NIEBIESKA': 'BLUE',
        'ZOLTA': 'YELLOW',
        'FIOLETOWA': 'PURPLE',
        'POMARANCZOWA': 'ORANGE',
    }.get(name, name[:8])


def _match_gt_by_color(gt: List[Dict[str, Any]], preds: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
    by_color: Dict[str, List[Dict[str, Any]]] = {}
    for rec in gt:
        by_color.setdefault(str(rec['color']), []).append(rec)
    matched: List[Optional[Dict[str, Any]]] = []
    for pred in preds:
        arr = by_color.get(str(pred['color']), [])
        if arr:
            matched.append(arr.pop(0))
        else:
            matched.append(None)
    return matched


def _score_predictions(gt: List[Dict[str, Any]], preds: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    gt_color = [g['color'] for g in gt]
    pred_color = [p['color'] for p in preds]
    gt_cxy = [(g['color'], g['x'], g['y']) for g in gt]
    pred_cxy = [(p['color'], p['x'], p['y']) for p in preds]

    def overlap(a: List[Any], b: List[Any]) -> int:
        b_left = list(b)
        hits = 0
        for item in a:
            if item in b_left:
                hits += 1
                b_left.remove(item)
        return hits

    return overlap(gt_color, pred_color), overlap(gt_cxy, pred_cxy), len(gt)


def _draw_text_panel(lines: List[Tuple[str, Tuple[int, int, int]]], width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    y = 30
    for text, color in lines:
        cv2.putText(panel, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        y += 26
    return panel


def _draw_original(
    image: np.ndarray,
    det: List[Tuple[int, float, float, float, float]],
    corners: Optional[np.ndarray],
    preds: List[Dict[str, Any]],
    matched_gt: List[Optional[Dict[str, Any]]],
) -> np.ndarray:
    vis = image.copy()
    h, w = vis.shape[:2]
    if corners is not None and corners.shape == (4, 2):
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (0, 255, 255), 3)
        for idx, p in enumerate(corners.astype(np.int32)):
            cv2.circle(vis, tuple(p), 7, (0, 255, 255), -1)
            cv2.putText(vis, str(idx), (int(p[0]) + 7, int(p[1]) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    for idx, (cls_id, cx_n, cy_n, bw_n, bh_n) in enumerate(det):
        color_name = pc.CLASS_TO_COLOR.get(cls_id, 'UNKNOWN')
        bgr = _BGR_BY_COLOR.get(color_name, _BGR_BY_COLOR['UNKNOWN'])
        cx = int(round(cx_n * w))
        cy = int(round(cy_n * h))
        bw = int(round(bw_n * w))
        bh = int(round(bh_n * h))
        p1 = (max(0, cx - bw // 2), max(0, cy - bh // 2))
        p2 = (min(w - 1, cx + bw // 2), min(h - 1, cy + bh // 2))
        cv2.rectangle(vis, p1, p2, bgr, 2)
        cv2.circle(vis, (cx, cy), 4, bgr, -1)
        pred = preds[idx] if idx < len(preds) else None
        gt = matched_gt[idx] if idx < len(matched_gt) else None
        pred_xy = f"P:{pred['x']},{pred['y']}" if pred else 'P:-'
        gt_xy = f"GT:{gt['x']},{gt['y']}" if gt else 'GT:-'
        ok = bool(pred and gt and pred['x'] == gt['x'] and pred['y'] == gt['y'])
        status = 'OK' if ok else 'MISS'
        label = f"{_short_color(color_name)} {pred_xy} {gt_xy} {status}"
        text_color = (30, 170, 30) if ok else (30, 30, 230)
        cv2.putText(vis, label, (p1[0], max(18, p1[1] - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)
    return vis


def _draw_warped(
    warped: np.ndarray,
    homography: np.ndarray,
    det: List[Tuple[int, float, float, float, float]],
    image_shape: Tuple[int, int],
    preds: List[Dict[str, Any]],
    matched_gt: List[Optional[Dict[str, Any]]],
) -> np.ndarray:
    vis = warped.copy()
    h, w = vis.shape[:2]
    step_x = w / 10.0
    step_y = h / 10.0
    for i in range(11):
        x = int(round(i * step_x))
        y = int(round(i * step_y))
        cv2.line(vis, (x, 0), (x, h - 1), (230, 230, 230), 1)
        cv2.line(vis, (0, y), (w - 1, y), (230, 230, 230), 1)
    for i in range(10):
        cv2.putText(vis, str(i + 1), (int(i * step_x + step_x * 0.42), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(vis, str(10 - i), (5, int(i * step_y + step_y * 0.58)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    img_h, img_w = image_shape
    src_pts = np.array([[[cx * img_w, cy * img_h]] for _, cx, cy, _, _ in det], dtype=np.float32)
    if len(src_pts):
        warped_pts = cv2.perspectiveTransform(src_pts, homography).reshape(-1, 2)
    else:
        warped_pts = np.zeros((0, 2), dtype=np.float32)

    for idx, (cls_id, _cx, _cy, _bw, _bh) in enumerate(det):
        color_name = pc.CLASS_TO_COLOR.get(cls_id, 'UNKNOWN')
        bgr = _BGR_BY_COLOR.get(color_name, _BGR_BY_COLOR['UNKNOWN'])
        pred = preds[idx] if idx < len(preds) else None
        gt = matched_gt[idx] if idx < len(matched_gt) else None
        if gt is not None:
            gx = int(round((float(gt['x']) - 0.5) * step_x))
            gy = int(round(h - (float(gt['y']) - 0.5) * step_y))
            cv2.drawMarker(vis, (gx, gy), (255, 255, 255), cv2.MARKER_CROSS, 18, 3)
            cv2.drawMarker(vis, (gx, gy), (0, 0, 0), cv2.MARKER_CROSS, 18, 1)
        if idx < len(warped_pts):
            px, py = warped_pts[idx]
            p = (int(round(px)), int(round(py)))
            ok = bool(pred and gt and pred['x'] == gt['x'] and pred['y'] == gt['y'])
            cv2.circle(vis, p, 9, bgr, -1)
            cv2.circle(vis, p, 11, (30, 170, 30) if ok else (30, 30, 230), 2)
            label = f"{_short_color(color_name)} {pred['x']},{pred['y']}" if pred else _short_color(color_name)
            cv2.putText(vis, label, (p[0] + 10, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
    return vis


def _resize_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / float(w)
    return cv2.resize(img, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def _compose_page(original: np.ndarray, warped: np.ndarray, text_panel: np.ndarray) -> np.ndarray:
    left = _resize_to_width(original, 900)
    right = _resize_to_width(warped, 900)
    text = _resize_to_width(text_panel, 900)
    gap = np.full((16, 900, 3), 255, dtype=np.uint8)
    page = np.vstack([left, gap, right, gap, text])
    return page


def _write_index(out_dir: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    parts = [
        '<!doctype html><html><head><meta charset="utf-8">',
        '<title>Droniada dataset debug</title>',
        '<style>body{font-family:system-ui,sans-serif;margin:24px;background:#f7f7f7;color:#111}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px}'
        '.card{background:white;border:1px solid #ddd;padding:12px}'
        'img{max-width:100%;height:auto;border:1px solid #ddd}'
        '.bad{color:#b00020;font-weight:700}.ok{color:#166534;font-weight:700}'
        'code{background:#eee;padding:2px 4px}</style></head><body>',
        '<h1>Droniada dataset debug</h1>',
        '<p>Overlay: zolte rogi = panel, kolorowe prostokaty = YOLO, P = predykcja, GT = raport. '
        'Na wyprostowanym panelu krzyzyk pokazuje srodek komorki GT, kropka pokazuje wykryta kartke.</p>',
        '<pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>',
        '<div class="grid">',
    ]
    for row in rows:
        cls = 'ok' if row['cxy_hits'] == row['gt_cards'] and row['angle_ok'] else 'bad'
        parts.append(
            '<div class="card">'
            f"<h3>{html.escape(row['stem'])}</h3>"
            f"<p class='{cls}'>CXY {row['cxy_hits']}/{row['gt_cards']} | "
            f"angle pred={row['pred_angle']} gt={row['gt_angle']} | "
            f"cat pred={html.escape(row['pred_category'])} gt={html.escape(row['gt_category'])}</p>"
            f"<p>reproj={row['reproj']:.2f}px | corner={html.escape(row['corner_source'])}</p>"
            f"<a href='{html.escape(row['file'])}'><img src='{html.escape(row['file'])}'></a>"
            '</div>'
        )
    parts.append('</div></body></html>')
    (out_dir / 'index.html').write_text('\n'.join(parts), encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset')
    ap.add_argument('--out-dir', default='dataset/debug_module_b')
    ap.add_argument('--max-images', type=int, default=300)
    ap.add_argument('--angle-source', default='rmat_linear')
    ap.add_argument('--xy-mode', default='grid_geom_white', choices=['grid_geom', 'grid_geom_white', 'warp_grid', 'geom_grid', 'line_grid'])
    ap.add_argument('--reliable-only', action='store_true', help='tylko grid_xy_reliable v2')
    ap.add_argument('--orbit-steps', default='any', help='frontal (=0), 0,11, any')
    ap.add_argument('--only-errors', action='store_true')
    ap.add_argument('--limit-output', type=int, default=0, help='0 = save all selected images')
    ap.add_argument('--corner-source', default='panel', choices=['panel', 'yolo'], help='panel=nasz detektor, yolo=szybka ramka z kartek do debugowania XY')
    args = ap.parse_args()
    allowed_orbit = _parse_orbit_steps(args.orbit_steps)

    base = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    summary = {
        'images': 0,
        'gt_cards': 0,
        'color_hits': 0,
        'cxy_hits': 0,
        'angle_hits': 0,
        'category_hits': 0,
        'xy_mode': args.xy_mode,
        'angle_source': args.angle_source,
        'orbit_steps': sorted(allowed_orbit) if allowed_orbit is not None else None,
        'reliable_only': bool(args.reliable_only),
    }

    saved = 0
    for i in range(args.max_images):
        stem = f'img_{i}'
        img_path = base / 'images' / f'{stem}.png'
        yolo_path = base / 'labels_yolo' / f'{stem}.txt'
        report_path = base / 'labels_raport' / f'{stem}.txt'
        pose_path = base / 'labels_pose' / f'{stem}.json'
        if not all(p.is_file() for p in (img_path, yolo_path, report_path)):
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        det = pc.load_yolo(str(yolo_path))
        gt = pc.load_gt(str(report_path))
        if not gt:
            continue
        pose_json = load_pose_gt_json(str(pose_path)) if pose_path.is_file() else None
        if allowed_orbit is not None:
            step = orbit_step_from_pose(pose_json if isinstance(pose_json, dict) else None)
            if step is not None and int(step) not in allowed_orbit:
                continue
        if isinstance(pose_json, dict) and pose_json.get('intrinsics') is not None:
            k, dist = intrinsics_from_pose_json(pose_json)
        else:
            h, w = img.shape[:2]
            k = np.array([[1000.0, 0.0, w / 2.0], [0.0, 1000.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            dist = np.zeros((4, 1), dtype=np.float32)
        if args.corner_source == 'yolo':
            corners = pc.detect_corners_yolo(det)
            corner_source = 'yolo_bbox_fast'
            if corners is not None:
                from module_panel.result import PanelAnalyzeResult
                from module_panel.warp import warp_panel_rect
                from module_panel.analyze import analyze_panel_from_warped
                warped, h_mat = warp_panel_rect(img, corners)
                preds = analyze_panel_from_warped(warped, det, corners, xy_mode=args.xy_mode, src_wh=(img.shape[1], img.shape[0]))
                res = PanelAnalyzeResult(predictions=preds, warped_bgr=warped, homography=h_mat, report_angle_deg=pc.angle_from_geom(corners), panel_angle_category='debug_yolo', meta={'corner_source': corner_source, 'reproj_mean_px': float('nan'), 'grid_xy_reliable': False})
            else:
                res = analyze_panel_image(
                    img, det, k=k, dist=dist, xy_mode=args.xy_mode, angle_source=args.angle_source,
                    pose_json=pose_json if isinstance(pose_json, dict) else None,
                    allowed_orbit_steps=allowed_orbit,
                )
        else:
            res = analyze_panel_image(
                img, det, k=k, dist=dist, xy_mode=args.xy_mode, angle_source=args.angle_source,
                pose_json=pose_json if isinstance(pose_json, dict) else None,
                allowed_orbit_steps=allowed_orbit,
            )
            corners, corner_source = detect_panel_corners_for_module_b(
                img,
                det,
                prefer_geom_vp=(args.xy_mode == 'geom_grid'),
                prefer_line_grid=(args.xy_mode == 'line_grid'),
                k=k,
                dist=dist,
            )
            corner_source = str(res.meta.get('corner_source', corner_source))
        if args.reliable_only and not bool(res.meta.get('grid_xy_reliable', False)):
            continue
        matched_gt = _match_gt_by_color(gt, res.predictions)
        color_hits, cxy_hits, gt_cards = _score_predictions(gt, res.predictions)
        gt_angle = int(gt[0]['angle_deg'])
        gt_category = str((pose_json or {}).get('panel', {}).get('panel_angle_category', 'unknown')) if isinstance(pose_json, dict) else 'unknown'
        angle_ok = int(res.report_angle_deg == gt_angle)
        category_ok = int(res.panel_angle_category == gt_category) if gt_category != 'unknown' else angle_ok

        summary['images'] += 1
        summary['gt_cards'] += gt_cards
        summary['color_hits'] += color_hits
        summary['cxy_hits'] += cxy_hits
        summary['angle_hits'] += angle_ok
        summary['category_hits'] += category_ok

        is_error = cxy_hits < gt_cards or not angle_ok
        if args.only_errors and not is_error:
            continue
        if args.limit_output and saved >= args.limit_output:
            continue

        original_vis = _draw_original(img, det, corners, res.predictions, matched_gt)
        warped_vis = _draw_warped(res.warped_bgr, res.homography, det, img.shape[:2], res.predictions, matched_gt)
        lines = [
            (f'{stem}', (0, 0, 0)),
            (f'cards CXY: {cxy_hits}/{gt_cards} | colors: {color_hits}/{gt_cards}', (30, 170, 30) if cxy_hits == gt_cards else (30, 30, 230)),
            (f'angle: pred {res.report_angle_deg} / gt {gt_angle} | category {res.panel_angle_category} / {gt_category}', (30, 170, 30) if angle_ok else (30, 30, 230)),
            (f"corner source: {corner_source} | reproj: {float(res.meta.get('reproj_mean_px', float('nan'))):.2f}px", (0, 0, 0)),
            (
                f"reliable v2: {bool(res.meta.get('grid_xy_reliable', False))} "
                f"(legacy {bool(res.meta.get('grid_xy_reliable_legacy', False))})",
                (0, 0, 0),
            ),
            (
                f"inliers: {res.meta.get('homography_inliers', '?')} | "
                f"orbit: {res.meta.get('orbit_step_index', '?')} | xy: {res.meta.get('xy_source', '?')}",
                (0, 0, 0),
            ),
            (f"panel cat: {gt_category}", (0, 0, 0)),
        ]
        reasons = res.meta.get('grid_xy_reliable_reasons')
        if isinstance(reasons, dict):
            lines.append((f"reliable gates: {reasons}", (0, 0, 0)))
        for idx, pred in enumerate(res.predictions):
            gt_i = matched_gt[idx] if idx < len(matched_gt) else None
            gt_txt = f"{gt_i['x']},{gt_i['y']}" if gt_i else '-'
            ok = bool(gt_i and pred['x'] == gt_i['x'] and pred['y'] == gt_i['y'])
            lines.append((f"{_short_color(str(pred['color']))}: pred {pred['x']},{pred['y']} gt {gt_txt} {'OK' if ok else 'MISS'}", (30, 170, 30) if ok else (30, 30, 230)))
        page = _compose_page(original_vis, warped_vis, _draw_text_panel(lines, 900, max(190, 30 + 26 * len(lines))))
        file_name = f'{stem}_debug.png'
        cv2.imwrite(str(out_dir / file_name), page)
        rows.append({
            'stem': stem,
            'file': file_name,
            'cxy_hits': cxy_hits,
            'gt_cards': gt_cards,
            'pred_angle': int(res.report_angle_deg),
            'gt_angle': gt_angle,
            'angle_ok': bool(angle_ok),
            'pred_category': str(res.panel_angle_category),
            'gt_category': gt_category,
            'reproj': float(res.meta.get('reproj_mean_px', float('nan'))),
            'corner_source': corner_source,
            'grid_reliable': bool(res.meta.get('grid_xy_reliable', False)),
            'homography_inliers': res.meta.get('homography_inliers'),
            'orbit_step': res.meta.get('orbit_step_index'),
        })
        saved += 1
        if saved % 10 == 0:
            print(f'[visualize] saved={saved} processed={summary["images"]}', flush=True)

    if summary['gt_cards']:
        summary['color_acc_pct'] = round(100.0 * summary['color_hits'] / summary['gt_cards'], 2)
        summary['cxy_acc_pct'] = round(100.0 * summary['cxy_hits'] / summary['gt_cards'], 2)
    if summary['images']:
        summary['angle_acc_pct'] = round(100.0 * summary['angle_hits'] / summary['images'], 2)
        summary['category_acc_pct'] = round(100.0 * summary['category_hits'] / summary['images'], 2)
    summary['saved_visualizations'] = saved
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    _write_index(out_dir, rows, summary)
    print(json.dumps({'out_dir': str(out_dir), 'index': str(out_dir / 'index.html'), **summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
