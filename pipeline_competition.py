import argparse
import csv
import json
import math
import os
import re
import cv2
import numpy as np
CLASS_TO_COLOR = {0: 'CZERWONA', 1: 'ZIELONA', 2: 'NIEBIESKA', 3: 'ZOLTA', 4: 'FIOLETOWA', 5: 'POMARANCZOWA'}
COLOR_TO_CLASS = {v: k for k, v in CLASS_TO_COLOR.items()}
ANGLE_CLASSES = [0, 45, 90]
RECT_W = 1000
RECT_H = 500

def parse_report_line(line):
    """Parsuje linię raportu (regulamin 2026 + starsze logi z datasetu)."""
    from module_panel.competition_report import (
        normalize_color_name,
        parse_competition_report_line,
        snap_panel_angle_deg,
    )
    rec = parse_competition_report_line(line)
    if rec is not None and rec.get('event') == 'card_detected':
        ck = rec.get('color_key')
        color = ck if ck else str(rec.get('color', '')).upper()
        return {
            'panel': rec['panel'],
            'angle_deg': int(rec['angle_deg']),
            'x': int(rec['x']),
            'y': int(rec['y']),
            'color': color,
        }
    line = line.strip()
    if not line:
        return None
    m = re.search(
        'Panel:\\s*([A-Z])\\s*\\((\\d+)°(?:\\s*\\|\\s*[^)]+)?\\)\\s*\\|\\s*'
        'Pozycja:\\s*Wiersz\\s*(\\d+),\\s*Kolumna\\s*(\\d+)\\s*\\|\\s*'
        'Kolor:\\s*([A-Za-zĄąĆćĘęŁłŃńÓóŚśŹźŻż]+)',
        line,
    )
    if not m:
        return None
    ck = normalize_color_name(m.group(5))
    return {
        'panel': m.group(1),
        'angle_deg': snap_panel_angle_deg(int(m.group(2))),
        'x': int(m.group(4)),
        'y': int(m.group(3)),
        'color': ck if ck else m.group(5).upper(),
    }

def load_gt(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = parse_report_line(line)
            if rec is not None:
                out.append(rec)
    return out

def load_yolo(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5:
                continue
            out.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def detect_corners_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 85, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = gray.shape[0] * gray.shape[1]
    best = None
    best_area = 0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.08 * area_img:
            continue
        if a > best_area:
            best_area = a
            best = cv2.boxPoints(cv2.minAreaRect(c))
    if best is None:
        return None
    return order_points(best)

def detect_corners_yolo(det):
    if not det:
        return None
    centers = np.array([(cx * 1024.0, cy * 1024.0) for _, cx, cy, _, _ in det], dtype=np.float32)
    sizes = np.array([(w * 1024.0, h * 1024.0) for _, _, _, w, h in det], dtype=np.float32)
    c = centers.mean(axis=0)
    bw = max(120.0, float(np.median(sizes[:, 0])) * 12.5)
    bh = max(60.0, float(np.median(sizes[:, 1])) * 12.5)
    rect = cv2.minAreaRect(centers)
    ang = np.deg2rad(rect[2]) if len(centers) >= 3 else 0.0
    ca, sa = (np.cos(ang), np.sin(ang))
    ux = np.array([ca, sa], dtype=np.float32)
    uy = np.array([-sa, ca], dtype=np.float32)
    hw, hh = (bw / 2.0, bh / 2.0)
    pts = np.array([c - ux * hw - uy * hh, c + ux * hw - uy * hh, c + ux * hw + uy * hh, c - ux * hw + uy * hh], dtype=np.float32)
    return order_points(pts)

def homography_from_corners(corners):
    dst = np.array([[0, 0], [RECT_W - 1, 0], [RECT_W - 1, RECT_H - 1], [0, RECT_H - 1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(corners, dst)

def map_xy_geom(det, h):
    out = []
    for cls_id, cx, cy, _w, _h in det:
        src = np.array([[[cx * 1024.0, cy * 1024.0]]], dtype=np.float32)
        p = cv2.perspectiveTransform(src, h)[0][0]
        gx = int(np.clip(p[0] / (RECT_W / 10), 0, 9)) + 1
        gy = int(np.clip((RECT_H - p[1]) / (RECT_H / 10), 0, 9)) + 1
        out.append({'x': gx, 'y': gy, 'color': CLASS_TO_COLOR.get(cls_id, 'UNKNOWN')})
    return out

def project_points(det, h):
    pts = []
    for cls_id, cx, cy, _w, _h in det:
        src = np.array([[[cx * 1024.0, cy * 1024.0]]], dtype=np.float32)
        p = cv2.perspectiveTransform(src, h)[0][0]
        pts.append({'px': float(p[0]), 'py': float(p[1]), 'color': CLASS_TO_COLOR.get(cls_id, 'UNKNOWN')})
    return pts

def snap_lines_to_grid(lines, expected=11, low=0.0, high=1.0):
    if len(lines) < 2:
        return np.linspace(low, high, expected)
    vals = np.array(sorted(lines), dtype=np.float32)
    vals = vals[(vals >= low) & (vals <= high)]
    if len(vals) < 2:
        return np.linspace(low, high, expected)
    q = np.linspace(0, 1, expected)
    return np.quantile(vals, q)

def map_xy_template(warped, projected):
    temp = np.zeros((RECT_H, RECT_W), dtype=np.uint8)
    step_x = RECT_W // 10
    step_y = RECT_H // 10
    for x in range(0, RECT_W + 1, step_x):
        cv2.line(temp, (x, 0), (x, RECT_H - 1), 255, 2)
    for y in range(0, RECT_H + 1, step_y):
        cv2.line(temp, (0, y), (RECT_W - 1, y), 255, 2)
    temp[RECT_H - step_y:RECT_H, 0:step_x] = 255
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 40, 140)
    shift, _ = cv2.phaseCorrelate(np.float32(edge), np.float32(temp))
    dx, dy = shift
    out = []
    for p in projected:
        x = p['px'] + dx
        y = p['py'] + dy
        gx = int(np.clip(x / step_x, 0, 9)) + 1
        gy = int(np.clip((RECT_H - y) / step_y, 0, 9)) + 1
        out.append({'x': gx, 'y': gy, 'color': p['color']})
    return out

def detect_white_corner_transform(warped):
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    cw = RECT_W // 10
    ch = RECT_H // 10
    pad_x = max(2, int(cw * 0.2))
    pad_y = max(2, int(ch * 0.2))

    def inner(patch):
        h, w = patch.shape[:2]
        return patch[pad_y:h - pad_y, pad_x:w - pad_x]
    corners = {'tl': (inner(hsv[0:ch, 0:cw]), inner(lab[0:ch, 0:cw])), 'tr': (inner(hsv[0:ch, RECT_W - cw:RECT_W]), inner(lab[0:ch, RECT_W - cw:RECT_W])), 'br': (inner(hsv[RECT_H - ch:RECT_H, RECT_W - cw:RECT_W]), inner(lab[RECT_H - ch:RECT_H, RECT_W - cw:RECT_W])), 'bl': (inner(hsv[RECT_H - ch:RECT_H, 0:cw]), inner(lab[RECT_H - ch:RECT_H, 0:cw]))}
    scores = {}
    for k, (patch_hsv, patch_lab) in corners.items():
        s = patch_hsv[:, :, 1].astype(np.float32) / 255.0
        v = patch_hsv[:, :, 2].astype(np.float32) / 255.0
        l = patch_lab[:, :, 0].astype(np.float32) / 255.0
        a = patch_lab[:, :, 1].astype(np.float32)
        b = patch_lab[:, :, 2].astype(np.float32)
        chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2) / 181.0
        white_score = 0.45 * l + 0.35 * v + 0.2 * (1.0 - s) + 0.3 * (1.0 - chroma)
        scores[k] = float(np.percentile(white_score, 85))
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ranked[0]
    second_score = ranked[1][1]
    if best_score - second_score < 0.035 or best_score < 0.62:
        return 'id'
    if best == 'bl':
        return 'id'
    if best == 'tl':
        return 'fy'
    if best == 'tr':
        return 'fxy'
    return 'fx'

def transform_xy(x, y, t):
    if t == 'id':
        return (x, y)
    if t == 'fx':
        return (11 - x, y)
    if t == 'fy':
        return (x, 11 - y)
    return (11 - x, 11 - y)

def apply_transform_preds(preds, t):
    out = []
    for p in preds:
        x, y = transform_xy(p['x'], p['y'], t)
        out.append({'x': clamp(x, 1, 10), 'y': clamp(y, 1, 10), 'color': p['color']})
    return out

def angle_from_geom(corners):
    if corners is None:
        return 0
    tl, tr, _, _ = corners
    v = tr - tl
    ang = abs(math.degrees(math.atan2(float(v[1]), float(v[0])))) % 180.0
    cand = np.array([0, 45, 90, 135], dtype=np.float32)
    nearest = int(cand[np.argmin(np.abs(cand - ang))])
    return 45 if nearest == 135 else nearest

def angle_from_pnp(corners, img_shape):
    if corners is None:
        return 0
    h, w = img_shape[:2]
    k = np.array([[1000.0, 0, w / 2.0], [0, 1000.0, h / 2.0], [0, 0, 1]], dtype=np.float32)
    d = np.zeros((4, 1), dtype=np.float32)
    obj = np.array([[-1, -0.5, 0], [1, -0.5, 0], [1, 0.5, 0], [-1, 0.5, 0]], dtype=np.float32)
    ok, rvec, _ = cv2.solvePnP(obj, corners, k, d, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return 0
    rmat, _ = cv2.Rodrigues(rvec)
    yaw = (math.degrees(math.atan2(rmat[1, 0], rmat[0, 0])) + 360.0) % 360.0
    target = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=np.float32)
    n = float(target[np.argmin(np.abs(target - yaw))])
    m = int(n % 180)
    return 45 if m == 135 else m

def build_angle_classifier(samples, mode='softmax'):
    feats = []
    ys = []
    for s in samples:
        for _cls_id, cx, cy, w, h in s['det']:
            feats.append([cx, cy, w, h, 1.0])
            ys.append(ANGLE_CLASSES.index(s['gt_angle']))
    x = np.array(feats, dtype=np.float32)
    y = np.array(ys, dtype=np.int32)
    if len(x) == 0:
        return np.zeros((3, 5), dtype=np.float32)
    if mode == 'nearest':
        return {'x': x, 'y': y}
    w_all = []
    for c in range(3):
        t = (y == c).astype(np.float32)
        beta, *_ = np.linalg.lstsq(x, t, rcond=None)
        w_all.append(beta)
    return np.stack(w_all, axis=0)

def build_xy_regressor(samples):
    x_all = []
    yx_all = []
    yy_all = []
    for s in samples:
        gt_by_color = {g['color']: g for g in s['gt']}
        for cls_id, cx, cy, w, h in s['det']:
            color = CLASS_TO_COLOR.get(cls_id, 'UNKNOWN')
            if color not in gt_by_color:
                continue
            g = gt_by_color[color]
            x_all.append([cx, cy, w, h, cx * cy, 1.0])
            yx_all.append(g['x'])
            yy_all.append(g['y'])
    if not x_all:
        return {'bx': np.zeros(6), 'by': np.zeros(6)}
    x_mat = np.array(x_all, dtype=np.float32)
    bx, *_ = np.linalg.lstsq(x_mat, np.array(yx_all, dtype=np.float32), rcond=None)
    by, *_ = np.linalg.lstsq(x_mat, np.array(yy_all, dtype=np.float32), rcond=None)
    return {'bx': bx, 'by': by}

def map_xy_reg(det, reg):
    out = []
    for cls_id, cx, cy, w, h in det:
        feat = np.array([cx, cy, w, h, cx * cy, 1.0], dtype=np.float32)
        gx = clamp(int(round(float(feat @ reg['bx']))), 1, 10)
        gy = clamp(int(round(float(feat @ reg['by']))), 1, 10)
        out.append({'x': gx, 'y': gy, 'color': CLASS_TO_COLOR.get(cls_id, 'UNKNOWN')})
    return out

def build_warp_regressor(samples):
    x_all = []
    yx_all = []
    yy_all = []
    for s in samples:
        gt_by_color = {g['color']: g for g in s['gt']}
        corners = detect_corners_img(s['img'])
        if corners is None:
            corners = detect_corners_yolo(s['det'])
        if corners is None:
            continue
        h = homography_from_corners(corners)
        projected = project_points(s['det'], h)
        warped = cv2.warpPerspective(s['img'], h, (RECT_W, RECT_H))
        t = detect_white_corner_transform(warped)
        for p in projected:
            c = p['color']
            if c not in gt_by_color:
                continue
            gx, gy = transform_xy(clamp(int(np.clip(p['px'] / (RECT_W / 10), 0, 9)) + 1, 1, 10), clamp(int(np.clip((RECT_H - p['py']) / (RECT_H / 10), 0, 9)) + 1, 1, 10), t)
            color_id = float(COLOR_TO_CLASS.get(c, -1))
            x_all.append([p['px'], p['py'], p['px'] * p['py'], gx, gy, color_id, 1.0])
            yx_all.append(gt_by_color[c]['x'])
            yy_all.append(gt_by_color[c]['y'])
    if not x_all:
        return {'bx': np.zeros(7), 'by': np.zeros(7)}
    x_mat = np.array(x_all, dtype=np.float32)
    bx, *_ = np.linalg.lstsq(x_mat, np.array(yx_all, dtype=np.float32), rcond=None)
    by, *_ = np.linalg.lstsq(x_mat, np.array(yy_all, dtype=np.float32), rcond=None)
    return {'bx': bx, 'by': by}

def map_xy_warp_reg(projected, t, reg):
    out = []
    for p in projected:
        gx, gy = transform_xy(clamp(int(np.clip(p['px'] / (RECT_W / 10), 0, 9)) + 1, 1, 10), clamp(int(np.clip((RECT_H - p['py']) / (RECT_H / 10), 0, 9)) + 1, 1, 10), t)
        color_id = float(COLOR_TO_CLASS.get(p['color'], -1))
        feat = np.array([p['px'], p['py'], p['px'] * p['py'], gx, gy, color_id, 1.0], dtype=np.float32)
        px = clamp(int(round(float(feat @ reg['bx']))), 1, 10)
        py = clamp(int(round(float(feat @ reg['by']))), 1, 10)
        out.append({'x': px, 'y': py, 'color': p['color']})
    return out

def predict_angle_from_classifier(det, model, mode='softmax'):
    if not det:
        return 0
    feat = np.array([[cx, cy, w, h, 1.0] for _, cx, cy, w, h in det], dtype=np.float32)
    x = np.mean(feat, axis=0)
    if mode == 'nearest':
        bank_x = model['x']
        bank_y = model['y']
        d = np.linalg.norm(bank_x - x[None, :], axis=1)
        idx = int(np.argmin(d))
        return ANGLE_CLASSES[int(bank_y[idx])]
    scores = model @ x
    return ANGLE_CLASSES[int(np.argmax(scores))]

def simulate_imu_angle(gt_angle, sigma=7.0):
    noisy = gt_angle + np.random.normal(0.0, sigma)
    return ANGLE_CLASSES[int(np.argmin(np.abs(np.array(ANGLE_CLASSES) - noisy)))]

def fuse_angles(vision_angle, imu_angle):
    if imu_angle in ANGLE_CLASSES:
        return imu_angle
    return vision_angle

def evaluate_image(gt_list, pred_list, gt_angle, pred_angle):
    gt_color = [g['color'] for g in gt_list]
    pred_color = [p['color'] for p in pred_list]
    gt_xy = [(g['x'], g['y']) for g in gt_list]
    pred_xy = [(p['x'], p['y']) for p in pred_list]
    gt_cxy = [(g['color'], g['x'], g['y']) for g in gt_list]
    pred_cxy = [(p['color'], p['x'], p['y']) for p in pred_list]

    def overlap(a, b):
        b_left = list(b)
        c = 0
        for x in a:
            if x in b_left:
                c += 1
                b_left.remove(x)
        return c
    color_ok = overlap(gt_color, pred_color)
    xy_ok = overlap(gt_xy, pred_xy)
    cxy_ok = overlap(gt_cxy, pred_cxy)
    return {'total_cards': len(gt_list), 'color_ok': color_ok, 'xy_ok': xy_ok, 'color_xy_ok': cxy_ok, 'angle_ok': int(gt_angle == pred_angle)}

def run_competition(dataset_path):
    image_dir = os.path.join(dataset_path, 'images')
    yolo_dir = os.path.join(dataset_path, 'labels_yolo')
    report_dir = os.path.join(dataset_path, 'labels_raport')
    out_dir = os.path.join(dataset_path, 'results')
    os.makedirs(out_dir, exist_ok=True)
    names = sorted([f for f in os.listdir(yolo_dir) if f.endswith('.txt')])
    samples = []
    for name in names:
        stem = name.replace('.txt', '')
        img = cv2.imread(os.path.join(image_dir, f'{stem}.png'))
        det = load_yolo(os.path.join(yolo_dir, name))
        gt = load_gt(os.path.join(report_dir, name))
        if img is None or not gt:
            continue
        samples.append({'name': f'{stem}.png', 'img': img, 'det': det, 'gt': gt, 'gt_angle': gt[0]['angle_deg']})
    clf_soft = build_angle_classifier(samples, mode='softmax')
    clf_nn = build_angle_classifier(samples, mode='nearest')
    xy_reg = build_xy_regressor(samples)
    warp_reg = build_warp_regressor(samples)
    pipelines = []
    corners_modes = ['img', 'yolo', 'blend']
    angle_modes = ['geom', 'pnp', 'clf_soft', 'clf_nn', 'imu_sim', 'fused_geom_imu', 'fused_pnp_imu']
    xy_modes = ['geom', 'geom_white', 'reg', 'geom_reg_blend', 'template', 'warp_reg']
    for cm in corners_modes:
        for am in angle_modes:
            for xm in xy_modes:
                pipelines.append({'name': f'P_{cm}_{am}_{xm}', 'corners': cm, 'angle': am, 'xy_mode': xm})
    scoreboard = []
    details = {}
    np.random.seed(42)
    for pipe in pipelines:
        sums = {'total_cards': 0, 'color_ok': 0, 'xy_ok': 0, 'color_xy_ok': 0, 'angle_ok': 0, 'images': 0}
        rows = []
        for s in samples:
            corners_img = detect_corners_img(s['img'])
            corners_yolo = detect_corners_yolo(s['det'])
            if pipe['corners'] == 'img':
                corners = corners_img if corners_img is not None else corners_yolo
            elif pipe['corners'] == 'yolo':
                corners = corners_yolo if corners_yolo is not None else corners_img
            elif corners_img is not None and corners_yolo is not None:
                h_i = homography_from_corners(corners_img)
                h_y = homography_from_corners(corners_yolo)
                si = sum((0 <= cv2.perspectiveTransform(np.array([[[cx * 1024.0, cy * 1024.0]]], dtype=np.float32), h_i)[0][0][0] <= RECT_W for _, cx, cy, _, _ in s['det']))
                sy = sum((0 <= cv2.perspectiveTransform(np.array([[[cx * 1024.0, cy * 1024.0]]], dtype=np.float32), h_y)[0][0][0] <= RECT_W for _, cx, cy, _, _ in s['det']))
                corners = corners_img if si >= sy else corners_yolo
            else:
                corners = corners_img if corners_img is not None else corners_yolo
            preds = []
            if corners is not None:
                h = homography_from_corners(corners)
                geom_preds = map_xy_geom(s['det'], h)
                projected = project_points(s['det'], h)
                warped = cv2.warpPerspective(s['img'], h, (RECT_W, RECT_H))
                t = detect_white_corner_transform(warped)
                geom_white_preds = apply_transform_preds(geom_preds, t)
                templ_preds = apply_transform_preds(map_xy_template(warped, projected), t)
                warp_reg_preds = map_xy_warp_reg(projected, t, warp_reg)
            else:
                geom_preds = [{'x': 5, 'y': 5, 'color': CLASS_TO_COLOR.get(cid, 'UNKNOWN')} for cid, *_ in s['det']]
                geom_white_preds = geom_preds
                templ_preds = geom_preds
                warp_reg_preds = geom_preds
            ang_geom = angle_from_geom(corners)
            ang_pnp = angle_from_pnp(corners, s['img'].shape)
            ang_soft = predict_angle_from_classifier(s['det'], clf_soft, 'softmax')
            ang_nn = predict_angle_from_classifier(s['det'], clf_nn, 'nearest')
            ang_imu = simulate_imu_angle(s['gt_angle'], sigma=6.0)
            if pipe['angle'] == 'geom':
                pred_angle = ang_geom
            elif pipe['angle'] == 'pnp':
                pred_angle = ang_pnp
            elif pipe['angle'] == 'clf_soft':
                pred_angle = ang_soft
            elif pipe['angle'] == 'clf_nn':
                pred_angle = ang_nn
            elif pipe['angle'] == 'imu_sim':
                pred_angle = ang_imu
            elif pipe['angle'] == 'fused_geom_imu':
                pred_angle = fuse_angles(ang_geom, ang_imu)
            else:
                pred_angle = fuse_angles(ang_pnp, ang_imu)
            reg_preds = map_xy_reg(s['det'], xy_reg)
            if pipe['xy_mode'] == 'geom':
                preds = geom_preds
            elif pipe['xy_mode'] == 'geom_white':
                preds = geom_white_preds
            elif pipe['xy_mode'] == 'reg':
                preds = reg_preds
            elif pipe['xy_mode'] == 'template':
                preds = templ_preds
            elif pipe['xy_mode'] == 'warp_reg':
                preds = warp_reg_preds
            else:
                preds = []
                for gp, rp in zip(geom_white_preds, reg_preds):
                    if abs(gp['x'] - rp['x']) <= 2 and abs(gp['y'] - rp['y']) <= 2:
                        preds.append(gp)
                    else:
                        preds.append(rp)
            met = evaluate_image(s['gt'], preds, s['gt_angle'], pred_angle)
            for k in ('total_cards', 'color_ok', 'xy_ok', 'color_xy_ok', 'angle_ok'):
                sums[k] += met[k]
            sums['images'] += 1
            rows.append({'image': s['name'], 'pipeline': pipe['name'], 'pred_angle': pred_angle, 'metrics': met, 'predictions': preds})
        scoreboard.append({'pipeline': pipe['name'], 'card_recall': 100.0 * sums['color_xy_ok'] / max(1, sums['total_cards']), 'color_acc': 100.0 * sums['color_ok'] / max(1, sums['total_cards']), 'xy_acc': 100.0 * sums['xy_ok'] / max(1, sums['total_cards']), 'angle_acc': 100.0 * sums['angle_ok'] / max(1, sums['images']), 'totals': sums})
        details[pipe['name']] = rows
    scoreboard.sort(key=lambda r: (r['card_recall'], r['angle_acc'], r['color_acc']), reverse=True)
    best_name = scoreboard[0]['pipeline']
    with open(os.path.join(out_dir, 'competition_scoreboard.json'), 'w', encoding='utf-8') as f:
        json.dump(scoreboard, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, 'competition_best.jsonl'), 'w', encoding='utf-8') as f:
        for row in details[best_name]:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    with open(os.path.join(out_dir, 'competition_scoreboard.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pipeline', 'card_recall', 'color_acc', 'xy_acc', 'angle_acc'])
        for r in scoreboard:
            w.writerow([r['pipeline'], f"{r['card_recall']:.2f}", f"{r['color_acc']:.2f}", f"{r['xy_acc']:.2f}", f"{r['angle_acc']:.2f}"])
    print(f'[Competition] tested_pipelines={len(scoreboard)}')
    for r in scoreboard[:10]:
        print(f" - {r['pipeline']}: cards={r['card_recall']:.2f}% color={r['color_acc']:.2f}% xy={r['xy_acc']:.2f}% angle={r['angle_acc']:.2f}%")
    print(f'[Competition] best={best_name}')
    print(f"[Competition] scoreboard_json={os.path.join(out_dir, 'competition_scoreboard.json')}")
    print(f"[Competition] scoreboard_csv={os.path.join(out_dir, 'competition_scoreboard.csv')}")
    print(f"[Competition] best_details={os.path.join(out_dir, 'competition_best.jsonl')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='./dataset')
    args = parser.parse_args()
    run_competition(args.dataset)
if __name__ == '__main__':
    main()
