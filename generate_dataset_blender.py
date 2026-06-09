import bpy
import bpy_extras
import json
import math
import os
import random
import sys
import mathutils

_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)


def _load_format_card_detected_line():
    """Bez importu module_panel.__init__ (wymaga cv2 w Blenderze)."""
    import importlib.util

    path = os.path.join(_BASE, 'module_panel', 'competition_report.py')
    spec = importlib.util.spec_from_file_location('droniada_competition_report', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Nie można załadować {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.format_card_detected_line


format_card_detected_line = _load_format_card_detected_line()
DATASET_PATH = os.path.join(_BASE, os.environ.get('DRONIADA_DATASET_SUBDIR', 'dataset'))
IMAGES_PATH = os.path.join(DATASET_PATH, 'images')
LABELS_YOLO_PATH = os.path.join(DATASET_PATH, 'labels_yolo')
LABELS_RAPORT_PATH = os.path.join(DATASET_PATH, 'labels_raport')
LABELS_POSE_PATH = os.path.join(DATASET_PATH, 'labels_pose')
for path in [IMAGES_PATH, LABELS_YOLO_PATH, LABELS_RAPORT_PATH, LABELS_POSE_PATH]:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
PANEL_MIN_Z_M = 1.0
CAMERA_HEIGHT_Z_M = float(os.environ.get('DRONIADA_CAMERA_Z_M', '3.0'))
CAMERA_XY_DIST_M = float(os.environ.get('DRONIADA_CAMERA_XY_DIST_M', '8.25'))
CAMERA_LOOK_BELOW_CENTER_M = 0.42
CAMERA_LATERAL_M = 0.58
CAMERA_TARGET_LATERAL_M = 0.24
ORBIT_ARC_DEG = float(os.environ.get('DRONIADA_ORBIT_ARC_DEG', '50'))
VIEWS_PER_SCENE = int(os.environ.get('DRONIADA_ORBIT_STEPS', '9'))
# Minimalny cos(kąta) między wektorem do kamery a normalną przodu panelu (1=prostopadle).
# 0.58 ≈ kąt do ~54° — odrzuca tył i czysty bok (cienka krawędź).
CAMERA_FRONT_MIN_DOT = float(os.environ.get('DRONIADA_MIN_FRONT_DOT', '0.72'))
MIN_PANEL_SPAN_PX = float(os.environ.get('DRONIADA_MIN_PANEL_SPAN_PX', '200'))
MIN_PANEL_AREA_FRAC = float(os.environ.get('DRONIADA_MIN_PANEL_AREA_FRAC', '0.06'))


def _parse_orbit_azimuth_deg_list():
    """Lista kątów względem frontu panelu (0=frontal). Pusta/full/360 = łuk ORBIT_ARC_DEG."""
    raw = os.environ.get(
        'DRONIADA_ORBIT_AZIMUTHS',
        '-22,-14,-7,0,7,14,22',
    )
    if raw.strip().lower() in ('full', '360', 'legacy'):
        return None
    return [float(x.strip()) for x in raw.split(',') if x.strip()]


ORBIT_AZIMUTH_DEG_LIST = _parse_orbit_azimuth_deg_list()
CARD_CELLS_BY_ANGLE = {'horizontal': [(3, 2), (8, 5), (5, 8), (9, 3)], 'vertical': [(2, 4), (7, 3), (4, 9), (10, 7)], '45_deg': [(3, 4), (8, 3), (5, 7), (10, 5)]}

def panel_angle_category(stand_id: str) -> str:
    if stand_id == 'long_edge_upright_tv':
        return 'horizontal'
    if stand_id in ('long_edge_upright_portrait', 'short_edge_upright'):
        return 'vertical'
    if stand_id == 'long_edge_laptop_45':
        return '45_deg'
    return 'horizontal'

def report_skew_deg_from_stand(stand_id: str) -> int:
    if stand_id == 'long_edge_upright_tv':
        return 0
    if stand_id in ('long_edge_upright_portrait', 'short_edge_upright'):
        return 90
    if stand_id == 'long_edge_laptop_45':
        return 45
    return 0
COLORS = {'czerwona': (0.75, 0.15, 0.15, 1), 'zielona': (0.15, 0.6, 0.2, 1), 'niebieska': (0.1, 0.3, 0.85, 1), 'zolta': (0.85, 0.65, 0.1, 1), 'fioletowa': (0.5, 0.15, 0.85, 1), 'pomaranczowa': (0.85, 0.35, 0.1, 1)}
COLOR_TO_CLASS = {'czerwona': 0, 'zielona': 1, 'niebieska': 2, 'zolta': 3, 'fioletowa': 4, 'pomaranczowa': 5}

def cleanup_data():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def create_grid_material():
    mat = bpy.data.materials.new(name='Mat_Board_Grid')
    try:
        mat.use_nodes = True
    except:
        pass
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Roughness'].default_value = 1.0
        try:
            if 'Specular IOR Level' in bsdf.inputs:
                bsdf.inputs['Specular IOR Level'].default_value = 0.0
            elif 'Specular' in bsdf.inputs:
                bsdf.inputs['Specular'].default_value = 0.0
        except:
            pass
    res = 500
    img_name = 'Custom_Grid_Tex'
    if img_name in bpy.data.images:
        img = bpy.data.images[img_name]
    else:
        img = bpy.data.images.new(img_name, width=res, height=res)
        pixels = [0.12, 0.12, 0.13, 1.0] * (res * res)
        base_fill = (0.12, 0.12, 0.13, 1.0)
        for y in range(res):
            y_line = y % 50 < 1 or y % 50 > 48 or y == 0 or (y == res - 1)
            for x in range(res):
                x_line = x % 50 < 1 or x % 50 > 48 or x == 0 or (x == res - 1)
                idx = (y * res + x) * 4
                if x_line or y_line:
                    pixels[idx] = 0.78
                    pixels[idx + 1] = 0.78
                    pixels[idx + 2] = 0.8
                else:
                    pixels[idx] = base_fill[0]
                    pixels[idx + 1] = base_fill[1]
                    pixels[idx + 2] = base_fill[2]
        for y in range(0, 50):
            for x in range(0, 50):
                idx = (y * res + x) * 4
                pixels[idx] = 0.95
                pixels[idx + 1] = 0.95
                pixels[idx + 2] = 0.95
        img.pixels = pixels
    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.image = img
    if bsdf:
        links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
    return mat

def get_yolo_bbox(scene, cam, obj):
    mat = obj.matrix_world
    me = obj.data
    coords_2d = []
    for v in me.vertices:
        co = mat @ v.co
        co2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, co)
        coords_2d.append(co2d)
    xs = [c.x for c in coords_2d]
    ys = [c.y for c in coords_2d]
    min_x, max_x = (max(0.0, min(1.0, min(xs))), max(0.0, min(1.0, max(xs))))
    min_y, max_y = (max(0.0, min(1.0, min(ys))), max(0.0, min(1.0, max(ys))))
    width = max_x - min_x
    height = max_y - min_y
    center_x = min_x + width / 2.0
    center_y = 1.0 - (min_y + height / 2.0)
    return (center_x, center_y, width, height)

def setup_sky_world_gradient():
    world = bpy.context.scene.world
    try:
        world.use_nodes = True
    except Exception:
        pass
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    bg = nodes.new('ShaderNodeBackground')
    out = nodes.new('ShaderNodeOutputWorld')
    try:
        sky = nodes.new('ShaderNodeTexSky')
        if hasattr(sky, 'sky_type'):
            sky.sky_type = 'PREETHAM'
        if hasattr(sky, 'sun_elevation'):
            sky.sun_elevation = math.radians(22.0 + random.uniform(-10.0, 18.0))
        if hasattr(sky, 'sun_rotation'):
            sky.sun_rotation = random.uniform(-0.45, 0.45)
        if hasattr(sky, 'turbidity'):
            sky.turbidity = random.uniform(2.2, 5.0)
        links.new(sky.outputs['Color'], bg.inputs['Color'])
    except Exception:
        bg.inputs['Color'].default_value = (0.38, 0.55, 0.82, 1.0)
    bg.inputs['Strength'].default_value = 0.55 + random.uniform(0.0, 0.25)
    links.new(bg.outputs['Background'], out.inputs['Surface'])

def create_grass_material():
    mat = bpy.data.materials.new(name='Mat_Grass')
    try:
        mat.use_nodes = True
    except Exception:
        return mat
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    out = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = random.uniform(18.0, 45.0)
    noise.inputs['Detail'].default_value = random.uniform(6.0, 12.0)
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = (0.06, 0.18, 0.05, 1.0)
    ramp.color_ramp.elements[1].color = (0.12, 0.38, 0.12, 1.0)
    links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.92
    try:
        bsdf.inputs['Specular IOR Level'].default_value = 0.15
    except Exception:
        try:
            bsdf.inputs['Specular'].default_value = 0.2
        except Exception:
            pass
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def add_grass_ground():
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0.0, 0.0, 0.0))
    grass = bpy.context.active_object
    grass.name = 'Ground_Grass'
    grass.scale = (55.0, 55.0, 1.0)
    bpy.ops.object.transform_apply(scale=True)
    grass.data.materials.append(create_grass_material())
    return grass

def panel_front_normal_and_center(board_obj):
    mw = board_obj.matrix_world
    center = mw @ mathutils.Vector((0.0, 0.0, 0.0))
    n = (mw.to_3x3() @ mathutils.Vector((0.0, 0.0, 1.0))).normalized()
    return (center, n)

def camera_sees_panel_front(cam_obj, board_obj, min_dot=None):
    """Kamera w półkuli przed siatką (+local Z panelu)."""
    center, fn = panel_front_normal_and_center(board_obj)
    to_cam = mathutils.Vector(cam_obj.location) - center
    if to_cam.length < 1e-06:
        return False
    to_cam.normalize()
    thr = CAMERA_FRONT_MIN_DOT if min_dot is None else float(min_dot)
    return to_cam.dot(fn) >= thr


def white_anchor_in_front(scene, cam, board_obj):
    """Biały znacznik na siatce musi być przed kamerą (nie plecy panelu)."""
    anchor = None
    for ch in board_obj.children:
        if ch.type != 'MESH' or not ch.data:
            continue
        for mat in ch.data.materials:
            if mat and 'White_Anchor' in mat.name:
                anchor = ch
                break
        if anchor is not None:
            break
    if anchor is None:
        return True
    co = anchor.matrix_world @ mathutils.Vector((0.0, 0.0, 0.0))
    c = bpy_extras.object_utils.world_to_camera_view(scene, cam, co)
    if c.z <= 0.05:
        return False
    return 0.04 <= c.x <= 0.96 and 0.04 <= c.y <= 0.96


def projected_panel_bbox_px(scene, cam, board_obj):
    """Rozpiętość rzutu panelu w px oraz ułamek pola kadru."""
    mw = board_obj.matrix_world
    w_img = float(scene.render.resolution_x)
    h_img = float(scene.render.resolution_y)
    xs, ys = [], []
    for corner in board_obj.bound_box:
        co = mw @ mathutils.Vector(corner)
        c = bpy_extras.object_utils.world_to_camera_view(scene, cam, co)
        if c.z <= 0.0:
            return 0.0, 0.0, 0.0, 0.0
        xs.append(float(c.x) * w_img)
        ys.append((1.0 - float(c.y)) * h_img)
    if not xs:
        return 0.0, 0.0, 0.0, 0.0
    w_px = float(max(xs) - min(xs))
    h_px = float(max(ys) - min(ys))
    area_frac = (w_px * h_px) / max(1.0, w_img * h_img)
    return w_px, h_px, float(min(w_px, h_px)), area_frac


def camera_view_is_usable(scene, cam, board_obj, margin=0.04):
    """Odrzuca tył panelu, czysty bok (cienka krawędź) i zbyt mały panel w kadrze."""
    if not camera_sees_panel_front(cam, board_obj):
        return False, 'back_or_grazing'
    if not white_anchor_in_front(scene, cam, board_obj):
        return False, 'grid_anchor_behind'
    if not board_corners_in_frame(scene, cam, board_obj, margin=margin):
        return False, 'corners_oob'
    w_px, h_px, min_span, area_frac = projected_panel_bbox_px(scene, cam, board_obj)
    if min_span < MIN_PANEL_SPAN_PX:
        return False, f'thin_side_span={min_span:.0f}'
    if area_frac < MIN_PANEL_AREA_FRAC:
        return False, f'small_area={area_frac:.3f}'
    # Panel 2x1 m: z boku jeden wymiar w px zapada — max/min bbox nie może być ekstremalny.
    short_side = max(1.0, min(w_px, h_px))
    long_side = max(w_px, h_px)
    if long_side / short_side > 5.5:
        return False, f'edge_aspect={long_side / short_side:.1f}'
    return True, 'ok'

def board_world_z_range(board_obj):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = board_obj.evaluated_get(dg)
    mw = ev.matrix_world
    zs = []
    for v in ev.data.vertices:
        zs.append((mw @ v.co).z)
    return (min(zs), max(zs))

def raise_panel_root_above_ground(panel_root, board_obj, min_z=PANEL_MIN_Z_M):
    bpy.context.view_layer.update()
    z_lo, _z_hi = board_world_z_range(board_obj)
    dz = float(min_z) - z_lo
    if dz > 0.0001:
        panel_root.location.z += dz
    bpy.context.view_layer.update()

def grass_looks_below_panel(scene, cam):
    p = bpy_extras.object_utils.world_to_camera_view(scene, cam, mathutils.Vector((0.0, 0.0, 0.0)))
    if p.z <= 0.0:
        return False
    return float(p.y) <= 0.48

def board_corners_in_frame(scene, cam, board_obj, margin=0.06):
    mw = board_obj.matrix_world
    for corner in board_obj.bound_box:
        co = mw @ mathutils.Vector(corner)
        c = bpy_extras.object_utils.world_to_camera_view(scene, cam, co)
        if not (margin <= c.x <= 1.0 - margin and margin <= c.y <= 1.0 - margin and (c.z > 0.0)):
            return False
    return True

def apply_camera_look_at_world_z_up(cam_obj, target_world):
    cam_loc = mathutils.Vector(cam_obj.location)
    tgt = mathutils.Vector(target_world)
    forward = tgt - cam_loc
    if forward.length < 1e-08:
        forward = mathutils.Vector((0.0, -1.0, 0.0))
    else:
        forward.normalize()
    wup = mathutils.Vector((0.0, 0.0, 1.0))
    up = wup - forward * forward.dot(wup)
    if up.length < 1e-08:
        aux = mathutils.Vector((0.0, 1.0, 0.0))
        up = aux - forward * forward.dot(aux)
    up.normalize()
    right = forward.cross(up)
    if right.length < 1e-08:
        return
    right.normalize()
    rot = mathutils.Matrix(((right.x, up.x, -forward.x), (right.y, up.y, -forward.y), (right.z, up.z, -forward.z)))
    cam_obj.matrix_world = mathutils.Matrix.Translation(cam_loc) @ rot.to_4x4()

def model_to_camera_opencv(board_obj, cam_obj):
    t_cw = cam_obj.matrix_world.inverted()
    r_world_to_cam_bl = t_cw.to_3x3()
    r_model_to_world = board_obj.matrix_world.to_3x3()
    r_model_to_cam_bl = r_world_to_cam_bl @ r_model_to_world
    p_model_origin_world = board_obj.matrix_world @ mathutils.Vector((0.0, 0.0, 0.0))
    t_model_to_cam_bl = t_cw @ p_model_origin_world
    r_bcam_to_cv = mathutils.Matrix(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)))
    r_model_to_cam_cv = r_bcam_to_cv @ r_model_to_cam_bl
    t_model_to_cam_cv = r_bcam_to_cv @ t_model_to_cam_bl
    return (r_model_to_cam_cv, t_model_to_cam_cv)

def place_camera_fixed_drone_view(scene, cam, target, board_obj):
    center, fn = panel_front_normal_and_center(board_obj)
    fn.normalize()
    world_up = mathutils.Vector((0.0, 0.0, 1.0))
    fn_xy = mathutils.Vector((fn.x, fn.y, 0.0))
    if fn_xy.length < 0.0001:
        fn_xy = mathutils.Vector((0.0, -1.0, 0.0))
    fn_xy.normalize()
    right = world_up.cross(mathutils.Vector((fn_xy.x, fn_xy.y, 0.0))).normalized()
    d0 = CAMERA_XY_DIST_M
    lc0 = CAMERA_LATERAL_M
    lt0 = CAMERA_TARGET_LATERAL_M
    below = CAMERA_LOOK_BELOW_CENTER_M
    reject_reason = 'no_valid_pose'
    for scale in (1.0, 0.9, 1.1, 0.82, 1.18):
        d = d0 * scale
        lc = lc0 * scale
        lt = lt0 * scale
        cam.location = (center.x + fn_xy.x * d + right.x * lc, center.y + fn_xy.y * d + right.y * lc, CAMERA_HEIGHT_Z_M)
        target.location = (center.x + right.x * lt, center.y + right.y * lt, center.z - below)
        apply_camera_look_at_world_z_up(cam, target.location)
        bpy.context.view_layer.update()
        ok_view, reject_reason = camera_view_is_usable(scene, cam, board_obj, margin=0.038)
        if ok_view:
            return {
                'placement': 'fixed_z_oblique',
                'accepted': True,
                'height_world_z_m': CAMERA_HEIGHT_Z_M,
                'horizontal_dist_xy_m': float(d),
                'lateral_cam_m': float(lc),
                'look_below_panel_center_m': float(below),
                'target_lateral_m': float(lt),
                'world_z_up_no_roll': True,
            }
    return {'placement': 'fixed_rejected', 'accepted': False, 'reject_reason': reject_reason}

def place_camera_orbit_step(
    scene,
    cam,
    target,
    board_obj,
    step_index,
    num_steps,
    arc_deg=360.0,
    azimuth_deg=None,
):
    center, fn = panel_front_normal_and_center(board_obj)
    fn.normalize()
    world_up = mathutils.Vector((0.0, 0.0, 1.0))
    fn_xy = mathutils.Vector((fn.x, fn.y, 0.0))
    if fn_xy.length < 0.0001:
        fn_xy = mathutils.Vector((0.0, -1.0, 0.0))
    fn_xy.normalize()
    tangent = world_up.cross(mathutils.Vector((fn_xy.x, fn_xy.y, 0.0))).normalized()
    az_offset = None
    if azimuth_deg is not None:
        theta = math.radians(float(azimuth_deg))
        az_offset = float(azimuth_deg)
    elif num_steps <= 1:
        theta = 0.0
        az_offset = 0.0
    elif arc_deg >= 359.99:
        theta = 2.0 * math.pi * float(step_index) / float(num_steps)
        az_offset = math.degrees(theta)
    else:
        # Symetryczny łuk wokół frontu: -arc/2 .. +arc/2 (nie 0..arc — to schodziło na tył/bok).
        half = float(arc_deg) / 2.0
        az_offset = -half + float(arc_deg) * float(step_index) / float(max(1, num_steps - 1))
        theta = math.radians(az_offset)
    dir_h = mathutils.Vector((fn_xy.x * math.cos(theta) + tangent.x * math.sin(theta), fn_xy.y * math.cos(theta) + tangent.y * math.sin(theta), 0.0))
    if dir_h.length < 1e-06:
        dir_h = fn_xy.copy()
    else:
        dir_h.normalize()
    lateral = world_up.cross(dir_h)
    if lateral.length > 1e-08:
        lateral.normalize()
    lc0 = CAMERA_LATERAL_M * 0.35
    lt0 = CAMERA_TARGET_LATERAL_M * 0.35
    below = CAMERA_LOOK_BELOW_CENTER_M
    reject_reason = 'no_valid_pose'
    for scale in (1.0, 0.88, 1.05, 0.75, 1.12, 1.2):
        d = CAMERA_XY_DIST_M * scale
        lc = lc0 * scale
        lt = lt0 * scale
        cam.location = (center.x + dir_h.x * d + lateral.x * lc, center.y + dir_h.y * d + lateral.y * lc, CAMERA_HEIGHT_Z_M)
        target.location = (center.x + lateral.x * lt, center.y + lateral.y * lt, center.z - below)
        apply_camera_look_at_world_z_up(cam, target.location)
        bpy.context.view_layer.update()
        ok_view, reject_reason = camera_view_is_usable(scene, cam, board_obj, margin=0.03)
        if ok_view:
            _w, _h, span_px, area_frac = projected_panel_bbox_px(scene, cam, board_obj)
            return {
                'placement': 'orbit_z_oblique',
                'accepted': True,
                'panel_projected_min_span_px': round(span_px, 1),
                'panel_projected_area_frac': round(area_frac, 4),
                'height_world_z_m': CAMERA_HEIGHT_Z_M,
                'orbit_azimuth_deg': round(math.degrees(theta) % 360.0, 4),
                'orbit_azimuth_offset_deg': round(float(az_offset), 4) if az_offset is not None else None,
                'orbit_step_index': int(step_index),
                'orbit_num_steps': int(num_steps),
                'orbit_arc_target_deg': float(arc_deg),
                'horizontal_dist_xy_m': float(d),
                'lateral_cam_m': float(lc),
                'look_below_panel_center_m': float(below),
                'target_lateral_m': float(lt),
                'world_z_up_no_roll': True,
            }
    # Bez flipu na tył panelu — lepiej pominąć klatkę niż renderować plecy/bok.
    return {
        'placement': 'orbit_rejected',
        'accepted': False,
        'reject_reason': reject_reason,
        'orbit_azimuth_offset_deg': round(float(az_offset), 4) if az_offset is not None else None,
        'orbit_step_index': int(step_index),
        'orbit_num_steps': int(num_steps),
    }

def add_panel_back_face(board_obj):
    """Gruba plecy tylko gdy DRONIADA_PANEL_THICKNESS_M > 0 (domyślnie wyłączone — mylą z tyłem)."""
    thick = float(os.environ.get('DRONIADA_PANEL_THICKNESS_M', '0'))
    if thick <= 1e-06:
        return
    back = bpy.data.materials.new(name='Mat_Panel_Back')
    try:
        back.use_nodes = True
        bn = back.node_tree.nodes
        bb = bn.get('Principled BSDF')
        if bb:
            bb.inputs['Base Color'].default_value = (0.04, 0.04, 0.045, 1.0)
            bb.inputs['Roughness'].default_value = 1.0
    except Exception:
        pass
    board_obj.data.materials.append(back)
    sol = board_obj.modifiers.new(name='PanelSolidify', type='SOLIDIFY')
    sol.thickness = thick
    sol.offset = 0.0
    sol.material_offset = 1
COMPETITION_STAND_MODES = ({'id': 'long_edge_upright_tv', 'label_pl': 'dłuższy bok na ziemi, pion jak TV (szeroka krawędź poziomo w kadrze)', 'long_hinge_rx_deg': 90.0}, {'id': 'long_edge_laptop_45', 'label_pl': 'dłuższy bok na ziemi, nachylenie ~45° (jak ekran laptopa)', 'long_hinge_rx_deg': 45.0}, {'id': 'short_edge_upright', 'label_pl': 'krótszy bok na ziemi, pion (wąski + siatka portret)'}, {'id': 'long_edge_upright_portrait', 'label_pl': 'dłuższy bok na ziemi, siatka obrócona 90° (portret, komórka 1,1 na dole)', 'long_hinge_rx_deg': 90.0})

def compute_intrinsics_dict(scene, cam):
    camd = cam.data
    w = float(scene.render.resolution_x)
    h = float(scene.render.resolution_y)
    sw = float(camd.sensor_width)
    sh = float(camd.sensor_height) if camd.sensor_height > 0 else sw * h / max(w, 1.0)
    lens = float(camd.lens)
    fx = w * lens / sw
    fy = h * lens / sh
    return {'width': int(w), 'height': int(h), 'fx': fx, 'fy': fy, 'cx': w / 2.0, 'cy': h / 2.0, 'lens_mm': lens, 'sensor_width_mm': sw, 'sensor_height_mm': sh, 'dist_coeffs': [0.0, 0.0, 0.0, 0.0, 0.0]}

def generate_scene_multiview(scene_idx, num_views, global_idx_start):
    cleanup_data()
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 1024 
    scene.render.resolution_y = 1024
    bpy.context.scene.world = bpy.data.worlds.new('World')
    setup_sky_world_gradient()
    bpy.ops.object.light_add(type='SUN', location=(0, -5, 5))
    sun = bpy.context.active_object
    sun.rotation_euler = (math.radians(random.uniform(40, 80)), math.radians(random.uniform(-45, 45)), 0)
    sun.data.energy = random.uniform(2.2, 5.0)
    try:
        sun.data.use_shadow = True
    except Exception:
        pass
    try:
        sun.data.shadow_soft_size = random.uniform(0.02, 0.12)
    except Exception:
        pass
    try:
        scene.eevee.use_shadows = True
    except Exception:
        pass
    add_grass_ground()
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    board = bpy.context.active_object
    board.scale = (2.0, 1.0, 1.0) 
    bpy.ops.object.transform_apply(scale=True)
    board.data.materials.append(create_grid_material())
    add_panel_back_face(board)
    anchor_x = (1 - 5.5) * 0.2
    anchor_y = (1 - 5.5) * 0.1
    bpy.ops.mesh.primitive_plane_add(size=1, location=(anchor_x, anchor_y, 0.012))
    white_anchor = bpy.context.active_object
    white_anchor.scale = (0.16, 0.08, 1)
    bpy.ops.object.transform_apply(scale=True)
    white_anchor.parent = board
    white_mat = bpy.data.materials.new(name='Mat_White_Anchor')
    try:
        white_mat.use_nodes = True
    except:
        pass
    wn = white_mat.node_tree.nodes
    wl = white_mat.node_tree.links
    for node in list(wn):
        wn.remove(node)
    em = wn.new('ShaderNodeEmission')
    em.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    em.inputs['Strength'].default_value = 1.8
    out = wn.new('ShaderNodeOutputMaterial')
    wl.new(em.outputs['Emission'], out.inputs['Surface'])
    white_anchor.data.materials.append(white_mat)
    cards_data = []
    log_entries = []
    panel_id = random.choice(['A', 'B', 'C'])
    if os.environ.get('DRONIADA_RANDOM_STAND', '0') == '1':
        stand_cfg = random.choice(COMPETITION_STAND_MODES)
    else:
        stand_cfg = COMPETITION_STAND_MODES[scene_idx % len(COMPETITION_STAND_MODES)]
    angle_cat = panel_angle_category(stand_cfg['id'])
    if os.environ.get('DRONIADA_RANDOM_PLANE_TILT', '0') == '1':
        tilt_deg = random.choice([0, 45, 90])
        report_skew_deg = int(tilt_deg)
    else:
        tilt_deg = 0
        report_skew_deg = report_skew_deg_from_stand(stand_cfg['id'])
    num_cards = 4
    chosen_colors = random.sample(list(COLORS.keys()), num_cards)
    cells = list(CARD_CELLS_BY_ANGLE[angle_cat])
    for i, (color_name, (col, row)) in enumerate(zip(chosen_colors, cells)):
        lx = (col - 5.5) * 0.2
        ly = (row - 5.5) * 0.1
        bpy.ops.mesh.primitive_plane_add(size=1, location=(lx, ly, 0.022))
        card = bpy.context.active_object
        card.scale = (0.17, 0.085, 1.0)
        bpy.ops.object.transform_apply(scale=True)
        card.parent = board 
        c_mat = bpy.data.materials.new(name=f'Mat_Card_{i}')
        try:
            c_mat.use_nodes = True
        except Exception:
            pass
        cn = c_mat.node_tree.nodes
        cl = c_mat.node_tree.links
        for node in list(cn):
            cn.remove(node)
        em_c = cn.new('ShaderNodeEmission')
        em_c.inputs['Color'].default_value = COLORS[color_name]
        em_c.inputs['Strength'].default_value = 1.15 + random.uniform(0.0, 0.35)
        out_c = cn.new('ShaderNodeOutputMaterial')
        cl.new(em_c.outputs['Emission'], out_c.inputs['Surface'])
        card.data.materials.append(c_mat)
        cards_data.append((card, COLOR_TO_CLASS[color_name]))
        log_str = format_card_detected_line(
            panel_id, report_skew_deg, row, col, color_name,
            timestamp_literal='HH:MM:SS.mmm',
        )
        log_entries.append(log_str)
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
    panel_root = bpy.context.active_object
    panel_root.name = 'PanelRoot'
    long_hinge = None
    short_hinge = None
    portrait_twist = None
    tilt_rz = math.radians(float(tilt_deg))
    long_rx = stand_cfg.get('long_hinge_rx_deg')
    if long_rx is not None:
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
        long_hinge = bpy.context.active_object
        long_hinge.name = 'LongEdgeHinge'
        long_hinge.parent = panel_root
        long_hinge.location = (0.0, -0.5, 0.0)
        long_hinge.rotation_euler = (0.0, 0.0, 0.0)
        if stand_cfg['id'] == 'long_edge_upright_portrait':
            bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
            portrait_twist = bpy.context.active_object
            portrait_twist.name = 'PortraitRzPivot'
            portrait_twist.parent = long_hinge
            portrait_twist.location = (-1.0, 0.0, 0.0)
            portrait_twist.rotation_euler = (0.0, 0.0, math.radians(90.0))
            board.parent = portrait_twist
            board.location = (1.0, 0.5, 0.0)
            board.rotation_euler = (0.0, 0.0, tilt_rz)
        else:
            board.parent = long_hinge
            board.location = (0.0, 0.5, 0.0)
            board.rotation_euler = (0.0, 0.0, tilt_rz)
        long_hinge.rotation_euler = (math.radians(float(long_rx)), 0.0, 0.0)
        panel_root.rotation_euler = (0.0, 0.0, 0.0)
    elif stand_cfg['id'] == 'short_edge_upright':
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
        short_hinge = bpy.context.active_object
        short_hinge.name = 'ShortEdgeHinge'
        short_hinge.parent = panel_root
        short_hinge.location = (-1.0, 0.0, 0.0)
        short_hinge.rotation_euler = (0.0, 0.0, 0.0)
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
        portrait_twist = bpy.context.active_object
        portrait_twist.name = 'PortraitRzPivot'
        portrait_twist.parent = short_hinge
        portrait_twist.location = (0.0, -0.5, 0.0)
        portrait_twist.rotation_euler = (0.0, 0.0, math.radians(90.0))
        board.parent = portrait_twist
        board.location = (1.0, 0.5, 0.0)
        board.rotation_euler = (0.0, 0.0, tilt_rz)
        short_hinge.rotation_euler = (math.radians(90.0), 0.0, 0.0)
        panel_root.rotation_euler = (0.0, 0.0, 0.0)
    else:
        raise RuntimeError(f'Nieznany tryb stojaka: {stand_cfg!r}')
    bpy.context.view_layer.update()
    raise_panel_root_above_ground(panel_root, board, min_z=PANEL_MIN_Z_M)
    bpy.ops.object.camera_add(location=(0.0, -7.0, 0.0))
    cam = bpy.context.active_object
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
    target = bpy.context.active_object
    scene.camera = cam
    if ORBIT_AZIMUTH_DEG_LIST:
        view_specs = [(i, float(az)) for i, az in enumerate(ORBIT_AZIMUTH_DEG_LIST)]
    else:
        view_specs = [(i, None) for i in range(num_views)]
    written = 0
    for view_idx, azimuth_deg in view_specs:
        cam_placement = place_camera_orbit_step(
            scene,
            cam,
            target,
            board,
            view_idx,
            len(view_specs),
            ORBIT_ARC_DEG,
            azimuth_deg=azimuth_deg,
        )
        if not cam_placement.get('accepted', False):
            az_lab = azimuth_deg if azimuth_deg is not None else view_idx
            print(
                f'skip scene={scene_idx} azimuth={az_lab} reason={cam_placement.get("reject_reason")}',
            )
            continue
        if not white_anchor_in_front(scene, cam, board):
            print(f'skip scene={scene_idx} azimuth={az_lab} reason=grid_anchor_behind_final')
            continue
        g = global_idx_start + written
        written += 1
        yolo_path = os.path.join(LABELS_YOLO_PATH, f'img_{g}.txt')
        with open(yolo_path, 'w', encoding='utf-8') as f:
            for card_obj, cls_id in cards_data:
                cx, cy, w, h = get_yolo_bbox(scene, cam, card_obj)
                if w > 0 and h > 0:
                    f.write(f'{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
        raport_path = os.path.join(LABELS_RAPORT_PATH, f'img_{g}.txt')
        az = float(cam_placement.get('orbit_azimuth_deg', 0.0))
        with open(raport_path, 'w', encoding='utf-8') as f:
            for log in log_entries:
                f.write(f'{log} | orbit_azimuth_deg={az:.2f}\n')
        intr = compute_intrinsics_dict(scene, cam)
        panel_pose = {'id': panel_id, 'competition_stand_index': scene_idx % len(COMPETITION_STAND_MODES), 'competition_stand_mode': stand_cfg['id'], 'competition_stand_label_pl': stand_cfg['label_pl'], 'panel_angle_category': angle_cat, 'root_rotation_euler_xyz_deg': [math.degrees(float(panel_root.rotation_euler.x)), math.degrees(float(panel_root.rotation_euler.y)), math.degrees(float(panel_root.rotation_euler.z))], 'board_plane_rotation_euler_xyz_deg': [math.degrees(float(board.rotation_euler.x)), math.degrees(float(board.rotation_euler.y)), math.degrees(float(board.rotation_euler.z))], 'business_angle_xy_deg': int(report_skew_deg), 'panel_skew_report_deg': int(report_skew_deg), 'panel_z_min_world': board_world_z_range(board)[0]}
        if long_hinge is not None:
            panel_pose['long_edge_hinge_rotation_euler_xyz_deg'] = [math.degrees(float(long_hinge.rotation_euler.x)), math.degrees(float(long_hinge.rotation_euler.y)), math.degrees(float(long_hinge.rotation_euler.z))]
            panel_pose['long_edge_hinge_offset_panel_root_xyz_m'] = [0.0, -0.5, 0.0]
            if portrait_twist is not None and stand_cfg['id'] == 'long_edge_upright_portrait':
                panel_pose['portrait_in_plane_rz_deg'] = 90.0
                panel_pose['portrait_pivot_cell_11_bottom_left'] = True
                panel_pose['portrait_twist_offset_long_hinge_xyz_m'] = [-1.0, 0.0, 0.0]
                panel_pose['board_offset_portrait_twist_xyz_m'] = [1.0, 0.5, 0.0]
            else:
                panel_pose['board_offset_long_hinge_xyz_m'] = [0.0, 0.5, 0.0]
        if short_hinge is not None:
            panel_pose['short_edge_hinge_rotation_euler_xyz_deg'] = [math.degrees(float(short_hinge.rotation_euler.x)), math.degrees(float(short_hinge.rotation_euler.y)), math.degrees(float(short_hinge.rotation_euler.z))]
            panel_pose['short_edge_hinge_offset_panel_root_xyz_m'] = [-1.0, 0.0, 0.0]
            if portrait_twist is not None:
                panel_pose['portrait_in_plane_rz_deg'] = 90.0
                panel_pose['portrait_pivot_cell_11_bottom_left'] = True
                panel_pose['portrait_twist_offset_short_hinge_xyz_m'] = [0.0, -0.5, 0.0]
                panel_pose['board_offset_portrait_twist_xyz_m'] = [1.0, 0.5, 0.0]
            else:
                panel_pose['board_offset_hinge_xyz_m'] = [1.0, 0.0, 0.0]
        r_model_to_cam_cv, t_model_to_cam_cv = model_to_camera_opencv(board, cam)
        pose_doc = {'schema': 'droniada_pose_v1', 'image': f'img_{g}.png', 'environment': {'grass_ground_z': 0.0, 'sky': 'tex_sky_preetham', 'camera_front_only': True, 'panel_min_z_m': PANEL_MIN_Z_M, 'camera_orbit_arc_deg': ORBIT_ARC_DEG}, 'scene_id': scene_idx, 'view_index': view_idx, 'intrinsics': intr, 'camera': {'location_world': [float(cam.location.x), float(cam.location.y), float(cam.location.z)], 'rotation_euler_xyz_rad': [float(cam.rotation_euler.x), float(cam.rotation_euler.y), float(cam.rotation_euler.z)], 'camera_convention': 'opencv_x_right_y_down_z_forward', **cam_placement}, 'panel': panel_pose, 'model_to_camera_opencv': {'rotation_3x3': [[float(r_model_to_cam_cv[0][0]), float(r_model_to_cam_cv[0][1]), float(r_model_to_cam_cv[0][2])], [float(r_model_to_cam_cv[1][0]), float(r_model_to_cam_cv[1][1]), float(r_model_to_cam_cv[1][2])], [float(r_model_to_cam_cv[2][0]), float(r_model_to_cam_cv[2][1]), float(r_model_to_cam_cv[2][2])]], 'translation_m': [float(t_model_to_cam_cv.x), float(t_model_to_cam_cv.y), float(t_model_to_cam_cv.z)]}}
        pose_path = os.path.join(LABELS_POSE_PATH, f'img_{g}.json')
        with open(pose_path, 'w', encoding='utf-8') as f:
            json.dump(pose_doc, f, ensure_ascii=False, indent=2)
        img_path = os.path.join(IMAGES_PATH, f'img_{g}.png')
        scene.render.filepath = img_path
        bpy.ops.render.render(write_still=True)
    return written
NUM_SCENES = int(os.environ.get('DRONIADA_NUM_SCENES', '25'))
TARGET_IMAGES = int(os.environ.get('DRONIADA_TARGET_IMAGES', '0'))
if ORBIT_AZIMUTH_DEG_LIST:
    VIEWS_FOR_RUN = len(ORBIT_AZIMUTH_DEG_LIST)
else:
    VIEWS_FOR_RUN = VIEWS_PER_SCENE
if os.environ.get('DRONIADA_QUICK_TEST') == '1':
    NUM_SCENES = 5
    if ORBIT_AZIMUTH_DEG_LIST:
        quick_az = os.environ.get('DRONIADA_ORBIT_AZIMUTHS_QUICK', '-20,0,20')
        ORBIT_AZIMUTH_DEG_LIST = [float(x.strip()) for x in quick_az.split(',') if x.strip()]
        VIEWS_FOR_RUN = len(ORBIT_AZIMUTH_DEG_LIST)
    else:
        VIEWS_FOR_RUN = int(os.environ.get('DRONIADA_ORBIT_STEPS_QUICK', '5'))
orbit_label = ORBIT_AZIMUTH_DEG_LIST if ORBIT_AZIMUTH_DEG_LIST else f'arc_{ORBIT_ARC_DEG}'
print(
    NUM_SCENES,
    VIEWS_FOR_RUN,
    orbit_label,
    'min_dot',
    CAMERA_FRONT_MIN_DOT,
    'min_span_px',
    MIN_PANEL_SPAN_PX,
    'target_images',
    TARGET_IMAGES,
    DATASET_PATH,
)
if os.environ.get('DRONIADA_FRESH') == '1':
    import shutil
    for sub in ('images', 'labels_yolo', 'labels_raport', 'labels_pose'):
        p = os.path.join(DATASET_PATH, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)
        os.makedirs(p, exist_ok=True)
    print('DRONIADA_FRESH=1: wyczyszczono images + labels_*')
gidx = 0
for s in range(NUM_SCENES):
    if TARGET_IMAGES > 0 and gidx >= TARGET_IMAGES:
        break
    n = generate_scene_multiview(s, VIEWS_FOR_RUN, gidx)
    gidx += n
    print(s, gidx - n, gidx - 1)
    if TARGET_IMAGES > 0 and gidx >= TARGET_IMAGES:
        print(f'target_reached={TARGET_IMAGES}')
        break
print(gidx, DATASET_PATH)
