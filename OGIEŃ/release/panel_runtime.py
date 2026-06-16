from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import pipeline_competition as pc
from module_panel.analyze import analyze_panel_image
from module_panel.report import predictions_to_report_lines
from module_pose.api import intrinsics_from_pose_json, load_pose_gt_json
from release.frames import FrameInput

@dataclass
class PanelConfig:
    panel_id: str = 'A'
    xy_mode: str = 'grid_geom_white'
    angle_source: str = 'rmat_linear'
    angle_calibration_path: Optional[str] = None
    use_pose_json_intrinsics: bool = True
    require_reliable_report: bool = False

@dataclass
class PanelFrameOutput:
    frame_id: str
    ok: bool
    panel_id: str = 'A'
    report_angle_deg: int = 0
    panel_angle_category: str = 'horizontal'
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    report_lines: List[str] = field(default_factory=list)
    grid_xy_reliable: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

class PanelRuntime:
    __slots__ = ('cfg', '_k', '_dist', '_outputs', '_default_calib')

    def __init__(self, cfg: PanelConfig) -> None:
        self.cfg = cfg
        self._k = np.eye(3, dtype=np.float32)
        self._dist = np.zeros((4, 1), dtype=np.float32)
        self._outputs: List[PanelFrameOutput] = []
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self._default_calib = cfg.angle_calibration_path or os.path.join(root, 'module_panel', 'data', 'angle_linear_rmat.json')
        if not os.path.isfile(self._default_calib):
            self._default_calib = None

    @property
    def outputs(self) -> List[PanelFrameOutput]:
        return self._outputs

    def _intrinsics(self, pose_json: Optional[str], w: int, h: int) -> None:
        if self.cfg.use_pose_json_intrinsics and pose_json:
            gt = load_pose_gt_json(pose_json)
            if isinstance(gt, dict) and gt.get('intrinsics') is not None:
                self._k, self._dist = intrinsics_from_pose_json(gt)
                return
        self._k[0, 0] = 1000.0
        self._k[1, 1] = 1000.0
        self._k[0, 2] = w / 2.0
        self._k[1, 2] = h / 2.0
        self._k[2, 2] = 1.0
        self._dist.fill(0.0)

    def process_bgr(self, img: np.ndarray, frame_id: str, det: Optional[List]=None, pose_json: Optional[str]=None, *, record: bool=True) -> PanelFrameOutput:
        if img is None or img.size == 0:
            out = PanelFrameOutput(frame_id=frame_id, ok=False, panel_id=self.cfg.panel_id, meta={'err': 'no_image'})
            if record:
                self._outputs.append(out)
            return out
        h, w = img.shape[:2]
        self._intrinsics(pose_json, w, h)
        det = det if det is not None else []
        json_angle: Optional[int] = None
        panel_id = self.cfg.panel_id
        angle_source = self.cfg.angle_source
        if angle_source == 'json' and pose_json:
            gt = load_pose_gt_json(pose_json)
            if gt:
                json_angle = int(gt.get('panel', {}).get('panel_skew_report_deg', gt.get('panel', {}).get('business_angle_xy_deg', 0)))
                panel_id = str(gt.get('panel', {}).get('id', panel_id))
            else:
                angle_source = 'rmat_linear'
        pan = analyze_panel_image(img, det, k=self._k, dist=self._dist, xy_mode=self.cfg.xy_mode, angle_source=angle_source, json_report_angle_deg=json_angle, angle_calibration_path=self._default_calib)
        reliable = bool(pan.meta.get('grid_xy_reliable', False))
        predictions = list(pan.predictions)
        if self.cfg.require_reliable_report and not reliable:
            predictions = []
            pan.meta['report_suppressed'] = 'grid_xy_not_reliable'
        lines = predictions_to_report_lines(panel_id, pan.report_angle_deg, predictions)
        ok = pan.meta.get('err') != 'no_corners'
        out = PanelFrameOutput(frame_id=frame_id, ok=bool(ok), panel_id=panel_id, report_angle_deg=int(pan.report_angle_deg), panel_angle_category=str(pan.panel_angle_category), predictions=predictions, report_lines=lines, grid_xy_reliable=reliable, meta=dict(pan.meta))
        if record:
            self._outputs.append(out)
        return out

    def process_frame(self, frame: FrameInput) -> PanelFrameOutput:
        img = cv2.imread(frame.image_path)
        det = pc.load_yolo(frame.yolo_path) if frame.yolo_path else []
        return self.process_bgr(img, frame.frame_id, det=det, pose_json=frame.pose_json_path)

    def run_loop(self, frames: List[FrameInput]) -> List[PanelFrameOutput]:
        self._outputs.clear()
        for frame in frames:
            self.process_frame(frame)
        return self._outputs
