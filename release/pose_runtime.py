from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import pipeline_competition as pc
from module_pose.api import (
    intrinsics_from_pose_json,
    load_pose_gt_json,
    pose_from_corners,
    pose_from_image,
    pose_from_yolo_pose,
)
from release.frames import FrameInput

@dataclass
class PoseConfig:
    """corner_source: yolo_pose (real) | cv (legacy Blender) | auto (yolo, potem cv)."""
    corner_source: str = 'yolo_pose'
    use_pose_json_intrinsics: bool = False
    refine_corners_grid: bool = True
    prefer_img_corners: bool = False
    yolo_pose_fallback_cv: bool = True
    yolo_pose_use_tracker: bool = True
    stand_calibration_path: Optional[str] = None

@dataclass
class PoseFrameOutput:
    frame_id: str
    ok: bool
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    distance_m: float = 0.0
    report_angle_deg: int = 0
    panel_angle_category: str = 'horizontal'
    stand_confidence: float = 0.0
    confidence: float = 0.0
    method: str = ''
    reproj_mean_px: float = float('nan')
    corners_px: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_integration_dict(self, *, panel_id: Optional[str] = None) -> Dict[str, Any]:
        from module_pose.panel_stand import integration_dict_from_pose_fields
        return integration_dict_from_pose_fields(
            ok=self.ok,
            roll_deg=self.roll_deg,
            pitch_deg=self.pitch_deg,
            yaw_deg=self.yaw_deg,
            distance_m=self.distance_m,
            report_angle_deg=self.report_angle_deg,
            panel_angle_category=self.panel_angle_category,
            stand_confidence=self.stand_confidence,
            reproj_mean_px=self.reproj_mean_px,
            method=self.method,
            panel_id=panel_id,
        )

class PoseRuntime:
    __slots__ = ('cfg', '_k', '_dist', '_corners_buf', '_outputs')

    def __init__(self, cfg: PoseConfig) -> None:
        self.cfg = cfg
        self._k = np.eye(3, dtype=np.float32)
        self._dist = np.zeros((4, 1), dtype=np.float32)
        self._corners_buf = np.zeros((4, 2), dtype=np.float32)
        self._outputs: List[PoseFrameOutput] = []

    @property
    def outputs(self) -> List[PoseFrameOutput]:
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

    def _resolve_pose(
        self,
        img: np.ndarray,
        det: List,
        *,
        corners_px: Optional[np.ndarray] = None,
        corners_meta: Optional[Dict[str, Any]] = None,
    ):
        src = str(self.cfg.corner_source).strip().lower()
        if corners_px is not None and corners_px.shape == (4, 2):
            return pose_from_corners(
                img,
                corners_px,
                self._k.copy(),
                self._dist.copy(),
                base_method='yolo_pose',
                refine_corners_grid=self.cfg.refine_corners_grid,
                extra_meta=dict(corners_meta or {}),
            )
        if src == 'yolo_pose':
            res = pose_from_yolo_pose(
                img,
                self._k.copy(),
                self._dist.copy(),
                refine_corners_grid=self.cfg.refine_corners_grid,
                use_tracker=self.cfg.yolo_pose_use_tracker,
            )
            if res.ok or not self.cfg.yolo_pose_fallback_cv:
                return res
            cv_res = pose_from_image(
                img,
                det,
                k=self._k.copy(),
                dist=self._dist.copy(),
                prefer_img_corners=True,
                refine_corners_grid=self.cfg.refine_corners_grid,
            )
            if cv_res.ok:
                cv_res.meta = dict(cv_res.meta or {})
                cv_res.meta['corner_source'] = 'cv_fallback'
                return cv_res
            return res
        if src == 'cv':
            return pose_from_image(
                img,
                det,
                k=self._k.copy(),
                dist=self._dist.copy(),
                prefer_img_corners=self.cfg.prefer_img_corners,
                refine_corners_grid=self.cfg.refine_corners_grid,
            )
        # auto: yolo_pose z fallback cv
        res = pose_from_yolo_pose(
            img,
            self._k.copy(),
            self._dist.copy(),
            refine_corners_grid=self.cfg.refine_corners_grid,
            use_tracker=self.cfg.yolo_pose_use_tracker,
        )
        if res.ok:
            return res
        return pose_from_image(
            img,
            det,
            k=self._k.copy(),
            dist=self._dist.copy(),
            prefer_img_corners=True,
            refine_corners_grid=self.cfg.refine_corners_grid,
        )

    def process_bgr(
        self,
        img: np.ndarray,
        frame_id: str,
        det: Optional[List] = None,
        pose_json: Optional[str] = None,
        *,
        k: Optional[np.ndarray] = None,
        dist: Optional[np.ndarray] = None,
        corners_px: Optional[np.ndarray] = None,
        corners_meta: Optional[Dict[str, Any]] = None,
        record: bool = True,
    ) -> PoseFrameOutput:
        if img is None or img.size == 0:
            out = PoseFrameOutput(frame_id=frame_id, ok=False, method='none', meta={'reason': 'no_image'})
            if record:
                self._outputs.append(out)
            return out
        h, w = img.shape[:2]
        if k is not None and dist is not None:
            self._k = k.astype(np.float32).copy()
            self._dist = dist.astype(np.float32).copy()
        else:
            self._intrinsics(pose_json, w, h)
        det = det if det is not None else []
        res = self._resolve_pose(img, det, corners_px=corners_px, corners_meta=corners_meta)
        corners_out = None
        if res.corners_px is not None and res.corners_px.shape == (4, 2):
            corners_out = res.corners_px.astype(np.float32).copy()
            np.copyto(self._corners_buf, corners_out)
        d = res.to_dict()
        if res.ok and res.euler_cam_deg is not None:
            rx, ry, rz = res.euler_cam_deg
            out = PoseFrameOutput(
                frame_id=frame_id,
                ok=True,
                roll_deg=float(rx),
                pitch_deg=float(ry),
                yaw_deg=float(rz),
                distance_m=float(d.get('distance_camera_to_panel_center_m', 0.0)),
                report_angle_deg=int(res.report_angle_deg),
                panel_angle_category=str(res.panel_angle_category),
                stand_confidence=float(res.stand_confidence),
                confidence=float(res.confidence),
                method=str(res.method),
                reproj_mean_px=float((res.meta or {}).get('reproj_mean_px', float('nan'))),
                corners_px=corners_out,
                meta=dict(res.meta or {}),
            )
        else:
            out = PoseFrameOutput(
                frame_id=frame_id,
                ok=False,
                confidence=float(res.confidence),
                method=str(res.method),
                reproj_mean_px=float((res.meta or {}).get('reproj_mean_px', float('nan'))),
                corners_px=corners_out,
                meta=dict(res.meta or {}),
            )
        if record:
            self._outputs.append(out)
        return out

    def process_frame(self, frame: FrameInput) -> PoseFrameOutput:
        img = cv2.imread(frame.image_path)
        det = pc.load_yolo(frame.yolo_path) if frame.yolo_path else []
        return self.process_bgr(img, frame.frame_id, det=det, pose_json=frame.pose_json_path)

    def run_loop(self, frames: List[FrameInput]) -> List[PoseFrameOutput]:
        self._outputs.clear()
        for frame in frames:
            self.process_frame(frame)
        return self._outputs
