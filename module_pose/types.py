from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass
class PoseResult:
    ok: bool
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    corners_px: Optional[np.ndarray] = None
    euler_cam_deg: Optional[tuple] = None
    report_angle_deg: int = 0
    panel_angle_category: str = 'horizontal'
    stand_confidence: float = 0.0
    confidence: float = 0.0
    method: str = ''
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {'ok': self.ok, 'confidence': self.confidence, 'method': self.method}
        if self.rvec is not None:
            out['rvec'] = self.rvec.reshape(-1).tolist()
        if self.tvec is not None:
            out['tvec'] = self.tvec.reshape(-1).tolist()
        if self.corners_px is not None:
            out['corners_px'] = self.corners_px.astype(float).tolist()
        if self.euler_cam_deg is not None:
            rx, ry, rz = (float(self.euler_cam_deg[0]), float(self.euler_cam_deg[1]), float(self.euler_cam_deg[2]))
            out['euler_cam_deg'] = [rx, ry, rz]
            out['panel_euler_vs_drone_deg'] = [rx, ry, rz]
            out['roll_deg'] = rx
            out['pitch_deg'] = ry
            out['yaw_deg'] = rz
            out['panel_orientation_vs_drone'] = {'roll_deg': rx, 'pitch_deg': ry, 'yaw_deg': rz}
        if self.tvec is not None:
            tv = np.asarray(self.tvec, dtype=np.float64).reshape(3)
            out['distance_camera_to_panel_center_m'] = float(np.linalg.norm(tv))
            out['panel_center_in_camera_m'] = {'x': float(tv[0]), 'y': float(tv[1]), 'z': float(tv[2])}
        out['meta'] = self.meta
        out['report_angle_deg'] = int(self.report_angle_deg)
        out['panel_angle_category'] = str(self.panel_angle_category)
        out['stand_confidence'] = float(self.stand_confidence)
        return out

    def to_integration_dict(self, *, panel_id: Optional[str] = None) -> Dict[str, Any]:
        from module_pose.panel_stand import integration_dict_from_pose_fields

        euler = self.euler_cam_deg or (0.0, 0.0, 0.0)
        dist_m = 0.0
        if self.tvec is not None:
            dist_m = float(np.linalg.norm(np.asarray(self.tvec, dtype=np.float64).reshape(3)))
        reproj = float(self.meta.get('reproj_mean_px', float('nan')))
        return integration_dict_from_pose_fields(
            ok=self.ok,
            roll_deg=float(euler[0]),
            pitch_deg=float(euler[1]),
            yaw_deg=float(euler[2]),
            distance_m=dist_m,
            report_angle_deg=int(self.report_angle_deg),
            panel_angle_category=str(self.panel_angle_category),
            stand_confidence=float(self.stand_confidence),
            reproj_mean_px=reproj,
            method=str(self.method),
            panel_id=panel_id,
        )
