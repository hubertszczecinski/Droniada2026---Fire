from dataclasses import dataclass
from typing import Optional

@dataclass
class CompetitionPaths:
    image_png: str
    yolo_txt: str
    pose_json_intrinsics_only: Optional[str] = None
    angle_calibration_json: Optional[str] = None
