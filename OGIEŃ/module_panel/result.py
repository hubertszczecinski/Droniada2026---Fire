from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np

@dataclass
class PanelAnalyzeResult:
    predictions: List[Dict[str, Any]]
    warped_bgr: np.ndarray
    homography: np.ndarray
    report_angle_deg: int
    panel_angle_category: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def predictions_only(self) -> List[Dict[str, Any]]:
        return self.predictions
