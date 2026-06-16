"""Zatrzask CXY: wybierz klatkę, gdy siatka jest idealnie nałożona (grid_xy_reliable)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CxySnapshot:
    frame_id: str
    reproj_mean_px: float
    homography_inliers: int
    predictions: List[Dict[str, Any]]
    report_lines: List[str]
    score: float
    meta: Dict[str, Any] = field(default_factory=dict)


def reliability_score(
    *,
    reliable: bool,
    reproj_mean_px: float,
    homography_inliers: int = 0,
) -> float:
    """Niżej = lepiej. Niewiarygodne → +inf."""
    if not reliable:
        return float('inf')
    reproj = float(reproj_mean_px) if np.isfinite(reproj_mean_px) else 999.0
    return reproj - 0.02 * float(homography_inliers)


@dataclass
class CxyLatch:
    """Blokuje wynik dopiero po serii identycznych wyników reliable."""

    min_stable_frames: int = 3
    _stable_run: int = 0
    _best: Optional[CxySnapshot] = None
    _locked: Optional[CxySnapshot] = None
    _candidate_signature: Optional[str] = None

    @staticmethod
    def _predictions_signature(predictions: List[Dict[str, Any]]) -> str:
        """Deterministyczny podpis wyniku CXY niezależny od kolejności listy."""
        norm: List[tuple[int, int, str]] = []
        for p in predictions:
            x = int(p.get('x', -1))
            y = int(p.get('y', -1))
            color = str(p.get('color', 'UNKNOWN')).upper()
            norm.append((y, x, color))
        norm.sort()
        return '|'.join(f'{y}:{x}:{color}' for y, x, color in norm)

    def reset(self) -> None:
        self._stable_run = 0
        self._best = None
        self._locked = None
        self._candidate_signature = None

    @property
    def locked(self) -> bool:
        return self._locked is not None

    @property
    def snapshot(self) -> Optional[CxySnapshot]:
        return self._locked or self._best

    def update(
        self,
        *,
        frame_id: str,
        reliable: bool,
        reproj_mean_px: float,
        homography_inliers: int,
        predictions: List[Dict[str, Any]],
        report_lines: List[str],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        score = reliability_score(
            reliable=reliable,
            reproj_mean_px=reproj_mean_px,
            homography_inliers=homography_inliers,
        )
        out: Dict[str, Any] = {
            'cxy_latch_locked': self.locked,
            'cxy_stable_run': self._stable_run,
            'cxy_score': score,
        }
        if self._locked is not None:
            out['cxy_locked_frame'] = self._locked.frame_id
            out['cxy_locked_reproj'] = self._locked.reproj_mean_px
            return out

        if reliable:
            sig = self._predictions_signature(predictions)
            snap = CxySnapshot(
                frame_id=frame_id,
                reproj_mean_px=float(reproj_mean_px),
                homography_inliers=int(homography_inliers),
                predictions=list(predictions),
                report_lines=list(report_lines),
                score=score,
                meta=dict(meta or {}),
            )
            if self._candidate_signature == sig:
                self._stable_run += 1
            else:
                self._stable_run = 1
                self._candidate_signature = sig
                self._best = None
            if self._best is None or snap.score < self._best.score:
                self._best = snap
            if self._stable_run >= self.min_stable_frames:
                self._locked = self._best
                out['cxy_latch_locked'] = True
                out['cxy_locked_frame'] = self._locked.frame_id
        else:
            self._stable_run = 0
            self._best = None
            self._candidate_signature = None

        if self._best is not None:
            out['cxy_best_frame'] = self._best.frame_id
            out['cxy_best_reproj'] = self._best.reproj_mean_px
        out['cxy_stable_run'] = self._stable_run
        return out

    def set_locked_colors(
        self,
        predictions: List[Dict[str, Any]],
        report_lines: List[str],
    ) -> None:
        """Uzupełnij kolory po zatrzaśnięciu (detekcja HSV dopiero na zamkniętej klatce)."""
        if self._locked is None:
            return
        self._locked.predictions = list(predictions)
        self._locked.report_lines = list(report_lines)
        if self._best is not None and self._best.frame_id == self._locked.frame_id:
            self._best.predictions = list(predictions)
            self._best.report_lines = list(report_lines)

    def force_snapshot(
        self,
        *,
        frame_id: str,
        reproj_mean_px: float,
        homography_inliers: int,
        predictions: List[Dict[str, Any]],
        report_lines: List[str],
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Klawisz 's': zapisz bieżącą klatkę jeśli reliable byłby — wymaga reliable w wywołaniu z zewnątrz."""
        snap = CxySnapshot(
            frame_id=frame_id,
            reproj_mean_px=float(reproj_mean_px),
            homography_inliers=int(homography_inliers),
            predictions=list(predictions),
            report_lines=list(report_lines),
            score=reliability_score(
                reliable=True,
                reproj_mean_px=reproj_mean_px,
                homography_inliers=homography_inliers,
            ),
            meta=dict(meta or {}),
        )
        self._locked = snap
        self._best = snap
        self._stable_run = self.min_stable_frames
        self._candidate_signature = self._predictions_signature(predictions)
        return True
