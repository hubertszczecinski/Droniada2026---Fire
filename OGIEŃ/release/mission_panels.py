"""Misja wielopanelowa: N migawek na panel, zapis per panel, reset po wysyłce."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from module_panel.competition_report import parse_competition_report_line
from release.live_dashboard import LiveSnapshotStore


@dataclass
class PanelMissionState:
    panel_ids: List[str]
    current_index: int = 0
    snapshots_per_panel: int = 8
    submitted: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'panel_ids': list(self.panel_ids),
            'current_index': int(self.current_index),
            'current_panel': self.current_panel,
            'snapshots_per_panel': int(self.snapshots_per_panel),
            'submitted': dict(self.submitted),
            'updated_at': datetime.now().isoformat(timespec='seconds'),
        }

    @property
    def current_panel(self) -> str:
        if not self.panel_ids:
            return 'A'
        idx = max(0, min(int(self.current_index), len(self.panel_ids) - 1))
        return str(self.panel_ids[idx])

    def mission_done(self) -> bool:
        return int(self.current_index) >= len(self.panel_ids)


class MultiPanelMissionManager:
    """Osobna galeria migawek i konkurs CXY dla każdego panelu A/B/C."""

    def __init__(
        self,
        session_root: str,
        panel_ids: List[str],
        *,
        snapshots_per_panel: int,
        store_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        ids = [str(p).strip().upper() for p in panel_ids if str(p).strip()]
        if not ids:
            ids = ['A']
        self.session_root = session_root
        self.state = PanelMissionState(
            panel_ids=ids,
            snapshots_per_panel=max(1, int(snapshots_per_panel)),
        )
        kw = dict(store_kwargs or {})
        kw['max_snapshots'] = self.state.snapshots_per_panel
        self._store_kw = kw
        self.stores: Dict[str, LiveSnapshotStore] = {}
        for pid in ids:
            pdir = os.path.join(session_root, 'panels', pid)
            os.makedirs(pdir, exist_ok=True)
            self.stores[pid] = LiveSnapshotStore(pdir, **kw)
        self._mission_path = os.path.join(session_root, 'mission.json')
        self._load_mission()

    def _load_mission(self) -> None:
        if not os.path.isfile(self._mission_path):
            self.save_mission()
            return
        try:
            with open(self._mission_path, encoding='utf-8') as fh:
                data = json.load(fh)
            if data.get('panel_ids') == self.state.panel_ids:
                self.state.current_index = int(data.get('current_index', 0))
                self.state.submitted = dict(data.get('submitted') or {})
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    def save_mission(self) -> None:
        with open(self._mission_path, 'w', encoding='utf-8') as fh:
            json.dump(self.state.to_dict(), fh, ensure_ascii=False, indent=2)

    @property
    def current_panel(self) -> str:
        return self.state.current_panel

    def current_store(self) -> LiveSnapshotStore:
        return self.stores[self.state.current_panel]

    def store_for(self, panel_id: str) -> LiveSnapshotStore:
        return self.stores[str(panel_id).upper()]

    def snapshot_count(self, panel_id: Optional[str] = None) -> int:
        pid = str(panel_id or self.current_panel).upper()
        return len(self.stores[pid].entries)

    def panel_is_full(self, panel_id: Optional[str] = None) -> bool:
        return self.snapshot_count(panel_id) >= self.state.snapshots_per_panel

    def can_accept_snapshots(self) -> bool:
        if self.state.mission_done():
            return False
        return not self.panel_is_full(self.current_panel)

    def clear_panel_snapshots(self, panel_id: Optional[str] = None) -> int:
        pid = str(panel_id or self.current_panel).upper()
        n = self.stores[pid].clear_all()
        from release.snapshot_cxy_competition import update_session_competition

        comp_path = os.path.join(self.stores[pid].session_dir, 'cxy_competition.json')
        if os.path.isfile(comp_path):
            try:
                os.remove(comp_path)
            except OSError:
                pass
        comp_txt = os.path.join(self.stores[pid].session_dir, 'cxy_competition_report.txt')
        if os.path.isfile(comp_txt):
            try:
                os.remove(comp_txt)
            except OSError:
                pass
        return n

    def submit_panel(
        self,
        panel_id: Optional[str],
        *,
        report_lines: List[str],
        predictions: Optional[List[Dict[str, Any]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], int]:
        """
        Zapisz szkic wyniku, wyzeruj migawki panelu, przejdź do następnego.
        Zwraca (następny_panel_id | None, liczba_usuniętych_plików_migawek).
        """
        pid = str(panel_id or self.current_panel).upper()
        structured: List[Dict[str, Any]] = []
        for line in report_lines:
            rec = parse_competition_report_line(line)
            if rec is not None:
                structured.append(rec)
        payload = {
            'panel_id': pid,
            'submitted_at': datetime.now().isoformat(timespec='seconds'),
            'report_format': 'droniada_2026_ogien_v1',
            'report_lines': list(report_lines),
            'report_structured': structured,
            'predictions': list(predictions or []),
            'meta': dict(meta or {}),
            'snapshot_count_at_submit': self.snapshot_count(pid),
        }
        self.state.submitted[pid] = payload
        out_dir = os.path.join(self.session_root, 'panels', pid)
        os.makedirs(out_dir, exist_ok=True)
        draft_path = os.path.join(out_dir, 'submitted_draft.json')
        with open(draft_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        n_cleared = self.clear_panel_snapshots(pid)
        if pid == self.current_panel:
            idx = self.state.panel_ids.index(pid)
            if idx == self.state.current_index:
                self.state.current_index = min(idx + 1, len(self.state.panel_ids))
        self.save_mission()
        if self.state.mission_done():
            return None, n_cleared
        return self.current_panel, n_cleared

    def panel_status(self, panel_id: Optional[str] = None) -> Dict[str, Any]:
        pid = str(panel_id or self.current_panel).upper()
        return {
            'panel_id': pid,
            'snapshots': self.snapshot_count(pid),
            'snapshots_max': self.state.snapshots_per_panel,
            'full': self.panel_is_full(pid),
            'submitted': pid in self.state.submitted,
        }

    def mission_hud_text(self, *, pause_remaining: int = 0) -> str:
        total = len(self.state.panel_ids)
        submitted = len(self.state.submitted)
        if self.state.mission_done():
            return f'Misja zakończona · wysłano {submitted}/{total} paneli'
        panel = self.current_panel
        pos = f'({self.state.current_index + 1}/{total})'
        snap_n = self.snapshot_count()
        snap_max = self.state.snapshots_per_panel
        if pause_remaining > 0:
            return (
                f'Lot do panelu {panel} {pos} · pauza {pause_remaining}s · analiza OFF'
            )
        return f'Panel {panel} {pos} · migawki {snap_n}/{snap_max}'
