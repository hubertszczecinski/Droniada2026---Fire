"""
Konkurs CXY z migawek — odfiltrowanie jednorazowych błędów detekcji.

Każda migawka głosuje na trójki (wiersz, kolumna, kolor). Wynik konkursu
zostawia tylko pozycje powtórzone w co najmniej ``min_votes`` migawkach
(opcjonalnie z wagą 1/(reproj+1)).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pipeline_competition as pc
from module_panel.report import predictions_to_report_lines


@dataclass
class SnapshotObservation:
    frame_id: str
    reproj_b: float
    grid_overlap_ratio: float
    predictions: List[Dict[str, Any]]
    report_lines: List[str]
    panel_id: str = 'A'
    angle_deg: int = 0


@dataclass
class CompetitionCard:
    x: int
    y: int
    color: str
    votes: int
    weight: float
    support_ratio: float
    sources: List[str] = field(default_factory=list)


@dataclass
class SnapshotCxyCompetitionResult:
    panel_id: str
    angle_deg: int
    n_snapshots: int
    min_votes: int
    accepted: List[CompetitionCard]
    rejected: List[CompetitionCard]
    predictions: List[Dict[str, Any]]
    report_lines: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'panel_id': self.panel_id,
            'angle_deg': self.angle_deg,
            'n_snapshots': self.n_snapshots,
            'min_votes': self.min_votes,
            'predictions': list(self.predictions),
            'report_lines': list(self.report_lines),
            'accepted': [
                {
                    'x': c.x,
                    'y': c.y,
                    'color': c.color,
                    'votes': c.votes,
                    'weight': round(c.weight, 4),
                    'support_ratio': round(c.support_ratio, 4),
                    'sources': list(c.sources),
                }
                for c in self.accepted
            ],
            'rejected': [
                {
                    'x': c.x,
                    'y': c.y,
                    'color': c.color,
                    'votes': c.votes,
                    'weight': round(c.weight, 4),
                    'support_ratio': round(c.support_ratio, 4),
                    'sources': list(c.sources),
                }
                for c in self.rejected
            ],
        }


def _norm_color(color: Any) -> str:
    return str(color or 'UNKNOWN').strip().upper()


def _norm_prediction(p: Dict[str, Any]) -> Optional[Tuple[int, int, str]]:
    try:
        x = int(p.get('x', -1))
        y = int(p.get('y', -1))
    except (TypeError, ValueError):
        return None
    if not (1 <= x <= 10 and 1 <= y <= 10):
        return None
    color = _norm_color(p.get('color'))
    if color == 'UNKNOWN':
        return None
    return y, x, color


def predictions_from_report_lines(lines: Sequence[str]) -> List[Dict[str, Any]]:
    """Odwrotność ``predictions_to_report_lines`` (do zapisu po edycji operatora)."""
    out: List[Dict[str, Any]] = []
    for line in lines:
        rec = pc.parse_report_line(line)
        if rec is None:
            continue
        out.append({
            'x': int(rec['x']),
            'y': int(rec['y']),
            'color': _norm_color(rec['color']),
        })
    return out


def observation_from_snapshot_json(payload: Dict[str, Any]) -> SnapshotObservation:
    mod_b = payload.get('module_b') or {}
    preds = list(mod_b.get('live_predictions') or [])
    lines = list(mod_b.get('live_report_lines') or [])
    if not preds and lines:
        preds = predictions_from_report_lines(lines)
    panel_id = str(mod_b.get('panel_id') or payload.get('panel_id') or 'A')
    angle = int(mod_b.get('report_angle_deg') or payload.get('report_angle_deg') or 0)
    return SnapshotObservation(
        frame_id=str(payload.get('frame_id', '?')),
        reproj_b=float(payload.get('reproj_b', 999.0)),
        grid_overlap_ratio=float(payload.get('grid_overlap_ratio', 0.0)),
        predictions=preds,
        report_lines=lines,
        panel_id=panel_id,
        angle_deg=angle,
    )


def load_session_observations(session_dir: str) -> List[SnapshotObservation]:
    snap_dir = os.path.join(session_dir, 'snapshots')
    if not os.path.isdir(snap_dir):
        return []
    out: List[SnapshotObservation] = []
    for name in sorted(os.listdir(snap_dir)):
        if not name.endswith('_snapshot.json'):
            continue
        path = os.path.join(snap_dir, name)
        try:
            with open(path, encoding='utf-8') as fh:
                payload = json.load(fh)
            obs = observation_from_snapshot_json(payload)
            if obs.predictions or obs.report_lines:
                out.append(obs)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return out


def _snapshot_weight(
    obs: SnapshotObservation,
    *,
    use_reproj: bool,
    use_overlap: bool,
) -> float:
    w = 1.0
    if use_reproj:
        w *= 1.0 / (float(obs.reproj_b) + 1.0)
    if use_overlap:
        w *= 0.5 + 0.5 * float(np_clip01(obs.grid_overlap_ratio))
    return w


def np_clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _pick_best_per_cell(
    cards: Sequence[CompetitionCard],
) -> Tuple[List[CompetitionCard], List[CompetitionCard]]:
    """Regulamin: jedna kartka na komórkę — wygrywa kolor z największą liczbą głosów."""
    by_pos: Dict[Tuple[int, int], List[CompetitionCard]] = defaultdict(list)
    for card in cards:
        by_pos[(int(card.y), int(card.x))].append(card)
    kept: List[CompetitionCard] = []
    dropped: List[CompetitionCard] = []
    for group in by_pos.values():
        ordered = sorted(group, key=lambda c: (-c.votes, -c.weight, c.color))
        kept.append(ordered[0])
        dropped.extend(ordered[1:])
    return kept, dropped


def run_snapshot_cxy_competition(
    observations: Sequence[SnapshotObservation],
    *,
    min_votes: int = 2,
    min_support_ratio: float = 0.0,
    max_cards: int = 4,
    use_reproj_weight: bool = True,
    use_overlap_weight: bool = True,
    panel_id: Optional[str] = None,
    angle_deg: Optional[int] = None,
) -> SnapshotCxyCompetitionResult:
    """
    Głosowanie na (wiersz, kolumna, kolor).

    ``min_votes`` — ile migawek musi zgłosić tę samą kartę.
    ``min_support_ratio`` — dodatkowo ułamek migawek (0 = wyłączone).
    """
    obs_list = list(observations)
    n = len(obs_list)
    need = max(int(min_votes), 1)
    if min_support_ratio > 0 and n > 0:
        need = max(need, int(math.ceil(n * float(min_support_ratio))))

    if n == 0:
        pid = panel_id or 'A'
        ang = int(angle_deg or 0)
        return SnapshotCxyCompetitionResult(
            panel_id=pid,
            angle_deg=ang,
            n_snapshots=0,
            min_votes=need,
            accepted=[],
            rejected=[],
            predictions=[],
            report_lines=[],
        )

    pid = panel_id or obs_list[0].panel_id
    ang = int(angle_deg if angle_deg is not None else obs_list[0].angle_deg)

    vote_w: Dict[Tuple[int, int, str], float] = defaultdict(float)
    vote_n: Dict[Tuple[int, int, str], int] = defaultdict(int)
    vote_src: Dict[Tuple[int, int, str], List[str]] = defaultdict(list)

    for obs in obs_list:
        w = _snapshot_weight(
            obs, use_reproj=use_reproj_weight, use_overlap=use_overlap_weight,
        )
        seen_keys: set[Tuple[int, int, str]] = set()
        for p in obs.predictions:
            key = _norm_prediction(p)
            if key is None:
                continue
            y, x, color = key
            trip = (y, x, color)
            if trip in seen_keys:
                continue
            seen_keys.add(trip)
            vote_w[trip] += w
            vote_n[trip] += 1
            vote_src[trip].append(obs.frame_id)

    accepted: List[CompetitionCard] = []
    rejected: List[CompetitionCard] = []
    for (y, x, color), votes in vote_n.items():
        card = CompetitionCard(
            x=x,
            y=y,
            color=color,
            votes=int(votes),
            weight=float(vote_w[(y, x, color)]),
            support_ratio=float(votes) / float(n),
            sources=list(vote_src[(y, x, color)]),
        )
        if votes >= need:
            accepted.append(card)
        else:
            rejected.append(card)

    accepted.sort(key=lambda c: (-c.votes, -c.weight, c.y, c.x))
    rejected.sort(key=lambda c: (-c.votes, c.y, c.x))

    accepted, pos_losers = _pick_best_per_cell(accepted)
    rejected.extend(pos_losers)
    accepted.sort(key=lambda c: (-c.votes, -c.weight, c.y, c.x))

    cap = int(max_cards)
    if cap > 0 and len(accepted) > cap:
        overflow = accepted[cap:]
        accepted = accepted[:cap]
        rejected.extend(overflow)

    preds = [{'x': c.x, 'y': c.y, 'color': c.color} for c in accepted]
    lines = predictions_to_report_lines(pid, ang, preds)

    return SnapshotCxyCompetitionResult(
        panel_id=pid,
        angle_deg=ang,
        n_snapshots=n,
        min_votes=need,
        accepted=accepted,
        rejected=rejected,
        predictions=preds,
        report_lines=lines,
    )


def write_session_competition(
    session_dir: str,
    result: SnapshotCxyCompetitionResult,
) -> Dict[str, str]:
    os.makedirs(session_dir, exist_ok=True)
    json_path = os.path.join(session_dir, 'cxy_competition.json')
    txt_path = os.path.join(session_dir, 'cxy_competition_report.txt')
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(result.to_dict(), fh, ensure_ascii=False, indent=2)
    lines = [
        f'Konkurs CXY — {result.n_snapshots} migawek, min. głosów: {result.min_votes}',
        f'Panel {result.panel_id} ({result.angle_deg}°)',
        '',
        f'Zaakceptowano: {len(result.accepted)} kart',
    ]
    for c in result.accepted:
        lines.append(
            f'  • W{c.y} K{c.x} {c.color}  ({c.votes}/{result.n_snapshots} migawek, '
            f'w={c.weight:.2f})  [{", ".join(c.sources)}]',
        )
    if result.rejected:
        lines.append('')
        lines.append(f'Odrzucono (jednorazowe / słabe): {len(result.rejected)}')
        for c in result.rejected[:12]:
            lines.append(
                f'  × W{c.y} K{c.x} {c.color}  ({c.votes}/{result.n_snapshots})',
            )
    lines.append('')
    lines.extend(result.report_lines or ['(brak kart)'])
    with open(txt_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    return {'json': json_path, 'txt': txt_path}


def update_session_competition(
    session_dir: str,
    *,
    min_votes: int = 2,
    min_support_ratio: float = 0.0,
    max_cards: int = 4,
    use_reproj_weight: bool = True,
    use_overlap_weight: bool = True,
) -> Optional[SnapshotCxyCompetitionResult]:
    obs = load_session_observations(session_dir)
    if not obs:
        return None
    result = run_snapshot_cxy_competition(
        obs,
        min_votes=min_votes,
        min_support_ratio=min_support_ratio,
        max_cards=max_cards,
        use_reproj_weight=use_reproj_weight,
        use_overlap_weight=use_overlap_weight,
    )
    write_session_competition(session_dir, result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description='Konkurs CXY z migawek sesji live')
    ap.add_argument('--session-dir', required=True, help='katalog session_YYYYMMDD_…')
    ap.add_argument('--min-votes', type=int, default=2,
                    help='min. liczba migawek z tą samą kartą (dom. 2)')
    ap.add_argument('--min-ratio', type=float, default=0.0,
                    help='min. ułamek migawek (np. 0.35); 0 = tylko min-votes')
    ap.add_argument('--no-reproj-weight', action='store_true')
    ap.add_argument('--no-overlap-weight', action='store_true')
    args = ap.parse_args()
    session = args.session_dir
    if not os.path.isabs(session):
        session = os.path.abspath(session)
    result = update_session_competition(
        session,
        min_votes=int(args.min_votes),
        min_support_ratio=float(args.min_ratio),
        use_reproj_weight=not args.no_reproj_weight,
        use_overlap_weight=not args.no_overlap_weight,
    )
    if result is None:
        print('[cxy_competition] brak migawek z raportem')
        return
    paths = write_session_competition(session, result)
    print(f'[cxy_competition] zaakceptowano {len(result.accepted)} / '
          f'odrzucono {len(result.rejected)}')
    print(f'[cxy_competition] json={paths["json"]}')
    print(f'[cxy_competition] txt={paths["txt"]}')


if __name__ == '__main__':
    main()
