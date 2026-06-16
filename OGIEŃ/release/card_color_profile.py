"""Profil kolorów kartek — kalibracja pod zawody (HSV z próbek)."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import pipeline_competition as pc

ColorRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]
RangesByClass = Dict[int, List[ColorRange]]

_SCHEMA = 'droniada_card_colors_v1'
_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PATH = _ROOT / 'config' / 'card_colors.json'

# Domyślne zakresy (OpenCV HSV) — fallback bez kalibracji.
_DEFAULT_RANGES: RangesByClass = {
    0: [((0, 80, 80), (12, 255, 255)), ((168, 80, 80), (180, 255, 255))],
    1: [((35, 60, 60), (88, 255, 255))],
    2: [((95, 52, 52), (106, 255, 255))],
    3: [((20, 68, 65), (40, 255, 255))],
    4: [((98, 16, 26), (175, 255, 255))],
    5: [((8, 80, 75), (26, 255, 255))],
}

# Marginesy zakresów inRange (fallback); klasyfikacja głównie po centroidach próbek.
_H_MARGIN = 32
_S_MARGIN = 90
_V_MARGIN = 90
_TIGHT_H = 20
_TIGHT_S = 55
_TIGHT_V = 65

# Maks. odległość HSV od najbliższego centroidu kalibracji (znormalizowana).
CENTROID_MAX_DIST = 2.85

_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
_SUFFIX_RE = re.compile(r'(\d+)$')

_active_profile: Optional['CardColorProfile'] = None
_active_path: Optional[str] = None


def color_names() -> List[str]:
    return [pc.CLASS_TO_COLOR[i] for i in sorted(pc.CLASS_TO_COLOR)]


def _strip_numeric_suffix(stem: str) -> str:
    return _SUFFIX_RE.sub('', stem)


def _numeric_suffix(stem: str) -> int:
    match = _SUFFIX_RE.search(stem)
    return int(match.group(1)) if match else 0


def _normalize_stem(name: str) -> Optional[str]:
    from module_panel.competition_report import normalize_color_name

    stem = Path(name).stem.strip()
    candidates = [stem, _strip_numeric_suffix(stem)]
    for candidate in candidates:
        for part in candidate.replace('-', '_').split('_'):
            canon = normalize_color_name(part)
            if canon is not None:
                return canon
        canon = normalize_color_name(candidate)
        if canon is not None:
            return canon
    return None


def _clip_hsv(lo: Sequence[float], hi: Sequence[float]) -> ColorRange:
    return (
        (int(max(0, round(lo[0]))), int(max(0, round(lo[1]))), int(max(0, round(lo[2])))),
        (int(min(180, round(hi[0]))), int(min(255, round(hi[1]))), int(min(255, round(hi[2])))),
    )


def ranges_from_median_hsv(cls_id: int, h: float, s: float, v: float) -> List[ColorRange]:
    return ranges_from_medians(cls_id, [(h, s, v)])


def _hue_bounds(cls_id: int, hs: Sequence[float]) -> Tuple[float, float]:
    """Per-kolor granice H — czerwień nie zjada żółci/pomarańczy."""
    h_ctr = float(np.median(hs))
    cid = int(cls_id)
    if cid == 0 and h_ctr <= 18.0:
        return 0.0, min(15.0, h_ctr + 10.0)
    if cid == 5:
        return max(0.0, h_ctr - 16.0), min(35.0, h_ctr + 16.0)
    if cid == 3:
        return max(0.0, h_ctr - 24.0), min(48.0, h_ctr + 18.0)
    if cid == 1:
        return max(0.0, h_ctr - 28.0), min(115.0, h_ctr + 45.0)
    if cid == 2:
        return max(0.0, h_ctr - 22.0), min(125.0, h_ctr + 22.0)
    if cid == 4:
        return max(0.0, h_ctr - 30.0), min(180.0, h_ctr + 30.0)
    return max(0.0, min(hs) - _H_MARGIN), min(180.0, max(hs) + _H_MARGIN)


def hue_distance(h1: float, h2: float) -> float:
    d = abs(float(h1) - float(h2))
    return min(d, 180.0 - d)


def hsv_centroid_distance(
    h: float,
    s: float,
    v: float,
    centroid: Tuple[float, float, float],
) -> float:
    ch, cs, cv = centroid
    dh = hue_distance(h, ch) / 18.0
    ds = abs(s - cs) / 90.0
    dv = abs(v - cv) / 110.0
    return float(dh * dh + ds * ds + dv * dv * 0.55)


def prune_centroids_by_hue(
    centroids_by_cls: Dict[int, List[Tuple[float, float, float]]],
    *,
    max_hue_spread: float = 38.0,
) -> Dict[int, List[Tuple[float, float, float]]]:
    """Zachowaj *2 + klastry nagrań (łańcuch od najjaśniejszej próbki), odrzuć odstające outliery."""
    out: Dict[int, List[Tuple[float, float, float]]] = {}
    for cls_id, cents in centroids_by_cls.items():
        if not cents:
            continue
        ordered = sorted(cents, key=lambda m: (-float(m[2]), float(m[0])))
        kept: List[Tuple[float, float, float]] = [ordered[0]]
        for c in ordered[1:]:
            if any(hue_distance(float(c[0]), float(k[0])) <= float(max_hue_spread) for k in kept):
                if any(hsv_centroid_distance(c[0], c[1], c[2], k) < 0.22 for k in kept):
                    continue
                kept.append((float(c[0]), float(c[1]), float(c[2])))
        out[int(cls_id)] = kept
    return out


def tight_range_from_centroid(
    cls_id: int,
    h: float,
    s: float,
    v: float,
) -> ColorRange:
    """Wąski zakres wokół dopasowanego centroidu — tylko do liczenia color_frac."""
    cid = int(cls_id)
    h_m = 24.0 if cid in (1, 2, 4) else 18.0
    if cid == 3:
        h_m = 22.0
    if cid == 0 or h <= 22.0 or h >= 158.0:
        if h <= 22.0:
            return _clip_hsv(
                (0.0, max(28.0, s - _TIGHT_S), max(18.0, v - _TIGHT_V)),
                (min(18.0, h + h_m), min(255.0, s + _TIGHT_S), min(255.0, v + _TIGHT_V)),
            )
        if h >= 158.0:
            return _clip_hsv(
                (max(158.0, h - h_m), max(28.0, s - _TIGHT_S), max(18.0, v - _TIGHT_V)),
                (180.0, min(255.0, s + _TIGHT_S), min(255.0, v + _TIGHT_V)),
            )
    v_lo = max(16.0, v - (85.0 if cid == 3 else _TIGHT_V))
    return _clip_hsv(
        (max(0.0, h - h_m), max(28.0, s - _TIGHT_S), v_lo),
        (min(180.0, h + h_m), min(255.0, s + _TIGHT_S), min(255.0, v + _TIGHT_V)),
    )


def classify_hsv_centroid(
    h: float,
    s: float,
    v: float,
    centroids_by_cls: Dict[int, Sequence[Tuple[float, float, float]]],
) -> Tuple[Optional[int], float, Optional[Tuple[float, float, float]]]:
    """Najbliższy kolor po medianie HSV ze środka komórki."""
    best_cls: Optional[int] = None
    best_dist = float('inf')
    best_cent: Optional[Tuple[float, float, float]] = None
    for cls_id, cents in centroids_by_cls.items():
        for cent in cents:
            dist = hsv_centroid_distance(h, s, v, cent)
            if dist < best_dist:
                best_dist = dist
                best_cls = int(cls_id)
                best_cent = (float(cent[0]), float(cent[1]), float(cent[2]))
    if best_cls is None or best_dist > CENTROID_MAX_DIST:
        return None, best_dist, best_cent
    return best_cls, best_dist, best_cent


def ranges_from_medians(cls_id: int, medians: Sequence[Tuple[float, float, float]]) -> List[ColorRange]:
    """Szeroki zakres z jednej lub wielu próbek (np. CZERWONA2 + CZERWONA)."""
    if not medians:
        return []
    hs = [m[0] for m in medians]
    ss = [m[1] for m in medians]
    vs = [m[2] for m in medians]
    h_ctr = float(np.median(hs))
    s_lo = max(20.0, min(min(ss) - _S_MARGIN, min(ss) * 0.35))
    s_hi = min(255.0, max(ss) + _S_MARGIN)
    v_lo = max(18.0, min(min(vs) - _V_MARGIN, min(vs) * 0.30))
    v_hi = min(255.0, max(vs) + _V_MARGIN)
    if int(cls_id) == 0 or h_ctr <= 22.0 or h_ctr >= 158.0:
        if h_ctr <= 22.0:
            _, h_hi = _hue_bounds(0, hs)
            return [
                _clip_hsv((0.0, s_lo, v_lo), (h_hi, s_hi, v_hi)),
                _clip_hsv((168.0, s_lo, v_lo), (180.0, s_hi, v_hi)),
            ]
        if h_ctr >= 158.0:
            h_lo, _ = _hue_bounds(int(cls_id), hs)
            return [
                _clip_hsv((h_lo, s_lo, v_lo), (180.0, s_hi, v_hi)),
                _clip_hsv((0.0, s_lo, v_lo), (12.0, s_hi, v_hi)),
            ]
    h_lo, h_hi = _hue_bounds(int(cls_id), hs)
    return [_clip_hsv((h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))]


def sample_median_hsv(bgr: np.ndarray, *, center_frac: float = 0.55) -> Optional[Tuple[float, float, float]]:
    """Mediana HSV ze środka zdjęcia kartki (piksele nasycone)."""
    if bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
    mx = max(1, int(w * (1.0 - center_frac) * 0.5))
    my = max(1, int(h * (1.0 - center_frac) * 0.5))
    crop = bgr[my : h - my, mx : w - mx] if h > 2 * my and w > 2 * mx else bgr
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    for smin, vmin in ((35.0, 35.0), (25.0, 25.0), (15.0, 15.0)):
        sel = (sat >= smin) & (val >= vmin)
        if int(np.count_nonzero(sel)) >= 80:
            hues = hsv[:, :, 0][sel].astype(np.float32)
            return (
                float(np.median(hues)),
                float(np.median(sat[sel])),
                float(np.median(val[sel])),
            )
    return (
        float(np.median(hsv[:, :, 0])),
        float(np.median(sat)),
        float(np.median(val)),
    )


class CardColorProfile:
    def __init__(
        self,
        ranges_by_cls: RangesByClass,
        *,
        source: str = 'default',
        meta: Optional[Dict[str, Any]] = None,
        centroids_by_cls: Optional[Dict[int, List[Tuple[float, float, float]]]] = None,
    ) -> None:
        self.ranges_by_cls = {int(k): list(v) for k, v in ranges_by_cls.items()}
        self.source = str(source)
        self.meta = dict(meta or {})
        self.centroids_by_cls: Dict[int, List[Tuple[float, float, float]]] = {
            int(k): [tuple(map(float, c)) for c in v]
            for k, v in (centroids_by_cls or {}).items()
        }

    def has_centroids(self) -> bool:
        return bool(self.centroids_by_cls)

    def nearest_centroid(
        self,
        h: float,
        s: float,
        v: float,
    ) -> Tuple[Optional[int], float, Optional[Tuple[float, float, float]]]:
        return classify_hsv_centroid(h, s, v, self.centroids_by_cls)

    @classmethod
    def default(cls) -> 'CardColorProfile':
        return cls(_DEFAULT_RANGES, source='default')

    @classmethod
    def from_json(cls, path: str | Path) -> 'CardColorProfile':
        path = Path(path)
        with path.open('r', encoding='utf-8') as fh:
            data = json.load(fh)
        ranges: RangesByClass = {}
        colors = data.get('colors') or {}
        for name, spec in colors.items():
            cls_id = int(spec.get('cls_id', pc.COLOR_TO_CLASS.get(str(name).upper(), -1)))
            if cls_id < 0:
                continue
            raw = spec.get('ranges') or []
            parsed: List[ColorRange] = []
            for item in raw:
                lo, hi = item[0], item[1]
                parsed.append(
                    (
                        (int(lo[0]), int(lo[1]), int(lo[2])),
                        (int(hi[0]), int(hi[1]), int(hi[2])),
                    )
                )
            if parsed:
                ranges[cls_id] = parsed
        if not ranges:
            raise ValueError(f'Brak zakresów kolorów w {path}')
        centroids: Dict[int, List[Tuple[float, float, float]]] = {}
        for name, spec in colors.items():
            cls_id = int(spec.get('cls_id', pc.COLOR_TO_CLASS.get(str(name).upper(), -1)))
            if cls_id < 0:
                continue
            cents: List[Tuple[float, float, float]] = []
            for item in spec.get('calibration_samples') or []:
                cents.append((float(item['h']), float(item['s']), float(item['v'])))
            med = spec.get('median_hsv')
            if med and not cents:
                cents.append((float(med['h']), float(med['s']), float(med['v'])))
            if cents:
                centroids[cls_id] = cents
        centroids = prune_centroids_by_hue(centroids)
        return cls(
            ranges,
            source=str(path),
            meta={
                'schema': data.get('schema', _SCHEMA),
                'created': data.get('created'),
                'notes': data.get('notes'),
                'calibrated': bool(data.get('calibrated', True)),
                'source_files': data.get('source_files') or {},
            },
            centroids_by_cls=centroids,
        )

    def to_json_dict(
        self,
        *,
        source: str = '',
        notes: str = '',
        samples: Optional[Dict[str, Dict[str, float]]] = None,
        calibration_samples: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        colors: Dict[str, Any] = {}
        for cls_id in sorted(self.ranges_by_cls):
            name = pc.CLASS_TO_COLOR[int(cls_id)]
            entry: Dict[str, Any] = {
                'cls_id': int(cls_id),
                'ranges': [
                    [[int(lo[0]), int(lo[1]), int(lo[2])], [int(hi[0]), int(hi[1]), int(hi[2])]]
                    for lo, hi in self.ranges_by_cls[int(cls_id)]
                ],
            }
            if calibration_samples and name in calibration_samples:
                entry['calibration_samples'] = calibration_samples[name]
            if samples and name in samples:
                entry['median_hsv'] = samples[name]
            colors[name] = entry
        return {
            'schema': _SCHEMA,
            'created': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'source': source,
            'notes': notes,
            'calibrated': bool(self.meta.get('calibrated', True)),
            'source_files': self.meta.get('source_files') or {},
            'colors': colors,
        }

    def iter_ranges(self) -> List[Tuple[int, ColorRange]]:
        out: List[Tuple[int, ColorRange]] = []
        for cls_id in sorted(self.ranges_by_cls):
            for lo_hi in self.ranges_by_cls[int(cls_id)]:
                out.append((int(cls_id), lo_hi))
        return out

    def save(self, path: str | Path, **kwargs: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as fh:
            json.dump(self.to_json_dict(**kwargs), fh, indent=2, ensure_ascii=False)
            fh.write('\n')


def _collect_calibration_files(folder: Path) -> Dict[str, List[Path]]:
    by_color: Dict[str, List[Path]] = {name: [] for name in color_names()}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        canon = _normalize_stem(path.name)
        if canon is not None and canon in by_color:
            by_color[canon].append(path)
    return by_color


_VIDEO_DUAL_COLORS = frozenset({'ZOLTA'})


def _pick_calibration_paths(paths: List[Path]) -> List[Path]:
    """Zawody (*2); wariant bez cyfry tylko dla żółci (ciemniejsza kartka z filmu)."""
    if not paths:
        return []
    numbered = sorted(
        [p for p in paths if _numeric_suffix(p.stem) > 0],
        key=lambda p: _numeric_suffix(p.stem),
    )
    unnumbered = sorted(
        [p for p in paths if _numeric_suffix(p.stem) == 0],
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    out: List[Path] = []
    seen: set[str] = set()
    for p in numbered:
        key = p.name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    for p in unnumbered:
        canon = _normalize_stem(p.name)
        if numbered and canon not in _VIDEO_DUAL_COLORS:
            continue
        key = p.name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    if out:
        return out
    return [max(paths, key=lambda p: p.stat().st_size)]


def _primary_competition_median(medians: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Zakres inRange opieramy na najjaśniejszej próbce (zwykle *2 z zawodów)."""
    return max(medians, key=lambda m: float(m[2]))


def build_profile_from_folder(folder: str | Path) -> Tuple['CardColorProfile', Dict[str, Dict[str, float]]]:
    """Zdjęcia kartek: CZERWONA2.png, ZIELONA2.jpg, … (cyfra na końcu = zawody)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Brak katalogu: {folder}')
    by_color = _collect_calibration_files(folder)
    missing = [n for n in color_names() if not by_color[n]]
    if missing:
        raise ValueError(
            f'Brakuje zdjęć dla: {", ".join(missing)}. '
            f'Wrzuć pliki np. CZERWONA2.jpg … POMARANCZOWA2.jpg do {folder}'
        )
    ranges: RangesByClass = {}
    samples: Dict[str, Dict[str, float]] = {}
    calib_samples: Dict[str, List[Dict[str, Any]]] = {}
    centroids_by_cls: Dict[int, List[Tuple[float, float, float]]] = {}
    source_files: Dict[str, str] = {}
    for name in color_names():
        cls_id = int(pc.COLOR_TO_CLASS[name])
        use_paths = _pick_calibration_paths(by_color[name])
        medians: List[Tuple[float, float, float]] = []
        entries: List[Dict[str, Any]] = []
        for path in use_paths:
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f'Nie wczytano obrazu: {path}')
            med = sample_median_hsv(bgr)
            if med is None:
                raise ValueError(f'Za mało koloru na zdjęciu: {path}')
            medians.append(med)
            h, s, v = med
            entries.append({
                'file': path.name,
                'h': round(h, 1),
                's': round(s, 1),
                'v': round(v, 1),
            })
        primary = _primary_competition_median(medians)
        ranges[cls_id] = ranges_from_medians(cls_id, [primary])
        centroids_by_cls[cls_id] = list(medians)
        h, s, v = primary
        samples[name] = {'h': round(h, 1), 's': round(s, 1), 'v': round(v, 1)}
        calib_samples[name] = entries
        numbered = [p for p in use_paths if _numeric_suffix(p.stem) > 0]
        source_files[name] = (numbered[-1] if numbered else use_paths[-1]).name
    profile = CardColorProfile(
        ranges,
        source=str(folder),
        meta={'calibrated': True, 'source_files': source_files},
        centroids_by_cls=centroids_by_cls,
    )
    profile.meta['calibration_samples'] = calib_samples
    return profile, samples


def profile_val_floor(cls_id: int) -> Optional[float]:
    """Minimalny V — z najciemniejszego centroidu kalibracji (×0.52), nie z szerokiego inRange."""
    if not is_calibrated_profile():
        return None
    prof = load_active_profile()
    cents = prof.centroids_by_cls.get(int(cls_id), [])
    if not cents:
        ranges = prof.ranges_by_cls.get(int(cls_id))
        if not ranges:
            return None
        lo, _hi = ranges[0]
        return max(35.0, float(lo[2]) * 0.45)
    min_v = min(float(c[2]) for c in cents)
    # Ciemna pianka na filmie — dolna granica z najciemniejszego centroidu (min. 32).
    return max(32.0, min_v * 0.48)


def is_calibrated_profile() -> bool:
    prof = load_active_profile()
    if prof.source == 'default':
        return False
    return bool(prof.meta.get('calibrated', True))


def resolve_profile_path() -> Optional[str]:
    env = os.environ.get('DRONIADA_CARD_COLORS', '').strip()
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = _ROOT / env
        return str(p) if p.is_file() else env
    if _DEFAULT_PATH.is_file():
        return str(_DEFAULT_PATH)
    return None


def load_active_profile(*, force: bool = False) -> CardColorProfile:
    global _active_profile, _active_path
    path = resolve_profile_path()
    if not force and _active_profile is not None and path == _active_path:
        return _active_profile
    _active_path = path
    if path and os.path.isfile(path):
        _active_profile = CardColorProfile.from_json(path)
    else:
        _active_profile = CardColorProfile.default()
    return _active_profile


def active_color_ranges() -> List[Tuple[int, ColorRange]]:
    return load_active_profile().iter_ranges()


def active_profile_label() -> str:
    prof = load_active_profile()
    if prof.source == 'default':
        return 'domyślny (bez kalibracji)'
    notes = prof.meta.get('notes') or ''
    base = Path(prof.source).name if prof.source else 'profil'
    return f'{base}{(" — " + notes) if notes else ""}'
