"""
Podgląd HTTP na osobnym porcie (Jetson / headless).

Serwuje: live JPEG, parametry, migawki, edycję raportu z konkursu CXY.
Komunikacja z ``run_live_panel`` przez pliki w katalogu sesji:

- ``live_preview.jpg`` — ostatni kadr dashboardu
- ``web_state.json`` — parametry + migawki + raport
- ``web_commands.json`` — żądania z przeglądarki (zapis / wyślij)
- ``web_report_draft.txt`` — szkic raportu edytowany w UI
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2

_REPORT_TS_RE = re.compile(r'^\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\](.*)$')
_NOW_PLACEHOLDER_RE = re.compile(r'\{\{NOW[^}]*\}\}')
_NOW_OFFSET_RE = re.compile(r'\+(\d+)(ms|s)')


def _format_report_ts(dt: datetime) -> str:
    return dt.strftime('[%H:%M:%S.') + f'{dt.microsecond // 1000:03d}]'


def _offset_ms_from_now_token(token: str) -> int:
    total_ms = 0
    for m in _NOW_OFFSET_RE.finditer(str(token)):
        val = int(m.group(1))
        total_ms += val if m.group(2) == 'ms' else val * 1000
    return total_ms


def _resolve_now_placeholders(line: str, when: datetime) -> str:
    def repl(match: re.Match[str]) -> str:
        offset_ms = _offset_ms_from_now_token(match.group(0))
        return _format_report_ts(when + timedelta(milliseconds=offset_ms))

    return _NOW_PLACEHOLDER_RE.sub(repl, str(line))


def resolve_report_lines_at_click(lines: List[str], when: Optional[datetime] = None) -> List[str]:
    """Podstaw czas kliknięcia: {{NOW}} / {{NOW+1s}} albo legacy [HH:MM:SS.mmm]."""
    when = when or datetime.now()
    if not lines:
        return []
    if any('{{NOW' in str(ln) for ln in lines):
        return [_resolve_now_placeholders(ln, when) for ln in lines]
    return stamp_report_lines_at_now(lines, when)


def stamp_report_lines_at_now(lines: List[str], when: Optional[datetime] = None) -> List[str]:
    """Zamień prefiksy [HH:MM:SS.mmm] na czas kliknięcia (+ zachowaj odstępy między liniami)."""
    when = when or datetime.now()
    if not lines:
        return []
    base: Optional[timedelta] = None
    offsets: List[timedelta] = []
    for ln in lines:
        m = _REPORT_TS_RE.match(str(ln).strip())
        if not m:
            offsets.append(offsets[-1] if offsets else timedelta(0))
            continue
        h, mi, s, ms = (int(m.group(i)) for i in range(1, 5))
        td = timedelta(hours=h, minutes=mi, seconds=s, milliseconds=ms)
        if base is None:
            base = td
            offsets.append(timedelta(0))
        else:
            offsets.append(td - base)
    out: List[str] = []
    for ln, off in zip(lines, offsets):
        m = _REPORT_TS_RE.match(str(ln).strip())
        if m:
            ts = when + off
            out.append(f'{_format_report_ts(ts)}{m.group(5)}')
        else:
            out.append(ln)
    return out


def _overlay_corners_list(corners_px: Any) -> Optional[List[List[float]]]:
    if corners_px is None:
        return None
    try:
        import numpy as np

        arr = np.asarray(corners_px, dtype=np.float64)
        if arr.shape != (4, 2):
            return None
        return [[float(arr[i, 0]), float(arr[i, 1])] for i in range(4)]
    except (TypeError, ValueError):
        return None


_HTML_VIEW_PAGE = r"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Droniada — podgląd live</title>
<style>
:root { --bg:#0a0c10; --card:#1a1e28; --text:#e8eaef; --muted:#9aa3b5; --acc:#3d9a6a; --warn:#c07030; --bor:#2c3344; }
* { box-sizing: border-box; }
body { margin:0; font-family: system-ui, sans-serif; background:var(--bg); color:var(--text); min-height:100vh; display:flex; flex-direction:column; }
header { padding:10px 16px; border-bottom:1px solid var(--bor); display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:space-between; }
header h1 { margin:0; font-size:1.1rem; }
#status { color:var(--muted); font-size:0.88rem; }
main { flex:1; padding:8px; display:flex; flex-direction:column; min-height:0; position:relative; }
#liveStage { flex:1; min-height:60vh; position:relative; border-radius:6px; overflow:hidden; background:#000; }
#liveImg { width:100%; height:100%; min-height:60vh; object-fit:contain; display:block; }
#reportStack {
  position:absolute; left:10px; top:10px; bottom:10px; z-index:2;
  width:min(34vw, 400px); max-width:calc(100% - 20px);
  display:flex; flex-direction:column; gap:8px; overflow:auto;
  pointer-events:none;
}
#reportStack.hidden { display:none; }
.report-card {
  flex:0 0 auto; max-height:32%; min-height:0; overflow:auto;
  display:flex; flex-direction:column;
  padding:8px 12px;
  background:rgba(10,14,22,0.88); backdrop-filter:blur(4px);
  border-radius:8px; border:1px solid rgba(61,154,106,0.55);
  box-shadow:0 4px 18px rgba(0,0,0,0.4);
}
.report-card h2 { margin:0 0 6px; font-size:0.92rem; color:#7ee0a8; letter-spacing:0.05em; }
.report-card-a { border-color:rgba(74,122,176,0.7); }
.report-card-b { border-color:rgba(106,154,74,0.7); }
.report-card-c { border-color:rgba(176,128,64,0.7); }
.report-lines {
  font-family: ui-monospace, monospace; font-size:clamp(0.68rem,1vw,0.85rem);
  line-height:1.4; white-space:pre-wrap; color:var(--text);
}
.hint { font-size:0.8rem; color:var(--muted); padding:8px 16px; }
</style>
</head>
<body>
<header>
  <h1>Droniada — podgląd live</h1>
  <span id="status">Ładowanie…</span>
</header>
<main>
  <div id="liveStage">
    <img id="liveImg" alt="live dashboard" src="/stream.mjpg"/>
    <div id="reportStack" class="hidden"></div>
  </div>
</main>
<p class="hint">Stream: surowy MJPEG z kamery (GStreamer). Overlay YOLO + panel: <a href="/live.jpg" target="_blank">/live.jpg</a>. Operator <strong>8089</strong>: 1→A, 2→B, 3→C.</p>
<script>
let state = {};
let _mjpegOk = true;
function ts() { return Date.now(); }
function pollLiveJpg() {
  const img = document.getElementById('liveImg');
  if (!img || _mjpegOk) return;
  img.src = '/live.jpg?' + ts();
}
document.getElementById('liveImg').onerror = function() {
  if (_mjpegOk) {
    _mjpegOk = false;
    const st = document.getElementById('status');
    if (st) st.textContent = 'MJPEG niedostępny — podgląd /live.jpg';
    pollLiveJpg();
    setInterval(pollLiveJpg, 400);
  }
};
function applyBroadcast(bc) {
  const stack = document.getElementById('reportStack');
  const st = document.getElementById('status');
  if (!stack) return;
  const panels = (bc && bc.panels) || [];
  stack.innerHTML = '';
  if (!panels.length) {
    stack.classList.add('hidden');
    return;
  }
  stack.classList.remove('hidden');
  panels.forEach((p) => {
    const pid = String(p.panel_id || '?').toUpperCase();
    const card = document.createElement('div');
    card.className = 'report-card report-card-' + pid.toLowerCase();
    const h = document.createElement('h2');
    h.textContent = 'Panel ' + pid;
    const lines = document.createElement('div');
    lines.className = 'report-lines';
    const txt = (p.lines || []).join('\n');
    lines.textContent = txt || '(brak linii raportu)';
    card.appendChild(h);
    card.appendChild(lines);
    stack.appendChild(card);
  });
  const ids = panels.map((p) => p.panel_id).join(', ');
  st.textContent = 'Raporty na podglądzie: ' + ids + (bc.updated_at ? ' · ' + bc.updated_at : '');
}
async function fetchBroadcast() {
  try {
    const r = await fetch('/api/broadcast?' + ts());
    applyBroadcast(await r.json());
  } catch (e) { /* ignore */ }
}
async function fetchState() {
  const r = await fetch('/api/state?' + ts());
  const next = await r.json();
  const fid = next.frame_id || '—';
  const upd = next.updated_at || '';
  const st = document.getElementById('status');
  const stack = document.getElementById('reportStack');
  if (stack && stack.classList.contains('hidden')) {
    st.textContent = fid + ' · ' + upd;
  }
  state = next;
}
setInterval(fetchBroadcast, 400);
setInterval(fetchState, 800);
fetchBroadcast();
fetchState();
</script>
</body>
</html>
"""

_HTML_CONTROL_PAGE = r"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Droniada — sterowanie</title>
<style>
:root { --bg:#12141a; --card:#1e222c; --text:#e8eaef; --muted:#9aa3b5; --acc:#3d9a6a; --warn:#c07030; --bor:#2c3344; }
* { box-sizing: border-box; }
body { margin:0; font-family: system-ui, sans-serif; background:var(--bg); color:var(--text); }
header { padding:12px 18px; border-bottom:1px solid var(--bor); display:flex; flex-wrap:wrap; gap:12px; align-items:center; }
header h1 { margin:0; font-size:1.15rem; }
#status { color:var(--muted); font-size:0.9rem; }
main { display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap:14px; padding:14px; align-items:start; }
.card { background:var(--card); border:1px solid var(--bor); border-radius:8px; padding:12px; }
.card h2 { margin:0 0 10px; font-size:0.95rem; color:#b8c0d4; }
.params { font-size:0.82rem; line-height:1.45; white-space:pre-wrap; font-family: ui-monospace, monospace; max-height:320px; overflow-y:auto; }
.snaps { display:flex; flex-direction:column; gap:10px; max-height:min(50vh,520px); overflow-y:auto; }
.snaps a { display:block; width:100%; text-decoration:none; color:var(--text); font-size:0.8rem; line-height:1.35; }
.snaps img { width:100%; border-radius:6px; border:1px solid var(--bor); background:#000; min-height:80px; object-fit:contain; }
textarea { width:100%; min-height:180px; font-family: ui-monospace, monospace; font-size:0.8rem;
  line-height:1.45; white-space:pre-wrap;
  background:#0d1016; color:var(--text); border:1px solid var(--bor); border-radius:6px; padding:8px; }
.row { display:flex; gap:8px; margin-top:8px; flex-wrap:wrap; }
button { cursor:pointer; border:none; border-radius:6px; padding:8px 14px; font-weight:600; }
.btn-save { background:#3a4a6a; color:#fff; }
.btn-send { background:var(--acc); color:#fff; }
.btn-preset { background:#2a3348; color:#c8d0e0; font-size:0.78rem; padding:6px 10px; }
.mission { font-size:0.85rem; color:var(--muted); margin-bottom:8px; }
.report-hint { font-size:0.8rem; color:#8ec4a8; margin-bottom:6px; min-height:1.1em; }
.report-errors { font-size:0.78rem; color:#f0a8a8; margin-top:6px; white-space:pre-wrap; line-height:1.35; }
.btn-send:disabled { opacity:0.45; cursor:not-allowed; }
.tune-row { margin:10px 0 4px; font-size:0.82rem; }
.tune-row label { display:flex; justify-content:space-between; color:#b8c0d4; margin-bottom:4px; }
.tune-row input[type=range] { width:100%; accent-color: var(--acc); }
.tune-hint { font-size:0.75rem; color:var(--muted); margin-top:8px; line-height:1.35; }
.intro { font-size:0.82rem; color:var(--muted); line-height:1.45; margin:0 0 10px; }
.param-block { margin-bottom:12px; padding-bottom:10px; border-bottom:1px solid var(--bor); }
.param-block:last-child { border-bottom:none; margin-bottom:0; }
.param-block h3 { margin:0 0 4px; font-size:0.88rem; color:#b8c0d4; }
.param-desc { font-size:0.74rem; color:var(--muted); margin:0 0 8px; line-height:1.4; }
.param-row { display:grid; grid-template-columns:minmax(9em,34%) 1fr; gap:6px 10px; font-size:0.8rem; margin:3px 0; align-items:baseline; }
.param-row dt { color:var(--muted); margin:0; }
.param-row dd { margin:0; font-family:ui-monospace,monospace; color:var(--text); }
.val-ok { color:#6ecf9a; }
.val-bad { color:#e09090; }
.analysis-lamp {
  display:block; width:100%; max-width:360px; margin:8px 0 4px;
  padding:18px 24px; font-size:1.05rem; font-weight:700; border-radius:10px;
  border:2px solid transparent; cursor:default; text-align:center;
  transition:background 0.25s, box-shadow 0.25s, color 0.25s;
}
.analysis-lamp.on {
  background:#143d28; color:#7ee0a8; border-color:#3d9a6a;
  box-shadow:0 0 22px rgba(61,154,106,0.55);
}
.analysis-lamp.off {
  background:#3d1818; color:#f0a0a0; border-color:#b05050;
  box-shadow:0 0 22px rgba(176,80,80,0.45);
}
.cam-wrap {
  background:#000; border-radius:6px; overflow:hidden;
  max-height:min(72vh, 920px); border:1px solid var(--bor);
}
.cam-wrap img { width:100%; max-height:min(72vh, 920px); object-fit:contain; display:block; }
details.raw-params { margin-top:10px; font-size:0.78rem; color:var(--muted); }
details.raw-params pre { margin:8px 0 0; padding:8px; background:#0d1016; border-radius:6px; font-size:0.75rem; white-space:pre-wrap; }
</style>
</head>
<body>
<header>
  <h1>Droniada — sterowanie</h1>
  <span id="status">Ładowanie…</span>
</header>
<main>
  <div class="card" style="grid-column:1/-1">
    <h2>Podgląd live (dashboard)</h2>
    <div class="cam-wrap">
      <img id="controlCam" alt="live dashboard" src="/stream.mjpg"/>
    </div>
  </div>
  <div class="card" style="grid-column:1/-1">
    <h2>Analiza wizji</h2>
    <p class="param-desc">Status skanu YOLO (w tle).</p>
    <button type="button" id="analysisLamp" class="analysis-lamp off" disabled>Analiza — …</button>
    <p id="analysisLampHint" class="tune-hint">Ładowanie statusu…</p>
  </div>
  <div class="card" id="snapshotPanel">
    <h2>Migawki — kiedy zapisać klatkę</h2>
    <p class="param-desc">Migawka to zrzut do galerii, gdy panel jest dobrze ustawiony. System sprawdza zgodność <em>trzech siatek</em>
    (szara na kamerze, CLAHE, OpenCV), stabilność przez kilka klatek oraz reproj i pokrycie warpu.</p>
    <div class="tune-row">
      <label>Min. zgodność 3 siatek <span id="vSnapOverlap">75</span>%</label>
      <input id="snapOverlap" type="range" min="40" max="98" step="1" value="75"/>
    </div>
    <p class="tune-hint">Im wyżej, tym trudniej o migawkę, ale pewniejsze dopasowanie siatki 10×10.</p>
    <div class="tune-row">
      <label>Klatek z rzędu <span id="vSnapStable">2</span></label>
      <input id="snapStable" type="range" min="1" max="5" step="1" value="2"/>
    </div>
    <p class="tune-hint">Ile kolejnych klatek musi spełniać próg, zanim zapisze migawkę (filtr drgań).</p>
    <div class="tune-row">
      <label>Max reproj B <span id="vSnapReproj">15</span> px</label>
      <input id="snapReproj" type="range" min="5" max="40" step="1" value="15"/>
    </div>
    <p class="tune-hint">Maks. błąd reprojekcji rogów po warp (px). Niżej = ostrzej; typowo 12–18 px na nagraniu.</p>
    <div class="row">
      <button class="btn-save" onclick="applyRuntime()">Zastosuj progi migawek</button>
    </div>
  </div>
  <div class="card" id="tunePanel">
    <h2>Żółta ramka (tracker rogów)</h2>
    <p class="param-desc">Stabilizuje rogi panelu między klatkami YOLO. Wpływa na płynność żółtej ramki i siatki na podglądzie.</p>
    <div id="tuneHint" class="tune-hint">Zmiany bez restartu — widać po ~1 s.</div>
    <div class="tune-row">
      <label>Szybkość reakcji (alpha) <span id="vAlpha">0.38</span></label>
      <input id="tAlpha" type="range" min="0.05" max="0.80" step="0.01" value="0.38"/>
    </div>
    <p class="tune-hint">Wyższe alpha = szybsza reakcja, więcej szumu. Niższe = spokojniejsza ramka.</p>
    <div class="tune-row">
      <label>Hold po zgubieniu (klatki) <span id="vHold">20</span></label>
      <input id="tHold" type="range" min="0" max="40" step="1" value="20"/>
    </div>
    <p class="tune-hint">Ile klatek trzymać ostatnie dobre rogi, gdy YOLO na chwilę zgubi panel.</p>
    <div class="tune-row">
      <label>Analiza YOLO co (ms) <span id="vInterval">300</span></label>
      <input id="tInterval" type="range" min="120" max="600" step="10" value="300"/>
    </div>
    <p class="tune-hint">Co ile ms uruchamiać detekcję rogów. Większa wartość = mniejsze obciążenie CPU.</p>
    <div class="tune-row">
      <label>Dobry reproj (px) <span id="vGood">28</span></label>
      <input id="tGood" type="range" min="10" max="45" step="1" value="28"/>
    </div>
    <p class="tune-hint">Próg uznania rogów za „dobre” w trackerze (nie to samo co reproj B modułu skanowania).</p>
    <div class="row">
      <button class="btn-save" onclick="applyTracker()">Zastosuj tracker</button>
    </div>
    <div class="row" style="margin-top:6px">
      <button class="btn-preset" onclick="loadPreset('camera')">Kamera</button>
      <button class="btn-preset" onclick="loadPreset('fast')">Szybka</button>
      <button class="btn-preset" onclick="loadPreset('stable')">Stabilna</button>
      <button class="btn-preset" onclick="loadPreset('video')">Nagranie</button>
    </div>
  </div>
  <div class="card" id="missionPanel">
    <h2>Misja drona (WebSocket)</h2>
    <p class="param-desc">Automatyczne zatrzymania i prędkości między panelami A/B/C. Wymaga połączenia z dronem (speed 0–1).</p>
    <div id="flightStatus" class="tune-hint">speed=— · faza=—</div>
    <div class="tune-row">
      <label>Strefa min (m) <span id="vDistMin">5</span></label>
      <input id="mDistMin" type="range" min="1" max="30" step="0.5" value="5"/>
    </div>
    <p class="tune-hint">Od tej odległości moduł A zaczyna „widzieć” panel i zwalniać podjazd.</p>
    <div class="tune-row">
      <label>Strefa max (m) <span id="vDistMax">15</span></label>
      <input id="mDistMax" type="range" min="3" max="50" step="0.5" value="15"/>
    </div>
    <div class="tune-row">
      <label>Zatrzymanie 1 — daleko (m) <span id="vDistHold1">11</span></label>
      <input id="mDistHold1" type="range" min="2" max="40" step="0.5" value="11"/>
    </div>
    <div class="tune-row">
      <label>Zatrzymanie 2 — środek (m) <span id="vDistHold2">9</span></label>
      <input id="mDistHold2" type="range" min="2" max="35" step="0.5" value="9"/>
    </div>
    <div class="tune-row">
      <label>Zatrzymanie 3 — blisko (m) <span id="vDistHold3">7</span></label>
      <input id="mDistHold3" type="range" min="1" max="30" step="0.5" value="7"/>
    </div>
    <p class="tune-hint">Kolejne postoje przy panelu (np. 11→9→7 m). Brak migawek → creep do następnego tieru.</p>
    <div class="tune-row">
      <label>Stabilizacja + migawki (s) <span id="vHoldS">15</span></label>
      <input id="mHoldS" type="range" min="3" max="180" step="1" value="15"/>
    </div>
    <p class="tune-hint">Czas stania przy panelu na zbieranie migawek przed decyzją o raporcie.</p>
    <div class="tune-row">
      <label>Podjazd creep (speed) <span id="vCreep">0.90</span></label>
      <input id="mCreep" type="range" min="50" max="99" step="1" value="90"/>
    </div>
    <div class="tune-row">
      <label>Czas creep (s) <span id="vCreepS">2</span></label>
      <input id="mCreepS" type="range" min="0.5" max="30" step="0.5" value="2"/>
    </div>
    <div class="tune-row">
      <label>Prędkość między panelami <span id="vCruise">0.35</span></label>
      <input id="mCruise" type="range" min="5" max="95" step="1" value="35"/>
    </div>
    <div class="tune-row">
      <label>Pauza po Wyślij (s) <span id="vPauseS">5</span></label>
      <input id="mPauseS" type="range" min="0" max="120" step="1" value="5"/>
    </div>
    <div class="tune-row">
      <label>Migawek / panel <span id="vSnapN">5</span></label>
      <input id="mSnapN" type="range" min="1" max="20" step="1" value="5"/>
    </div>
    <div class="row">
      <button class="btn-save" onclick="applyMission()">Zastosuj misję</button>
    </div>
  </div>
  <div class="card" style="grid-column:1/-1">
    <h2>Parametry live — co widzi system</h2>
    <p class="param-desc">Wartości z bieżącej klatki wideo. <strong>Reliable (moduł B)</strong> to główny sygnał, czy panel jest w kadrze i dobrze dopasowany.</p>
    <div id="paramsStructured"></div>
    <details class="raw-params">
      <summary>Tekst surowy (debug)</summary>
      <pre id="params">—</pre>
    </details>
  </div>
  <div class="card">
    <h2>Migawki</h2>
    <div id="snaps" class="snaps"></div>
  </div>
  <div class="card" style="grid-column:1/-1">
    <h2>Misja</h2>
    <div id="mission" class="mission"></div>
    <div id="reportHint" class="report-hint"></div>
  </div>
</main>
<script>
let state = {};
let lastSnapKey = '';
const PRESETS = {
  camera: { smooth_alpha: 0.38, hold_frames: 20, interval_ms: 300, tracker_good_reproj: 28 },
  fast:   { smooth_alpha: 0.50, hold_frames: 24, interval_ms: 250, tracker_good_reproj: 24 },
  stable: { smooth_alpha: 0.22, hold_frames: 28, interval_ms: 350, tracker_good_reproj: 32 },
  video:  { smooth_alpha: 0.28, hold_frames: 14, interval_ms: 350, tracker_good_reproj: 28 },
};
function tuneEditing() { return document.getElementById('tunePanel')?.dataset.editing === '1'; }
function missionEditing() { return document.getElementById('missionPanel')?.dataset.editing === '1'; }
function runtimeEditing() { return document.getElementById('snapshotPanel')?.dataset.editing === '1'; }
function setRuntimeUi(rt) {
  if (!rt || runtimeEditing()) return;
  const pct = Math.round(100 * Number(rt.snapshot_min_grid_overlap || 0.75));
  document.getElementById('snapOverlap').value = pct;
  document.getElementById('vSnapOverlap').textContent = pct;
  const st = Number(rt.snapshot_min_stable || 2);
  document.getElementById('snapStable').value = st;
  document.getElementById('vSnapStable').textContent = st;
  const rp = Number(rt.snapshot_max_reproj || 15);
  document.getElementById('snapReproj').value = rp;
  document.getElementById('vSnapReproj').textContent = rp;
}
function bindRuntimeSliders() {
  [['snapOverlap','vSnapOverlap',0],['snapStable','vSnapStable',0],['snapReproj','vSnapReproj',0]].forEach(([id,vid]) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', () => {
      document.getElementById('snapshotPanel').dataset.editing = '1';
      document.getElementById(vid).textContent = el.value;
    });
  });
}
async function applyRuntime() {
  const body = {
    snapshot_min_grid_overlap: Number(document.getElementById('snapOverlap').value) / 100,
    snapshot_min_stable: Number(document.getElementById('snapStable').value),
    snapshot_max_reproj: Number(document.getElementById('snapReproj').value),
  };
  const r = await fetch('/api/runtime', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)});
  const j = await r.json();
  document.getElementById('snapshotPanel').dataset.editing = '';
  if (j.runtime) setRuntimeUi(j.runtime);
  await fetchState();
}
function setTrackerUi(t) {
  if (!t || tuneEditing()) return;
  const set = (id, v, dec) => {
    const el = document.getElementById(id);
    if (!el || v === undefined) return;
    el.value = v;
    const lab = id.replace('t', 'v');
    const ve = document.getElementById(lab);
    if (ve) ve.textContent = dec ? Number(v).toFixed(dec) : String(v);
  };
  set('tAlpha', t.smooth_alpha, 2);
  set('tHold', t.hold_frames, 0);
  set('tInterval', t.interval_ms, 0);
  set('tGood', t.tracker_good_reproj, 0);
}
function loadPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  document.getElementById('tunePanel').dataset.editing = '1';
  setTrackerUi(p);
  applyTracker(false);
}
async function applyTracker(showAlert) {
  const body = {
    smooth_alpha: Number(document.getElementById('tAlpha').value),
    hold_frames: Number(document.getElementById('tHold').value),
    interval_ms: Number(document.getElementById('tInterval').value),
    tracker_good_reproj: Number(document.getElementById('tGood').value),
  };
  const r = await fetch('/api/tracker', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)});
  const j = await r.json();
  if (showAlert !== false) {
    document.getElementById('tuneHint').textContent = j.message || 'Zastosowano.';
  }
  document.getElementById('tunePanel').dataset.editing = '';
  await fetchState();
}
function ts() { return Date.now(); }
const PRESENCE_REASONS = {
  ok: 'panel w kadrze',
  reliable_b: 'moduł B reliable — panel pewnie w kadrze',
  ok_reproj_b: 'niski reproj modułu B',
  no_corners: 'brak rogów YOLO',
  tracker_hold: 'tracker trzyma stare rogi (panel mógł zniknąć)',
  high_reproj: 'zbyt wysoki reproj rogów YOLO',
  low_yolo_conf: 'niska pewność detekcji YOLO',
  quad_too_small: 'quad za mały w kadrze',
};
function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function valClass(ok) { return ok ? 'val-ok' : 'val-bad'; }
function pct(v) { return v == null ? '—' : Math.round(100 * Number(v)) + '%'; }
function renderStructuredParams(next) {
  const el = document.getElementById('paramsStructured');
  if (!el) return;
  const p = next.params || {};
  const pp = p.panel_presence || {};
  const ma = p.module_a || {};
  const mb = p.module_b || {};
  const latch = p.latch || {};
  const present = !!pp.present;
  const reason = PRESENCE_REASONS[pp.reason] || pp.reason || '';
  let html = '';
  html += '<div class="param-block"><h3>Panel w kadrze</h3>';
  html += '<p class="param-desc">Czy konkursowy panel jest widoczny i śledzony. Najpewniejszy sygnał: <em>Reliable</em> w module B.</p>';
  html += '<dl class="param-row"><dt>Obecny</dt><dd class="' + valClass(present) + '">' + (present ? 'TAK' : 'NIE') + '</dd>';
  if (!present && reason) html += '<dt>Powód</dt><dd>' + esc(reason) + '</dd>';
  html += '</dl></div>';
  html += '<div class="param-block"><h3>Moduł A — podejście drona</h3>';
  html += '<p class="param-desc">Odległość i kąt stojaka z PnP (zielony trapez). Służy do hamowania i ustawienia drona przed skanem.</p>';
  if (ma.ok) {
    html += '<dl class="param-row">';
    html += '<dt>Status</dt><dd class="val-ok">OK</dd>';
    html += '<dt>Odległość</dt><dd>' + Number(ma.distance_m).toFixed(2) + ' m</dd>';
    html += '<dt>Stojak</dt><dd>' + esc(ma.stand_label) + ' (' + ma.report_angle_deg + '°)</dd>';
    html += '<dt>Reproj A</dt><dd>' + Number(ma.reproj_mean_px).toFixed(1) + ' px</dd>';
    html += '</dl>';
  } else if (Object.keys(ma).length) {
    html += '<dl class="param-row"><dt>Status</dt><dd class="val-bad">FAIL</dd>';
    html += '<dt>Powód</dt><dd>' + esc(ma.reason || '?') + '</dd></dl>';
  } else {
    html += '<p class="param-desc">Wyłączony w tej sesji.</p>';
  }
  html += '</div>';
  html += '<div class="param-block"><h3>Moduł B — skan panelu</h3>';
  html += '<p class="param-desc">Warp panelu, siatka 10×10 i dopasowanie homografii. <em>Reliable=TAK</em> oznacza, że siatka i reproj są w normie.</p>';
  html += '<dl class="param-row">';
  html += '<dt>Reliable</dt><dd class="' + valClass(mb.reliable) + '">' + (mb.reliable ? 'TAK' : 'NIE') + '</dd>';
  html += '<dt>Reproj B</dt><dd>' + Number(mb.reproj_mean_px || 0).toFixed(2) + ' px</dd>';
  html += '<dt>Inliers</dt><dd>' + (mb.homography_inliers ?? 0) + '</dd>';
  html += '<dt>3 siatki</dt><dd>' + pct(mb.grid_overlap_ratio) + '</dd>';
  if (mb.grid_line_match_ratio != null) html += '<dt>Pokrycie linii</dt><dd>' + pct(mb.grid_line_match_ratio) + '</dd>';
  if (mb.warp_panel_coverage != null) html += '<dt>Pokrycie warpu</dt><dd>' + pct(mb.warp_panel_coverage) + '</dd>';
  html += '<dt>Rogi / XY</dt><dd>' + esc(mb.corner_source) + ' / ' + esc(mb.xy_backend) + '</dd>';
  html += '<dt>Panel</dt><dd>' + esc(mb.panel_id) + ' · kąt ' + (mb.report_angle_deg || 0) + '°</dd>';
  html += '</dl></div>';
  if (latch.txt) {
    html += '<div class="param-block"><h3>Zatrzask CXY</h3>';
    html += '<p class="param-desc">Zamrożony wynik konkursu po stabilnej siatce (klawisz s w oknie podglądu).</p>';
    html += '<dl class="param-row"><dt>Stan</dt><dd>' + esc(latch.txt) + '</dd></dl></div>';
  }
  el.innerHTML = html;
}
function snapKey(list) {
  return (list || []).map(s => s.frame_id + ':' + (s.rank||'')).join('|');
}
function renderSnapshots(list) {
  const key = snapKey(list);
  if (key === lastSnapKey) return;
  lastSnapKey = key;
  const snaps = document.getElementById('snaps');
  snaps.replaceChildren();
  (list || []).forEach(s => {
    const a = document.createElement('a');
    const url = '/snap/' + encodeURIComponent(s.frame_id) + '?full=1';
    a.href = url;
    a.target = '_blank';
    const im = document.createElement('img');
    im.src = url;
    im.alt = s.frame_id;
    im.loading = 'lazy';
    a.appendChild(im);
    a.appendChild(document.createTextNode(
      '#' + (s.rank||'?') + ' ' + s.frame_id + ' ' + (s.reproj_b||0).toFixed(1) + 'px'
    ));
    snaps.appendChild(a);
  });
}
async function fetchState() {
  const r = await fetch('/api/state?' + ts());
  const next = await r.json();
  const fid = next.frame_id || '—';
  const upd = next.updated_at || '';
  const m = next.mission || {};
  document.getElementById('status').textContent = [m.text, fid, upd].filter(Boolean).join(' · ') || '—';
  renderStructuredParams(next);
  const pt = next.params_text || '—';
  if (pt !== (state.params_text||'')) {
    const pre = document.getElementById('params');
    if (pre) pre.textContent = pt;
  }
  renderSnapshots(next.snapshots || []);
  document.getElementById('mission').textContent = m.text || '';
  const hint = m.mission_done
    ? 'Misja zakończona — analiza wyłączona.'
    : next.report_pause_sec > 0
    ? `Lot do następnego panelu · pauza ${next.report_pause_sec}s (analiza OFF)…`
    : next.report_can_send
    ? `Panel ${m.current_panel || next.panel_id || 'A'} gotowy — raport z migawek na :8088 (klawisz ${{'A':'1','B':'2','C':'3'}[m.current_panel || next.panel_id || 'A'] || '?'} lub hold_started).`
    : m.panel_full === false && (next.snapshots || []).length
      ? 'Zbieranie migawek…'
      : 'Raporty tylko z migawek (CXY). hold_started / 1·2·3 → :8088. 0 = wyczyść.';
  document.getElementById('reportHint').textContent = hint;
  state = next;
  setTrackerUi(next.tracker || (next.params && next.params.tracker));
  const p = next.params || {};
  setMissionUi(
    next.mission_settings || p.mission_settings || (next.flight || p.flight || {}).settings,
  );
  setRuntimeUi(next.runtime_settings || {});
  const fl = next.flight || p.flight || {};
  document.getElementById('flightStatus').textContent =
    `speed=${fl.speed != null ? Number(fl.speed).toFixed(2) : '—'} · faza=${fl.phase || '—'}`
    + (fl.hold_tier_m != null ? ` · cel ${Number(fl.hold_tier_m).toFixed(1)} m` : '')
    + (fl.distance_m != null ? ` · ${Number(fl.distance_m).toFixed(1)} m` : '');
  updateAnalysisLamp(next);
}
function setMissionUi(m) {
  if (!m || missionEditing()) return;
  const set = (id, v, vid, dec) => {
    const el = document.getElementById(id);
    if (!el || v === undefined || v === null) return;
    const num = Number(v);
    const lo = el.min !== '' ? Number(el.min) : num;
    const hi = el.max !== '' ? Number(el.max) : num;
    const clamped = Math.min(hi, Math.max(lo, num));
    el.value = String(clamped);
    const lab = document.getElementById(vid);
    if (lab) lab.textContent = dec ? clamped.toFixed(dec) : String(clamped);
  };
  set('mDistMin', m.dist_min_m, 'vDistMin', 1);
  set('mDistMax', m.dist_max_m, 'vDistMax', 1);
  set('mDistHold1', m.dist_hold_tier1_m, 'vDistHold1', 1);
  set('mDistHold2', m.dist_hold_tier2_m, 'vDistHold2', 1);
  set('mDistHold3', m.dist_hold_tier3_m, 'vDistHold3', 1);
  set('mHoldS', m.hold_stabilize_s, 'vHoldS', 0);
  set('mCreep', Math.round(Number(m.creep_speed || 0.9) * 100), 'vCreep', 0);
  document.getElementById('vCreep').textContent = Number(m.creep_speed || 0.9).toFixed(2);
  set('mCreepS', m.creep_duration_s, 'vCreepS', 1);
  set('mCruise', Math.round(Number(m.cruise_speed || 0.35) * 100), 'vCruise', 0);
  document.getElementById('vCruise').textContent = Number(m.cruise_speed || 0.35).toFixed(2);
  set('mPauseS', m.report_send_pause_s, 'vPauseS', 0);
  set('mSnapN', m.snapshots_per_panel, 'vSnapN', 0);
}
function bindMissionSliders() {
  const map = [
    ['mDistMin', 'vDistMin', 1], ['mDistMax', 'vDistMax', 1],
    ['mDistHold1', 'vDistHold1', 1], ['mDistHold2', 'vDistHold2', 1], ['mDistHold3', 'vDistHold3', 1],
    ['mHoldS', 'vHoldS', 0], ['mCreepS', 'vCreepS', 1], ['mPauseS', 'vPauseS', 0], ['mSnapN', 'vSnapN', 0],
  ];
  map.forEach(([id, vid, dec]) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', () => {
      document.getElementById('missionPanel').dataset.editing = '1';
      document.getElementById(vid).textContent = dec ? Number(el.value).toFixed(dec) : el.value;
    });
  });
  ['mCreep', 'mCruise'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', () => {
      document.getElementById('missionPanel').dataset.editing = '1';
      const vid = id === 'mCreep' ? 'vCreep' : 'vCruise';
      document.getElementById(vid).textContent = (Number(el.value) / 100).toFixed(2);
    });
  });
}
async function applyMission() {
  const body = {
    dist_min_m: Number(document.getElementById('mDistMin').value),
    dist_max_m: Number(document.getElementById('mDistMax').value),
    dist_hold_tier1_m: Number(document.getElementById('mDistHold1').value),
    dist_hold_tier2_m: Number(document.getElementById('mDistHold2').value),
    dist_hold_tier3_m: Number(document.getElementById('mDistHold3').value),
    hold_stabilize_s: Number(document.getElementById('mHoldS').value),
    creep_speed: Number(document.getElementById('mCreep').value) / 100,
    creep_duration_s: Number(document.getElementById('mCreepS').value),
    cruise_speed: Number(document.getElementById('mCruise').value) / 100,
    report_send_pause_s: Number(document.getElementById('mPauseS').value),
    snapshots_per_panel: Number(document.getElementById('mSnapN').value),
  };
  const r = await fetch('/api/mission', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)});
  const res = await r.json();
  document.getElementById('missionPanel').dataset.editing = '';
  if (res.mission) setMissionUi(res.mission);
  await fetchState();
}
function operatorSendBlocked() {
  const m = state.mission || {};
  return !!(m.mission_done || (state.report_pause_sec || 0) > 0);
}
async function operatorPanelKey(panelId) {
  if (operatorSendBlocked()) return;
  const pid = String(panelId || 'A').toUpperCase().slice(0, 1);
  const r = await fetch('/api/operator-panel', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({panel_id: pid}),
  });
  const j = await r.json();
  const hint = document.getElementById('reportHint');
  if (hint && j.message) hint.textContent = j.message;
  if (!r.ok || !j.ok) return;
  await fetchState();
}
async function setBroadcastCamera() {
  await fetch('/api/broadcast', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode: 'camera'}),
  });
}
function broadcastHotkeyBlocked(target) {
  if (!target) return false;
  const tag = target.tagName;
  if (tag === 'SELECT') return true;
  if (tag === 'INPUT') {
    const type = (target.type || 'text').toLowerCase();
    return !['range', 'button', 'checkbox', 'radio', 'hidden'].includes(type);
  }
  return false;
}
function bindBroadcastHotkeys() {
  window.addEventListener('keydown', (e) => {
    if (broadcastHotkeyBlocked(e.target)) return;
    if (!'0123'.includes(e.key)) return;
    e.preventDefault();
    e.stopPropagation();
    if (e.key === '0') setBroadcastCamera();
    else operatorPanelKey({'1': 'A', '2': 'B', '3': 'C'}[e.key]);
  }, true);
}
bindBroadcastHotkeys();
function bindTuneSliders() {
  [['tAlpha','vAlpha',2],['tHold','vHold',0],['tInterval','vInterval',0],['tGood','vGood',0]].forEach(([id,vid,dec]) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', () => {
      document.getElementById('tunePanel').dataset.editing = '1';
      document.getElementById(vid).textContent = dec ? Number(el.value).toFixed(dec) : el.value;
    });
  });
}
function updateAnalysisLamp(next) {
  const lamp = document.getElementById('analysisLamp');
  const hint = document.getElementById('analysisLampHint');
  if (!lamp) return;
  let active = true;
  if (next.analysis_active != null) {
    active = !!next.analysis_active;
  } else {
    const fl = next.flight || (next.params || {}).flight || {};
    if (fl.vision_active === false) active = false;
  }
  lamp.classList.toggle('on', active);
  lamp.classList.toggle('off', !active);
  lamp.textContent = active ? 'Analiza YOLO — WŁĄCZONA' : 'Analiza YOLO — WYŁĄCZONA';
  if (hint) {
    hint.textContent = active
      ? 'System skanuje panel (migawki, raport). Kamera na :8088 bez nakładki.'
      : 'Pauza lotu / misja — bez skanu. Kamera na :8088 nadal działa.';
  }
}
setInterval(fetchState, 400);
bindTuneSliders();
bindMissionSliders();
bindRuntimeSliders();
async function loadSettingsFromDisk() {
  try {
    const [m, r, t] = await Promise.all([
      fetch('/api/mission?' + ts()).then(x => x.json()),
      fetch('/api/runtime?' + ts()).then(x => x.json()),
      fetch('/api/tracker?' + ts()).then(x => x.json()),
    ]);
    if (m.mission) setMissionUi(m.mission);
    if (r.runtime) setRuntimeUi(r.runtime);
    if (t.tracker) setTrackerUi(t.tracker);
  } catch (_) {}
}
loadSettingsFromDisk();
fetchState();
</script>
</body>
</html>
"""

# Zachowane dla kompatybilności (tryb combined = stary jeden panel).
_HTML_PAGE = _HTML_CONTROL_PAGE


def format_params_text(state: Dict[str, Any]) -> str:
    p = state.get('params') or {}
    lines: List[str] = []
    ma = p.get('module_a') or {}
    if ma:
        lines.append('=== MODUŁ A ===')
        lines.append(f"Status: {'OK' if ma.get('ok') else 'FAIL'}")
        if ma.get('ok'):
            lines.append(f"Odległość: {ma.get('distance_m', 0):.2f} m")
            lines.append(f"Stojak: {ma.get('stand_label', '?')} ({ma.get('report_angle_deg', 0)}°)")
            lines.append(
                f"Roll {ma.get('roll_deg', 0):.0f}°  Pitch {ma.get('pitch_deg', 0):.0f}°  "
                f"Yaw {ma.get('yaw_deg', 0):.0f}°",
            )
            lines.append(f"Reproj A: {ma.get('reproj_mean_px', 0):.1f} px")
        else:
            lines.append(str(ma.get('reason', '?')))
    pp = p.get('panel_presence') or {}
    if pp:
        lines.append('')
        lines.append('=== PANEL W KADRZE ===')
        lines.append(
            f"{'TAK' if pp.get('present') else 'NIE'}"
            + (f"  ({pp.get('reason', '')})" if not pp.get('present') else ''),
        )
    mb = p.get('module_b') or {}
    if mb:
        lines.append('')
        lines.append('=== MODUŁ B ===')
        lines.append(f"Reliable: {'TAK' if mb.get('reliable') else 'NIE'}")
        if mb.get('reliable_legacy') is not None:
            lines.append(
                f"Reliable (legacy reproj): {'TAK' if mb.get('reliable_legacy') else 'NIE'}",
            )
        lines.append(f"Reproj B: {mb.get('reproj_mean_px', 0):.2f} px  inl={mb.get('homography_inliers', 0)}")
        lines.append(f"Panel: {mb.get('panel_id', '?')}  kąt: {mb.get('report_angle_deg', 0)}°")
        lines.append(f"Rogi: {mb.get('corner_source', '?')}  XY: {mb.get('xy_backend', '?')}")
        lines.append(f"3 siatki: {100.0 * float(mb.get('grid_overlap_ratio', 0)):.0f}%")
        if mb.get('grid_line_match_ratio') is not None:
            lines.append(
                f"Pokrycie linii: {100.0 * float(mb['grid_line_match_ratio']):.0f}%",
            )
        if mb.get('roi_coverage') is not None:
            lines.append(f"ROI rogów: {100.0 * float(mb['roi_coverage']):.0f}%")
        if mb.get('warp_panel_coverage') is not None:
            lines.append(
                f"Pokrycie panelu (warp): {100.0 * float(mb['warp_panel_coverage']):.0f}%",
            )
    latch = p.get('latch') or {}
    if latch.get('txt'):
        lines.append('')
        lines.append('=== ZATRASK CXY ===')
        lines.append(str(latch['txt']))
    tr = p.get('tracker') or {}
    if tr:
        lines.append('')
        lines.append('=== TRACKER (WWW) ===')
        lines.append(
            f"alpha={float(tr.get('smooth_alpha', 0)):.2f}  "
            f"hold={int(tr.get('hold_frames', 0))}  "
            f"interval={int(tr.get('interval_ms', 0))}ms  "
            f"good={float(tr.get('tracker_good_reproj', 0)):.0f}px"
        )
    rt = state.get('runtime_settings') or {}
    if rt:
        lines.append('')
        lines.append('=== STEROWANIE (WWW) ===')
        lines.append(
            f"Próg migawek (3 siatki): {100.0 * float(rt.get('snapshot_min_grid_overlap', 0)):.0f}%  "
            f"stable={int(rt.get('snapshot_min_stable', 2))}  "
            f"reproj≤{float(rt.get('snapshot_max_reproj', 15)):.0f}px"
        )
    return '\n'.join(lines) if lines else '—'


def _safe_frame_id(frame_id: str) -> bool:
    return bool(re.match(r'^[A-Za-z0-9_.\-]+$', frame_id or ''))


def _snapshot_search_dirs(session_root: str) -> List[str]:
    """Katalogi migawek — główny + per panel (misja A/B/C)."""
    dirs: List[str] = []
    primary = os.path.join(session_root, 'snapshots')
    if os.path.isdir(primary):
        dirs.append(primary)
    panels_root = os.path.join(session_root, 'panels')
    if os.path.isdir(panels_root):
        for name in sorted(os.listdir(panels_root)):
            sub = os.path.join(panels_root, name, 'snapshots')
            if os.path.isdir(sub):
                dirs.append(sub)
    return dirs


def _find_snapshot_asset(session_root: str, frame_id: str, suffix: str) -> Optional[str]:
    stem = str(frame_id).replace('/', '_').replace(':', '_')
    fname = f'{stem}_{suffix}'
    for snap_dir in _snapshot_search_dirs(session_root):
        full = os.path.join(snap_dir, fname)
        if os.path.isfile(full):
            return full
    return None


def validate_report_payload(
    lines: List[str],
    *,
    panel_id: str = 'A',
    min_cards: int = 0,
    max_cards: int = 4,
) -> Tuple[bool, List[str]]:
    from module_panel.competition_report import validate_competition_report_lines

    return validate_competition_report_lines(
        lines,
        min_cards=min_cards,
        max_cards=max_cards,
        expected_panel=str(panel_id).upper()[:1],
        allow_empty=True,
    )


class LiveWebPublisher:
    """Zapis stanu na dysk (czytany przez serwer HTTP)."""

    def __init__(
        self,
        session_dir: str,
        *,
        min_interval_s: float = 0.3,
        cam_interval_s: float = 0.066,
        stream_interval_s: float = 0.0,
        stream_width: int = 0,
        stream_jpeg_quality: int = 0,
        report_mode: str = 'live',
        preset_reports: Optional[Dict[str, List[str]]] = None,
        preset_reports_path: str = '',
    ) -> None:
        self.session_dir = os.path.abspath(session_dir)
        os.makedirs(self.session_dir, exist_ok=True)
        self.report_mode = str(report_mode or 'live').strip().lower()
        if self.report_mode not in ('live', 'preset'):
            self.report_mode = 'live'
        self.preset_reports: Dict[str, List[str]] = {
            str(k).upper()[:1]: list(v)
            for k, v in (preset_reports or {}).items()
            if str(k).strip()
        }
        self.preset_reports_path = str(preset_reports_path or '')
        self.preview_path = os.path.join(self.session_dir, 'live_preview.jpg')
        self.camera_path = os.path.join(self.session_dir, 'live_cam.jpg')
        self.state_path = os.path.join(self.session_dir, 'web_state.json')
        self.commands_path = os.path.join(self.session_dir, 'web_commands.json')
        self.tracker_path = os.path.join(self.session_dir, 'web_tracker.json')
        self.mission_path = os.path.join(self.session_dir, 'web_mission.json')
        self.runtime_path = os.path.join(self.session_dir, 'web_runtime.json')
        self.draft_path = os.path.join(self.session_dir, 'web_report_draft.txt')
        self.broadcast_path = os.path.join(self.session_dir, 'web_broadcast.json')
        if not os.path.isfile(self.broadcast_path):
            self.clear_broadcast()
        _mi = os.environ.get('DRONIADA_WEB_PREVIEW_INTERVAL_S', str(min_interval_s))
        self.min_interval_s = max(0.1, float(_mi))
        self.cam_interval_s = float(cam_interval_s)
        _si = stream_interval_s or os.environ.get('DRONIADA_STREAM_INTERVAL_S', '0.04')
        self.stream_interval_s = max(0.02, float(_si))
        _sw = stream_width or os.environ.get('DRONIADA_STREAM_WIDTH', '854')
        self.stream_width = max(480, int(_sw))
        _sq = stream_jpeg_quality or os.environ.get('DRONIADA_STREAM_JPEG_QUALITY', '72')
        self.stream_jpeg_quality = max(50, min(95, int(_sq)))
        self._stream_source = os.environ.get('DRONIADA_STREAM_SOURCE', 'vis').strip().lower()
        self._last_publish = 0.0
        self._last_cam_publish = 0.0
        self._tracker_mtime = 0.0
        self._mission_mtime = 0.0
        self._runtime_mtime = 0.0
        self._lock = threading.Lock()
        self._cam_jpeg: Optional[bytes] = None
        self._cam_jpeg_seq = 0
        self._cam_cv = threading.Condition(self._lock)
        self._stream_passthrough = False
        self._stream_frame_lock = threading.Lock()
        self._stream_camera_buf: Optional[Any] = None
        self._stream_vis_buf: Optional[Any] = None
        self._stream_dashboard_buf: Optional[Any] = None
        self._stream_stop = threading.Event()
        self._stream_encoder_thread = threading.Thread(
            target=self._stream_encoder_loop,
            name='droniada-mjpeg-enc',
            daemon=True,
        )
        self._stream_encoder_thread.start()

    def _encode_stream_jpeg(self, bgr: Any, *, target_w: int = 960, quality: int = 78) -> Optional[bytes]:
        if bgr is None or not hasattr(bgr, 'shape'):
            return None
        h, w = bgr.shape[:2]
        tw = max(480, int(target_w))
        if w > tw:
            scale = tw / float(w)
            img = cv2.resize(
                bgr,
                (tw, max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            img = bgr
        ok_enc, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok_enc else None

    def enable_stream_passthrough(self, *, enabled: bool = True) -> None:
        self._stream_passthrough = bool(enabled)

    def _push_stream_jpeg(self, data: bytes, *, also_camera_path: bool = False) -> None:
        with self._cam_cv:
            self._cam_jpeg = data
            self._cam_jpeg_seq += 1
            self._cam_cv.notify_all()
            if also_camera_path:
                with open(self.camera_path, 'wb') as fh:
                    fh.write(data)

    def push_stream_jpeg_bytes(self, data: bytes) -> None:
        """Surowy JPEG z kamery (GStreamer passthrough) — bez cv2.imencode."""
        if data:
            self._push_stream_jpeg(data, also_camera_path=False)

    def _store_stream_buf(self, attr: str, bgr: Any) -> None:
        import numpy as np

        if bgr is None or not hasattr(bgr, 'shape'):
            return
        with self._stream_frame_lock:
            cur = getattr(self, attr, None)
            if cur is None or cur.shape != bgr.shape:
                setattr(self, attr, bgr.copy())
            else:
                np.copyto(cur, bgr)

    def set_stream_camera_frame(self, bgr: Any) -> None:
        """Wątek kamery — tylko podmiana bufora (bez JPEG)."""
        self._store_stream_buf('_stream_camera_buf', bgr)

    def set_stream_vis_frame(self, bgr: Any) -> None:
        """Wątek analizy — kamera + overlay (aktualizacja co YOLO)."""
        self._store_stream_buf('_stream_vis_buf', bgr)

    def set_stream_dashboard_frame(self, bgr: Any) -> None:
        """Pełny dashboard — tylko tryb DRONIADA_STREAM_SOURCE=dashboard."""
        self._store_stream_buf('_stream_dashboard_buf', bgr)

    def _pick_stream_bgr(self) -> Optional[Any]:
        with self._stream_frame_lock:
            if self._stream_source == 'dashboard':
                # Pełny dashboard (rzadko) — między klatkami YOLO użyj vis (płynny overlay).
                if self._stream_vis_buf is not None:
                    return self._stream_vis_buf
                if self._stream_dashboard_buf is not None:
                    return self._stream_dashboard_buf
                return self._stream_camera_buf
            if self._stream_source == 'vis':
                if self._stream_vis_buf is not None:
                    return self._stream_vis_buf
                return self._stream_camera_buf
            return self._stream_camera_buf

    def _stream_encoder_loop(self) -> None:
        """Osobny wątek: stały fps MJPEG dla :8088/stream.mjpg."""
        while not self._stream_stop.is_set():
            if self._stream_passthrough:
                self._stream_stop.wait(0.25)
                continue
            t0 = time.monotonic()
            frame = self._pick_stream_bgr()
            if frame is not None:
                data = self._encode_stream_jpeg(
                    frame,
                    target_w=self.stream_width,
                    quality=self.stream_jpeg_quality,
                )
                if data is not None:
                    self._push_stream_jpeg(data, also_camera_path=False)
            delay = self.stream_interval_s - (time.monotonic() - t0)
            if delay > 0:
                self._stream_stop.wait(delay)

    def shutdown(self) -> None:
        self._stream_stop.set()
        if self._stream_encoder_thread.is_alive():
            self._stream_encoder_thread.join(timeout=1.5)

    def publish_stream_frame(self, bgr: Any) -> None:
        """Legacy — użyj set_stream_camera_frame + wątek enkodera."""
        self.set_stream_camera_frame(bgr)

    def publish_camera(self, bgr: Any, *, target_w: int = 960) -> None:
        """Zapis live_cam.jpg (legacy)."""
        now = time.monotonic()
        if now - self._last_cam_publish < self.cam_interval_s:
            return
        data = self._encode_stream_jpeg(bgr, target_w=target_w, quality=78)
        if data is None:
            return
        with open(self.camera_path, 'wb') as fh:
            fh.write(data)
        self._last_cam_publish = now

    def wait_cam_jpeg(self, after_seq: int = 0, timeout: float = 1.0) -> Tuple[Optional[bytes], int]:
        """Czeka na nową klatkę (seq > after_seq). Bez busy-loop duplikatów."""
        with self._cam_cv:
            deadline = time.monotonic() + max(0.01, float(timeout))
            while self._cam_jpeg is None or self._cam_jpeg_seq <= int(after_seq):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cam_cv.wait(remaining)
            return self._cam_jpeg, self._cam_jpeg_seq

    def read_draft_lines(self) -> List[str]:
        if not os.path.isfile(self.draft_path):
            return []
        with open(self.draft_path, encoding='utf-8') as fh:
            return [ln.strip() for ln in fh.readlines() if ln.strip()]

    def write_draft_lines(self, lines: List[str]) -> None:
        with open(self.draft_path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + ('\n' if lines else ''))

    def write_tracker_settings(self, settings: Dict[str, Any]) -> None:
        from release.tracker_tuning import validate_settings

        prev = self.read_tracker_settings()
        merged = dict(prev)
        merged.update(validate_settings(settings))
        merged['updated_at'] = datetime.now().isoformat(timespec='seconds')
        with open(self.tracker_path, 'w', encoding='utf-8') as fh:
            json.dump(merged, fh, ensure_ascii=False, indent=2)

    def read_tracker_settings(self) -> Dict[str, Any]:
        if not os.path.isfile(self.tracker_path):
            return {}
        try:
            with open(self.tracker_path, encoding='utf-8') as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def poll_tracker_update(self) -> Optional[Dict[str, Any]]:
        if not os.path.isfile(self.tracker_path):
            return None
        try:
            mtime = os.path.getmtime(self.tracker_path)
        except OSError:
            return None
        if mtime <= self._tracker_mtime + 1e-6:
            return None
        self._tracker_mtime = mtime
        data = self.read_tracker_settings()
        return data or None

    def write_mission_settings(self, settings: Dict[str, Any]) -> None:
        from release.mission_settings import merge_mission_settings

        merged = merge_mission_settings(self.read_mission_settings(), settings)
        merged['updated_at'] = datetime.now().isoformat(timespec='seconds')
        with open(self.mission_path, 'w', encoding='utf-8') as fh:
            json.dump(merged, fh, ensure_ascii=False, indent=2)
        # _mission_mtime aktualizuje tylko poll_mission_update — inaczej pętla live nie widzi zmian z WWW.

    def read_mission_settings(self) -> Dict[str, Any]:
        if not os.path.isfile(self.mission_path):
            return {}
        try:
            with open(self.mission_path, encoding='utf-8') as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def poll_mission_update(self) -> Optional[Dict[str, Any]]:
        if not os.path.isfile(self.mission_path):
            return None
        try:
            mtime = os.path.getmtime(self.mission_path)
        except OSError:
            return None
        if mtime <= self._mission_mtime + 1e-6:
            return None
        self._mission_mtime = mtime
        data = self.read_mission_settings()
        return data or None

    def write_runtime_settings(self, settings: Dict[str, Any]) -> None:
        from release.web_runtime_settings import merge_runtime_settings

        merged = merge_runtime_settings(self.read_runtime_settings(), settings)
        merged['updated_at'] = datetime.now().isoformat(timespec='seconds')
        with open(self.runtime_path, 'w', encoding='utf-8') as fh:
            json.dump(merged, fh, ensure_ascii=False, indent=2)

    def read_runtime_settings(self) -> Dict[str, Any]:
        if not os.path.isfile(self.runtime_path):
            from release.web_runtime_settings import DEFAULT_RUNTIME_SETTINGS
            return dict(DEFAULT_RUNTIME_SETTINGS)
        try:
            with open(self.runtime_path, encoding='utf-8') as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            from release.web_runtime_settings import DEFAULT_RUNTIME_SETTINGS
            return dict(DEFAULT_RUNTIME_SETTINGS)

    def poll_runtime_update(self) -> Optional[Dict[str, Any]]:
        if not os.path.isfile(self.runtime_path):
            return None
        try:
            mtime = os.path.getmtime(self.runtime_path)
        except OSError:
            return None
        if mtime <= self._runtime_mtime + 1e-6:
            return None
        self._runtime_mtime = mtime
        from release.web_runtime_settings import validate_runtime_settings

        data = validate_runtime_settings(self.read_runtime_settings())
        return data or None

    def _build_payload(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from release.mission_settings import snapshot as mission_snapshot

        payload = dict(state)
        draft = self.read_draft_lines()
        if draft:
            payload['report_draft'] = draft
        params = payload.get('params') if isinstance(payload.get('params'), dict) else {}
        if not payload.get('flight') and isinstance(params, dict):
            payload['flight'] = dict(params.get('flight') or {})
        # Plik web_mission.json = źródło prawdy dla suwaków (nie flight_controller w RAM).
        mission_saved = mission_snapshot(self.read_mission_settings())
        payload['mission_settings'] = mission_saved
        if isinstance(payload.get('params'), dict):
            payload['params']['mission_settings'] = mission_saved
        payload['params_text'] = format_params_text(payload)
        payload['runtime_settings'] = self.read_runtime_settings()
        saved_tracker = self.read_tracker_settings()
        if saved_tracker:
            payload['tracker'] = dict(saved_tracker)
        payload['report_mode'] = self.report_mode
        payload['preset_panels'] = {
            pid: len(lines) for pid, lines in sorted(self.preset_reports.items())
        }
        payload['preset_reports_path'] = self.preset_reports_path
        payload['updated_at'] = datetime.now().isoformat(timespec='seconds')
        return payload

    def preset_lines(self, panel_id: str) -> List[str]:
        pid = str(panel_id or 'A').strip().upper()[:1]
        return list(self.preset_reports.get(pid, []))

    def read_broadcast(self) -> Dict[str, Any]:
        default: Dict[str, Any] = {
            'visible_panels': [],
            'panel_lines': {},
            'updated_at': None,
        }
        if not os.path.isfile(self.broadcast_path):
            return dict(default)
        try:
            with open(self.broadcast_path, encoding='utf-8') as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return dict(default)
            panels: List[str] = []
            raw = data.get('visible_panels')
            if isinstance(raw, list):
                for item in raw:
                    pid = str(item).strip().upper()[:1]
                    if pid in ('A', 'B', 'C') and pid not in panels:
                        panels.append(pid)
            elif str(data.get('mode', '')).strip().lower() == 'report':
                legacy = str(data.get('panel_id', '')).strip().upper()[:1]
                if legacy in ('A', 'B', 'C'):
                    panels = [legacy]
            panel_lines: Dict[str, List[str]] = {}
            raw_lines = data.get('panel_lines')
            if isinstance(raw_lines, dict):
                for key, val in raw_lines.items():
                    pid = str(key).strip().upper()[:1]
                    if pid in ('A', 'B', 'C') and isinstance(val, list):
                        panel_lines[pid] = [str(ln) for ln in val]
            return {
                'visible_panels': panels,
                'panel_lines': panel_lines,
                'updated_at': data.get('updated_at'),
            }
        except (OSError, json.JSONDecodeError):
            return dict(default)

    def clear_broadcast(self) -> List[str]:
        payload = {
            'visible_panels': [],
            'panel_lines': {},
            'updated_at': datetime.now().isoformat(timespec='seconds'),
        }
        with open(self.broadcast_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return []

    def add_broadcast_panel(self, panel_id: str) -> List[str]:
        """Legacy: preset z pliku (tylko tryb preset)."""
        return self.add_broadcast_report_lines(
            panel_id,
            resolve_report_lines_at_click(self.preset_lines(panel_id)),
        )

    def add_broadcast_report_lines(self, panel_id: str, lines: List[str]) -> List[str]:
        """Raport na :8088 — linie z konsensusu migawek (CXY)."""
        pid = str(panel_id).strip().upper()[:1]
        if pid not in ('A', 'B', 'C'):
            return self.read_broadcast().get('visible_panels', [])
        norm = [str(ln).strip() for ln in lines if str(ln).strip()]
        if not norm:
            return self.read_broadcast().get('visible_panels', [])
        bc = self.read_broadcast()
        panels = list(bc.get('visible_panels') or [])
        panel_lines = dict(bc.get('panel_lines') or {})
        panel_lines[pid] = norm
        if pid not in panels:
            panels.append(pid)
        payload = {
            'visible_panels': panels,
            'panel_lines': panel_lines,
            'updated_at': datetime.now().isoformat(timespec='seconds'),
        }
        with open(self.broadcast_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return panels

    def broadcast_lines_for_panel(self, panel_id: str) -> List[str]:
        pid = str(panel_id).strip().upper()[:1]
        bc = self.read_broadcast()
        stored = (bc.get('panel_lines') or {}).get(pid)
        if stored:
            return list(stored)
        return self.preset_lines(pid)

    def broadcast_payload(self) -> Dict[str, Any]:
        bc = self.read_broadcast()
        panels = list(bc.get('visible_panels') or [])
        return {
            'visible_panels': panels,
            'panels': [
                {'panel_id': pid, 'lines': self.broadcast_lines_for_panel(pid)}
                for pid in panels
            ],
            'updated_at': bc.get('updated_at'),
        }

    def read_published_state(self) -> Dict[str, Any]:
        if not os.path.isfile(self.state_path):
            return {}
        try:
            with open(self.state_path, encoding='utf-8') as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def operator_submit_panel(self, panel_id: str) -> Dict[str, Any]:
        """Raport z migawek (CXY) na :8088 + wysyłka misji."""
        pid = str(panel_id).strip().upper()[:1]
        if pid not in ('A', 'B', 'C'):
            return {'ok': False, 'message': 'Nieprawidłowy panel (A/B/C).'}
        state = self.read_published_state()
        mission = state.get('mission') if isinstance(state.get('mission'), dict) else {}
        current = str(mission.get('current_panel') or state.get('panel_id') or 'A').upper()[:1]
        pause = int(state.get('report_pause_sec') or 0)
        can_send = bool(state.get('report_can_send'))
        report_mode = str(state.get('report_mode') or 'live')
        if report_mode == 'preset':
            lines = resolve_report_lines_at_click(self.preset_lines(current))
        else:
            lines = self.read_draft_lines() or list(state.get('consensus_report_lines') or [])
        visible = self.add_broadcast_report_lines(current, lines)
        payload = self.broadcast_payload()
        mission_sent = False
        if mission.get('mission_done'):
            msg = 'Misja zakończona.'
        elif pause > 0:
            msg = f'Pauza misji ({pause}s).'
        elif pid != current:
            key_map = {'A': '1', 'B': '2', 'C': '3'}
            msg = (
                f'Misja na panelu {current} — użyj klawisza {key_map.get(current, current)}.'
            )
        elif not lines:
            msg = 'Brak raportu z migawek — zbierz pełny zestaw klatek.'
        elif not can_send:
            msg = 'Zbierz migawki przed przejściem misji.'
        else:
            ok, errors = validate_report_payload(lines, panel_id=current)
            if not ok:
                return {
                    'ok': False,
                    'message': 'Raport z migawek nie przeszedł walidacji.',
                    'errors': errors,
                    'visible_panels': visible,
                    'panels': payload['panels'],
                }
            self.enqueue_command({
                'action': 'send',
                'panel_id': current,
                'lines': lines,
                'source': 'operator-key',
                'at': datetime.now().isoformat(timespec='seconds'),
            })
            mission_sent = True
            msg = f'Panel {current}: raport z migawek na :8088 + następny panel.'
        return {
            'ok': True,
            'mission_sent': mission_sent,
            'message': msg,
            'visible_panels': visible,
            'panels': payload['panels'],
        }

    def publish_state(self, state: Dict[str, Any]) -> None:
        """Zapis JSON dla panelu sterowania — zawsze, bez throttlingu obrazu."""
        payload = self._build_payload(state)
        with self._lock:
            with open(self.state_path, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)

    def publish(self, dashboard_bgr: Any, state: Dict[str, Any], *, force: bool = False) -> None:
        now = time.monotonic()
        payload = self._build_payload(state)
        with self._lock:
            with open(self.state_path, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            if not force and now - self._last_publish < self.min_interval_s:
                return
            cv2.imwrite(
                self.preview_path,
                dashboard_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 82],
            )
            self._last_publish = now

    def poll_command(self) -> Optional[Dict[str, Any]]:
        if not os.path.isfile(self.commands_path):
            return None
        try:
            with open(self.commands_path, encoding='utf-8') as fh:
                cmd = json.load(fh)
            os.remove(self.commands_path)
            return cmd if isinstance(cmd, dict) else None
        except (OSError, json.JSONDecodeError):
            try:
                os.remove(self.commands_path)
            except OSError:
                pass
            return None

    def enqueue_command(self, cmd: Dict[str, Any]) -> None:
        with open(self.commands_path, 'w', encoding='utf-8') as fh:
            json.dump(cmd, fh, ensure_ascii=False)


def _read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get('Content-Length', 0))
    raw = handler.rfile.read(length) if length > 0 else b'{}'
    try:
        data = json.loads(raw.decode('utf-8'))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def make_handler(
    session_dir: str,
    publisher: LiveWebPublisher,
    *,
    ui_mode: str = 'control',
) -> type:
    root = os.path.abspath(session_dir)
    _BOUNDARY = b'--droniadaframe'
    _html_by_mode = {
        'view': _HTML_VIEW_PAGE,
        'control': _HTML_CONTROL_PAGE,
        'combined': _HTML_PAGE,
    }
    _index_html = _html_by_mode.get(ui_mode, _HTML_CONTROL_PAGE)
    _allow_stream = ui_mode in ('view', 'control', 'combined')
    _allow_snap = ui_mode in ('control', 'combined')
    _allow_report_api = ui_mode in ('control', 'combined')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def _send_bytes(self, data: bytes, content_type: str, *, code: int = 200) -> None:
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, obj: Any, *, code: int = 200) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
            self._send_bytes(data, 'application/json; charset=utf-8', code=code)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ('/', '/index.html'):
                self._send_bytes(_index_html.encode('utf-8'), 'text/html; charset=utf-8')
                return
            if path == '/live.jpg':
                jpeg, _ = publisher.wait_cam_jpeg(0, timeout=0.15)
                if jpeg:
                    self._send_bytes(jpeg, 'image/jpeg')
                    return
                for preview in (
                    os.path.join(root, 'live_preview.jpg'),
                    os.path.join(root, 'live_cam.jpg'),
                ):
                    if os.path.isfile(preview):
                        with open(preview, 'rb') as fh:
                            self._send_bytes(fh.read(), 'image/jpeg')
                        return
                self.send_error(404)
                return
            if path == '/api/broadcast':
                self._send_json(publisher.broadcast_payload())
                return
            if path == '/api/state':
                state_path = os.path.join(root, 'web_state.json')
                if os.path.isfile(state_path):
                    with open(state_path, encoding='utf-8') as fh:
                        self._send_bytes(fh.read().encode('utf-8'), 'application/json; charset=utf-8')
                else:
                    self._send_json({
                        'updated_at': None,
                        'frame_id': None,
                        'params_text': 'Oczekiwanie na pierwszą klatkę…',
                        'snapshots': [],
                    })
                return
            if path.startswith('/api/preset-report/') and _allow_report_api:
                if publisher.report_mode != 'preset':
                    self.send_error(404)
                    return
                pid = path.rsplit('/', 1)[-1].strip().upper()[:1]
                lines = publisher.preset_lines(pid)
                if not lines:
                    self.send_error(404)
                    return
                self._send_json({'ok': True, 'panel_id': pid, 'lines': lines})
                return
            if path == '/api/mission':
                if not _allow_report_api:
                    self.send_error(404)
                    return
                from release.mission_settings import snapshot as mission_snapshot

                mission = mission_snapshot(publisher.read_mission_settings())
                self._send_json({'ok': True, 'mission': mission})
                return
            if path == '/api/runtime':
                if not _allow_report_api:
                    self.send_error(404)
                    return
                from release.web_runtime_settings import runtime_snapshot

                rt = runtime_snapshot(publisher.read_runtime_settings())
                self._send_json({'ok': True, 'runtime': rt})
                return
            if path == '/api/tracker':
                if not _allow_report_api:
                    self.send_error(404)
                    return
                from release.tracker_tuning import snapshot as tracker_snapshot

                saved = publisher.read_tracker_settings()
                tr = tracker_snapshot() if not saved else saved
                self._send_json({'ok': True, 'tracker': tr})
                return
            if path.startswith('/stream.mjpg'):
                if not _allow_stream:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Connection', 'close')
                self.send_header(
                    'Content-Type',
                    'multipart/x-mixed-replace; boundary=droniadaframe',
                )
                self.end_headers()
                try:
                    last_seq = 0
                    while True:
                        jpeg, last_seq = publisher.wait_cam_jpeg(last_seq, timeout=1.0)
                        if jpeg is None:
                            continue
                        header = (
                            _BOUNDARY + b'\r\n'
                            b'Content-Type: image/jpeg\r\n'
                            + f'Content-Length: {len(jpeg)}\r\n\r\n'.encode('ascii')
                        )
                        self.wfile.write(header)
                        self.wfile.write(jpeg)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
            if path.startswith('/snap/'):
                if not _allow_snap:
                    self.send_error(404)
                    return
                fid = path.split('/snap/', 1)[-1].split('?', 1)[0]
                if not _safe_frame_id(fid):
                    self.send_error(400)
                    return
                q = urlparse(self.path).query or ''
                suffixes = (
                    ('dashboard.png', 'dashboard.jpg', 'thumb.jpg')
                    if 'full=1' in q
                    else ('dashboard.png', 'dashboard.jpg', 'thumb.jpg')
                )
                for suf in suffixes:
                    full = _find_snapshot_asset(root, fid, suf)
                    if full is not None:
                        ctype = mimetypes.guess_type(full)[0] or 'application/octet-stream'
                        with open(full, 'rb') as fh:
                            self._send_bytes(fh.read(), ctype)
                        return
                self.send_error(404)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            body = _read_json_body(self)
            if path == '/api/runtime':
                from release.web_runtime_settings import merge_runtime_settings

                current = publisher.read_runtime_settings()
                merged = merge_runtime_settings(current, body)
                publisher.write_runtime_settings(merged)
                self._send_json({
                    'ok': True,
                    'runtime': merged,
                    'message': 'Progi migawek zapisane',
                })
                return
            if path == '/api/report' and not _allow_report_api:
                self.send_error(404)
                return
            if path == '/api/report':
                lines = [str(ln).strip() for ln in body.get('lines', []) if str(ln).strip()]
                publisher.write_draft_lines(lines)
                self._send_json({'ok': True, 'n_lines': len(lines)})
                return
            if path == '/api/validate':
                if not _allow_report_api:
                    self.send_error(404)
                    return
                lines = [str(ln).strip() for ln in body.get('lines', []) if str(ln).strip()]
                panel_id = str(body.get('panel_id', 'A')).upper()[:1]
                ok, errors = validate_report_payload(lines, panel_id=panel_id)
                self._send_json({'ok': ok, 'errors': errors, 'n_lines': len(lines)})
                return
            if path == '/api/broadcast':
                if not _allow_report_api:
                    self.send_error(404)
                    return
                mode = str(body.get('mode', 'camera')).strip().lower()
                if mode == 'report':
                    panel_id = str(body.get('panel_id', 'A')).strip().upper()[:1]
                    lines = publisher.preset_lines(panel_id)
                    if not lines:
                        self._send_json({
                            'ok': False,
                            'message': (
                                f'Brak raportu dla panelu {panel_id} — '
                                'uzupełnij config/preset_reports.json'
                            ),
                        }, code=404)
                        return
                    publisher.add_broadcast_panel(panel_id)
                    payload = publisher.broadcast_payload()
                    self._send_json({
                        'ok': True,
                        'visible_panels': payload['visible_panels'],
                        'panels': payload['panels'],
                    })
                    return
                visible = publisher.clear_broadcast()
                self._send_json({'ok': True, 'visible_panels': visible, 'panels': []})
                return
            if path == '/api/operator-panel':
                if not _allow_report_api:
                    self.send_error(404)
                    return
                panel_id = str(body.get('panel_id', 'A')).strip().upper()[:1]
                result = publisher.operator_submit_panel(panel_id)
                code = 200 if result.get('ok') else 400
                self._send_json(result, code=code)
                return
            if path == '/api/send-preset':
                if not _allow_report_api:
                    self.send_error(404)
                    return
                if publisher.report_mode != 'preset':
                    self._send_json({
                        'ok': False,
                        'message': 'Tryb preset wyłączony — uruchom z DRONIADA_REPORT_MODE=preset',
                    }, code=400)
                    return
                panel_id = str(body.get('panel_id', 'A')).upper()[:1]
                lines = publisher.preset_lines(panel_id)
                if not lines:
                    self._send_json({
                        'ok': False,
                        'message': f'Brak raportu preset dla panelu {panel_id}',
                    }, code=404)
                    return
                ok, errors = validate_report_payload(lines, panel_id=panel_id)
                if not ok:
                    self._send_json({
                        'ok': False,
                        'message': 'Raport preset nie przeszedł walidacji.',
                        'errors': errors,
                    }, code=400)
                    return
                publisher.enqueue_command({
                    'action': 'send',
                    'panel_id': panel_id,
                    'lines': lines,
                    'source': 'preset',
                    'at': datetime.now().isoformat(timespec='seconds'),
                })
                self._send_json({
                    'ok': True,
                    'message': f'Wysłano preset panelu {panel_id} — lot do następnego panelu.',
                    'panel_id': panel_id,
                    'lines': lines,
                })
                return
            if path == '/api/send':
                if not _allow_report_api:
                    self.send_error(404)
                    return
                lines = [str(ln).strip() for ln in body.get('lines', []) if str(ln).strip()]
                panel_id = str(body.get('panel_id', 'A')).upper()[:1]
                ok, errors = validate_report_payload(lines, panel_id=panel_id)
                if not ok:
                    self._send_json({
                        'ok': False,
                        'message': 'Raport nie spełnia wymagań regulaminu — popraw linie poniżej.',
                        'errors': errors,
                    }, code=400)
                    return
                publisher.write_draft_lines(lines)
                publisher.enqueue_command({
                    'action': 'send',
                    'panel_id': panel_id,
                    'lines': lines,
                    'at': datetime.now().isoformat(timespec='seconds'),
                })
                self._send_json({
                    'ok': True,
                    'message': f'Wysłano panel {panel_id} — migawki zostaną zresetowane.',
                })
                return
            if path == '/api/mission':
                if not _allow_report_api:
                    self.send_error(404)
                    return
                from release.mission_settings import merge_mission_settings, snapshot as mission_snapshot

                settings = merge_mission_settings(publisher.read_mission_settings(), body)
                publisher.write_mission_settings(settings)
                saved = mission_snapshot(publisher.read_mission_settings())
                self._send_json({'ok': True, 'message': 'Ustawienia misji zapisane.', 'mission': saved})
                return
            if path == '/api/tracker':
                if not _allow_report_api:
                    self.send_error(404)
                    return
                from release.tracker_tuning import validate_settings

                settings = validate_settings(body)
                if not settings:
                    self._send_json({'ok': False, 'message': 'Brak parametrów trackera.'}, code=400)
                    return
                settings['updated_at'] = datetime.now().isoformat(timespec='seconds')
                publisher.write_tracker_settings(settings)
                self._send_json({
                    'ok': True,
                    'message': (
                        f"Tracker: alpha={settings.get('smooth_alpha', '?')} "
                        f"hold={int(settings.get('hold_frames', 0))} "
                        f"interval={int(settings.get('interval_ms', 0))}ms"
                    ),
                    'tracker': settings,
                })
                return
            self.send_error(404)

    return Handler


def start_web_server_thread(
    session_dir: str,
    *,
    host: str = '0.0.0.0',
    port: int = 8765,
    publisher: Optional[LiveWebPublisher] = None,
    ui_mode: str = 'control',
) -> Tuple[ThreadingHTTPServer, threading.Thread, LiveWebPublisher]:
    """Uruchom serwer w wątku daemon (nasłuch na wszystkich interfejsach)."""
    pub = publisher or LiveWebPublisher(session_dir)
    handler = make_handler(session_dir, pub, ui_mode=ui_mode)
    httpd = ThreadingHTTPServer((host, int(port)), handler)
    thread = threading.Thread(
        target=httpd.serve_forever,
        name=f'droniada-web-{ui_mode}-{port}',
        daemon=True,
    )
    thread.start()
    return httpd, thread, pub


def start_split_web_servers(
    session_dir: str,
    *,
    host: str = '0.0.0.0',
    view_port: int = 8088,
    control_port: int = 8089,
    publisher: Optional[LiveWebPublisher] = None,
) -> Tuple[LiveWebPublisher, ThreadingHTTPServer, ThreadingHTTPServer]:
    """Podgląd live (view_port) + panel sterowania (control_port), wspólna sesja."""
    pub = publisher or LiveWebPublisher(session_dir)
    view_httpd, _vt, _ = start_web_server_thread(
        session_dir, host=host, port=int(view_port), publisher=pub, ui_mode='view',
    )
    ctrl_httpd, _ct, _ = start_web_server_thread(
        session_dir, host=host, port=int(control_port), publisher=pub, ui_mode='control',
    )
    return pub, view_httpd, ctrl_httpd


def build_web_state(
    *,
    frame_id: str,
    panel_id: str,
    module_a_po: Any = None,
    reliable: bool = False,
    reproj_b: float = 999.0,
    homography_inliers: int = 0,
    corner_source: str = '-',
    xy_backend: str = '-',
    angle: int = 0,
    category: str = 'horizontal',
    grid_overlap_ratio: float = 0.0,
    grid_line_match_ratio: Optional[float] = None,
    roi_coverage: Optional[float] = None,
    warp_panel_coverage: Optional[float] = None,
    live_report_lines: Optional[List[str]] = None,
    latched_report_lines: Optional[List[str]] = None,
    consensus_report_lines: Optional[List[str]] = None,
    latch_txt: Optional[str] = None,
    latch_locked: bool = False,
    snapshots: Optional[List[Dict[str, Any]]] = None,
    mission: Optional[Dict[str, Any]] = None,
    report_ready: bool = False,
    report_can_send: bool = False,
    report_pause_sec: int = 0,
    report_validation: Optional[Dict[str, Any]] = None,
    report_mode: str = 'live',
    preset_panels: Optional[Dict[str, int]] = None,
    panel_present: bool = False,
    panel_presence_reason: str = 'no_corners',
    reliable_legacy: Optional[bool] = None,
    tracker: Optional[Dict[str, Any]] = None,
    flight: Optional[Dict[str, Any]] = None,
    mission_settings: Optional[Dict[str, Any]] = None,
    analysis_active: bool = True,
    overlay_corners_px: Any = None,
    overlay_frame_w: int = 0,
    overlay_frame_h: int = 0,
    overlay_tracker_hold: bool = False,
) -> Dict[str, Any]:
    from module_pose.panel_stand import STAND_LABEL_PL

    ma: Dict[str, Any] = {}
    if module_a_po is not None:
        if getattr(module_a_po, 'ok', False):
            ma = {
                'ok': True,
                'distance_m': float(module_a_po.distance_m),
                'report_angle_deg': int(module_a_po.report_angle_deg),
                'stand_label': STAND_LABEL_PL.get(
                    module_a_po.panel_angle_category,
                    module_a_po.panel_angle_category,
                ),
                'stand_confidence': float(module_a_po.stand_confidence),
                'roll_deg': float(module_a_po.roll_deg),
                'pitch_deg': float(module_a_po.pitch_deg),
                'yaw_deg': float(module_a_po.yaw_deg),
                'reproj_mean_px': float(module_a_po.reproj_mean_px),
            }
        else:
            ma = {
                'ok': False,
                'reason': str(
                    module_a_po.meta.get('reason', module_a_po.meta.get('fail', '?')),
                ),
            }
    return {
        'frame_id': frame_id,
        'panel_id': str(panel_id).upper()[:1],
        'analysis_active': bool(analysis_active),
        'params': {
            'panel_presence': {
                'present': bool(panel_present),
                'reason': str(panel_presence_reason),
            },
            'module_a': ma,
            'module_b': {
                'reliable': bool(reliable),
                'reliable_legacy': reliable_legacy,
                'reproj_mean_px': float(reproj_b),
                'homography_inliers': int(homography_inliers),
                'corner_source': corner_source,
                'xy_backend': xy_backend,
                'panel_id': str(panel_id).upper()[:1],
                'report_angle_deg': int(angle),
                'panel_angle_category': category,
                'grid_overlap_ratio': float(grid_overlap_ratio),
                'grid_line_match_ratio': (
                    float(grid_line_match_ratio) if grid_line_match_ratio is not None else None
                ),
                'roi_coverage': float(roi_coverage) if roi_coverage is not None else None,
                'warp_panel_coverage': (
                    float(warp_panel_coverage) if warp_panel_coverage is not None else None
                ),
            },
            'latch': {'txt': latch_txt, 'locked': bool(latch_locked)},
            'tracker': dict(tracker or {}),
            'flight': dict(flight or {}),
            'mission_settings': dict(mission_settings or {}),
        },
        'tracker': dict(tracker or {}),
        'live_report_lines': list(live_report_lines or []),
        'latched_report_lines': list(latched_report_lines or []),
        'consensus_report_lines': list(consensus_report_lines or []),
        'snapshots': list(snapshots or []),
        'mission': dict(mission or {}),
        'report_ready': bool(report_ready),
        'report_can_send': bool(report_can_send),
        'report_pause_sec': int(report_pause_sec),
        'report_validation': dict(report_validation or {}),
        'report_mode': str(report_mode or 'live'),
        'preset_panels': dict(preset_panels or {}),
        'flight': dict(flight or {}),
        'mission_settings': dict(mission_settings or {}),
        'overlay': {
            'frame_w': int(overlay_frame_w or 0),
            'frame_h': int(overlay_frame_h or 0),
            'corners': _overlay_corners_list(overlay_corners_px),
            'reliable': bool(reliable),
            'tracker_hold': bool(overlay_tracker_hold),
            'panel_present': bool(panel_present),
        },
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description='Podgląd HTTP sesji live (bez kamery)')
    ap.add_argument('--session-dir', required=True, help='katalog session_YYYYMMDD_…')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8765)
    args = ap.parse_args()
    httpd, _thread, _pub = start_web_server_thread(args.session_dir, host=args.host, port=args.port)
    print(f'Droniada web: http://{args.host}:{args.port}/  sesja={args.session_dir}')
    print('Ctrl+C = koniec')
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == '__main__':
    main()
