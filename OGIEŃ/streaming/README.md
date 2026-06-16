# Judge stream (historical)

> **Not used in the 2026 workflow.** Jetson preview and reports: **:8088** (public) + **:8089** (operator). See [`docs/JETSON_DOCKER.md`](../docs/JETSON_DOCKER.md).

Isolated module (outside `release/` vision core): UDP/RTP receive from drone,
preview on **Gigabyte** (`robot@192.168.100.249`), relay to **Twitch** (RTMP),
judge page (Twitch iframe / ngrok).

## Topology

```
┌─────────────┐     UDP RTP H.264/H.265      ┌──────────────────┐
│ Drone/Jetson│ ───────────────────────────► │ Gigabyte (robot) │
│  (sender)   │         :5600 (default)     │  receive + preview│
└─────────────┘                              │  + RTMP relay     │
                                             └────────┬─────────┘
                                                      │
                      ┌───────────────────────────────┼────────────────┐
                      ▼                               ▼                ▼
               Twitch Live #1                   local monitor    portal HTTP
               (screen / vision)                (GStreamer)        (Twitch iframe)

Mac (OGIEŃ competition — alternative):
  Multi-desktop → OBS → Twitch (:8088 preview)
  QGroundControl → OBS → Twitch #2 (optional second channel)
```

## Quick start — Gigabyte

```bash
# On Mac — copy module to Gigabyte
./streaming/scripts/sync_to_gigabyte.sh

# On Gigabyte (SSH robot@192.168.100.249)
cd ~/droniada_lastmile
cp config/stream.env.example config/stream.env
# fill TWITCH_STREAM_KEY — see TWITCH_SETUP.md
./gigabyte/install_deps.sh
./gigabyte/start_lastmile.sh
```

## OGIEŃ competition (Mac instead of drone UDP)

See [`mac/OGIEN_STREAM.md`](mac/OGIEN_STREAM.md) — stream from laptop (:8088 preview + QGC in OBS).

## Judge portal / ngrok

- [`TWITCH_SETUP.md`](TWITCH_SETUP.md) — **where to get the stream key**
- [`portal/README.md`](portal/README.md) — Twitch iframe + ngrok
- [`portal/index.html`](portal/index.html) — two-window template

## Reporting vs video (scoring system)

Video stream ≠ regulatory text report. Reports use a separate channel (web panel / scoring API).
This module covers **video only**. Report integration: `docs/INTEGRATION.md`, `release/web_dashboard.py`.

## Directories

| Path | Description |
|------|-------------|
| `config/` | `stream.env` — ports, codec, Twitch key (do not commit) |
| `gigabyte/` | receive and relay scripts on robot machine |
| `sender/` | example sender pipelines (Jetson / simulator) |
| `mac/` | OBS + multi-monitor |
| `portal/` | iframe page for judges |
| `scripts/` | sync to Gigabyte |
