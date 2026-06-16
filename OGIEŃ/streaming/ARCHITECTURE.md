# Architecture — reporting vs streaming

## Two independent channels

```
┌─────────────────────────────────────────────────────────────┐
│  CHANNEL A — REPORT (competition rules / scoring)           │
│  release/web_dashboard + run_live_panel                     │
│  live: CXY → textarea → Submit                              │
│  preset: config/preset_reports.json → keys 1/2/3            │
│  → flight_controller.on_report_sent() → WebSocket speed     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  CHANNEL B — VIDEO (judges / audience)                      │
│  streaming/ — UDP RTP or OBS → Twitch → portal iframe       │
└─────────────────────────────────────────────────────────────┘
```

Text reports do **not** go through GStreamer. Video does **not** replace the report.

## Video variants

### 1. Lastmile (Gigabyte) — UDP from drone

1. Drone/Jetson: `sender/jetson_udp_send_h264.sh` or custom pipeline.
2. Gigabyte: `gigabyte/start_lastmile.sh` — decode + window + optional RTMP.
3. Twitch: key in `config/stream.env`, `RTMP_RELAY=1` — see `TWITCH_SETUP.md`.
4. Portal: `portal/index.html` via ngrok with `?vision=channel`.

### 2. OGIEŃ — Mac + OBS

1. OBS capture of `:8088` preview → Twitch.
2. OBS capture QGC → second Twitch channel (optional).
3. Portal with two iframes (`?vision=&qgc=`).

### 3. Gigabyte without NVIDIA GPU

- Decode: `avdec_h264` / `avdec_h265` (FFmpeg, CPU).
- Encode to Twitch: `x264enc` software (`veryfast`, ~2.5 Mbps).
- For heavy 1080p60 consider lower resolution on sender (720p30).

## Example competition network

| Host | IP | Role |
|------|-----|------|
| Jetson | 192.168.100.200 | Droniada vision |
| Gigabyte | 192.168.100.249 | lastmile UDP → Twitch |
| Operator Mac | DHCP | OBS / ngrok portal |

Ensure Gigabyte firewall allows UDP `5600` from the drone segment.

## Pre-flight checklist

- [ ] `config/stream.env` on Gigabyte (port, codec, Twitch key)
- [ ] Test `videotestsrc → udpsink` + receive on Gigabyte
- [ ] Twitch live active, channel names in portal
- [ ] Ngrok / domain for judges (if used)
- [ ] Separately: `preset_reports.json` or live report mode
