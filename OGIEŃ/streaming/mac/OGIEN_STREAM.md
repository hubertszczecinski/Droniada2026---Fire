# OGIEŃ competition — Mac to Twitch stream

## Two-stream layout

| Stream | Source | Tool | Twitch |
|--------|--------|------|--------|
| **#1 Vision** | Droniada preview `:8088` | OBS (Display Capture) | Channel #1 |
| **#2 Telemetry** | QGroundControl | OBS (Window Capture) | Channel #2 (optional) |

## OBS → Twitch

1. Stream key: [`../TWITCH_SETUP.md`](../TWITCH_SETUP.md)
2. OBS → **Settings → Stream** → Service: **Twitch**
3. **Connect Account** or paste key manually
4. Encoder: **Apple VT H264**, bitrate 3000–4500 kbps, keyframe 2 s

## Multi-desktop on Mac

1. Displays → **Extend**
2. Move `http://<jetson>:8088/` to second desktop
3. OBS → Screen Capture (Display 2)

## Reports

Separate from video — `DRONIADA_REPORT_MODE=preset` + keys 1/2/3 on `:8089`.
