# Twitch — stream key

## 1. Twitch account

Sign up at [https://www.twitch.tv/signup](https://www.twitch.tv/signup) if needed.

Two streams (vision + QGC) = **two Twitch channels** or one channel with OBS scene switching. One channel is enough to start.

## 2. Stream key

### Method A — Creator Dashboard (recommended)

1. Log in at [https://www.twitch.tv](https://www.twitch.tv)
2. Avatar (top right) → **Creator Dashboard**
3. Left: **Settings** → **Stream** ([dashboard.twitch.tv/settings/stream](https://dashboard.twitch.tv/settings/stream))
4. **Stream configuration**:
   - **RTMP server:** `rtmp://live.twitch.tv/app` (already in `config/stream.env`)
   - **Primary stream key** → **Copy**

Paste into `config/stream.env`:

```bash
RTMP_RELAY=1
TWITCH_STREAM_KEY=live_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Do not publish the key** — anyone with it can broadcast to your channel. On leak: **Reset** in the same panel.

### Method B — OBS (quick login)

1. OBS → Settings → Stream → Twitch → Connect Account
2. OBS stores the key; for GStreamer scripts you still need the raw key in `stream.env`.

## 3. Test relay

```bash
./gigabyte/test_twitch_key.sh
```

## 4. Where to watch

- Twitch: `https://www.twitch.tv/<your_channel>`
- Portal embed: channel **name** (not key) in `portal/index.html?vision=<name>`

## Stream key vs channel name

| | Stream key | Channel name |
|---|------------|--------------|
| Where | Creator Dashboard → Stream | URL `twitch.tv/NAME` |
| Used for | GStreamer / OBS broadcast | Portal iframe `?vision=` |
| File | `config/stream.env` → `TWITCH_STREAM_KEY` | portal URL param |
