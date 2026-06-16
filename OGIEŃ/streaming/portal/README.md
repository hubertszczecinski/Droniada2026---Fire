# Judge portal — Twitch iframe + ngrok

## Embed Twitch on your domain

Twitch requires **`parent=`** = hostname of the page embedding the player (e.g. `your-sub.ngrok-free.app`).

`index.html` sets `parent` automatically from `location.hostname`.

## Ngrok

```bash
cd streaming/portal
python3 -m http.server 8766
# second terminal:
ngrok http 8766
```

Judges open e.g.:

`https://xxxx.ngrok-free.app/?vision=your_channel&qgc=second_channel`

`vision` / `qgc` = **Twitch channel name** (not stream key).

## Stream key vs channel name

| | Stream key | Channel name |
|---|------------|--------------|
| Where | Creator Dashboard → Stream | URL `twitch.tv/NAME` |
| Used for | GStreamer / OBS broadcast | Portal embed `?vision=` |
| File | `config/stream.env` → `TWITCH_STREAM_KEY` | portal URL param |

Details: [`../TWITCH_SETUP.md`](../TWITCH_SETUP.md)
