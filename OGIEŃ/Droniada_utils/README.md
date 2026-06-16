# Droniada_utils — gimbal control GUI

Small Tkinter app (`gui.py`) for gimbal control over WebSocket.  
Connects to **`ws://192.168.100.200:6100`** by default and sends move commands (`pitch`, `yaw`, `zoom`) with angle limits.

### Install

```bash
cd Droniada_utils
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.venv/` is local and gitignored.

### Run

Ensure the gimbal backend listens on the WebSocket URL (edit in `gui.py` → `ws_loop` if needed):

```bash
python gui.py
```

### Controls

- Top bar: connection status (`Connected` / `Disconnected`) and live **`pitch`**, **`roll`**, **`yaw`**.
- **↑ Pitch Up / ↓ Pitch Down**, **← Yaw Left / → Yaw Right**, **Zoom In / Out**.
- Speed fields: `0–255` for pitch/yaw, `0–8` for zoom.

Move commands repeat ~every 100 ms while a button is held (`timeout: 0.15` on server). GUI enforces `LIMITS` in `gui.py`.
