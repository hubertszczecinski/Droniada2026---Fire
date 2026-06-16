import os
import json
import tkinter as tk
from tkinter import ttk
import asyncio
import threading
import websockets

LIMITS = {
    "pitch": [-85.0, 43.0],
    "yaw": [-140.0, 140.0]
}

class GimbalGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gimbal Controller GUI")
        self.geometry("400x250")
        
        self.angles = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}
        self.active_cmds = {"pitch": None, "yaw": None, "zoom": None}
        self.cmd_queue = None
        
        self.setup_ui()
        self.update_movement()
        
        threading.Thread(target=self.run_async_loop, daemon=True).start()

    def run_async_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.cmd_queue = asyncio.Queue()
        self.loop.run_until_complete(self.ws_loop())
        
    async def ws_loop(self):
        uri = "ws://192.168.100.200:6100"
        while True:
            try:
                async with websockets.connect(uri) as ws:
                    self.lbl_status.config(text="Status: Connected", foreground="green")
                    
                    async def read_task():
                        async for message in ws:
                            data = json.loads(message)
                            if "pitch" in data:
                                self.angles = data
                                self.lbl_pitch.config(text=f"Pitch: {data['pitch']:.2f}")
                                self.lbl_roll.config(text=f"Roll: {data['roll']:.2f}")
                                self.lbl_yaw.config(text=f"Yaw: {data['yaw']:.2f}")
                                
                    async def write_task():
                        while True:
                            cmd = await self.cmd_queue.get()
                            if cmd == 'QUIT': break
                            await ws.send(json.dumps(cmd))
                            
                    await asyncio.gather(read_task(), write_task())
            except Exception:
                self.lbl_status.config(text="Status: Disconnected", foreground="red")
                await asyncio.sleep(2)

    def send_ws_command(self, cmd):
        if self.cmd_queue and hasattr(self, 'loop'):
            try:
                self.loop.call_soon_threadsafe(self.cmd_queue.put_nowait, cmd)
            except Exception as e:
                print(f"Error queuing command: {e}")

    def setup_ui(self):
        frame = ttk.Frame(self, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.lbl_status = ttk.Label(frame, text="Status: Disconnected", foreground="red")
        self.lbl_status.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        # Feedback row (Row 1)
        self.lbl_pitch = ttk.Label(frame, text="Pitch: 0.0")
        self.lbl_pitch.grid(row=1, column=0, padx=5, pady=5)
        self.lbl_roll = ttk.Label(frame, text="Roll: 0.0")
        self.lbl_roll.grid(row=1, column=1, padx=5, pady=5)
        self.lbl_yaw = ttk.Label(frame, text="Yaw: 0.0")
        self.lbl_yaw.grid(row=1, column=2, padx=5, pady=5)
        
        # Row 2: UP / ZOOM IN / LEFT
        btn_up = ttk.Button(frame, text="↑ Pitch Up")
        btn_up.grid(row=2, column=0, pady=5)
        btn_up.bind("<ButtonPress-1>", lambda e: self.on_press("pitch", "up"))
        btn_up.bind("<ButtonRelease-1>", lambda e: self.on_release("pitch"))

        btn_zoom_in = ttk.Button(frame, text="🔍 Zoom In")
        btn_zoom_in.grid(row=2, column=1, pady=5)
        btn_zoom_in.bind("<ButtonPress-1>", lambda e: self.on_press("zoom", "in"))
        btn_zoom_in.bind("<ButtonRelease-1>", lambda e: self.on_release("zoom"))
        
        btn_left = ttk.Button(frame, text="← Yaw Left")
        btn_left.grid(row=2, column=2, pady=5)
        btn_left.bind("<ButtonPress-1>", lambda e: self.on_press("yaw", "left"))
        btn_left.bind("<ButtonRelease-1>", lambda e: self.on_release("yaw"))

        # Row 3: DOWN / ZOOM OUT / RIGHT
        btn_down = ttk.Button(frame, text="↓ Pitch Down")
        btn_down.grid(row=3, column=0, pady=5)
        btn_down.bind("<ButtonPress-1>", lambda e: self.on_press("pitch", "down"))
        btn_down.bind("<ButtonRelease-1>", lambda e: self.on_release("pitch"))

        btn_zoom_out = ttk.Button(frame, text="🔎 Zoom Out")
        btn_zoom_out.grid(row=3, column=1, pady=5)
        btn_zoom_out.bind("<ButtonPress-1>", lambda e: self.on_press("zoom", "out"))
        btn_zoom_out.bind("<ButtonRelease-1>", lambda e: self.on_release("zoom"))
        
        btn_right = ttk.Button(frame, text="→ Yaw Right")
        btn_right.grid(row=3, column=2, pady=5)
        btn_right.bind("<ButtonPress-1>", lambda e: self.on_press("yaw", "right"))
        btn_right.bind("<ButtonRelease-1>", lambda e: self.on_release("yaw"))

        # Row 4: SPEEDS
        ttk.Label(frame, text="Speed (0-255):").grid(row=4, column=0, pady=(10, 0))
        self.pitch_speed = tk.StringVar(value="50")
        ttk.Entry(frame, textvariable=self.pitch_speed, width=5).grid(row=5, column=0)

        ttk.Label(frame, text="Speed (0-8):").grid(row=4, column=1, pady=(10, 0))
        self.zoom_speed = tk.StringVar(value="4")
        ttk.Entry(frame, textvariable=self.zoom_speed, width=5).grid(row=5, column=1)
        
        ttk.Label(frame, text="Speed (0-255):").grid(row=4, column=2, pady=(10, 0))
        self.yaw_speed = tk.StringVar(value="50")
        ttk.Entry(frame, textvariable=self.yaw_speed, width=5).grid(row=5, column=2)

    def on_press(self, axis, dir_):
        self.active_cmds[axis] = dir_
        self.send_movement(axis)

    def on_release(self, axis):
        self.active_cmds[axis] = None
        self.send_ws_command({"action": "stop", "axis": axis})

    def update_movement(self):
        # Refresh the active movement command every 100ms
        for axis, dir_ in self.active_cmds.items():
            if dir_:
                self.send_movement(axis)
        self.after(100, self.update_movement)

    def send_movement(self, axis):
        dir_ = self.active_cmds[axis]
        if not dir_: return
        
        cur_angle = self.angles.get(axis, 0.0)
        limits = LIMITS.get(axis, [-180, 180])
        
        # Check limits before sending command
        if axis != "zoom":
            if dir_ in ["up", "right"]:  
                if cur_angle >= limits[1]: 
                    self.send_ws_command({"action": "stop", "axis": axis})
                    return
            if dir_ in ["down", "left"]: 
                if cur_angle <= limits[0]: 
                    self.send_ws_command({"action": "stop", "axis": axis})
                    return
            
        if axis == 'pitch':
            speed_str = self.pitch_speed.get()
        elif axis == 'yaw':
            speed_str = self.yaw_speed.get()
        else:
            speed_str = self.zoom_speed.get()

        try:
            speed = int(speed_str)
        except ValueError:
            speed = 50 if axis != 'zoom' else 4
            
        self.send_ws_command({
            "action": "move",
            "axis": axis,
            "dir": dir_,
            "speed": speed,
            "timeout": 0.15  # 150ms timeout on the server side
        })

if __name__ == "__main__":
    app = GimbalGUI()
    app.mainloop()
