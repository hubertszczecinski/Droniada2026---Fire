from __future__ import annotations

import argparse
import os
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk


class SnapshotBrowser:
    def __init__(self, root: tk.Tk, snapshots_dir: str) -> None:
        self.root = root
        self.snapshots_dir = snapshots_dir
        self.paths: list[str] = []
        self.idx = 0
        self.photo: ImageTk.PhotoImage | None = None
        self.last_mtime = 0.0
        self._configure_after: str | None = None

        self.root.title("droniada_snapshots")
        self.root.geometry("1400x900")

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")

        self.prev_btn = ttk.Button(top, text="← Poprzednie", command=self.prev)
        self.prev_btn.pack(side="left")
        self.next_btn = ttk.Button(top, text="Następne →", command=self.next)
        self.next_btn.pack(side="left", padx=(8, 12))

        self.var = tk.StringVar(value="")
        self.combo = ttk.Combobox(top, textvariable=self.var, state="readonly", width=70)
        self.combo.pack(side="left", fill="x", expand=True)
        self.combo.bind("<<ComboboxSelected>>", self.on_select)

        self.info = ttk.Label(top, text="Brak migawek…")
        self.info.pack(side="left", padx=10)

        self.canvas = tk.Label(root, background="#202020")
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.root.bind("<Left>", lambda _e: self.prev())
        self.root.bind("<Right>", lambda _e: self.next())
        self.root.bind("a", lambda _e: self.prev())
        self.root.bind("d", lambda _e: self.next())
        self.root.bind("<Configure>", self._on_configure)

        self.root.update_idletasks()
        self.refresh_files()
        self.root.after(600, self.poll)

    def poll(self) -> None:
        self.refresh_files()
        self.root.after(600, self.poll)

    def refresh_files(self) -> None:
        if not os.path.isdir(self.snapshots_dir):
            return
        mtime = os.path.getmtime(self.snapshots_dir)
        if mtime == self.last_mtime and self.paths:
            return
        self.last_mtime = mtime
        files = sorted(
            os.path.join(self.snapshots_dir, f)
            for f in os.listdir(self.snapshots_dir)
            if f.endswith("_dashboard.png")
        )
        if not files:
            self.paths = []
            self.combo["values"] = []
            self.info.configure(text="Brak migawek…")
            return
        prev_path = self.paths[self.idx] if self.paths and self.idx < len(self.paths) else None
        self.paths = files
        names = [os.path.basename(p) for p in self.paths]
        self.combo["values"] = names
        if prev_path in self.paths:
            self.idx = self.paths.index(prev_path)
        else:
            self.idx = len(self.paths) - 1
        self.show_current()

    def _display_area(self) -> tuple[int, int]:
        max_w = max(400, self.root.winfo_width() - 40)
        max_h = max(300, self.root.winfo_height() - 120)
        return max_w, max_h

    def _on_configure(self, event: tk.Event) -> None:
        if event.widget is not self.root or not self.paths:
            return
        if self._configure_after is not None:
            self.root.after_cancel(self._configure_after)
        self._configure_after = self.root.after(150, self.show_current)

    def show_current(self) -> None:
        if not self.paths:
            return
        self.idx = max(0, min(self.idx, len(self.paths) - 1))
        path = self.paths[self.idx]
        try:
            pil = Image.open(path)
        except OSError:
            self.info.configure(text=f"Nie mogę otworzyć: {os.path.basename(path)}")
            return
        orig_w, orig_h = pil.size
        max_w, max_h = self._display_area()
        scale = min(max_w / orig_w, max_h / orig_h)
        disp_w = max(1, int(orig_w * scale))
        disp_h = max(1, int(orig_h * scale))
        if (disp_w, disp_h) != (orig_w, orig_h):
            pil = pil.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.configure(image=self.photo)
        self.combo.current(self.idx)
        self.info.configure(
            text=f"{self.idx + 1}/{len(self.paths)}  "
            f"{disp_w}x{disp_h}  (oryg. {orig_w}x{orig_h})"
        )

    def prev(self) -> None:
        if not self.paths:
            return
        self.idx = (self.idx - 1) % len(self.paths)
        self.show_current()

    def next(self) -> None:
        if not self.paths:
            return
        self.idx = (self.idx + 1) % len(self.paths)
        self.show_current()

    def on_select(self, _event: object) -> None:
        if not self.paths:
            return
        i = self.combo.current()
        if i >= 0:
            self.idx = i
            self.show_current()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots-dir", required=True)
    args = ap.parse_args()
    root = tk.Tk()
    SnapshotBrowser(root, args.snapshots_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
