#!/usr/bin/env python3
"""Probe raw MJPEG z /dev/video0 (diagnoza szarego ekranu)."""
import sys
import time

import cv2
import numpy as np

dev = sys.argv[1] if len(sys.argv) > 1 else '/dev/video0'

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(None)
    pipe = Gst.parse_launch(
        f'v4l2src device={dev} io-mode=2 ! '
        'image/jpeg,width=1920,height=1080,framerate=30/1 ! '
        'jpegparse ! appsink name=s emit-signals=true max-buffers=1 drop=true'
    )
    sink = pipe.get_by_name('s')
    chunks: list[bytes] = []

    def on_sample(_sink):
        sample = _sink.emit('pull-sample')
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if ok:
            chunks.append(bytes(info.data))
        buf.unmap(info)
        return Gst.FlowReturn.EOS

    sink.connect('new-sample', on_sample)
    pipe.set_state(Gst.State.PLAYING)
    time.sleep(2)
    pipe.set_state(Gst.State.NULL)
    if chunks:
        arr = np.frombuffer(chunks[0], dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            print(f'gst_mjpeg {img.shape} mean={img.mean():.1f} std={img.std():.1f}')
        else:
            print('gst_mjpeg decode failed')
except Exception as exc:
    print(f'gst_mjpeg error: {exc}')

cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
for _ in range(15):
    ok, frame = cap.read()
    if ok and frame is not None:
        print(f'opencv {frame.shape} mean={frame.mean():.1f} std={frame.std():.1f}')
        break
cap.release()
