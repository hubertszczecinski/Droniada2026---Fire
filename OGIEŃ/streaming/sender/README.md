# UDP sender (drone / Jetson)

Pipeline depends on what you control on the drone side. This folder contains **templates**.

## H.264 from USB camera (Jetson)

```bash
./sender/jetson_udp_send_h264.sh
```

Set in `config/stream.env`:
- `UDP_DEST_HOST=192.168.100.249` (Gigabyte)
- `UDP_DEST_PORT=5600`

## Custom drone pipeline

If the drone already sends RTP on a known port, on Gigabyte:

```bash
./gigabyte/start_lastmile.sh
```

Match in `stream.env`:
- `RTP_CODEC=H264` or `H265`
- `UDP_PORT` — same as sender
- `RTP_PAYLOAD` — PT from SDP (often 96)

## Local test (Mac → Gigabyte)

```bash
gst-launch-1.0 videotestsrc is-live=true ! \
  video/x-raw,width=1280,height=720,framerate=30/1 ! \
  videoconvert ! x264enc tune=zerolatency ! rtph264pay pt=96 ! \
  udpsink host=192.168.100.249 port=5600
```
