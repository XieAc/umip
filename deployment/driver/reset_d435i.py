"""Hardware-reset the D435i and check that color and depth frames arrive.

Use when the policy runner reports "RGB-D frame did not arrive", e.g. after
the USB hub was re-plugged (the depth stream can stall while color still works).
Close every program that uses the camera first.

    python driver/reset_d435i.py
"""
import sys
import time

import pyrealsense2 as rs


def count_frames(stream, fmt, n=30):
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(stream, 640, 480, fmt, 30)
    pipe.start(cfg)
    got = 0
    try:
        for _ in range(n):
            try:
                pipe.wait_for_frames(500)
                got += 1
            except RuntimeError:
                pass
    finally:
        pipe.stop()
    return got


def main():
    devices = rs.context().query_devices()
    if not len(devices):
        print("No RealSense device found: check the USB cable/hub.")
        return 1
    print("Resetting", devices[0].get_info(rs.camera_info.name), "...")
    devices[0].hardware_reset()
    for _ in range(20):
        time.sleep(1)
        try:
            if len(rs.context().query_devices()):
                break
        except RuntimeError:
            pass
    else:
        print("Camera did not come back within 20 s: re-plug it.")
        return 1
    time.sleep(2)
    color = count_frames(rs.stream.color, rs.format.bgr8)
    depth = count_frames(rs.stream.depth, rs.format.z16)
    print(f"color {color}/30, depth {depth}/30")
    if color < 25 or depth < 25:
        print("Still unstable: re-plug the camera directly into a USB3 port (not through the hub) and retry.")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
