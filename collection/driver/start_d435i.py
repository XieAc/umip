"""Start an Intel RealSense D435i and preview color/depth streams.

Controls:
    q or Esc: quit
    s: save the current color and depth frames under ./captures
"""

from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs


WIDTH = 1280
HEIGHT = 720
FPS = 30
WINDOW_NAME = "Intel RealSense D435i"


def main() -> None:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    align = rs.align(rs.stream.color)
    captures_dir = Path(__file__).resolve().parent / "captures"

    try:
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"D435i started: {WIDTH}x{HEIGHT}@{FPS}, depth scale={depth_scale} m")
        print("Press 's' to save frames, 'q' or Esc to quit.")

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Convert depth to an 8-bit color visualization for display only.
            depth_display = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET,
            )
            preview = np.hstack((color_image, depth_display))
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                captures_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                color_path = captures_dir / f"color_{stamp}.png"
                depth_path = captures_dir / f"depth_{stamp}.png"
                cv2.imwrite(str(color_path), color_image)
                cv2.imwrite(str(depth_path), depth_image)
                print(f"Saved: {color_path.name}, {depth_path.name}")
    except RuntimeError as exc:
        print(f"Unable to start/read D435i: {exc}")
        print("Check the USB connection and that pyrealsense2 is installed.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("D435i stopped.")


if __name__ == "__main__":
    main()
