"""Start and inspect the local Intel RealSense D435i.

This is a small command-line front end for :mod:`driver.d435i`. It uses the
same 640x480 RGB-D profile as the collection pipeline and can run headless,
which is useful on the UR5 computer. The replay command uses the same class
with ``--enable-d435i`` and drains frames in the background; recorded camera
frames are not used by the open-loop replay controller.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

try:
    from driver.d435i import D435iConfig, D435iConnectionError, D435iStream
except ModuleNotFoundError:  # direct ``python driver/start_d435i.py`` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from driver.d435i import D435iConfig, D435iConnectionError, D435iStream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--fps", type=int, default=30,
        help="requested hardware RGB-D FPS; 15 Hz is used if 30 is unavailable",
    )
    parser.add_argument("--serial", help="RealSense serial number; otherwise select the first D435i")
    parser.add_argument("--no-imu", action="store_true", help="start only color/depth streams")
    parser.add_argument("--no-preview", action="store_true", help="headless startup check; Ctrl+C stops it")
    parser.add_argument("--captures-dir", type=Path, default=Path("captures"))
    return parser.parse_args()


def run_preview(stream: D435iStream, captures_dir: Path) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise D435iConnectionError(
            "Preview requires opencv-python and numpy; use --no-preview instead"
        ) from exc

    window = "Intel RealSense D435i"
    captures_dir.mkdir(parents=True, exist_ok=True)
    print("Press 's' to save aligned RGB/depth PNGs; press 'q' or Esc to stop.", flush=True)
    try:
        while True:
            _frames, color_frame, depth_frame = stream.aligned_frames(1000)
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            depth_display = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET
            )
            cv2.imshow(window, np.hstack((color, depth_display)))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return
            if key == ord("s"):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                color_path = captures_dir / f"color_{stamp}.png"
                depth_path = captures_dir / f"depth_{stamp}.png"
                cv2.imwrite(str(color_path), color)
                cv2.imwrite(str(depth_path), depth)
                print(f"Saved: {color_path.name}, {depth_path.name}", flush=True)
    finally:
        cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    stream = D435iStream(
        D435iConfig(
            width=args.width,
            height=args.height,
            fps=args.fps,
            serial=args.serial,
            enable_imu=not args.no_imu,
        )
    )
    try:
        info = stream.start(drain=False)
        print(info, flush=True)
        if args.no_preview:
            print("D435i is running headlessly; press Ctrl+C to stop.", flush=True)
            while True:
                stream.wait_for_frames(1000)
                time.sleep(0.01)
        else:
            run_preview(stream, args.captures_dir)
        return 0
    except KeyboardInterrupt:
        return 130
    except (D435iConnectionError, RuntimeError) as exc:
        print(f"D435i start/read failed: {exc}")
        return 1
    finally:
        stream.stop()
        print("D435i stopped.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
