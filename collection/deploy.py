"""Portable XAC collection, processing and UR5 replay entry point."""

import argparse
import subprocess
import sys
from pathlib import Path


COMMANDS = {
    "collect": "gripper_grasp_control.py",
    "prepare": "prepare_session.py",
    "build": "build_dataset.py",
    "export": "export_umi_zarr.py",
    "replay": "replay_xac_ur5.py",
    "check": "check_pipeline.py",
    "calibrate-width": "calibrate_gripper_width.py",
    "calibrate-tig": "calibrate_tig.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script = Path(__file__).resolve().parent / COMMANDS[args.command]
    return subprocess.call([sys.executable, str(script), *args.arguments])


if __name__ == "__main__":
    raise SystemExit(main())
