"""Collect synchronized XC330 positions and D435i data for the XAC device.

Keys: s start/stop recording, q quit.
The iPhone records independently; copy its CSV files into the resulting session later.
"""

from __future__ import annotations

import csv
import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
from terminal_input import keyboard

from xac_config import DATA_ROOT, DXL_BAUDRATE, DXL_IDS, DXL_PORT, load_width_calibration, positions_to_width

PROTOCOL = 2.0
ADDR_TORQUE, ADDR_POSITION = 64, 132
LOG_HZ = 10.0


def signed(value: int, bits: int) -> int:
    limit = 1 << (bits - 1)
    return value - (1 << bits) if value >= limit else value


class Collector:
    def __init__(self) -> None:
        self.port = PortHandler(DXL_PORT)
        self.packet = PacketHandler(PROTOCOL)
        self.io_lock = threading.Lock()
        self.log_lock = threading.RLock()
        self.log_error = None
        self.recording = False
        self.session: Path | None = None
        self.camera: subprocess.Popen | None = None
        self.log_stream = None
        self.log_writer = None
        self.calibration = None
        self.running = True

    def check(self, result: int, error: int, action: str) -> None:
        if result != COMM_SUCCESS:
            raise RuntimeError(f"{action}: {self.packet.getTxRxResult(result)}")
        if error:
            raise RuntimeError(f"{action}: {self.packet.getRxPacketError(error)}")

    def write1(self, motor: int, address: int, value: int) -> None:
        result, error = self.packet.write1ByteTxRx(self.port, motor, address, value)
        self.check(result, error, f"写入 ID={motor}")

    def read4(self, motor: int, address: int) -> int:
        value, result, error = self.packet.read4ByteTxRx(self.port, motor, address)
        self.check(result, error, f"读取 ID={motor}")
        return signed(value, 32)

    def connect(self) -> None:
        if not self.port.openPort() or not self.port.setBaudRate(DXL_BAUDRATE):
            raise RuntimeError(f"无法打开 {DXL_PORT} @ {DXL_BAUDRATE}")
        with self.io_lock:
            for motor in DXL_IDS:
                # The operator moves the gripper by hand. Torque must remain off;
                # changing operating mode here could unexpectedly move the motor.
                self.write1(motor, ADDR_TORQUE, 0)

    def new_session(self) -> Path:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        index = max([int(p.name) for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.isdigit()] or [0]) + 1
        path = DATA_ROOT / f"{index:03d}"
        path.mkdir()
        return path

    def start_recording(self) -> None:
        if self.recording:
            return
        self.calibration = load_width_calibration()
        self.session = self.new_session()
        try:
            self.log_stream = (self.session / "gripper_width_log.csv").open("w", encoding="utf-8-sig", newline="")
            self.log_writer = csv.writer(self.log_stream)
            self.log_writer.writerow(["computer_time_s", "m_left", "m_right", "width_m", "d_left", "d_right", "sync_error"])
            camera_dir = self.session / "d435i"
            self.camera = subprocess.Popen([sys.executable, str(Path(__file__).with_name("record_435i.py")), "--output-dir", str(camera_dir)], stdin=subprocess.PIPE)
            deadline = time.monotonic() + 30.0
            while True:
                if self.camera.poll() is not None:
                    raise RuntimeError(f"D435i recorder exited with code {self.camera.returncode}")
                try:
                    events = json.loads((camera_dir / "recording_events.json").read_text(encoding="utf-8"))
                    if "camera_ready" in events.get("events", {}):
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("D435i did not save its first RGB-D frame within 30 seconds")
                time.sleep(0.05)
        except BaseException:
            if self.camera is not None:
                self.camera.terminate()
                try:
                    self.camera.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.camera.kill()
                    self.camera.wait()
                if self.camera.stdin:
                    self.camera.stdin.close()
                self.camera = None
            if self.log_stream is not None:
                self.log_stream.close()
                self.log_stream = None
            self.log_writer = None
            self.session = None
            raise
        self.recording = True
        (self.session / "session.json").write_text(json.dumps({"started_at": datetime.now().astimezone().isoformat(), "port": DXL_PORT, "baudrate": DXL_BAUDRATE}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"开始采集：{self.session}（现在启动 iPhone 记录并执行一次 shake）")

    def stop_recording(self) -> None:
        with self.log_lock:
            self._stop_recording()

    def _stop_recording(self) -> None:
        if not self.recording:
            return
        self.recording = False
        if self.camera and self.camera.stdin:
            try:
                self.camera.stdin.write(b"stop\n"); self.camera.stdin.flush()
                self.camera.wait(timeout=20)
            except Exception:
                self.camera.terminate()
                try:
                    self.camera.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.camera.kill()
                    self.camera.wait()
            finally:
                self.camera.stdin.close()
        self.camera = None
        if self.log_stream:
            self.log_stream.flush(); self.log_stream.close()
        self.log_stream = None
        self.log_writer = None
        print(
            f"采集结束：{self.session}\n"
            "下一步：导出 iPhone 数据，然后运行：\n"
            f"  python prepare_session.py \"{self.session}\" <odometry.csv> <imu.csv>"
        )

    def log_loop(self) -> None:
        while self.running:
            begin = time.monotonic()
            with self.log_lock:
                if self.recording and self.log_writer:
                    try:
                        with self.io_lock:
                            left, right = (self.read4(m, ADDR_POSITION) for m in DXL_IDS)
                        width, d_left, d_right = positions_to_width(left, right, self.calibration)
                        self.log_writer.writerow([f"{time.time():.9f}", left, right, f"{width:.9f}", f"{d_left:.9f}", f"{d_right:.9f}", f"{abs(d_left-d_right):.9f}"])
                        self.log_stream.flush()
                    except Exception as exc:
                        self.log_error = exc
                        self.running = False
            time.sleep(max(0.0, 1.0 / LOG_HZ - (time.monotonic() - begin)))

    def run(self) -> None:
        worker = None
        try:
            with keyboard() as read_key:
                self.connect()
                worker = threading.Thread(target=self.log_loop, daemon=True)
                worker.start()
                print("手动带动夹爪完成开合；按 s=开始/停止采集，q=退出")
                while self.running:
                    if self.recording and self.camera and self.camera.poll() is not None:
                        raise RuntimeError(f"D435i recorder exited with code {self.camera.returncode}")
                    key = read_key().lower()
                    if key == "s":
                        with self.log_lock:
                            self.stop_recording() if self.recording else self.start_recording()
                    elif key == "q":
                        self.running = False
                    time.sleep(0.02)
                if self.log_error is not None:
                    raise RuntimeError("Gripper logging failed") from self.log_error
        finally:
            self.running = False
            try:
                self.stop_recording()
            finally:
                if worker is not None:
                    worker.join(timeout=5)
                try:
                    with self.io_lock:
                        for motor in DXL_IDS:
                            try:
                                self.write1(motor, ADDR_TORQUE, 0)
                            except Exception:
                                pass
                finally:
                    self.port.closePort()


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        Collector().run()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"退出：{exc}")
        raise SystemExit(1)
