"""
Intel RealSense D435i RGB-D 录制程序。

职责边界：
    1. 只负责 D435i 相机采集。
    2. 不读取 iPhone 位姿。
    3. 不读取 DYNAMIXEL 串口。
    4. 不控制夹爪、不控制 UR5、不做训练。

输出结构：
    d435i/
    ├── rgb/
    │   ├── 000000.png
    │   └── ...
    ├── depth/
    │   ├── 000000.png
    │   └── ...
    ├── timestamps.csv
    ├── imu.csv
    ├── recording_events.json
    ├── intrinsics.json
    └── meta.json

停止方式：
    gripper_grasp_control.py 会向本程序 stdin 写入 stop，或者关闭 stdin。
    本程序收到后会停止采集、关闭 pipeline、写入 meta.json 并退出。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path


DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 10
MAX_RGBD_SAVE_FPS = 10
IMU_OUTPUT_FPS = 10.0
PREVIEW_WINDOW_NAME = "D435i RGB Preview"


class RgbPreview:
    """在采集线程刷新预览；关闭窗口或 GUI 不可用时继续录制。"""

    def __init__(self, cv2, enabled: bool):
        self.cv2 = cv2
        self.active = enabled
        self.window_created = False

    def poll(self) -> None:
        if not self.active or not self.window_created:
            return
        try:
            self.cv2.waitKey(1)
            if self.cv2.getWindowProperty(PREVIEW_WINDOW_NAME, self.cv2.WND_PROP_VISIBLE) < 1:
                self.active = False
        except self.cv2.error as exc:
            self._disable(exc)

    def update(self, color_image, saved_frames: int, save_fps: float, hardware_fps: float) -> None:
        if not self.active:
            return
        try:
            if not self.window_created:
                self.cv2.namedWindow(PREVIEW_WINDOW_NAME, self.cv2.WINDOW_NORMAL)
                self.window_created = True
                height, width = color_image.shape[:2]
                self.cv2.resizeWindow(PREVIEW_WINDOW_NAME, width, height)
                print("[record_435i] RGB实时预览已开启；停止采集请回到终端按 s。", flush=True)
            preview_image = color_image.copy()
            height, width = color_image.shape[:2]
            self.cv2.putText(
                preview_image,
                f"LIVE {hardware_fps:g} Hz | SAVE {save_fps:g} Hz | Frames: {saved_frames}",
                (12, 28), self.cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 255), 2, self.cv2.LINE_AA,
            )
            self.cv2.imshow(PREVIEW_WINDOW_NAME, preview_image)
        except self.cv2.error as exc:
            self._disable(exc)

    def _disable(self, exc) -> None:
        self.active = False
        self.close()
        print(f"[record_435i] 预览不可用，继续录制：{exc}", file=sys.stderr, flush=True)

    def close(self) -> None:
        if self.window_created:
            try:
                self.cv2.destroyWindow(PREVIEW_WINDOW_NAME)
                self.cv2.waitKey(1)
            except self.cv2.error:
                pass
            self.window_created = False

# 正式数据集落盘统一使用 640x480@10 Hz。D435i RGB-D 硬件流通常不提供
# 10 Hz profile，因此硬件打开 color/depth 共同支持的最高帧率，再按
# RGB-D 时间戳降采样保存到 10 Hz。

# 子进程通过 stdout 向 gripper_grasp_control.py 发送结构化事件。
# 普通诊断输出不使用这个前缀，父进程只会对带前缀的 JSON 做状态解析。
D435I_EVENT_PREFIX = "D435I_EVENT "

# 相机真正保存第一张 RGB-D 后才开始建立静止基线；用户看到“相机已就绪”后应保持静止。
SHAKE_BASELINE_DURATION_SECONDS = 1.0
SHAKE_SMOOTH_WINDOW_SECONDS = 0.12

# 连续超过开始阈值至少这段时间才认为 shake 开始，过滤单个 IMU 毛刺。
SHAKE_MIN_ACTIVE_DURATION_SECONDS = 0.06

# shake 后连续安静这段时间，才认为整次 shake 已完全结束。
SHAKE_END_STILL_DURATION_SECONDS = 0.35
VALID_DATA_DELAY_AFTER_SHAKE_SECONDS = 1.5

# 开始阈值同时受静止基线/MAD 和绝对陀螺仪强度约束。
#
# 2026-08-02 的五次专用采样中，平滑 gyro 模长的最弱有效 shake 峰值为
# 3.709 rad/s，而最大静止噪声上界（baseline + 8*MAD）仅为 0.508 rad/s。
# 因此把绝对开始下限设为 3.0 rad/s：它保留约 19% 的最弱样本余量，同时比
# 已测静止噪声上界高约 5.9 倍。只有接近正常 shake 力度或更强的动作才会触发。
SHAKE_START_ABSOLUTE_MIN_RAD_S = 3.0
SHAKE_START_MIN_DELTA_RAD_S = 0.03
SHAKE_END_MIN_DELTA_RAD_S = 0.015
SHAKE_START_MAD_MULTIPLIER = 8.0
SHAKE_END_MAD_MULTIPLIER = 4.0


class D435iConnectionError(RuntimeError):
    """D435i 未连接、依赖缺失、USB 不满足要求或 stream 启动失败。"""


class D435iImageWriteError(RuntimeError):
    """D435i 已经正常出帧，但 RGB/depth 图片保存失败。"""


class ImuRateLimiter:
    """将 accel/gyro 分别按时间戳降采样到目标频率。"""

    def __init__(self, target_fps: float):
        if target_fps <= 0:
            raise ValueError("IMU 目标频率必须大于 0。")
        self.target_fps = float(target_fps)
        self.interval_seconds = 1.0 / self.target_fps
        self.next_emit_time_by_stream: dict[str, float] = {}
        self.written_counts = {"accel": 0, "gyro": 0}
        self.skipped_counts = {"accel": 0, "gyro": 0}

    def should_emit(self, stream_name: str, sample_time_s: float) -> bool:
        self.written_counts.setdefault(stream_name, 0)
        self.skipped_counts.setdefault(stream_name, 0)
        next_emit_time_s = self.next_emit_time_by_stream.get(stream_name)
        if next_emit_time_s is None or sample_time_s < next_emit_time_s - 2.0 * self.interval_seconds:
            self.next_emit_time_by_stream[stream_name] = sample_time_s + self.interval_seconds
            self.written_counts[stream_name] += 1
            return True

        # 允许 5% 周期误差，避免硬件时间戳量化导致长期少一帧。
        tolerance_seconds = 0.05 * self.interval_seconds
        if sample_time_s + tolerance_seconds < next_emit_time_s:
            self.skipped_counts[stream_name] += 1
            return False

        while next_emit_time_s <= sample_time_s + tolerance_seconds:
            next_emit_time_s += self.interval_seconds
        self.next_emit_time_by_stream[stream_name] = next_emit_time_s
        self.written_counts[stream_name] += 1
        return True


class RealtimeShakeDetector:
    """
    使用 D435i gyro 三轴模长实时检测本次录制的第一次 shake。

    状态顺序：
        waiting_camera -> collecting_baseline -> armed -> shaking
        -> waiting_valid_data -> collecting_valid_data

    设计约束：
        - 必须在第一张 RGB-D 成功落盘后才开始静止基线，避免“IMU已启动但RGB还没出图”。
        - 只检测第一次 shake；进入 collecting_valid_data 后忽略任务中的正常快速运动。
        - 所有事件既写入 recording_events.json，也通过 stdout 发给 control 程序。
        - RGB、Depth 和降采样后的10 Hz IMU 从录制开始一直保存；
          valid_data_started 只是训练起点标记。
    """

    def __init__(self, events_path: Path):
        self.events_path = events_path
        self.state = "waiting_camera"

        self.smooth_samples = deque()
        self.smooth_sum = 0.0
        self.baseline_started_time_s = None
        self.baseline_values = []
        self.baseline_signal_rad_s = None
        self.baseline_mad_rad_s = None
        self.start_threshold_rad_s = None
        self.end_threshold_rad_s = None

        self.shake_candidate_start_time_s = None
        self.shake_start_time_s = None
        self.shake_peak_time_s = None
        self.shake_peak_signal_rad_s = None
        self.shake_end_candidate_time_s = None
        self.shake_end_time_s = None
        self.valid_data_start_time_s = None

        self.event_data = {
            "version": 1,
            "detector": "realtime_d435i_gyro_norm_first_shake",
            "state": self.state,
            "parameters": {
                "baseline_duration_seconds": SHAKE_BASELINE_DURATION_SECONDS,
                "smooth_window_seconds": SHAKE_SMOOTH_WINDOW_SECONDS,
                "min_active_duration_seconds": SHAKE_MIN_ACTIVE_DURATION_SECONDS,
                "end_still_duration_seconds": SHAKE_END_STILL_DURATION_SECONDS,
                "valid_data_delay_after_shake_seconds": VALID_DATA_DELAY_AFTER_SHAKE_SECONDS,
                "start_absolute_min_rad_s": SHAKE_START_ABSOLUTE_MIN_RAD_S,
                "start_min_delta_rad_s": SHAKE_START_MIN_DELTA_RAD_S,
                "end_min_delta_rad_s": SHAKE_END_MIN_DELTA_RAD_S,
                "start_mad_multiplier": SHAKE_START_MAD_MULTIPLIER,
                "end_mad_multiplier": SHAKE_END_MAD_MULTIPLIER,
            },
            "events": {},
        }

    def _write_event_file(self) -> None:
        """用临时文件替换的方式更新事件JSON，降低中途断电留下半个JSON的概率。"""

        self.event_data["state"] = self.state
        self.event_data["updated_at"] = datetime.now().isoformat(timespec="milliseconds")
        temporary_path = self.events_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(self.event_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.events_path)

    def _emit_event(self, event_name: str, system_time_s: float, **details) -> None:
        """保存事件并向父进程发送一行可解析JSON；事件文件失败不应中断相机录制。"""

        event_payload = {
            "event": event_name,
            "system_time_s": float(system_time_s),
            **details,
        }
        self.event_data["events"][event_name] = event_payload
        try:
            self._write_event_file()
        except OSError as exc:
            print(
                f"[record_435i] recording_events.json 写入失败：{exc}",
                file=sys.stderr,
                flush=True,
            )

        print(
            D435I_EVENT_PREFIX + json.dumps(event_payload, ensure_ascii=False),
            flush=True,
        )

    def on_camera_ready(self, system_time_s: float, first_frame_idx: int) -> None:
        """第一张RGB-D成功保存后调用，开始收集一秒静止gyro基线。"""

        if self.state != "waiting_camera":
            return
        self.state = "collecting_baseline"
        self.baseline_started_time_s = None
        self._emit_event(
            "camera_ready",
            system_time_s,
            first_rgbd_frame_idx=int(first_frame_idx),
        )

    def _append_smoothed_signal(self, system_time_s: float, gyro_norm_rad_s: float) -> float:
        """维护短滑动窗口并返回gyro模长的实时均值。"""

        self.smooth_samples.append((system_time_s, gyro_norm_rad_s))
        self.smooth_sum += gyro_norm_rad_s
        minimum_time_s = system_time_s - SHAKE_SMOOTH_WINDOW_SECONDS
        while self.smooth_samples and self.smooth_samples[0][0] < minimum_time_s:
            _, removed_value = self.smooth_samples.popleft()
            self.smooth_sum -= removed_value
        return self.smooth_sum / max(1, len(self.smooth_samples))

    def _finish_baseline(self, system_time_s: float) -> None:
        """根据静止段中位数和MAD生成本条录制自己的开始/结束阈值。"""

        if not self.baseline_values:
            return
        baseline = float(statistics.median(self.baseline_values))
        absolute_deviations = [abs(value - baseline) for value in self.baseline_values]
        mad = float(statistics.median(absolute_deviations))

        self.baseline_signal_rad_s = baseline
        self.baseline_mad_rad_s = mad
        dynamic_start_threshold = baseline + max(
            SHAKE_START_MIN_DELTA_RAD_S,
            SHAKE_START_MAD_MULTIPLIER * mad,
        )
        # 动态阈值继续适应不同场地的静止噪声；绝对下限则防止在噪声很小的
        # 环境中，一点轻微晃动就被错误认作同步 shake。
        self.start_threshold_rad_s = max(
            dynamic_start_threshold,
            SHAKE_START_ABSOLUTE_MIN_RAD_S,
        )
        self.end_threshold_rad_s = baseline + max(
            SHAKE_END_MIN_DELTA_RAD_S,
            SHAKE_END_MAD_MULTIPLIER * mad,
        )
        # 确保回落阈值严格低于开始阈值，形成滞回，避免阈值附近来回跳动。
        self.end_threshold_rad_s = min(
            self.end_threshold_rad_s,
            baseline + 0.8 * (self.start_threshold_rad_s - baseline),
        )
        self.state = "armed"
        self._emit_event(
            "shake_armed",
            system_time_s,
            baseline_signal_rad_s=baseline,
            baseline_mad_rad_s=mad,
            start_threshold_rad_s=self.start_threshold_rad_s,
            end_threshold_rad_s=self.end_threshold_rad_s,
        )

    def on_gyro_sample(self, system_time_s: float, x: float, y: float, z: float) -> None:
        """每个gyro样本调用一次，推进实时shake状态机。"""

        if self.state == "waiting_camera" or self.state == "collecting_valid_data":
            return

        gyro_norm_rad_s = math.sqrt(x * x + y * y + z * z)
        smoothed_signal = self._append_smoothed_signal(system_time_s, gyro_norm_rad_s)

        if self.state == "collecting_baseline":
            if self.baseline_started_time_s is None:
                self.baseline_started_time_s = system_time_s
            self.baseline_values.append(smoothed_signal)
            if system_time_s - self.baseline_started_time_s >= SHAKE_BASELINE_DURATION_SECONDS:
                self._finish_baseline(system_time_s)
            return

        if self.state == "armed":
            if smoothed_signal >= self.start_threshold_rad_s:
                if self.shake_candidate_start_time_s is None:
                    self.shake_candidate_start_time_s = system_time_s
                    self.shake_peak_time_s = system_time_s
                    self.shake_peak_signal_rad_s = smoothed_signal
                elif smoothed_signal > self.shake_peak_signal_rad_s:
                    self.shake_peak_time_s = system_time_s
                    self.shake_peak_signal_rad_s = smoothed_signal

                active_duration = system_time_s - self.shake_candidate_start_time_s
                if active_duration >= SHAKE_MIN_ACTIVE_DURATION_SECONDS:
                    self.shake_start_time_s = self.shake_candidate_start_time_s
                    self.state = "shaking"
                    self._emit_event(
                        "shake_started",
                        self.shake_start_time_s,
                        detected_at_system_time_s=system_time_s,
                        smoothed_signal_rad_s=smoothed_signal,
                    )
            else:
                self.shake_candidate_start_time_s = None
                self.shake_peak_time_s = None
                self.shake_peak_signal_rad_s = None
            return

        if self.state == "shaking":
            if (
                self.shake_peak_signal_rad_s is None
                or smoothed_signal > self.shake_peak_signal_rad_s
            ):
                self.shake_peak_time_s = system_time_s
                self.shake_peak_signal_rad_s = smoothed_signal

            if smoothed_signal <= self.end_threshold_rad_s:
                if self.shake_end_candidate_time_s is None:
                    self.shake_end_candidate_time_s = system_time_s
                quiet_duration = system_time_s - self.shake_end_candidate_time_s
                if quiet_duration >= SHAKE_END_STILL_DURATION_SECONDS:
                    self.shake_end_time_s = self.shake_end_candidate_time_s
                    self.valid_data_start_time_s = (
                        self.shake_end_time_s + VALID_DATA_DELAY_AFTER_SHAKE_SECONDS
                    )
                    self.state = "waiting_valid_data"
                    self._emit_event(
                        "shake_ended",
                        self.shake_end_time_s,
                        confirmed_at_system_time_s=system_time_s,
                        shake_start_time_s=self.shake_start_time_s,
                        shake_peak_time_s=self.shake_peak_time_s,
                        shake_peak_signal_rad_s=self.shake_peak_signal_rad_s,
                        valid_data_start_time_s=self.valid_data_start_time_s,
                    )
            else:
                self.shake_end_candidate_time_s = None
            return

        if self.state == "waiting_valid_data":
            if system_time_s >= self.valid_data_start_time_s:
                self.state = "collecting_valid_data"
                self._emit_event(
                    "valid_data_started",
                    self.valid_data_start_time_s,
                    emitted_at_system_time_s=system_time_s,
                    delay_after_shake_seconds=VALID_DATA_DELAY_AFTER_SHAKE_SECONDS,
                )

    def on_recording_stopped(self, system_time_s: float) -> None:
        """无论是否检测到shake，都记录第二次按s对应的停止时刻和最终检测状态。"""

        previous_state = self.state
        self.state = "stopped"
        self._emit_event(
            "recording_stopped",
            system_time_s,
            detector_state_before_stop=previous_state,
            valid_data_started=previous_state == "collecting_valid_data",
        )


def import_realsense_dependencies():
    """
    延迟导入 RealSense 依赖。

    这样即使当前环境暂时没有 D435i，脚本仍然可以通过语法检查；
    只有真正运行录制时才要求 pyrealsense2、cv2、numpy 可用。
    """

    try:
        import cv2
        import numpy as np
        import pyrealsense2 as rs
    except ImportError as exc:
        raise D435iConnectionError(
            "record_435i.py 需要 pyrealsense2、opencv-python、numpy。"
        ) from exc

    return rs, cv2, np


def listen_for_stop_command(stop_event: threading.Event) -> None:
    """
    监听 stdin 停止命令。

    支持：
        stop / q / quit / exit

    如果父进程关闭 stdin，for 循环自然结束，也会触发停止。
    """

    try:
        for raw_line in sys.stdin:
            command = raw_line.strip().lower()
            if command in ("stop", "q", "quit", "exit"):
                stop_event.set()
                return
    finally:
        stop_event.set()


def intrinsics_to_dict(intrinsics) -> dict:
    """把 RealSense intrinsics 对象转成 JSON 可保存的普通字典。"""

    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "model": str(intrinsics.model),
        "coeffs": [float(value) for value in intrinsics.coeffs],
    }


def get_device_info(rs, device) -> dict:
    """
    读取 RealSense 设备基础信息。

    和 camera.py 一样，这里会检查设备名和 USB 连接类型；
    D435i RGB-D 建议使用 USB 3.x，否则高分辨率高帧率容易不稳定。
    """

    name = device.get_info(rs.camera_info.name)
    serial = device.get_info(rs.camera_info.serial_number)
    usb_type = device.get_info(rs.camera_info.usb_type_descriptor)

    return {
        "name": name,
        "serial": serial,
        "usb_type": usb_type,
    }


def prepare_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    """创建 d435i/rgb 和 d435i/depth 输出目录。"""

    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    return rgb_dir, depth_dir


def write_png_unicode_safe(cv2, image_path: Path, image) -> None:
    """
    保存 PNG，并兼容 Windows 中文路径。

    OpenCV 的 cv2.imwrite(str(path), image) 在 Windows 中文路径下可能失败。
    这里先用 cv2.imencode 在内存里编码 PNG，再用 pathlib.Path.write_bytes 写文件；
    Python 自己的文件 API 可以正确处理 Unicode 路径。
    """

    encode_ok, encoded_image = cv2.imencode(".png", image)
    if not encode_ok or encoded_image is None:
        raise D435iImageWriteError(f"PNG 编码失败：{image_path}")

    try:
        image_path.write_bytes(encoded_image.tobytes())
    except OSError as exc:
        raise D435iImageWriteError(f"PNG 写入失败：{image_path}，原因：{exc}") from exc

    if not image_path.exists() or image_path.stat().st_size <= 0:
        raise D435iImageWriteError(f"PNG 文件未正确生成：{image_path}")


def get_next_frame_index(rgb_dir: Path, depth_dir: Path) -> int:
    """
    获取同一 d435i 目录下下一帧编号。

    这样同一个 control 进程里多次开始/停止记录时，不会覆盖旧 PNG；
    新记录会从已有最大编号之后继续写。
    """

    existing_indices = []
    for image_dir in (rgb_dir, depth_dir):
        for image_path in image_dir.glob("*.png"):
            try:
                existing_indices.append(int(image_path.stem))
            except ValueError:
                continue

    if not existing_indices:
        return 0

    return max(existing_indices) + 1


def get_existing_csv_data_row_count(csv_path: Path) -> int:
    """
    统计已有 CSV 的数据行数，不包含表头。

    同一个 control 进程中多次开始/停止记录时，RGB-D 图片会追加编号；
    IMU CSV 也需要继续追加 row index，避免覆盖或重复从 0 开始。
    """

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        row_count = sum(1 for _ in reader)

    return max(0, row_count - 1)


def write_intrinsics_json(
    intrinsics_path: Path,
    color_intrinsics,
    depth_intrinsics,
    depth_scale: float,
) -> None:
    """
    保存相机内参。

    本程序使用 rs.align(rs.stream.color)，因此保存出来的 depth 图已经对齐到 color 坐标系；
    后续如果用 depth 做点云/投影，通常优先使用 color_intrinsics。
    """

    data = {
        "color": intrinsics_to_dict(color_intrinsics),
        "depth_original": intrinsics_to_dict(depth_intrinsics),
        "aligned_depth_to_color": True,
        "depth_scale": depth_scale,
        "depth_unit": "meter = raw_depth_uint16 * depth_scale",
    }
    intrinsics_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_meta_json(meta_path: Path, meta: dict) -> None:
    """保存采集参数、设备信息、开始/结束时间和帧数。"""

    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def find_motion_profile(rs, device, stream_type):
    """
    在 RealSense 设备上查找指定 IMU 流配置。

    D435i 的 accel/gyro 不是普通 video stream，而是 motion stream；
    这里参考 camera.py，从设备 sensor profiles 里选出可用配置。
    """

    candidates = []
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() == stream_type:
                candidates.append(profile.as_motion_stream_profile())

    if not candidates:
        return None

    # D435i 通常最低也只提供 accel 63 Hz / gyro 200 Hz，并没有 10 Hz
    # motion profile。选最低硬件频率可以先降低 USB/回调负载，后面再降采样。
    return min(candidates, key=lambda profile: int(profile.fps()))


def list_video_fps_for_profile(rs, device, stream_type, width: int, height: int, fmt) -> set[int]:
    """列出指定 video stream 在目标分辨率/格式下支持的硬件 FPS。"""

    supported_fps: set[int] = set()
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != stream_type:
                continue
            try:
                video_profile = profile.as_video_stream_profile()
            except RuntimeError:
                continue
            if (
                int(video_profile.width()) == int(width)
                and int(video_profile.height()) == int(height)
                and profile.format() == fmt
            ):
                supported_fps.add(int(profile.fps()))
    return supported_fps


def choose_rgbd_hardware_fps(rs, device, width: int, height: int, target_save_fps: int) -> int:
    """选择 color/depth 共同支持的最高 FPS，用于流畅预览并降采样保存。"""

    color_fps = list_video_fps_for_profile(rs, device, rs.stream.color, width, height, rs.format.bgr8)
    depth_fps = list_video_fps_for_profile(rs, device, rs.stream.depth, width, height, rs.format.z16)
    common_fps = color_fps & depth_fps
    if not common_fps:
        raise D435iConnectionError(
            f"D435i does not expose common color/depth profiles at {width}x{height}."
        )

    usable = sorted(fps for fps in common_fps if fps >= target_save_fps)
    if usable:
        return usable[-1]

    raise D435iConnectionError(
        f"No RGB-D hardware FPS >= target {target_save_fps} Hz at {width}x{height}; "
        f"common profiles: {sorted(common_fps)}"
    )


def configure_d435i_pipeline(rs, width: int, height: int, fps: int):
    """
    配置并启动 D435i pipeline。

    参考 camera.py 的做法：
        - 通过 rs.context() 查找设备。
        - 检查是否为 D435i。
        - 根据 USB 连接类型选择 RGB-D 帧率。
        - 启用 color、depth、accel 和 gyro stream。
    """

    context = rs.context()
    devices = context.query_devices()
    if len(devices) == 0:
        raise D435iConnectionError("No Intel RealSense device found.")

    device = devices[0]
    device_info = get_device_info(rs, device)
    name = device_info["name"]
    serial = device_info["serial"]
    usb_type = device_info["usb_type"]

    if "D435I" not in name.upper():
        print(f"Warning: expected a RealSense D435i, but opened: {name}", flush=True)

    # 数据集保存固定不高于 10 FPS；硬件 FPS 单独选择，避免请求 D435i 不支持的 10Hz profile。
    target_save_fps = min(int(fps), MAX_RGBD_SAVE_FPS)
    hardware_fps = choose_rgbd_hardware_fps(
        rs=rs,
        device=device,
        width=width,
        height=height,
        target_save_fps=target_save_fps,
    )

    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, hardware_fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, hardware_fps)

    accel_profile = find_motion_profile(rs, device, rs.stream.accel)
    gyro_profile = find_motion_profile(rs, device, rs.stream.gyro)
    if accel_profile is None or gyro_profile is None:
        raise D435iConnectionError("This device does not expose both accel and gyro streams.")

    config.enable_stream(rs.stream.accel, accel_profile.format(), accel_profile.fps())
    config.enable_stream(rs.stream.gyro, gyro_profile.format(), gyro_profile.fps())

    pipeline = rs.pipeline()
    frame_queue = rs.frame_queue(1000)
    try:
        pipeline_profile = pipeline.start(config, frame_queue)
    except RuntimeError as exc:
        raise D435iConnectionError(f"Failed to start D435i stream: {exc}") from exc

    align_to_color = rs.align(rs.stream.color)

    depth_sensor = pipeline_profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    color_profile = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = pipeline_profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_intrinsics = color_profile.get_intrinsics()
    depth_intrinsics = depth_profile.get_intrinsics()

    print(f"Opened RealSense device: {name}, serial: {serial}, USB: {usb_type}", flush=True)
    print(
        f"Color stream: hardware {width}x{height}@{hardware_fps} -> saved {target_save_fps} Hz",
        flush=True,
    )
    print(
        f"Depth stream: hardware {width}x{height}@{hardware_fps} -> saved {target_save_fps} Hz, "
        f"depth_scale={depth_scale}",
        flush=True,
    )
    print(
        f"Accel stream: hardware {accel_profile.fps()} Hz -> saved {IMU_OUTPUT_FPS:g} Hz",
        flush=True,
    )
    print(
        f"Gyro stream: hardware {gyro_profile.fps()} Hz -> saved {IMU_OUTPUT_FPS:g} Hz",
        flush=True,
    )

    return {
        "pipeline": pipeline,
        "frame_queue": frame_queue,
        "align": align_to_color,
        "device_info": device_info,
        "depth_scale": depth_scale,
        "color_intrinsics": color_intrinsics,
        "depth_intrinsics": depth_intrinsics,
        "rgbd_fps": target_save_fps,
        "rgbd_hardware_fps": hardware_fps,
        "accel_hardware_fps": int(accel_profile.fps()),
        "gyro_hardware_fps": int(gyro_profile.fps()),
        "imu_output_fps": float(IMU_OUTPUT_FPS),
    }


def sensor_timestamp_to_wall_time_s(sensor_timestamp_ms: float) -> float | None:
    """把 RealSense 传感器时间戳(ms)换算成电脑墙钟秒(拍摄时刻)。

    开启了 global timestamp 的 D435i, ``get_timestamp()`` 返回与主机时钟同步的
    epoch 毫秒(约 1.7e12, 例如 1785636721918.2), 除以 1000 即拍摄时刻的墙钟秒。
    若设备未开启 global timestamp, 返回的是小量级相对时间(无法换算), 返回 None
    表示调用方应回退到出队时刻的 ``system_time_s``。
    """

    if sensor_timestamp_ms is None:
        return None
    try:
        value = float(sensor_timestamp_ms)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 1e11:
        return None
    return value / 1000.0


def write_imu_csv_header_if_needed(csv_file, writer: csv.writer, should_write_header: bool) -> None:
    """按需写入 IMU CSV 表头。"""

    if not should_write_header:
        return

    writer.writerow(
        [
            "imu_row_idx",
            "system_time_ns",
            "system_time_s",
            "sensor_timestamp_ms",
            "capture_system_time_s",
            "frame_number",
            "stream",
            "x",
            "y",
            "z",
            "unit",
            "nearest_rgbd_frame_idx",
        ]
    )
    csv_file.flush()


def write_imu_frame_to_csv(
    rs,
    imu_writer: csv.writer,
    motion_frame,
    imu_row_idx: int,
    latest_rgbd_frame_idx: int | None,
    rate_limiter: ImuRateLimiter,
    shake_detector: RealtimeShakeDetector | None = None,
) -> int:
    """
    将单个 accel/gyro motion frame 写入 imu.csv。

    字段含义：
        system_time_ns/s      IMU 帧出队时刻的电脑系统时间，与夹爪 width 对齐用。
        sensor_timestamp_ms   RealSense 设备时间戳(开启 global timestamp 时为墙钟 ms)。
        capture_system_time_s IMU 真实拍摄时刻的电脑墙钟秒；等于
                              sensor_timestamp_ms/1000。设备不支持 global timestamp
                              时为空字符串, 调用方应回退到 system_time_s。
        stream                accel 或 gyro。
        x/y/z                 IMU 三轴读数。
        unit                  accel 是 m/s^2，gyro 是 rad/s。
        nearest_rgbd_frame_idx 写入该 IMU 行时最近一次保存的 RGB-D 帧编号。
    """

    motion = motion_frame.as_motion_frame()
    motion_data = motion.get_motion_data()
    stream_type = motion.profile.stream_type()

    if stream_type == rs.stream.accel:
        stream_name = "accel"
        unit = "m/s^2"
    elif stream_type == rs.stream.gyro:
        stream_name = "gyro"
        unit = "rad/s"
    else:
        return imu_row_idx

    system_time_ns = time.time_ns()
    system_time_s = system_time_ns / 1_000_000_000.0
    capture_system_time_s = sensor_timestamp_to_wall_time_s(motion.get_timestamp())
    capture_text = (
        "" if capture_system_time_s is None else f"{capture_system_time_s:.9f}"
    )
    # shake 检测与 recording_events.json 使用与 timestamps.csv 同轴的拍摄时刻；
    # 设备不支持 global timestamp 时才退回到出队时刻。
    detector_time_s = (
        capture_system_time_s if capture_system_time_s is not None else system_time_s
    )
    if not rate_limiter.should_emit(stream_name, detector_time_s):
        return imu_row_idx

    nearest_frame_text = "" if latest_rgbd_frame_idx is None else str(latest_rgbd_frame_idx)
    motion_x = float(motion_data.x)
    motion_y = float(motion_data.y)
    motion_z = float(motion_data.z)

    imu_writer.writerow(
        [
            imu_row_idx,
            system_time_ns,
            f"{system_time_s:.9f}",
            f"{float(motion.get_timestamp()):.6f}",
            capture_text,
            int(motion.get_frame_number()),
            stream_name,
            f"{motion_x:.9f}",
            f"{motion_y:.9f}",
            f"{motion_z:.9f}",
            unit,
            nearest_frame_text,
        ]
    )

    # shake同步使用gyro旋转速度；accel仍照常写入CSV，但不参与实时判断。
    if stream_name == "gyro" and shake_detector is not None:
        shake_detector.on_gyro_sample(
            system_time_s=detector_time_s,
            x=motion_x,
            y=motion_y,
            z=motion_z,
        )

    return imu_row_idx + 1


def record_rgbd(
    output_dir: Path,
    width: int,
    height: int,
    fps: int,
    preview_enabled: bool = True,
) -> int:
    """
    主录制循环。

    每拿到一组对齐后的 color/depth 帧：
        1. 记录出队时刻的电脑系统时间 time.time_ns()。
        2. 用 color 流传感器时间戳换算真实拍摄时刻 capture_system_time_s
           (开启 global timestamp 时即墙钟, 与电脑时间同轴, 无队列延迟)。
        3. 保存 rgb/000000.png。
        4. 保存 depth/000000.png，原始 uint16，不做伪彩色。
        5. 写 timestamps.csv。

    同时，accel/gyro motion frame 会被写入 imu.csv。
    """

    rs, cv2, np = import_realsense_dependencies()
    stop_event = threading.Event()
    stdin_thread = threading.Thread(
        target=listen_for_stop_command,
        args=(stop_event,),
        daemon=True,
    )
    stdin_thread.start()

    started_at = datetime.now().isoformat(timespec="seconds")
    pipeline_bundle = configure_d435i_pipeline(rs, width=width, height=height, fps=fps)
    pipeline = pipeline_bundle["pipeline"]
    frame_queue = pipeline_bundle["frame_queue"]
    align_to_color = pipeline_bundle["align"]

    rgb_dir, depth_dir = prepare_output_dirs(output_dir)
    timestamps_path = output_dir / "timestamps.csv"
    imu_path = output_dir / "imu.csv"
    intrinsics_path = output_dir / "intrinsics.json"
    meta_path = output_dir / "meta.json"
    events_path = output_dir / "recording_events.json"
    should_write_timestamp_header = not timestamps_path.exists() or timestamps_path.stat().st_size == 0
    should_write_imu_header = not imu_path.exists() or imu_path.stat().st_size == 0
    frame_idx = get_next_frame_index(rgb_dir=rgb_dir, depth_dir=depth_dir)
    start_frame_idx = frame_idx
    imu_row_idx = get_existing_csv_data_row_count(imu_path)
    start_imu_row_idx = imu_row_idx

    write_intrinsics_json(
        intrinsics_path=intrinsics_path,
        color_intrinsics=pipeline_bundle["color_intrinsics"],
        depth_intrinsics=pipeline_bundle["depth_intrinsics"],
        depth_scale=pipeline_bundle["depth_scale"],
    )

    dropped_frames = 0
    first_system_time_s = None
    last_system_time_s = None
    capture_frame_count = 0
    first_capture_system_time_s = None
    last_capture_system_time_s = None
    latest_rgbd_frame_idx = None
    shake_detector = RealtimeShakeDetector(events_path=events_path)
    imu_rate_limiter = ImuRateLimiter(target_fps=IMU_OUTPUT_FPS)
    rgbd_rate_limiter = ImuRateLimiter(target_fps=pipeline_bundle["rgbd_fps"])
    preview = RgbPreview(cv2, enabled=bool(preview_enabled))

    meta = {
        "camera": "Intel RealSense D435i",
        "device_info": pipeline_bundle["device_info"],
        "width": width,
        "height": height,
        # fps 是最终落盘频率；rgbd_hardware_fps 是 RealSense 实际打开的硬件流频率。
        "fps": pipeline_bundle["rgbd_fps"],
        "rgbd_hardware_fps": pipeline_bundle["rgbd_hardware_fps"],
        "color_format": "bgr8_png",
        "depth_format": "z16_uint16_png",
        "depth_scale": pipeline_bundle["depth_scale"],
        "depth_unit": "meter = raw_depth_uint16 * depth_scale",
        "aligned_depth_to_color": True,
        "imu_csv": "imu.csv",
        # accel_fps/gyro_fps 保留为 CSV 实际目标频率，兼容旧处理脚本。
        "accel_fps": float(IMU_OUTPUT_FPS),
        "gyro_fps": float(IMU_OUTPUT_FPS),
        "accel_hardware_fps": pipeline_bundle["accel_hardware_fps"],
        "gyro_hardware_fps": pipeline_bundle["gyro_hardware_fps"],
        "imu_output_fps": float(IMU_OUTPUT_FPS),
        "accel_unit": "m/s^2",
        "gyro_unit": "rad/s",
        "recording_events_json": "recording_events.json",
        "realtime_shake_detector": "d435i_gyro_norm_first_shake",
        "valid_data_delay_after_shake_seconds": VALID_DATA_DELAY_AFTER_SHAKE_SECONDS,
        "started_at": started_at,
        "stopped_at": None,
        "start_frame_idx": start_frame_idx,
        "start_imu_row_idx": start_imu_row_idx,
        "frame_count_this_run": 0,
        "frame_count_total_in_folder": 0,
        "imu_row_count_this_run": 0,
        "imu_row_count_total_in_folder": 0,
        "dropped_frames": 0,
        "first_system_time_s": None,
        "last_system_time_s": None,
        "timestamp_clock": "global_timestamp_wallclock",
        "timestamp_clock_note": (
            "capture_system_time_s 为传感器拍摄时刻的电脑墙钟秒; "
            "设备不支持 global timestamp 时该列为空, 请回退到 system_time_s。"
        ),
        "first_capture_system_time_s": None,
        "last_capture_system_time_s": None,
        "capture_frame_count": 0,
        "preview_enabled": bool(preview_enabled),
        "preview_fps": float(pipeline_bundle["rgbd_hardware_fps"]),
    }

    try:
        with (
            timestamps_path.open("a", encoding="utf-8-sig", newline="") as csv_file,
            imu_path.open("a", encoding="utf-8-sig", newline="") as imu_csv_file,
        ):
            writer = csv.writer(csv_file)
            imu_writer = csv.writer(imu_csv_file)

            if should_write_timestamp_header:
                writer.writerow(
                    [
                        "frame_idx",
                        "system_time_ns",
                        "system_time_s",
                        "color_timestamp_ms",
                        "depth_timestamp_ms",
                        "capture_system_time_ns",
                        "capture_system_time_s",
                        "color_frame_number",
                        "depth_frame_number",
                        "rgb_path",
                        "depth_path",
                    ]
                )
                csv_file.flush()
            write_imu_csv_header_if_needed(
                csv_file=imu_csv_file,
                writer=imu_writer,
                should_write_header=should_write_imu_header,
            )

            while not stop_event.is_set():
                preview.poll()
                try:
                    frame = frame_queue.wait_for_frame(1000)
                except RuntimeError:
                    dropped_frames += 1
                    continue

                # IMU 帧会以 motion frame 的形式进入同一个 frame_queue。
                if frame.is_motion_frame():
                    imu_row_idx = write_imu_frame_to_csv(
                        rs=rs,
                        imu_writer=imu_writer,
                        motion_frame=frame,
                        imu_row_idx=imu_row_idx,
                        latest_rgbd_frame_idx=latest_rgbd_frame_idx,
                        rate_limiter=imu_rate_limiter,
                        shake_detector=shake_detector,
                    )
                    imu_csv_file.flush()
                    continue

                # RGB-D 通常以 frameset 形式进入队列；其他单独 video frame 不参与保存。
                if not (hasattr(frame, "is_frameset") and frame.is_frameset()):
                    continue

                frames = frame.as_frameset()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    dropped_frames += 1
                    continue

                # 每个硬件 RGB 帧都预览，不受 10 Hz 保存降采样限制。
                color_image = np.asanyarray(color_frame.get_data())
                preview.update(
                    color_image,
                    saved_frames=frame_idx - start_frame_idx,
                    save_fps=pipeline_bundle["rgbd_fps"],
                    hardware_fps=pipeline_bundle["rgbd_hardware_fps"],
                )

                system_time_ns = time.time_ns()
                system_time_s = system_time_ns / 1_000_000_000.0
                if first_system_time_s is None:
                    first_system_time_s = system_time_s
                last_system_time_s = system_time_s

                # 真实拍摄时刻: 使用 color 流传感器时间戳(global timestamp 时为墙钟)。
                # color_timestamp_ms 与 system_time_s 同坐标系, 除以 1000 即拍摄时刻墙钟。
                color_capture_s = sensor_timestamp_to_wall_time_s(
                    color_frame.get_timestamp()
                )
                if color_capture_s is None:
                    color_capture_ns = None
                    color_capture_text = ""
                else:
                    color_capture_ns = int(round(color_capture_s * 1_000_000_000.0))
                    color_capture_text = f"{color_capture_s:.9f}"

                rgbd_sample_time_s = color_capture_s if color_capture_s is not None else system_time_s
                if not rgbd_rate_limiter.should_emit("rgbd", rgbd_sample_time_s):
                    continue

                # 仅对需要保存的帧进行深度对齐，降低高帧率预览的 CPU 开销。
                aligned_frames = align_to_color.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    dropped_frames += 1
                    continue

                capture_frame_count += 1
                if first_capture_system_time_s is None and color_capture_s is not None:
                    first_capture_system_time_s = color_capture_s
                if color_capture_s is not None:
                    last_capture_system_time_s = color_capture_s

                rgb_name = f"{frame_idx:06d}.png"
                depth_name = f"{frame_idx:06d}.png"
                rgb_path = rgb_dir / rgb_name
                depth_path = depth_dir / depth_name

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                write_png_unicode_safe(cv2=cv2, image_path=rgb_path, image=color_image)
                write_png_unicode_safe(cv2=cv2, image_path=depth_path, image=depth_image)

                writer.writerow(
                    [
                        frame_idx,
                        system_time_ns,
                        f"{system_time_s:.9f}",
                        f"{float(color_frame.get_timestamp()):.6f}",
                        f"{float(depth_frame.get_timestamp()):.6f}",
                        "" if color_capture_ns is None else str(color_capture_ns),
                        color_capture_text,
                        int(color_frame.get_frame_number()),
                        int(depth_frame.get_frame_number()),
                        f"rgb/{rgb_name}",
                        f"depth/{depth_name}",
                    ]
                )
                csv_file.flush()

                # 只有RGB和Depth都成功保存、timestamps.csv也已flush，才宣布相机就绪。
                # 这样control终端出现提示时，训练所需的图像时间轴已经真实存在。
                if latest_rgbd_frame_idx is None:
                    shake_detector.on_camera_ready(
                        system_time_s=system_time_s,
                        first_frame_idx=frame_idx,
                    )

                latest_rgbd_frame_idx = frame_idx
                frame_idx += 1

    except KeyboardInterrupt:
        stop_event.set()
    finally:
        # 事件停止时间使用电脑系统时钟，与timestamps.csv、imu.csv和夹爪日志同源。
        shake_detector.on_recording_stopped(time.time())
        pipeline.stop()
        preview.close()

        meta["stopped_at"] = datetime.now().isoformat(timespec="seconds")
        meta["frame_count_this_run"] = frame_idx - start_frame_idx
        meta["frame_count_total_in_folder"] = frame_idx
        meta["imu_row_count_this_run"] = imu_row_idx - start_imu_row_idx
        meta["imu_row_count_total_in_folder"] = imu_row_idx
        meta["dropped_frames"] = dropped_frames
        meta["first_system_time_s"] = first_system_time_s
        meta["last_system_time_s"] = last_system_time_s
        meta["first_capture_system_time_s"] = first_capture_system_time_s
        meta["last_capture_system_time_s"] = last_capture_system_time_s
        meta["capture_frame_count"] = capture_frame_count
        meta["imu_written_counts"] = imu_rate_limiter.written_counts
        meta["imu_skipped_hardware_counts"] = imu_rate_limiter.skipped_counts
        meta["realtime_shake_final_state"] = shake_detector.event_data.get("state")
        write_meta_json(meta_path, meta)

        print(
            "[record_435i] stopped, "
            f"frames={frame_idx - start_frame_idx}, "
            f"imu_rows={imu_row_idx - start_imu_row_idx}, "
            f"dropped={dropped_frames}, output={output_dir}",
            flush=True,
        )

    return 0


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Record D435i aligned RGB-D image sequence.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录，例如 D:\\Users\\dell\\Desktop\\umi\\0811data\\data_0811_2100\\d435i",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="关闭按相机硬件帧率刷新的 RGB 实时预览（保存频率不变）。",
    )
    return parser.parse_args()


def main() -> int:
    """程序入口。"""

    args = parse_args()
    try:
        return record_rgbd(
            output_dir=args.output_dir,
            width=args.width,
            height=args.height,
            fps=args.fps,
            preview_enabled=not args.no_preview,
        )
    except D435iConnectionError as exc:
        print(f"435i没有连接或启动失败：{exc}", file=sys.stderr, flush=True)
        return 1
    except D435iImageWriteError as exc:
        print(f"D435i保存失败：{exc}", file=sys.stderr, flush=True)
        return 1
    except RuntimeError as exc:
        print(f"D435i记录失败：{exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
