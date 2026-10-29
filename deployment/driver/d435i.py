"""Reusable Intel RealSense D435i startup interface.

The old ``driver/start_d435i.py`` was a GUI-only demo with a hard-coded
1280x720 profile.  The local collection pipeline uses 640x480 RGB-D, a 30 Hz
hardware profile (or 15 Hz fallback), and optional accelerometer/gyro streams.
This module exposes those choices without importing ``pyrealsense2`` until a
camera is explicitly started.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any


class D435iConnectionError(RuntimeError):
    """The requested RealSense device or stream profile is unavailable."""


@dataclass(frozen=True)
class D435iConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    serial: str | None = None
    enable_imu: bool = True
    queue_size: int = 64

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("D435i width and height must be positive")
        if self.fps <= 0:
            raise ValueError("D435i fps must be positive")
        if self.queue_size < 2:
            raise ValueError("D435i queue size must be at least 2")


def _realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise D435iConnectionError("Install pyrealsense2 before starting the D435i") from exc
    return rs


def _device_info(rs: Any, device: Any) -> dict[str, str]:
    def get(option: Any, default: str) -> str:
        try:
            if device.supports(option):
                return str(device.get_info(option))
        except RuntimeError:
            pass
        return default

    return {
        "name": get(rs.camera_info.name, "unknown"),
        "serial": get(rs.camera_info.serial_number, "unknown"),
        "firmware": get(rs.camera_info.firmware_version, "unknown"),
        "usb_type": get(rs.camera_info.usb_type_descriptor, "unknown"),
    }


def _select_device(rs: Any, serial: str | None) -> tuple[Any, dict[str, str]]:
    devices = list(rs.context().query_devices())
    if not devices:
        raise D435iConnectionError("No Intel RealSense device found")
    selected: Any | None = None
    selected_info: dict[str, str] | None = None
    for device in devices:
        info = _device_info(rs, device)
        if serial is not None and info["serial"] == serial:
            selected, selected_info = device, info
            break
        if serial is None and "D435I" in info["name"].upper() and selected is None:
            selected, selected_info = device, info
    if selected is None:
        if serial is not None:
            available = [_device_info(rs, device)["serial"] for device in devices]
            raise D435iConnectionError(
                f"D435i serial {serial} was not found; available serials: {available}"
            )
        selected = devices[0]
        selected_info = _device_info(rs, selected)
    if "D435I" not in selected_info["name"].upper():
        # A different RealSense model may still be usable, but make the choice
        # visible instead of silently treating it as the local D435i.
        print(f"Warning: selected RealSense device is {selected_info['name']}", flush=True)
    return selected, selected_info


def _video_fps(rs: Any, device: Any, stream: Any, width: int, height: int, fmt: Any) -> set[int]:
    values: set[int] = set()
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != stream or profile.format() != fmt:
                continue
            try:
                video = profile.as_video_stream_profile()
            except RuntimeError:
                continue
            if int(video.width()) == int(width) and int(video.height()) == int(height):
                values.add(int(profile.fps()))
    return values


def _choose_video_fps(rs: Any, device: Any, config: D435iConfig) -> int:
    color = _video_fps(rs, device, rs.stream.color, config.width, config.height, rs.format.bgr8)
    depth = _video_fps(rs, device, rs.stream.depth, config.width, config.height, rs.format.z16)
    common = color & depth
    if not common:
        raise D435iConnectionError(
            f"No common color/depth profile at {config.width}x{config.height}; "
            f"color={sorted(color)}, depth={sorted(depth)}"
        )
    for candidate in (config.fps, 30, 15, 6):
        if candidate in common:
            return candidate
    lower = [value for value in common if value <= config.fps]
    return max(lower) if lower else min(common)


def _motion_profile(rs: Any, device: Any, stream: Any) -> Any | None:
    profiles = []
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() == stream:
                profiles.append(profile)
    if not profiles:
        return None
    return max(profiles, key=lambda profile: int(profile.fps()))


class D435iStream:
    """Start and own one D435i pipeline.

    ``drain=True`` starts a small background reader, which is useful while an
    open-loop robot replay is running.  ``drain=False`` leaves frame ownership
    to ``wait_for_frames`` and is used by the preview/check command.
    """

    def __init__(self, config: D435iConfig | None = None) -> None:
        self.config = config or D435iConfig()
        self.config.validate()
        self.rs: Any | None = None
        self.pipeline: Any | None = None
        self.queue: Any | None = None
        self.align: Any | None = None
        self.info: dict[str, Any] = {}
        self.latest_frames: Any | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self.last_error: Exception | None = None

    @property
    def running(self) -> bool:
        return self._running

    def start(self, *, drain: bool = False) -> dict[str, Any]:
        if self._running:
            return dict(self.info)
        self.rs = _realsense()
        device, device_info = _select_device(self.rs, self.config.serial)
        hardware_fps = _choose_video_fps(self.rs, device, self.config)
        pipeline_config = self.rs.config()
        if device_info["serial"] != "unknown":
            pipeline_config.enable_device(device_info["serial"])
        pipeline_config.enable_stream(
            self.rs.stream.color, self.config.width, self.config.height, self.rs.format.bgr8, hardware_fps
        )
        pipeline_config.enable_stream(
            self.rs.stream.depth, self.config.width, self.config.height, self.rs.format.z16, hardware_fps
        )

        accel = _motion_profile(self.rs, device, self.rs.stream.accel) if self.config.enable_imu else None
        gyro = _motion_profile(self.rs, device, self.rs.stream.gyro) if self.config.enable_imu else None
        if self.config.enable_imu and (accel is None or gyro is None):
            raise D435iConnectionError("Selected D435i does not expose both accel and gyro profiles")
        if accel is not None:
            pipeline_config.enable_stream(self.rs.stream.accel, accel.format(), accel.fps())
        if gyro is not None:
            pipeline_config.enable_stream(self.rs.stream.gyro, gyro.format(), gyro.fps())

        pipeline = self.rs.pipeline()
        queue = self.rs.frame_queue(self.config.queue_size)
        try:
            profile = pipeline.start(pipeline_config, queue)
        except RuntimeError as exc:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise D435iConnectionError(f"Failed to start D435i stream: {exc}") from exc

        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        self.pipeline = pipeline
        self.queue = queue
        self.align = self.rs.align(self.rs.stream.color)
        self.info = {
            **device_info,
            "width": self.config.width,
            "height": self.config.height,
            "hardware_fps": hardware_fps,
            "depth_scale_m": depth_scale,
            "imu_enabled": bool(self.config.enable_imu),
            "accel_hardware_fps": None if accel is None else int(accel.fps()),
            "gyro_hardware_fps": None if gyro is None else int(gyro.fps()),
        }
        self._running = True
        self.last_error = None
        self._stop_event.clear()
        if drain:
            self._thread = threading.Thread(target=self._drain_loop, name="d435i-drain", daemon=True)
            self._thread.start()
        print(
            f"D435i started: {device_info['name']} serial={device_info['serial']} "
            f"RGB-D={self.config.width}x{self.config.height}@{hardware_fps}Hz",
            flush=True,
        )
        return dict(self.info)

    def wait_for_frames(self, timeout_ms: int = 1000) -> Any:
        if not self._running or self.queue is None:
            raise D435iConnectionError("D435i is not running")
        # ``frame_queue`` exposes ``wait_for_frame`` (singular).  The
        # pipeline object has ``wait_for_frames``; this class uses an explicit
        # queue so that optional background draining and direct reads share the
        # same frame source.
        return self.queue.wait_for_frame(timeout_ms)

    def aligned_frames(self, timeout_ms: int = 1000, retries: int = 3,
                       *, latest: bool = False) -> tuple[Any, Any, Any]:
        """Read and align one RGB-D frameset.

        Immediately after ``pipeline.start`` the queue can take a few frame
        periods to produce its first synchronized color/depth frameset.  The
        recorder naturally tolerates this because it keeps reading in a loop;
        callers such as online inference need the same bounded tolerance.
        Motion or standalone video frames are ignored because ``align.process``
        only accepts a composite frameset.

        ``latest=True`` discards queued older framesets for online inference.
        Recording callers keep the default FIFO behavior.
        """
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if retries < 0:
            raise ValueError("retries must be nonnegative")
        last_error: Exception | None = None
        for _ in range(retries + 1):
            try:
                frames = self.wait_for_frames(timeout_ms)
                if latest:
                    while True:
                        newer = self.queue.poll_for_frame()
                        if not newer:
                            break
                        if hasattr(newer, "is_frameset") and newer.is_frameset():
                            frames = newer
            except RuntimeError as exc:
                last_error = exc
                continue
            if not hasattr(frames, "is_frameset") or not frames.is_frameset():
                continue
            try:
                # A frame_queue returns ``frameset``.  Some pyrealsense2
                # builds require the corresponding ``composite_frame`` from
                # ``as_frameset()`` for ``align.process``.
                frames = frames.as_frameset()
                aligned = self.align.process(frames) if self.align is not None else frames
                color = aligned.get_color_frame()
                depth = aligned.get_depth_frame()
                if not color or not depth:
                    last_error = RuntimeError("aligned RGB-D frameset has no color or depth frame")
                    continue
                return aligned, color, depth
            except RuntimeError as exc:
                last_error = exc
                continue
        detail = f"; last error: {last_error}" if last_error is not None else ""
        raise D435iConnectionError(
            f"D435i RGB-D frame did not arrive within {(retries + 1) * timeout_ms} ms"
            f" (camera startup/USB stream may be unstable){detail}"
        ) from last_error

    def _drain_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frames = self.wait_for_frames(100)
            except Exception as exc:
                if not self._stop_event.is_set():
                    self.last_error = D435iConnectionError(
                        f"D435i frame drain stopped unexpectedly: {exc}"
                    )
                    self._stop_event.set()
                break
            with self._lock:
                self.latest_frames = frames

    def stop(self) -> None:
        self._stop_event.set()
        pipeline, thread = self.pipeline, self._thread
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None
        self.pipeline = None
        self.queue = None
        self.align = None
        self._running = False

    @property
    def healthy(self) -> bool:
        """Whether a background-drained stream has not reported an error."""

        return self.last_error is None

    def __enter__(self) -> "D435iStream":
        self.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.stop()
