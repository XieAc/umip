"""Shared hardware and calibration configuration for the XAC collector."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CALIBRATION_DIR = ROOT / "calibration"


def _configured_path(variable: str, default: Path) -> Path:
    """Resolve relative XAC paths next to this project, not process CWD."""

    raw_value = os.environ.get(variable)
    path = Path(raw_value).expanduser() if raw_value else default
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


DATA_ROOT = _configured_path("XAC_DATA_ROOT", ROOT / "data")
DATASET_ROOT = _configured_path("XAC_DATASET_ROOT", ROOT / "dataset")
TIG_ROOT = _configured_path("XAC_TIG_ROOT", ROOT / "tig_calibration_data")


def _default_dxl_port() -> str:
    """Prefer the local FTDI-by-id path, then fall back to ttyUSB0/COM6."""

    if os.name == "nt":
        return "COM6"
    by_id = Path("/dev/serial/by-id")
    if by_id.is_dir():
        candidates = sorted(
            path for path in by_id.iterdir()
            if path.is_symlink() and "FTDI" in path.name.upper()
        )
        if candidates:
            return str(candidates[0])
    return "/dev/ttyUSB0"


DXL_PORT = os.environ.get("XAC_DXL_PORT", _default_dxl_port())
try:
    DXL_BAUDRATE = int(os.environ.get("XAC_DXL_BAUDRATE", "57600"))
except ValueError as exc:
    raise ValueError("XAC_DXL_BAUDRATE must be an integer") from exc
if DXL_BAUDRATE <= 0:
    raise ValueError("XAC_DXL_BAUDRATE must be positive")
DXL_IDS = (1, 2)
COUNTS_PER_REV = 4096


def load_width_calibration() -> dict:
    import numpy as np

    path = CALIBRATION_DIR / "gripper_width.json"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少夹爪宽度标定：{path}\n请先运行 python calibrate_gripper_width.py"
        )
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("schema_version") != 1 or "samples" not in data:
        raise ValueError(f"夹爪宽度标定格式无效：{path}")
    samples = data.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError(f"width calibration needs at least two samples: {path}")
    required = ("m1", "m2", "width_m")
    import math
    for index, row in enumerate(samples, 1):
        if not isinstance(row, dict) or any(key not in row for key in required):
            raise ValueError(f"calibration sample {index} is missing required fields: {path}")
        try:
            values = [float(row[key]) for key in required]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"calibration sample {index} contains non-numeric data: {path}") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"calibration sample {index} contains non-finite data: {path}")
    if data.get("model") == "paired_difference_tick":
        difference_ticks = [float(row["m1"]) - float(row["m2"]) for row in samples]
        if len(set(difference_ticks)) != len(difference_ticks):
            raise ValueError(f"calibration difference ticks must be unique: {path}")
        order = np.argsort(difference_ticks)
        sorted_widths = np.asarray([float(samples[i]["width_m"]) for i in order])
        delta = np.diff(sorted_widths)
        if len(delta) and not (np.all(delta > 0) or np.all(delta < 0)):
            raise ValueError(f"calibration width is not monotonic against difference ticks: {path}")
    return data


def positions_to_width(position_1: int, position_2: int, calibration: dict) -> tuple[float, float, float]:
    """Convert paired motor positions to measured fingertip opening.

    XAC is hinged, so the default model interpolates the directly measured
    opening against the encoder difference (m1-m2). Legacy per-jaw calibration
    files remain supported when ``model`` is omitted. Widths saturate at the
    calibration endpoints and are limited to 0–100 mm.
    """
    import numpy as np

    samples = calibration["samples"]
    widths = np.clip(np.asarray([float(row["width_m"]) for row in samples], dtype=float), 0.0, 0.1)
    if calibration.get("model") == "paired_difference_tick":
        # Opposing hinge motors: m1-m2 tracks the opening monotonically.
        ticks = np.asarray([float(row["m1"]) - float(row["m2"]) for row in samples])
        order = np.argsort(ticks)
        if np.any(np.diff(ticks[order]) <= 0):
            raise ValueError("paired calibration difference ticks must be strictly increasing")
        # Select the calibrated revolution for each encoder independently.
        # This prevents 4095 -> 0 or extended-position revolutions changing
        # the apparent aperture by a complete turn.
        center1 = np.mean([float(row["m1"]) for row in samples])
        center2 = np.mean([float(row["m2"]) for row in samples])
        position_1 += COUNTS_PER_REV * round((center1 - position_1) / COUNTS_PER_REV)
        position_2 += COUNTS_PER_REV * round((center2 - position_2) / COUNTS_PER_REV)
        difference = position_1 - position_2
        # np.interp saturates outside the measured tick range; enforce 0–100 mm.
        width = float(np.interp(difference, ticks[order], widths[order]))
        return width, width * 0.5, width * 0.5
    if calibration.get("model") == "paired_mean_tick":
        # Compatibility with early calibration files. New files use the
        # difference model above because the motors rotate in opposite ways.
        ticks = np.asarray([(float(row["m1"]) + float(row["m2"])) * 0.5 for row in samples])
        order = np.argsort(ticks)
        if np.any(np.diff(ticks[order]) <= 0):
            raise ValueError("legacy paired calibration mean ticks must be strictly increasing")
        width = float(np.interp((position_1 + position_2) * 0.5, ticks[order], widths[order]))
        return width, width * 0.5, width * 0.5
    halves = []
    for motor_id, position in ((1, position_1), (2, position_2)):
        ticks = np.asarray([float(row[f"m{motor_id}"]) for row in samples], dtype=float)
        order = np.argsort(ticks)
        if np.any(np.diff(ticks[order]) <= 0):
            raise ValueError(f"电机 {motor_id} 的标定 tick 必须互不相同")
        halves.append(float(np.interp(position, ticks[order], widths[order] * 0.5)))
    return halves[0] + halves[1], halves[0], halves[1]
