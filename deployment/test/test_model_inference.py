"""Run the trained XAC Diffusion Policy on the real UR5.

The default mode is a hardware-connected preview: it reads the robot/camera,
runs inference, and never sends motion.  Motion requires ``--enable-motion``
and an interactive ``RUN`` confirmation.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver.d435i import D435iConfig, D435iConnectionError, D435iStream
from driver.xc330 import XC330Controller, clamp_width_m
from replay.camera_preview import CameraPreview
from replay.replay_xac_ur5 import pose_to_transform, transform_to_pose
from replay.verify_tcp_transform import (
    gripper_from_active,
    load_calibration,
    read_controller,
    ready,
)
from replay.policy_control import (
    BoundedPoseFollower, Observation, ObservationStream, PredictionWorker,
    TimedPoseTrajectory,
)
from umi.common.pose_util import mat_to_pose10d, pose10d_to_mat


DEFAULT_CHECKPOINT = PROJECT_ROOT / "data_model/outputs/xac_bf16_cached_lr1e-3_10k_20261011/checkpoints/best.ckpt"
DEFAULT_TCP = PROJECT_ROOT / "calibration/tcp_gripper_01_gx_down_candidate.json"
DEFAULT_GRIPPER_CALIBRATION = PROJECT_ROOT / "calibration/gripper_width.json"


def resize_rgb_with_letterbox(bgr_image: np.ndarray, image_size: int = 224) -> np.ndarray:
    """Match build_dataset.py: BGR camera frame -> RGB 224x224 letterbox."""

    import cv2

    if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 camera image, got {bgr_image.shape}")
    height, width = bgr_image.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized = cv2.resize(
        bgr_image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    y = (image_size - resized.shape[0]) // 2
    x = (image_size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def load_policy(checkpoint: Path, device: str):
    """Instantiate the exact policy from the checkpoint and load its EMA weights."""

    import hydra
    import torch
    from omegaconf import OmegaConf

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "cfg" not in payload:
        raise ValueError("Checkpoint is not a TrainXacWorkspace checkpoint (missing cfg)")
    cfg = OmegaConf.create(payload["cfg"])
    if "policy" not in cfg:
        raise ValueError("Checkpoint config does not contain policy")
    # The checkpoint contains the complete encoder weights.  Avoid a needless
    # ImageNet download (and make offline deployment deterministic) when
    # constructing the module before loading those weights.
    # Rejects checkpoints from the removed training pipeline and drops
    # encoder options that no longer exist.
    from diffusion_policy.common.checkpoint_compat import policy_cfg_for_loading
    policy_cfg = OmegaConf.create(policy_cfg_for_loading(
        OmegaConf.to_container(cfg.policy, resolve=True)))
    if "obs_encoder" in policy_cfg and "pretrained" in policy_cfg.obs_encoder:
        encoder_cfg = policy_cfg.obs_encoder
        # TimmObsEncoder replaces BatchNorm only when use_group_norm=True
        # and pretrained=False. Preserve the trained architecture before
        # disabling the initial weight download.
        if encoder_cfg.pretrained and encoder_cfg.get("use_group_norm", False):
            encoder_cfg.use_group_norm = False
        encoder_cfg.pretrained = False
    policy = hydra.utils.instantiate(policy_cfg)
    state = payload.get("ema_model") or payload.get("model")
    if state is None:
        raise ValueError("Checkpoint contains neither ema_model nor model weights")
    policy.load_state_dict(state, strict=True)
    policy.to(device)
    policy.eval()
    return policy, payload


def warm_up_camera(camera: Any, frames: int = 3, timeout_s: float = 5.0,
                   settle_s: float = 1.0) -> None:
    """Wait until the D435i delivers RGB-D framesets; the first ones can take
    longer than a policy-rate read allows right after the pipeline starts.
    Keep streaming for ``settle_s``: the first ~0.2 s are dark and green
    until auto exposure / white balance converge."""
    started = time.monotonic()
    deadline = started + timeout_s
    got, last_error = 0, None
    while got < frames or time.monotonic() - started < settle_s:
        if time.monotonic() > deadline:
            raise D435iConnectionError(
                f"D435i delivered {got}/{frames} RGB-D frames in {timeout_s:g} s ({last_error}). "
                "Close realsense-viewer and other camera programs, run `python driver/reset_d435i.py`, "
                "and use --d435i-no-imu.")
        try:
            camera.aligned_frames(timeout_ms=500, retries=0, latest=True)
            got += 1
        except D435iConnectionError as exc:
            last_error = exc


# Gravity direction expressed in the gripper frame G at the start of the
# training demonstrations (mean of 18): fingertips lean ~11 deg toward +Z_G
# (toward the bowl).
DEMO_START_GRAVITY_IN_G = np.array([0.981, -0.062, -0.180])
# Finger direction (G+X) in UR base coordinates at go_home, where the gripper
# points straight down; the base itself is mounted at 45 deg.
GO_HOME_DOWN_IN_BASE = (-0.7071, 0.0, -0.7071)


DEMO_APPROACH_DISTANCE_M = 0.106   # mean start->grasp distance along G+X in the demonstrations


def _rotate_about_pivot(pose: np.ndarray, correction: Rotation, pivot_m: float) -> np.ndarray:
    """New G pose = G * T(p) * C * T(-p): rotates about the point p = (pivot_m, 0, 0)
    in the gripper frame, i.e. a point on the finger axis that stays fixed in the world."""
    shift = np.eye(4); shift[0, 3] = pivot_m
    rot = np.eye(4); rot[:3, :3] = correction.as_matrix()
    return transform_to_pose(pose_to_transform(pose) @ shift @ rot @ np.linalg.inv(shift))


def demo_posture_correction(pose: np.ndarray, down_in_base) -> Rotation:
    """Minimal rotation C (in G) after which gravity expressed in G matches the
    demonstrations' start; no yaw about gravity."""
    down = np.asarray(down_in_base, dtype=float)
    down = down / np.linalg.norm(down)
    gravity_g = Rotation.from_rotvec(pose[3:]).inv().apply(down)
    demo = DEMO_START_GRAVITY_IN_G / np.linalg.norm(DEMO_START_GRAVITY_IN_G)
    # new G = old G * C with C^T gravity_g = demo  <=>  C maps demo onto gravity_g
    correction, _ = Rotation.align_vectors([gravity_g], [demo])
    return correction


def demo_posture_target(pose: np.ndarray, down_in_base,
                        pivot_m: float = 0.0) -> tuple[np.ndarray, float]:
    """Target G pose with the demonstrations' start tilt, rotating about the
    point ``pivot_m`` along the fingers (the cube when the gripper starts right
    above it).  Returns (target G pose, rotation angle in degrees)."""
    correction = demo_posture_correction(pose, down_in_base)
    return _rotate_about_pivot(pose, correction, pivot_m), float(np.degrees(correction.magnitude()))


def tilt_to_demo_posture(control: Any, receiver: Any, start_pose: np.ndarray, correction: Rotation,
                         pivot_m: float, to_controller: Callable[[np.ndarray], np.ndarray],
                         to_gripper: Callable[[np.ndarray], np.ndarray],
                         duration_s: float, args: argparse.Namespace) -> np.ndarray:
    """Rotate about the pivot with a minimum-jerk profile at 125 Hz; returns the final G pose."""
    period = 1.0 / 125.0
    rotvec = correction.as_rotvec()
    started = time.monotonic()
    while True:
        cycle = control.initPeriod()
        u = min(1.0, (time.monotonic() - started) / duration_s)
        s = u ** 3 * (10 - 15 * u + 6 * u ** 2)
        command_g = _rotate_about_pivot(start_pose, Rotation.from_rotvec(s * rotvec), pivot_m)
        check_pose_limits(command_g, start_pose, args)
        if receiver.isProtectiveStopped() or receiver.isEmergencyStopped():
            raise RuntimeError("UR5 safety stop during start-posture tilt")
        actual_g = to_gripper(np.asarray(receiver.getActualTCPPose(), dtype=np.float64))
        following = float(np.linalg.norm(actual_g[:3] - command_g[:3]))
        rotation_error = float((Rotation.from_rotvec(actual_g[3:]).inv()
                                * Rotation.from_rotvec(command_g[3:])).magnitude())
        if following > args.max_following_error_m or rotation_error > args.max_rotation_following_error_rad:
            raise RuntimeError(f"Following error during start-posture tilt: {following:.4f} m / {rotation_error:.4f} rad")
        if not control.servoL(to_controller(pose_to_transform(command_g)).tolist(), 0.0, 0.0, period,
                              args.lookahead_time_s, args.servo_gain):
            raise RuntimeError("UR5 rejected servoL command during start-posture tilt")
        control.waitPeriod(cycle)
        if u >= 1.0 and rotation_error < np.radians(0.5) and following < 0.002:
            break
        if time.monotonic() - started > duration_s + 3.0:
            raise RuntimeError("Start-posture tilt did not settle")
    control.servoStop(2.0)
    return command_g


# Fixed rotation between the robot's G frame (TCP calibration) and the training
# data's G frame (T_I_G), data_G = robot_G * C.  Estimated 2026-10-22 from the
# demonstrations' gravity (vertical hand-held gripper -> gravity in data G =
# DEMO_START_GRAVITY_IN_G) and confirmed with the camera focus of expansion of
# +X motion (11.2 deg / 5.9 deg offset vs 10.4 / 3.6 from gravity).
DATA_FRAME_CORRECTION = Rotation.align_vectors(
    [[1.0, 0.0, 0.0]], [DEMO_START_GRAVITY_IN_G / np.linalg.norm(DEMO_START_GRAVITY_IN_G)])[0]


def _homogeneous(rotation: Rotation) -> np.ndarray:
    m = np.eye(4); m[:3, :3] = rotation.as_matrix(); return m


def robot_to_data_pose(pose: np.ndarray, correction: np.ndarray | None) -> np.ndarray:
    """Robot G pose -> pose of the training-data G frame (same origin)."""
    if correction is None:
        return np.asarray(pose, dtype=float)
    return transform_to_pose(pose_to_transform(pose) @ correction)


def data_to_robot_actions(actions: np.ndarray, correction: np.ndarray | None) -> np.ndarray:
    """Relative actions predicted in data-G -> relative actions in robot-G: C A C^-1."""
    if correction is None:
        return actions
    inverse = np.linalg.inv(correction)
    out = np.array(actions, dtype=float, copy=True)
    for row in out:
        row[:9] = mat_to_pose10d(correction @ pose10d_to_mat(row[:9]) @ inverse)
    return out


def report_start_posture(pose: np.ndarray, down_in_base, auto_tilt: bool) -> float:
    """Compare the start orientation with the demonstrations; returns the
    rotation (deg) that tilt_to_demo_posture would apply.

    Replay is anchored to the robot's start, so absolute tilt never mattered
    there; the closed-loop policy sees the camera image, so it does."""
    down = np.asarray(down_in_base, dtype=float)
    down = down / np.linalg.norm(down)
    gravity_g = Rotation.from_rotvec(pose[3:]).inv().apply(down)
    lean = lambda g: (float(np.degrees(np.arctan2(-g[2], g[0]))), float(np.degrees(np.arctan2(-g[1], g[0]))))
    (lean_z, lean_y), (demo_z, demo_y) = lean(gravity_g), lean(DEMO_START_GRAVITY_IN_G)
    angle = float(np.degrees(demo_posture_correction(pose, down).magnitude()))
    print(f"Start posture: fingertips lean {lean_z:+.1f} deg toward +Z_G (bowl side), {lean_y:+.1f} deg toward +Y_G; "
          f"demonstrations: {demo_z:+.1f} / {demo_y:+.1f} deg", flush=True)
    if angle > 2.0:
        if auto_tilt:
            print(f"After RUN the gripper first rotates {angle:.1f} deg about the point {DEMO_APPROACH_DISTANCE_M*100:.1f} cm "
                  "along the fingers (place the cube there) to the demo start posture.",
                  flush=True)
        else:
            print(f"WARNING: start posture differs from the demonstrations by {angle:.1f} deg and --no-auto-tilt is set.",
                  flush=True)
    return angle


def read_width_with_retry(gripper: Any, attempts: int = 3) -> float:
    """The Dynamixel bus occasionally drops a status packet; retry only those
    communication errors (TxRxResult), not current/calibration errors."""
    for attempt in range(attempts):
        try:
            return float(gripper.read_width_m())
        except RuntimeError as exc:
            if "TxRxResult" not in str(exc) or attempt == attempts - 1:
                raise
            time.sleep(0.005)
    raise AssertionError("unreachable")


def read_gripper_pose_and_width(
    receiver: Any,
    flange_to_gripper: np.ndarray | None,
    flange_to_active: np.ndarray,
    gripper: XC330Controller | None,
) -> tuple[np.ndarray, float]:
    active_pose = np.asarray(receiver.getActualTCPPose(), dtype=np.float64)
    if active_pose.shape != (6,) or not np.isfinite(active_pose).all():
        raise RuntimeError(f"UR5 returned invalid TCP pose: {active_pose}")
    active_matrix = pose_to_transform(active_pose)
    if flange_to_gripper is None:
        gripper_pose = active_pose
    else:
        gripper_pose = transform_to_pose(
            gripper_from_active(active_matrix, flange_to_gripper, flange_to_active)
        )
    width = float(gripper.read_width_m()) if gripper is not None else float("nan")
    return gripper_pose, width


def make_observation(
    images: deque[np.ndarray], poses: deque[np.ndarray], widths: deque[float],
    img_horizon: int = 2, low_dim_horizon: int = 2,
) -> dict[str, Any]:
    """Build the [B=1, horizon] tensors expected by XacDataset.

    Two frames are always acquired; models trained with a horizon of 1 use
    only the latest one (no previous-frame motion cue)."""

    import torch

    if len(images) != 2 or len(poses) != 2 or len(widths) != 2:
        raise ValueError("Exactly two observation frames are required")
    if img_horizon not in (1, 2) or low_dim_horizon not in (1, 2):
        raise ValueError("Observation horizons of 1 or 2 are supported")
    poses, widths = list(poses)[-low_dim_horizon:], list(widths)[-low_dim_horizon:]
    latest = pose_to_transform(poses[-1])
    relative = np.stack([
        mat_to_pose10d(np.linalg.inv(latest) @ pose_to_transform(pose))
        for pose in poses
    ]).astype(np.float32)
    image = np.stack(list(images)[-img_horizon:]).astype(np.float32) / 255.0
    return {
        "camera0_rgb": torch.from_numpy(np.moveaxis(image, -1, 1))[None],
        "robot0_eef_pos": torch.from_numpy(relative[:, :3])[None],
        "robot0_eef_rot_axis_angle": torch.from_numpy(relative[:, 3:])[None],
        "robot0_gripper_width": torch.from_numpy(
            np.asarray(widths, dtype=np.float32).reshape(low_dim_horizon, 1))[None],
    }


def observation_horizons(payload: dict) -> tuple[int, int]:
    """(image horizon, low-dim horizon) the checkpoint was trained with; 2/2 if unspecified."""
    obs = payload.get("cfg", {}).get("shape_meta", {}).get("obs", {})
    return (int(obs.get("camera0_rgb", {}).get("horizon", 2)),
            int(obs.get("robot0_eef_pos", {}).get("horizon", 2)))


def infer_one(
    policy: Any,
    observation: dict[str, Any],
    device: str,
    inference_seed: int | None = 0,
) -> np.ndarray:
    import torch
    from contextlib import nullcontext

    observation = {
        key: value.to(device=device, dtype=torch.float32)
        for key, value in observation.items()
    }
    if inference_seed is None:
        rng_context = nullcontext()
    else:
        cuda_devices = []
        if device.startswith("cuda") and torch.cuda.is_available():
            cuda_devices = [torch.device(device).index or torch.cuda.current_device()]
        rng_context = torch.random.fork_rng(devices=cuda_devices)
    with rng_context:
        if inference_seed is not None:
            torch.manual_seed(inference_seed)
        with torch.inference_mode():
            result = policy.predict_action(observation)
    action = result["action"]
    action = action.detach().cpu().numpy()
    if (action.ndim != 3 or action.shape[0] != 1 or action.shape[1] < 1
            or action.shape[2] != 10 or not np.isfinite(action).all()):
        raise RuntimeError(f"Policy returned invalid action shape/value: {action.shape}")
    return action[0]


def decode_action(
    action_sequence: np.ndarray,
    action_index: int,
) -> tuple[np.ndarray, float]:
    """Decode one [position, rotation-6D, width] row of a predicted horizon."""

    action_sequence = np.asarray(action_sequence)
    if (action_sequence.ndim != 2 or action_sequence.shape[0] < 1
            or action_sequence.shape[1] != 10 or not np.isfinite(action_sequence).all()):
        raise ValueError(f"Expected finite action sequence [horizon, 10], got {action_sequence.shape}")
    if not 0 <= action_index < action_sequence.shape[0]:
        raise ValueError(f"Action index {action_index} is outside predicted horizon {action_sequence.shape[0]}")
    selected = action_sequence[action_index]
    return pose10d_to_mat(selected[:9]), clamp_width_m(selected[9])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--robot-ip", required=True)
    tcp = parser.add_mutually_exclusive_group(required=True)
    tcp.add_argument("--tcp-calibration", type=Path, default=None,
                     help="Computer-side T_F_G JSON; supports any active UR TCP")
    tcp.add_argument("--tcp-offset", type=float, nargs=6, metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                     help="Legacy mode: active UR TCP is the trained G frame (m/rad)")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--enable-gripper", action="store_true")
    parser.add_argument("--gripper-port", default=None)
    parser.add_argument("--gripper-baudrate", type=int, default=57600)
    parser.add_argument("--gripper-calibration", type=Path, default=DEFAULT_GRIPPER_CALIBRATION)
    parser.add_argument("--gripper-current-limit-raw", type=int, default=600)
    parser.add_argument("--gripper-start-tolerance-ticks", type=int, default=80)
    parser.add_argument("--gripper-goal-current-raw", type=int, default=600)
    parser.add_argument("--gripper-hold-current-raw", type=int, default=100)
    parser.add_argument("--gripper-profile-velocity", type=int, default=30)
    parser.add_argument("--gripper-profile-acceleration", type=int, default=20)
    parser.add_argument("--gripper-mode", choices=("position", "switch"), default="position",
                        help="position: follow the predicted width (current-based position mode); "
                             "switch: grip/release with driver/gripper_torque_control.py, "
                             "triggered by the predicted width with hysteresis")
    parser.add_argument("--grip-close-below-m", type=float, default=0.088,
                        help="switch mode: grip when the predicted width drops below this (demo approach >= 92.6 mm)")
    parser.add_argument("--grip-close-below-at-floor-m", type=float, default=None,
                        help="switch mode: looser grip threshold used while the soft floor holds the arm "
                             "(e.g. 0.093; on the floor the policy predicted 80-93 mm, while descending >= 96 mm)")
    parser.add_argument("--grip-open-above-m", type=float, default=0.06,
                        help="switch mode: release when the predicted width rises above this")
    parser.add_argument("--d435i-serial")
    parser.add_argument("--d435i-width", type=int, default=640)
    parser.add_argument("--d435i-height", type=int, default=480)
    parser.add_argument("--d435i-fps", type=int, default=30)
    parser.add_argument("--d435i-no-imu", action="store_true")
    parser.add_argument("--no-camera-preview", action="store_true",
                        help="disable the live camera window (enabled by default on desktop)")
    parser.add_argument("--policy-hz", type=float, default=10.0)
    parser.add_argument(
        "--action-index", type=int, default=1,
        help="earliest usable horizon row; motion skips expired rows using the observation time",
    )
    parser.add_argument(
        "--inference-seed", type=int, default=0,
        help="fixed diffusion sampling seed to reduce replanning jitter; use -1 for fresh random noise",
    )
    parser.add_argument("--max-duration-s", type=float, default=30.0)
    parser.add_argument("--max-distance-from-start-m", type=float, default=0.12)
    parser.add_argument("--minimum-base-z-m", type=float, required=True)
    parser.add_argument("--maximum-base-z-m", type=float, required=True)
    # The base is mounted at 45 deg, so base Z is not vertical: sideways motion
    # changes base Z too (2026-10-29: base-Z floor held, fingertips still 4.5 mm
    # deeper and on the table).  The gripper also tilts a few degrees, which puts
    # one fingertip below the G origin (the midpoint of the two fingertips), so
    # the limit applies to the lower fingertip along --world-down-in-base.
    parser.add_argument("--floor-depth-m", type=float, default=None,
                        help="soft floor for the lower fingertip along world down: deeper targets are lifted "
                             "straight up; arm sideways motion and the gripper continue")
    parser.add_argument("--max-depth-m", type=float, default=None,
                        help="hard stop for the lower fingertip; must be >= --floor-depth-m + 0.5 mm")
    parser.add_argument("--fingertip-half-span-m", type=float, default=0.05,
                        help="fingertips sit this far either side of the G origin along G Y (0.05 = fully open)")
    parser.add_argument("--max-following-error-m", type=float, default=0.03)
    parser.add_argument("--max-rotation-following-error-rad", type=float, default=0.3)
    parser.add_argument("--lookahead-time-s", type=float, default=0.1)
    parser.add_argument("--servo-gain", type=float, default=300.0)
    parser.add_argument("--trajectory-blend-s", type=float, default=0.1,
                        help="minimum transition time when installing a new prediction")
    parser.add_argument("--max-linear-speed-m-s", type=float, default=0.05)
    parser.add_argument("--max-angular-speed-rad-s", type=float, default=0.3)
    parser.add_argument("--max-linear-acceleration-m-s2", type=float, default=0.1)
    parser.add_argument("--max-angular-acceleration-rad-s2", type=float, default=0.6)
    parser.add_argument("--prediction-timeout-s", type=float, default=1.0,
                        help="stop if the latest prediction observation is older than this")
    # Demonstrations start fully open (95-99 mm); a half-closed value makes the
    # policy believe the cube is already held.
    parser.add_argument("--default-width-m", type=float, default=0.098,
                        help="width observation used when --enable-gripper is absent; default = open, as at demo start")
    parser.add_argument("--preview-steps", type=int, default=1,
                        help="Inference steps in dry-run mode")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--world-down-in-base", type=float, nargs=3, default=GO_HOME_DOWN_IN_BASE,
                        metavar=("X", "Y", "Z"),
                        help="gravity direction in UR base coordinates (default: finger direction at go_home)")
    # Default: the demonstrations were hand-held with the fingers leaning ~10 deg
    # toward the bowl (user, 2026-10-22), so the robot tilts physically after RUN.
    parser.add_argument("--auto-tilt", action=argparse.BooleanOptionalAction, default=True,
                        help="after RUN, rotate about the cube to the demonstrations' start tilt before inference")
    parser.add_argument("--frame-correction", action=argparse.BooleanOptionalAction, default=False,
                        help="alternative hypothesis: treat the demo tilt as a T_I_G calibration offset and "
                             "convert frames in software instead (use with --no-auto-tilt)")
    parser.add_argument("--tilt-pivot-m", type=float, default=DEMO_APPROACH_DISTANCE_M,
                        help="rotate about this point along the fingers (the cube); 0 = about the gripper origin")
    parser.add_argument("--tilt-duration-s", type=float, default=2.0,
                        help="duration of the start-posture rotation (at least 1 s per 10 deg)")
    parser.add_argument("--save-images", type=Path, default=None,
                        help="directory for the 224x224 images the policy sees; "
                             "file name = observation_monotonic_s in the trace")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if (not np.isfinite([args.minimum_base_z_m, args.maximum_base_z_m]).all()
            or args.minimum_base_z_m >= args.maximum_base_z_m):
        raise ValueError("--minimum-base-z-m must be smaller than --maximum-base-z-m")
    down = np.asarray(args.world_down_in_base, dtype=np.float64)
    if down.shape != (3,) or not np.isfinite(down).all() or np.linalg.norm(down) < 0.5:
        raise ValueError("--world-down-in-base must be a finite direction")
    args.world_down_unit = down / np.linalg.norm(down)
    if (args.floor_depth_m is None) != (args.max_depth_m is None):
        raise ValueError("--floor-depth-m and --max-depth-m must be given together")
    if args.floor_depth_m is not None:
        if (not np.isfinite([args.floor_depth_m, args.max_depth_m]).all()
                or args.max_depth_m < args.floor_depth_m + FLOOR_MIN_MARGIN_M):
            raise ValueError("--max-depth-m must be at least 0.5 mm deeper than --floor-depth-m")
    if not 0.0 <= args.fingertip_half_span_m <= 0.1:
        raise ValueError("--fingertip-half-span-m must be between 0 and 0.1 m")
    for name in ("policy_hz", "max_duration_s", "max_distance_from_start_m", "max_following_error_m",
                 "max_rotation_following_error_rad", "lookahead_time_s", "servo_gain", "default_width_m",
                 "trajectory_blend_s", "max_linear_speed_m_s", "max_angular_speed_rad_s", "tilt_duration_s",
                 "max_linear_acceleration_m_s2", "max_angular_acceleration_rad_s2", "prediction_timeout_s"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    if not 0.03 <= args.lookahead_time_s <= 0.2 or not 100 <= args.servo_gain <= 2000:
        raise ValueError("UR servo lookahead must be 0.03..0.2 s and gain 100..2000")
    if args.frame_correction and args.auto_tilt:
        raise ValueError("--frame-correction and --auto-tilt compensate the same 10 deg; use --frame-correction --no-auto-tilt")
    if not 0.0 <= args.tilt_pivot_m <= 0.3:
        raise ValueError("--tilt-pivot-m must be between 0 and 0.3 m")
    if not 0.0 < args.grip_open_above_m < args.grip_close_below_m <= 0.1:
        raise ValueError("--grip-open-above-m must be positive and smaller than --grip-close-below-m (<= 0.1 m)")
    if args.grip_close_below_at_floor_m is not None:
        if not args.grip_close_below_m <= args.grip_close_below_at_floor_m <= 0.1:
            raise ValueError("--grip-close-below-at-floor-m must lie between --grip-close-below-m and 0.1 m")
        if args.floor_depth_m is None:
            raise ValueError("--grip-close-below-at-floor-m needs the soft floor (--floor-depth-m)")
    if args.preview_steps < 1:
        raise ValueError("--preview-steps must be positive")
    if args.action_index < 0:
        raise ValueError("--action-index must be nonnegative")
    if args.prediction_timeout_s <= args.trajectory_blend_s:
        raise ValueError("--prediction-timeout-s must exceed --trajectory-blend-s")
    if args.enable_gripper and args.gripper_port is None:
        from xac_config import DXL_PORT
        args.gripper_port = DXL_PORT


def check_pose_limits(pose: np.ndarray, initial: np.ndarray,
                      args: argparse.Namespace, *, actual: bool = False):
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise RuntimeError("Non-finite robot or policy pose")
    distance = float(np.linalg.norm(pose[:3] - initial[:3]))
    allowance = 0.005 if actual else 0.0
    if distance > args.max_distance_from_start_m + allowance:
        raise RuntimeError(f"{'Runtime ' if actual else ''}Distance protection: {distance:.4f} m")
    if not args.minimum_base_z_m <= pose[2] <= args.maximum_base_z_m:
        raise RuntimeError(f"{'Runtime ' if actual else ''}Base-Z protection: {pose[2]:.4f} m")
    if getattr(args, "max_depth_m", None) is not None:
        depth = world_depth(pose, args.world_down_unit, args.fingertip_half_span_m)
        if depth > args.max_depth_m:
            raise RuntimeError(f"{'Runtime ' if actual else ''}Depth protection: {depth:.4f} m "
                               f"> {args.max_depth_m:.4f} m")


FLOOR_MIN_MARGIN_M = 0.0005


def world_depth(pose_g: np.ndarray, down: np.ndarray, half_span: float = 0.0) -> float:
    """Depth along the world-down unit vector (base frame, m) of the lower of the
    two fingertips at +-half_span along G Y; half_span=0 gives the G origin."""
    origin = float(np.dot(np.asarray(pose_g[:3], dtype=np.float64), down))
    if half_span == 0.0:
        return origin
    side = Rotation.from_rotvec(np.asarray(pose_g[3:], dtype=np.float64)).apply([0.0, 1.0, 0.0])
    return origin + half_span * abs(float(np.dot(side, down)))


def apply_depth_floor(pose_g: np.ndarray, floor_depth: float | None,
                      down: np.ndarray, half_span: float = 0.0) -> tuple[np.ndarray, bool]:
    """Lift a G pose whose lower fingertip is deeper than floor_depth straight
    up (against world down).

    Sideways motion and orientation are kept, so the policy can still align
    and close the gripper while resting on the floor."""
    if floor_depth is None:
        return pose_g, False
    excess = world_depth(pose_g, down, half_span) - floor_depth
    if excess <= 0:
        return pose_g, False
    clamped = np.array(pose_g, dtype=np.float64)
    clamped[:3] -= excess * down
    return clamped, True


def prediction_trace(prediction, source_index: int, now: float) -> dict[str, Any]:
    previous, latest = prediction.observations
    return {
        "policy_step": prediction.step,
        "observation_g_m_rad": latest.pose.tolist(),
        "observation_width_m": latest.width,
        "relative_action_pose10d": prediction.actions[source_index].tolist(),
        "source_action_index": source_index,
        "observation_monotonic_s": latest.captured_at,
        "camera_timestamp_ms": latest.frame_timestamp_ms,
        "observation_gap_s": latest.captured_at - previous.captured_at,
        "inference_time_s": prediction.ready_at - prediction.started_at,
        "policy_age_s": now - prediction.observed_at,
    }


def execute_policy_servo(args, control, receiver, worker, initial_pose, initial_width,
                         to_controller, to_gripper, set_width, trace, action_frequency,
                         gripper_busy: Callable[[], bool] | None = None,
                         on_floor: Callable[[], None] | None = None):
    """Keep servoL running while acquisition and inference run in workers.

    ``gripper_busy`` (switch gripper) returns True while it grips or releases;
    the arm then holds its pose."""
    period = 1.0 / 125.0
    follower = BoundedPoseFollower(
        to_controller(pose_to_transform(initial_pose)), args.max_linear_speed_m_s,
        args.max_angular_speed_rad_s, args.max_linear_acceleration_m_s2,
        args.max_angular_acceleration_rad_s2,
    )
    width = initial_width
    trajectory = None
    installed = None
    source_index = args.action_index
    started = time.monotonic()
    previous_cycle = None
    last_state_stamp = receiver.getTimestamp()
    last_state_update = started
    step = 0
    hold_since = resume_after = None
    while True:
        cycle = control.initPeriod()
        now = time.monotonic()
        if now - started >= args.max_duration_s:
            break
        prediction = worker.latest()
        # Switch gripper: hold the arm while it grips/releases, as in the demos
        # (they move ~3 mm while closing).  Serial traffic during the grip also
        # slows inference, so the stale-prediction check pauses meanwhile and
        # motion resumes only with a prediction observed after the grip ended.
        if gripper_busy is not None and gripper_busy():
            if hold_since is None:
                hold_since = now
                print("[arm] holding while the gripper moves", flush=True)
            resume_after = None
            trajectory = None
        elif hold_since is not None:
            hold_since, resume_after = None, now
        if hold_since is not None:
            pass
        elif resume_after is not None and (prediction is None or prediction.observed_at <= resume_after):
            trajectory = None
            if now - resume_after > args.prediction_timeout_s:
                raise RuntimeError("No fresh policy prediction after the gripper action")
        elif prediction is None:
            if now - started > args.prediction_timeout_s:
                raise RuntimeError("Timed out waiting for the first prediction")
        else:
            if resume_after is not None:
                resume_after = None
                print("[arm] resuming with a prediction observed after the gripper action", flush=True)
                # Blend the width from the measured opening, not from the
                # pre-grip prediction (10-29: a stale 84 mm released the grip).
                width = float(prediction.observations[-1].width)
            if now - prediction.observed_at > args.prediction_timeout_s:
                raise RuntimeError(
                    f"Policy prediction is stale; stopping servo: "
                    f"observation age={now-prediction.observed_at:.3f}s, "
                    f"limit={args.prediction_timeout_s:.3f}s, "
                    f"inference={prediction.ready_at-prediction.started_at:.3f}s"
                )
            if installed != prediction.step:
                trajectory, source_index = TimedPoseTrajectory.from_prediction(
                    prediction, follower.pose, width, now, action_frequency,
                    args.trajectory_blend_s, args.action_index, to_controller,
                )
                first_target, _ = apply_depth_floor(to_gripper(trajectory.poses[1]), args.floor_depth_m,
                                                    args.world_down_unit, args.fingertip_half_span_m)
                check_pose_limits(first_target, initial_pose, args)
                installed = prediction.step
        if previous_cycle is not None and now - previous_cycle > args.prediction_timeout_s:
            raise RuntimeError("Servo loop missed its update deadline")
        interval = period if previous_cycle is None else now - previous_cycle
        previous_cycle = now
        if trajectory is None:
            raw_command = follower.pose.copy()
        else:
            raw_command, width = trajectory.sample(now + period)
        width = clamp_width_m(width)
        raw_target_g = to_gripper(raw_command)
        down = args.world_down_unit
        span = args.fingertip_half_span_m
        raw_target_g, raw_floored = apply_depth_floor(raw_target_g, args.floor_depth_m, down, span)
        if raw_floored:
            raw_command = to_controller(pose_to_transform(raw_target_g))
        check_pose_limits(raw_target_g, initial_pose, args)
        command = follower.step(raw_command, period)
        target_g = to_gripper(command)
        target_g, command_floored = apply_depth_floor(target_g, args.floor_depth_m, down, span)
        if command_floored:
            # The follower carries velocity across replans; drop the part that
            # would keep pushing deeper (a pure translation moves G and TCP alike).
            command = to_controller(pose_to_transform(target_g))
            follower.pose = command.copy()
            descent = float(follower.velocity @ down)
            if descent > 0:
                follower.velocity = follower.velocity - descent * down
        check_pose_limits(target_g, initial_pose, args)
        if on_floor is not None and (raw_floored or command_floored):
            on_floor()
        if receiver.isProtectiveStopped() or receiver.isEmergencyStopped():
            raise RuntimeError("UR5 safety stop during policy execution")
        state_stamp = receiver.getTimestamp()
        if state_stamp != last_state_stamp:
            last_state_stamp, last_state_update = state_stamp, now
        if now - last_state_update > args.prediction_timeout_s:
            raise RuntimeError("RTDE robot state is stale")
        active_actual = np.asarray(receiver.getActualTCPPose(), dtype=np.float64)
        if active_actual.shape != (6,) or not np.isfinite(active_actual).all():
            raise RuntimeError("UR5 returned an invalid TCP pose")
        actual_g = to_gripper(active_actual)
        check_pose_limits(actual_g, initial_pose, args, actual=True)
        following_error = float(np.linalg.norm(actual_g[:3] - target_g[:3]))
        rotation_error = float((Rotation.from_rotvec(actual_g[3:]).inv()
                                * Rotation.from_rotvec(target_g[3:])).magnitude())
        if (following_error > args.max_following_error_m
                or rotation_error > args.max_rotation_following_error_rad):
            raise RuntimeError(f"Following error too large: {following_error:.4f} m / {rotation_error:.4f} rad")
        if not control.servoL(command.tolist(), 0.0, 0.0, period,
                              args.lookahead_time_s, args.servo_gain):
            raise RuntimeError("UR5 rejected servoL command")
        if hold_since is None and resume_after is None:
            set_width(width)
        else:
            # Holding for the gripper: never decide from the pre-grip width,
            # it would undo the grip at once (10-29 16:46 and 16:49 loops).
            set_width(None)
        row = {} if prediction is None else prediction_trace(prediction, source_index, now)
        row.update({
            "step": step,
            "elapsed_s": now - started,
            "command_monotonic_s": now,
            "control_interval_s": interval,
            "raw_target_g_m_rad": raw_target_g.tolist(),
            "target_g_m_rad": target_g.tolist(),
            "controller_target_active_tcp_m_rad": command.tolist(),
            "width_m": width,
            "actual_g_m_rad": actual_g.tolist(),
            "actual_active_tcp_m_rad": active_actual.tolist(),
            "following_error_m": following_error,
            "rotation_error_rad": rotation_error,
            "floor_active": bool(raw_floored or command_floored),
            "actual_world_depth_m": world_depth(actual_g, down),
            "actual_tip_depth_m": world_depth(actual_g, down, span),
            "gripper_hold": hold_since is not None or resume_after is not None,
        })
        trace.append(row)
        control.waitPeriod(cycle)
        step += 1


def run(args: argparse.Namespace) -> int:
    import torch
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: {args.device} is unavailable; falling back to CPU", file=sys.stderr)
        args.device = "cpu"
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    policy, payload = load_policy(args.checkpoint.expanduser().resolve(), args.device)
    checkpoint_contract = payload.get("contract", {})
    action_frequency = float(checkpoint_contract.get(
        "frequency_hz", payload["cfg"].get("task", {}).get("dataset_frequeny", 10.0)))
    if not np.isfinite(action_frequency) or action_frequency <= 0:
        raise ValueError("Checkpoint has an invalid action frequency")
    if args.action_index >= policy.action_horizon:
        raise ValueError("--action-index is outside the predicted horizon")
    print(f"Loaded EMA checkpoint: {args.checkpoint} (epoch={payload.get('epoch', '?')})")
    horizons = observation_horizons(payload)
    print(f"Observation horizon: image={horizons[0]}, pose/width={horizons[1]}")
    frame_correction = _homogeneous(DATA_FRAME_CORRECTION) if args.frame_correction else None
    print("Robot-G -> data-G frame correction: "
          + (f"{np.degrees(DATA_FRAME_CORRECTION.magnitude()):.1f} deg (software, gripper stays as placed)"
             if frame_correction is not None else "off"), flush=True)
    print(f"Action frequency={action_frequency:g} Hz; policy={args.policy_hz:g} Hz; servo=125 Hz; device={args.device}")

    state = read_controller(args.robot_ip)
    ready(state)
    flange_to_active = pose_to_transform(np.asarray(state["tcp_offset_m_rad"], dtype=np.float64))
    flange_to_gripper = None
    if args.tcp_calibration is not None:
        flange_to_gripper, _ = load_calibration(args.tcp_calibration.expanduser().resolve())
    else:
        expected = np.asarray(args.tcp_offset, dtype=np.float64)
        if not np.allclose(flange_to_active, pose_to_transform(expected), atol=1e-8, rtol=0):
            raise RuntimeError("Active UR TCP differs from --tcp-offset; stop and correct the pendant TCP")
    g_to_active = None if flange_to_gripper is None else np.linalg.inv(flange_to_gripper) @ flange_to_active
    active_to_g = None if flange_to_gripper is None else np.linalg.inv(flange_to_active) @ flange_to_gripper

    def to_controller(target_g):
        return transform_to_pose(target_g if flange_to_gripper is None else
                                 target_g @ g_to_active)

    def to_gripper(active_pose):
        if flange_to_gripper is None:
            return np.asarray(active_pose, dtype=np.float64).copy()
        return transform_to_pose(pose_to_transform(active_pose) @ active_to_g)

    camera = D435iStream(D435iConfig(
        width=args.d435i_width, height=args.d435i_height, fps=args.d435i_fps,
        serial=args.d435i_serial, enable_imu=not args.d435i_no_imu, queue_size=2,
    ))
    control = receiver = gripper = observations = worker = None
    trace: list[dict[str, Any]] = []
    gripper_lock = threading.Lock()
    width_lock = threading.Lock()
    requested_width = None
    gripper_decisions_paused = False
    preview = None
    try:
        image_dir = None
        if args.save_images is not None:
            image_dir = args.save_images.expanduser().resolve()
            image_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving policy input images to {image_dir}")
        camera.start(drain=False)
        warm_up_camera(camera)
        if not args.no_camera_preview:
            preview = CameraPreview()
            preview.start()
        if args.enable_gripper and args.gripper_mode == "switch":
            from driver.xc330_switch import XC330SwitchGripper
            gripper = XC330SwitchGripper(
                args.gripper_port, args.gripper_baudrate,
                args.gripper_calibration.expanduser().resolve(), (1, 2),
                args.gripper_current_limit_raw, args.gripper_start_tolerance_ticks,
                close_below_m=args.grip_close_below_m, open_above_m=args.grip_open_above_m,
                close_below_at_floor_m=args.grip_close_below_at_floor_m,
            )
            at_floor = ("" if args.grip_close_below_at_floor_m is None else
                        f" ({args.grip_close_below_at_floor_m*1000:.0f} mm on the floor)")
            print(f"Gripper switch mode: grip below {args.grip_close_below_m*1000:.0f} mm{at_floor}, "
                  f"release above {args.grip_open_above_m*1000:.0f} mm (gripper_torque_control)", flush=True)
        elif args.enable_gripper:
            gripper = XC330Controller(
                port_name=args.gripper_port, baudrate=args.gripper_baudrate,
                calibration_path=args.gripper_calibration.expanduser().resolve(),
                motor_ids=(1, 2), current_limit_raw=args.gripper_current_limit_raw,
                start_tolerance_ticks=args.gripper_start_tolerance_ticks,
                goal_current_raw=args.gripper_goal_current_raw,
                hold_current_raw=args.gripper_hold_current_raw,
                profile_velocity=args.gripper_profile_velocity,
                profile_acceleration=args.gripper_profile_acceleration,
            )
        receiver = RTDEReceiveInterface(args.robot_ip, 125.0)

        def read_sample():
            _, color, _ = camera.aligned_frames(timeout_ms=100, retries=2, latest=True)
            captured_at = time.monotonic()
            pose, _ = read_gripper_pose_and_width(receiver, flange_to_gripper, flange_to_active, None)
            width = args.default_width_m
            if gripper is not None:
                with gripper_lock:
                    width = read_width_with_retry(gripper)
                    with width_lock:
                        goal, paused = requested_width, gripper_decisions_paused
                    if goal is not None and paused and args.gripper_mode == "switch":
                        gripper.command_width(None, check_current=True)   # finish the action, decide nothing
                    elif goal is not None:
                        gripper.command_width(goal, check_current=True)
            image = np.asanyarray(color.get_data())
            if preview is not None:
                preview.publish(image)
            policy_image = resize_rgb_with_letterbox(image)
            if image_dir is not None:
                import cv2
                cv2.imwrite(str(image_dir / f"{captured_at:.3f}.jpg"),
                            cv2.cvtColor(policy_image, cv2.COLOR_RGB2BGR))
            return Observation(policy_image,
                               pose, width, captured_at, float(color.get_timestamp()))

        def set_width(width):
            """None = the arm is holding for the gripper: keep any grip/release
            running but make no decision from the stale pre-grip width."""
            nonlocal requested_width, gripper_decisions_paused
            if width is None:
                with width_lock:
                    gripper_decisions_paused = True
                return
            width = clamp_width_m(width)
            if gripper is not None:
                if not gripper.widths[0] <= width <= gripper.widths[-1]:
                    raise RuntimeError(f"Policy gripper width is outside calibration range: {width:.5f} m")
                with width_lock:
                    requested_width = float(width)
                    gripper_decisions_paused = False

        def infer_samples(samples):
            poses = deque(robot_to_data_pose(sample.pose, frame_correction) for sample in samples)
            actions = infer_one(policy, make_observation(
                deque(sample.image for sample in samples), poses,
                deque(sample.width for sample in samples), *horizons), args.device,
                inference_seed=None if args.inference_seed < 0 else args.inference_seed)
            return data_to_robot_actions(actions, frame_correction)

        # A single late camera frame is retried; >1 s without any sample stops.
        observations = ObservationStream(read_sample, action_frequency, min(0.5, args.prediction_timeout_s),
                                         transient_errors=(D435iConnectionError,), max_transient_s=1.0)
        observations.start()
        samples = observations.wait_ready()
        print("Warming up model...")
        infer_samples(samples)
        samples = observations.wait_ready()
        initial_pose = samples[-1].pose.copy()
        check_pose_limits(initial_pose, initial_pose, args)
        print(f"Initial G pose (m/rad): {initial_pose.tolist()}")
        if args.floor_depth_m is not None:
            start_depth = world_depth(initial_pose, args.world_down_unit, args.fingertip_half_span_m)
            if start_depth > args.floor_depth_m:
                raise RuntimeError(f"Start pose depth {start_depth:.4f} m is already below the floor "
                                   f"{args.floor_depth_m:.4f} m")
            print(f"Soft floor (lower fingertip, world vertical): depth <= {args.floor_depth_m:.4f} m, hard stop "
                  f"{args.max_depth_m:.4f} m; start depth {start_depth:.4f} m "
                  f"({(args.floor_depth_m - start_depth)*1000:.0f} mm above the floor)", flush=True)
        report_start_posture(robot_to_data_pose(initial_pose, frame_correction), args.world_down_in_base, args.auto_tilt)
        if not args.enable_gripper:
            print(f"Gripper not connected: the policy sees a fixed width of {args.default_width_m:.3f} m", flush=True)
        if not args.enable_motion:
            worker = PredictionWorker(observations, infer_samples, args.policy_hz)
            worker.start()
            previous = None
            deadline = time.monotonic() + 5.0
            while len(trace) < args.preview_steps:
                prediction = worker.latest()
                if prediction is None or prediction.step == previous:
                    if time.monotonic() > deadline:
                        raise RuntimeError("Timed out waiting for preview inference")
                    time.sleep(0.005)
                    continue
                now = time.monotonic()
                if now - prediction.observed_at > args.prediction_timeout_s:
                    raise RuntimeError("Preview prediction is stale")
                relative_target, width = decode_action(prediction.actions, args.action_index)
                target_g = pose_to_transform(prediction.observations[-1].pose) @ relative_target
                target_pose, floored = apply_depth_floor(transform_to_pose(target_g), args.floor_depth_m,
                                                         args.world_down_unit, args.fingertip_half_span_m)
                check_pose_limits(target_pose, initial_pose, args)
                row = prediction_trace(prediction, args.action_index, now)
                row.update({"step": len(trace), "target_g_m_rad": target_pose.tolist(), "width_m": width,
                            "floor_active": floored})
                trace.append(row)
                print(f"step={len(trace)-1} target={target_pose.round(5).tolist()} width={width:.4f} m "
                      f"inference={row['inference_time_s']*1000:.1f} ms", flush=True)
                previous = prediction.step
                deadline = now + 5.0
            return 0

        print("LIVE MODE: clear the workspace, set the pendant speed low, and keep E-stop reachable.")
        if input('Type exactly "RUN" to enable UR5 motion: ').strip() != "RUN":
            print("Cancelled; no motion command was sent.")
            return 2
        ready(read_controller(args.robot_ip))
        control = RTDEControlInterface(args.robot_ip, 125.0)
        active_offset_now = np.asarray(control.getTCPOffset(), dtype=np.float64)
        if not np.allclose(pose_to_transform(active_offset_now), flange_to_active, atol=1e-8, rtol=0):
            raise RuntimeError("Active TCP changed between preview and motion connection")
        if args.auto_tilt:
            current = observations.wait_ready(after=time.monotonic())[-1].pose.copy()
            correction = demo_posture_correction(current, args.world_down_in_base)
            angle = float(np.degrees(correction.magnitude()))
            if angle > 2.0:
                print(f"Tilting {angle:.1f} deg about the point {args.tilt_pivot_m*100:.1f} cm along the fingers...",
                      flush=True)
                tilt_to_demo_posture(control, receiver, current, correction, args.tilt_pivot_m, to_controller,
                                     to_gripper, max(args.tilt_duration_s, angle / 10.0), args)
                time.sleep(0.3)   # let the camera settle on the new view
                report_start_posture(observations.wait_ready(after=time.monotonic())[-1].pose,
                                     args.world_down_in_base, args.auto_tilt)
        # Connecting RTDE control can briefly pause the Python acquisition
        # thread. Require a new observation pair before configuring the
        # gripper; the pre-RUN pair may no longer describe its current width.
        samples = observations.wait_ready(after=time.monotonic())
        if gripper is not None:
            with gripper_lock:
                gripper.initialize(samples[-1].width)
            # Serial setup also holds the acquisition lock. Start inference
            # only once both input frames were acquired after setup finished.
            samples = observations.wait_ready(after=time.monotonic())
        initial_pose = samples[-1].pose.copy()
        check_pose_limits(initial_pose, initial_pose, args)
        worker = PredictionWorker(observations, infer_samples, args.policy_hz)
        worker.start()
        print('Waiting for the first fresh prediction...', flush=True)
        first_prediction = worker.wait_ready(args.prediction_timeout_s)
        print(f'First inference={(first_prediction.ready_at-first_prediction.started_at)*1000:.1f} ms; '
              f'prediction timeout={args.prediction_timeout_s:g}s.', flush=True)
        print(f"Executing interpolated trajectory at 125 Hz for up to {args.max_duration_s:g} s.", flush=True)
        switch = gripper is not None and args.gripper_mode == "switch"
        execute_policy_servo(args, control, receiver, worker, initial_pose, samples[-1].width,
                             to_controller, to_gripper, set_width, trace, action_frequency,
                             gripper_busy=gripper.is_busy if switch else None,
                             on_floor=gripper.mark_at_floor if switch else None)
        return 0
    finally:
        if control is not None:
            for cleanup in (lambda: control.servoStop(2.0), control.stopScript, control.disconnect):
                try:
                    cleanup()
                except Exception as exc:
                    print(f"UR5 cleanup: {exc}", file=sys.stderr)
        with width_lock:
            requested_width = None
        if worker is not None:
            worker.close()
        cleanups = []
        if observations is not None:
            cleanups.append(observations.close)
        if preview is not None:
            cleanups.append(preview.close)
        cleanups.append(camera.stop)
        if receiver is not None:
            cleanups.append(receiver.disconnect)
        if gripper is not None:
            cleanups.append(gripper.close)
        for cleanup in cleanups:
            try:
                cleanup()
            except Exception as exc:
                print(f"Device cleanup: {exc}", file=sys.stderr)
        if args.trace is not None:
            path = args.trace.expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
            events = getattr(gripper, "events", None)
            if events:
                path.with_name(path.stem + "_gripper.json").write_text(
                    json.dumps(events, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_args(args)
        return run(args)
    except KeyboardInterrupt:
        print("Stopped by operator.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
