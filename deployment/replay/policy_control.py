"""Timed policy trajectories and workers independent of robot hardware."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from replay.replay_xac_ur5 import pose_to_transform
from umi.common.pose_util import pose10d_to_mat


@dataclass(frozen=True)
class Observation:
    image: np.ndarray
    pose: np.ndarray
    width: float
    captured_at: float
    frame_timestamp_ms: float


@dataclass(frozen=True)
class Prediction:
    step: int
    observations: tuple[Observation, Observation]
    actions: np.ndarray
    started_at: float
    ready_at: float

    @property
    def observed_at(self) -> float:
        return self.observations[-1].captured_at


class StaleObservationError(RuntimeError):
    """An observation pair is too old to use for policy execution."""


class ObservationStream:
    """Acquire matched image/pose samples while inference and servo run."""

    def __init__(self, read_sample: Callable[[], Observation], frequency: float,
                 max_age_s: float = 0.5, *,
                 transient_errors: tuple[type[BaseException], ...] = (),
                 max_transient_s: float = 1.0):
        """``transient_errors`` (e.g. a single camera frame timeout) are
        retried until no sample has succeeded for ``max_transient_s``; any
        other error stops acquisition at once.  Stale samples are still
        rejected by snapshot(), so retrying never feeds old data to control."""
        self.read_sample = read_sample
        self.period = 1.0 / frequency
        self.max_age_s = max_age_s
        self.transient_errors = tuple(transient_errors)
        self.max_transient_s = max_transient_s
        self.samples: deque[Observation] = deque(maxlen=16)
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, name="policy-observations", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            last_success = time.monotonic()
            while not self.stop_event.is_set():
                started = time.monotonic()
                try:
                    sample = self.read_sample()
                except self.transient_errors:
                    if time.monotonic() - last_success > self.max_transient_s:
                        raise
                    continue
                last_success = time.monotonic()
                with self.condition:
                    self.samples.append(sample)
                    self.condition.notify_all()
                self.stop_event.wait(max(0.0, started + self.period - time.monotonic()))
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"Observation acquisition failed: {self.error}") from self.error

    def snapshot(self, *, after: float | None = None) -> tuple[Observation, Observation] | None:
        with self.condition:
            self.check()
            samples = [sample for sample in self.samples
                       if after is None or sample.captured_at > after]
            if len(samples) < 2:
                return None
            latest = samples[-1]
            age = time.monotonic() - latest.captured_at
            if age > self.max_age_s:
                raise StaleObservationError(
                    f"Camera/robot observations are stale: age={age:.3f}s, limit={self.max_age_s:.3f}s"
                )
            previous = min(samples[:-1],
                           key=lambda sample: abs(sample.captured_at - (latest.captured_at - self.period)))
            gap = latest.captured_at - previous.captured_at
            if gap > self.max_age_s:
                raise StaleObservationError(
                    f"Observation history has a stale frame: gap={gap:.3f}s, limit={self.max_age_s:.3f}s"
                )
            return previous, latest

    def wait_ready(self, timeout_s: float = 5.0, *,
                   after: float | None = None) -> tuple[Observation, Observation]:
        """Wait for fresh samples, optionally both acquired after setup ended.

        Hardware setup can pause publication while old frames remain in the
        buffer. Waiting here permits that startup pause; snapshot() still
        rejects stale observations immediately during policy execution.
        """
        deadline = time.monotonic() + timeout_s
        stale_error = None
        with self.condition:
            while True:
                try:
                    samples = self.snapshot(after=after)
                except StaleObservationError as exc:
                    stale_error = exc
                    samples = None
                if samples is not None:
                    return samples
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = f"; {stale_error}" if stale_error is not None else ""
                    raise RuntimeError(f"Timed out waiting for fresh camera/robot observations{detail}")
                self.condition.wait(min(remaining, 0.1))

    def close(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                raise RuntimeError("Observation worker did not stop")


class PredictionWorker:
    """Publish only the newest completed prediction; never block servo."""

    def __init__(self, observations: ObservationStream, infer: Callable,
                 frequency: float):
        self.observations = observations
        self.infer = infer
        self.period = 1.0 / frequency
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.prediction: Prediction | None = None
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, name="policy-inference", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            step = 0
            last_observed_at = None
            while not self.stop_event.is_set():
                started = time.monotonic()
                samples = self.observations.snapshot()
                if samples is None or samples[-1].captured_at == last_observed_at:
                    self.stop_event.wait(0.002)
                    continue
                actions = self.infer(samples)
                prediction = Prediction(step, samples, actions, started, time.monotonic())
                with self.condition:
                    self.prediction = prediction
                    self.condition.notify_all()
                step += 1
                last_observed_at = samples[-1].captured_at
                self.stop_event.wait(max(0.0, started + self.period - time.monotonic()))
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def latest(self) -> Prediction | None:
        self.observations.check()
        with self.condition:
            if self.error is not None:
                raise RuntimeError(f"Policy inference failed: {self.error}") from self.error
            return self.prediction

    def wait_ready(self, max_age_s: float, timeout_s: float = 5.0) -> Prediction:
        """Wait before servo starts; discard expired startup predictions."""
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while True:
                self.observations.check()
                if self.error is not None:
                    raise RuntimeError(f"Policy inference failed: {self.error}") from self.error
                prediction = self.prediction
                now = time.monotonic()
                if prediction is not None and now - prediction.observed_at <= max_age_s:
                    return prediction
                if now >= deadline:
                    detail = ('no prediction completed' if prediction is None else
                              f'latest observation age={now-prediction.observed_at:.3f}s, '
                              f'inference={prediction.ready_at-prediction.started_at:.3f}s')
                    raise RuntimeError(f'Timed out waiting for a fresh first prediction: {detail}')
                self.condition.wait(min(deadline - now, 0.05))

    def close(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=2.0)


class TimedPoseTrajectory:
    def __init__(self, times: np.ndarray, poses: np.ndarray, widths: np.ndarray):
        self.times = np.asarray(times, dtype=float)
        self.poses = np.asarray(poses, dtype=float)
        self.widths = np.asarray(widths, dtype=float)
        self.rotations = Slerp(self.times, Rotation.from_rotvec(self.poses[:, 3:]))

    def sample(self, now: float) -> tuple[np.ndarray, float]:
        at = float(np.clip(now, self.times[0], self.times[-1]))
        pose = np.r_[[np.interp(at, self.times, self.poses[:, axis]) for axis in range(3)],
                     self.rotations([at]).as_rotvec()[0]]
        return pose, float(np.interp(at, self.times, self.widths))

    @classmethod
    def from_prediction(cls, prediction: Prediction, current_pose: np.ndarray,
                        current_width: float, now: float, action_frequency: float,
                        blend_time_s: float, minimum_action_index: int,
                        to_controller: Callable[[np.ndarray], np.ndarray]):
        actions = np.asarray(prediction.actions)
        if (actions.ndim != 2 or actions.shape[1] != 10
                or not np.isfinite(actions).all()):
            raise ValueError("Expected finite policy actions [horizon, 10]")
        if not 0 <= minimum_action_index < len(actions):
            raise ValueError("--action-index is outside the predicted horizon")
        times = prediction.observed_at + np.arange(len(actions)) / action_frequency
        first = max(minimum_action_index, int(np.searchsorted(times, now + blend_time_s)))
        if first >= len(actions):
            raise RuntimeError("Prediction has no unexpired future actions")
        anchor = pose_to_transform(prediction.observations[-1].pose)
        poses = np.array([to_controller(anchor @ pose10d_to_mat(action[:9]))
                          for action in actions[first:]])
        trajectory = cls(np.r_[now, times[first:]],
                         np.vstack([current_pose, poses]),
                         np.r_[np.clip(current_width, 0.0, 0.1), np.clip(actions[first:, 9], 0.0, 0.1)])
        return trajectory, first


def _limit_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector * min(1.0, limit / max(norm, 1e-12))


class BoundedPoseFollower:
    """Bound commanded active-TCP speed and acceleration across replans."""

    def __init__(self, pose: np.ndarray, linear_speed: float, angular_speed: float,
                 linear_acceleration: float, angular_acceleration: float):
        self.pose = np.array(pose, dtype=float)
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.linear_acceleration = linear_acceleration
        self.angular_acceleration = angular_acceleration

    def step(self, target: np.ndarray, dt: float) -> np.ndarray:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Servo interval must be positive and finite")
        target = np.asarray(target, dtype=float)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("Servo target must be a finite UR pose")
        position_error = target[:3] - self.pose[:3]
        current_rotation = Rotation.from_rotvec(self.pose[3:])
        rotation_error = (Rotation.from_rotvec(target[3:]) * current_rotation.inv()).as_rotvec()
        # Brake as the error shrinks and low-pass the tracking error. Retain
        # velocity on replanning rather than restarting from rest each time.
        linear_limit = min(self.linear_speed,
                           np.sqrt(2 * self.linear_acceleration * np.linalg.norm(position_error)))
        angular_limit = min(self.angular_speed,
                            np.sqrt(2 * self.angular_acceleration * np.linalg.norm(rotation_error)))
        desired_velocity = _limit_norm(position_error / max(0.05, dt), linear_limit)
        desired_angular_velocity = _limit_norm(rotation_error / max(0.05, dt), angular_limit)
        self.velocity += _limit_norm(desired_velocity - self.velocity, self.linear_acceleration * dt)
        self.angular_velocity += _limit_norm(desired_angular_velocity - self.angular_velocity,
                                            self.angular_acceleration * dt)
        self.pose[:3] += self.velocity * dt
        self.pose[3:] = (Rotation.from_rotvec(self.angular_velocity * dt) * current_rotation).as_rotvec()
        return self.pose.copy()
