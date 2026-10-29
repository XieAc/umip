"""Offline checks for continuous policy servo, timing and shutdown."""

import importlib.util
import json
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation

from driver.d435i import D435iStream
from replay.policy_control import (
    BoundedPoseFollower, Observation, ObservationStream, Prediction,
    PredictionWorker, TimedPoseTrajectory,
)
from replay.replay_xac_ur5 import pose_to_transform, transform_to_pose
from umi.common.pose_util import mat_to_pose10d


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("policy_inference_test", ROOT / "test/test_model_inference.py")
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)


def sample(stamp, pose=None):
    return Observation(np.zeros((224, 224, 3), dtype=np.uint8),
                       np.zeros(6) if pose is None else np.array(pose), 0.05, stamp, stamp * 1000)


def actions():
    result = np.zeros((16, 10))
    for i in range(16):
        pose = pose_to_transform(np.array([i * 0.001, 0, 0, 0, 0, i * 0.002]))
        result[i, :9] = mat_to_pose10d(pose)
    result[:, 9] = 0.05
    return result


def args(*extra):
    parsed = inference.build_parser().parse_args([
        "--robot-ip", "mock", "--tcp-offset", *(["0"] * 6),
        "--minimum-base-z-m", "-1", "--maximum-base-z-m", "1", "--no-camera-preview",
        *(extra if any(e in extra for e in ("--auto-tilt", "--frame-correction")) else ("--no-auto-tilt", *extra)),
    ])
    inference.validate_args(parsed)
    return parsed


class PolicyControlTest(unittest.TestCase):
    def test_first_prediction_wait_skips_expired_startup_result(self):
        observations = ObservationStream(lambda: sample(0), 10)
        worker = PredictionWorker(observations, lambda _: actions(), 10)
        worker.prediction = Prediction(0, (sample(8.9), sample(9)), actions(), 9, 9.3)
        fresh = Prediction(1, (sample(9.8), sample(9.9)), actions(), 9.9, 10)
        def complete(_):
            worker.prediction = fresh
        with patch('replay.policy_control.time.monotonic', return_value=10), \
             patch.object(worker.condition, 'wait', side_effect=complete):
            self.assertIs(worker.wait_ready(0.5), fresh)
        worker.prediction = None
        with self.assertRaisesRegex(RuntimeError, 'fresh first prediction'):
            worker.wait_ready(0.5, timeout_s=0)

    def test_camera_window_displays_bgr_and_quit_cleans_up(self):
        from replay.camera_preview import _show_frames
        frames, stop, gui = Mock(), Mock(), Mock()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frames.get.return_value = frame
        stop.is_set.return_value = False
        gui.waitKey.return_value = ord('q')
        with patch.dict('sys.modules', {'cv2': gui}):
            _show_frames(frames, stop)
        self.assertIs(gui.imshow.call_args.args[1], frame)
        gui.destroyAllWindows.assert_called_once()

    def test_camera_preview_queue_drops_frames_without_waiting(self):
        import queue
        from replay.camera_preview import CameraPreview
        preview = object.__new__(CameraPreview)
        preview.started = True
        preview.process = Mock()
        preview.process.is_alive.return_value = True
        preview.frames = Mock()
        preview.frames.put_nowait.side_effect = [queue.Full, None]
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        preview.publish(frame)
        preview.frames.get_nowait.assert_called_once()
        self.assertEqual(preview.frames.put_nowait.call_count, 2)
        preview.frames.put.assert_not_called()
        self.assertIsNot(preview.frames.put_nowait.call_args.args[0], frame)

    def test_startup_waits_for_fresh_observations_after_a_pause(self):
        clock = {"now": 10.0}
        observations = ObservationStream(lambda: sample(0), 10)
        observations.samples.extend([sample(8.9), sample(9.0)])

        def resume_sampling(_):
            clock["now"] = 10.2
            observations.samples.extend([sample(10.1), sample(10.2)])

        with patch("replay.policy_control.time.monotonic", side_effect=lambda: clock["now"]), \
             patch.object(observations.condition, "wait", side_effect=resume_sampling) as wait:
            previous, latest = observations.wait_ready(timeout_s=1)
        wait.assert_called_once()
        self.assertEqual((previous.captured_at, latest.captured_at), (10.1, 10.2))

    def test_startup_requires_two_samples_after_hardware_initialization(self):
        observations = ObservationStream(lambda: sample(0), 10)
        observations.samples.extend([sample(9.9), sample(10.0), sample(10.1)])

        def next_frame(_):
            observations.samples.append(sample(10.2))

        with patch("replay.policy_control.time.monotonic", return_value=10.2), \
             patch.object(observations.condition, "wait", side_effect=next_frame) as wait:
            previous, latest = observations.wait_ready(timeout_s=1, after=10.05)
        wait.assert_called_once()
        self.assertEqual((previous.captured_at, latest.captured_at), (10.1, 10.2))

    def test_startup_timeout_and_acquisition_errors_still_stop(self):
        observations = ObservationStream(lambda: sample(0), 10)
        observations.samples.extend([sample(8.9), sample(9.0)])
        clock = {"now": 10.0}
        def advance(_):
            clock["now"] += 0.1
        with patch("replay.policy_control.time.monotonic", side_effect=lambda: clock["now"]), \
             patch.object(observations.condition, "wait", side_effect=advance):
            with self.assertRaisesRegex(RuntimeError, "Timed out.*fresh"):
                observations.wait_ready(timeout_s=0.2)
            with self.assertRaisesRegex(RuntimeError, "observations are stale"):
                observations.snapshot()
            observations.error = RuntimeError("USB disconnected")
            with self.assertRaisesRegex(RuntimeError, "acquisition failed: USB disconnected"):
                observations.wait_ready()

    def test_expired_actions_skipped_with_continuous_pose_and_width(self):
        anchor = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.7])
        prediction = Prediction(0, (sample(9.9, anchor), sample(10.0, anchor)), actions(), 10.0, 10.15)
        current = np.array([0.11, -0.2, 0.3, 0.0, 0.0, 0.8])
        trajectory, first = TimedPoseTrajectory.from_prediction(
            prediction, current, 0.03, 10.15, 10.0, 0.1, 1, transform_to_pose)
        self.assertEqual(first, 3)
        beginning, width = trajectory.sample(10.15)
        np.testing.assert_allclose(beginning, current, atol=1e-12)
        self.assertAlmostEqual(width, 0.03)
        expected = pose_to_transform(anchor) @ pose_to_transform(np.array([0.003, 0, 0, 0, 0, 0.006]))
        end, width = trajectory.sample(10.3)
        np.testing.assert_allclose(pose_to_transform(end), expected, atol=1e-9)
        self.assertAlmostEqual(width, 0.05)
        midpoint, _ = trajectory.sample(10.225)
        np.testing.assert_allclose(midpoint[:3], (current[:3] + end[:3]) / 2, atol=1e-12)
        total_angle = (Rotation.from_rotvec(current[3:]).inv() * Rotation.from_rotvec(end[3:])).magnitude()
        half_angle = (Rotation.from_rotvec(current[3:]).inv() * Rotation.from_rotvec(midpoint[3:])).magnitude()
        self.assertAlmostEqual(total_angle / 2, half_angle)
        with self.assertRaisesRegex(RuntimeError, "unexpired"):
            TimedPoseTrajectory.from_prediction(prediction, current, 0.03, 11.6, 10, 0.1, 1, transform_to_pose)

    def test_speed_acceleration_bounded_through_target_reversals(self):
        dt = 0.008
        follower = BoundedPoseFollower(np.zeros(6), 0.05, 0.3, 0.1, 0.6)
        poses = [follower.pose.copy()]
        velocities = [np.zeros(3)]
        angular_velocities = [np.zeros(3)]
        for i in range(400):
            target = np.array([0.2 if i < 200 else -0.2, 0.1, 0.05, 0, 0, 0.8 if i < 200 else -0.8])
            poses.append(follower.step(target, dt))
            velocities.append(follower.velocity.copy())
            angular_velocities.append(follower.angular_velocity.copy())
        speed = np.linalg.norm(np.diff(np.array(poses)[:, :3], axis=0) / dt, axis=1)
        acceleration = np.linalg.norm(np.diff(velocities, axis=0) / dt, axis=1)
        angular_speed = np.linalg.norm(angular_velocities, axis=1)
        angular_acceleration = np.linalg.norm(np.diff(angular_velocities, axis=0) / dt, axis=1)
        self.assertLessEqual(max(speed), 0.05 + 1e-10)
        self.assertLessEqual(max(acceleration), 0.1 + 1e-10)
        self.assertLessEqual(max(angular_speed), 0.3 + 1e-10)
        self.assertLessEqual(max(angular_acceleration), 0.6 + 1e-10)

    def test_slow_inference_does_not_pause_servo(self):
        observations = ObservationStream(lambda: sample(time.monotonic()), 10.0)
        observations.start()
        self.addCleanup(observations.close)
        observations.wait_ready()
        def infer(_):
            time.sleep(0.34)
            return actions()
        worker = PredictionWorker(observations, infer, 10.0)
        worker.start()
        self.addCleanup(worker.close)
        control, receiver = Mock(), Mock()
        state = {"pose": np.zeros(6)}
        calls = []
        def servo(command, *_):
            calls.append(time.monotonic())
            state["pose"] = np.array(command)
            return True
        control.initPeriod.side_effect = time.monotonic
        control.servoL.side_effect = servo
        control.waitPeriod.side_effect = lambda start: time.sleep(max(0, start + 0.008 - time.monotonic()))
        receiver.getActualTCPPose.side_effect = lambda: state["pose"].tolist()
        receiver.getTimestamp.side_effect = time.monotonic
        receiver.isProtectiveStopped.return_value = False
        receiver.isEmergencyStopped.return_value = False
        trace = []
        inference.execute_policy_servo(args("--max-duration-s", "1.1"), control, receiver, worker,
            np.zeros(6), 0.05, transform_to_pose, lambda pose: pose, lambda width: None, trace, 10)
        self.assertGreater(len(calls), 50)
        self.assertLess(max(np.diff(calls)), 0.10)
        self.assertGreater(len({row.get("policy_step") for row in trace if "policy_step" in row}), 1)
        moved = np.array([row["controller_target_active_tcp_m_rad"] for row in trace])
        self.assertGreater(np.max(moved[:, 0]), 0.0001)
        self.assertLessEqual(np.max(np.linalg.norm(np.diff(moved[:, :3], axis=0), axis=1)), 0.05 * 0.008 + 1e-10)

    def test_stale_prediction_and_robot_stop_block_servo(self):
        control, receiver, worker = Mock(), Mock(), Mock()
        now = time.monotonic()
        worker.latest.return_value = Prediction(0, (sample(now-1.1), sample(now-1)), actions(), now-1, now-0.9)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            inference.execute_policy_servo(args(), control, receiver, worker, np.zeros(6), 0.05,
                                          transform_to_pose, lambda pose: pose, lambda width: None, [], 10)
        control.servoL.assert_not_called()
        worker.latest.return_value = None
        receiver.isProtectiveStopped.return_value = True
        with self.assertRaisesRegex(RuntimeError, "safety stop"):
            inference.execute_policy_servo(args(), control, receiver, worker, np.zeros(6), 0.05,
                                          transform_to_pose, lambda pose: pose, lambda width: None, [], 10)
        control.servoL.assert_not_called()

    def test_camera_latest_mode_discards_queued_old_images(self):
        old, fresh = Mock(), Mock()
        old.is_frameset.return_value = fresh.is_frameset.return_value = True
        old.as_frameset.return_value = old
        fresh.as_frameset.return_value = fresh
        camera = D435iStream()
        camera._running = True
        camera.queue = Mock()
        camera.queue.wait_for_frame.return_value = old
        camera.queue.poll_for_frame.side_effect = [fresh, None]
        camera.align = Mock()
        camera.align.process.side_effect = lambda frames: frames
        camera.aligned_frames(latest=True)
        camera.align.process.assert_called_once_with(fresh)
        camera.align.process.reset_mock()
        camera.queue.poll_for_frame.reset_mock()
        camera.aligned_frames()
        camera.align.process.assert_called_once_with(old)
        camera.queue.poll_for_frame.assert_not_called()

    def test_delayed_motion_connection_and_gripper_setup_use_new_frames(self):
        receiver, control, camera, color, gripper = (Mock() for _ in range(5))
        receiver.getActualTCPPose.return_value = [0.0] * 6
        control.getTCPOffset.return_value = [0.0] * 6
        color.get_data.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        color.get_timestamp.side_effect = lambda: time.monotonic() * 1000
        camera.aligned_frames.return_value = (None, color, None)
        gripper.read_width_m.return_value = 0.05
        streams = []
        initialized_at = []
        acquired_predictions = []

        def create_observations(*parameters, **options):
            stream = ObservationStream(*parameters, **options)
            streams.append(stream)
            return stream

        def connect_control(*_):
            # Model a native connection call that prevents the Python
            # acquisition thread from publishing frames for >0.5 seconds.
            with streams[0].condition:
                time.sleep(0.6)
            return control

        def initialize_gripper(_):
            self.assertFalse(any(t.name == "policy-inference" for t in threading.enumerate()))
            # run() holds the serial lock while this blocks the next sample.
            time.sleep(0.6)
            initialized_at.append(time.monotonic())
        gripper.initialize.side_effect = initialize_gripper

        def execute(parsed, control, receiver, worker, initial_pose, initial_width, *_, **_kwargs):
            deadline = time.monotonic() + 1
            with worker.condition:
                while worker.prediction is None and worker.error is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.fail("No prediction after hardware initialization")
                    worker.condition.wait(remaining)
            prediction = worker.latest()
            self.assertIsNotNone(prediction)
            self.assertTrue(all(obs.captured_at > initialized_at[0]
                                for obs in prediction.observations))
            self.assertLess(time.monotonic() - prediction.observed_at, parsed.prediction_timeout_s)
            acquired_predictions.append(prediction)

        modules = {
            "rtde_receive": types.SimpleNamespace(RTDEReceiveInterface=Mock(return_value=receiver)),
            "rtde_control": types.SimpleNamespace(RTDEControlInterface=Mock(side_effect=connect_control)),
        }
        payload = {"cfg": {"task": {"dataset_frequeny": 10}}, "contract": {"frequency_hz": 10}}
        parsed = args("--enable-motion", "--enable-gripper")
        with patch.dict("sys.modules", modules), \
             patch.object(inference, "D435iStream", return_value=camera), \
             patch.object(inference, "XC330Controller", return_value=gripper), \
             patch.object(inference, "ObservationStream", side_effect=create_observations), \
             patch.object(inference, "read_controller", return_value={"tcp_offset_m_rad": [0.0]*6}), \
             patch.object(inference, "ready"), \
             patch.object(inference, "load_policy", return_value=(types.SimpleNamespace(action_horizon=16), payload)), \
             patch.object(inference, "infer_one", return_value=actions()), \
             patch("builtins.input", return_value="RUN"), \
             patch.object(inference, "execute_policy_servo", side_effect=execute):
            self.assertEqual(inference.run(parsed), 0)
        self.assertEqual(len(acquired_predictions), 1)
        gripper.initialize.assert_called_once_with(0.05)
        gripper.command_width.assert_not_called()
        gripper.close.assert_called_once()
        control.servoStop.assert_called_once_with(2.0)
        camera.stop.assert_called_once()

    def test_motion_failure_stops_and_disconnects_robot(self):
        receiver, control, camera, color = Mock(), Mock(), Mock(), Mock()
        receiver.getActualTCPPose.return_value = [0.0] * 6
        control.getTCPOffset.return_value = [0.0] * 6
        color.get_data.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        color.get_timestamp.side_effect = lambda: time.monotonic() * 1000
        camera.aligned_frames.return_value = (None, color, None)
        modules = {
            "rtde_receive": types.SimpleNamespace(RTDEReceiveInterface=Mock(return_value=receiver)),
            "rtde_control": types.SimpleNamespace(RTDEControlInterface=Mock(return_value=control)),
        }
        payload = {"cfg": {"task": {"dataset_frequeny": 10}}, "contract": {"frequency_hz": 10}}
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.json"
            parsed = args("--enable-motion", "--trace", str(trace))
            with patch.dict("sys.modules", modules), patch.object(inference, "D435iStream", return_value=camera), \
                 patch.object(inference, "read_controller", return_value={"tcp_offset_m_rad": [0.0]*6}), \
                 patch.object(inference, "ready"), patch.object(inference, "load_policy", return_value=(types.SimpleNamespace(action_horizon=16), payload)), \
                 patch.object(inference, "infer_one", return_value=actions()), patch("builtins.input", return_value="RUN"), \
                 patch.object(inference, "execute_policy_servo", side_effect=RuntimeError("simulated inference failure")), \
                 self.assertRaisesRegex(RuntimeError, "simulated inference failure"):
                inference.run(parsed)
            self.assertEqual(json.loads(trace.read_text()), [])
        control.servoStop.assert_called_once_with(2.0)
        control.stopScript.assert_called_once()
        control.disconnect.assert_called_once()
        receiver.disconnect.assert_called_once()
        camera.stop.assert_called_once()

    def test_preview_and_cancel_never_construct_motion_control(self):
        for extra in [[], ["--enable-motion"]]:
            with self.subTest(motion=bool(extra)), tempfile.TemporaryDirectory() as directory:
                receiver = Mock()
                receiver.getActualTCPPose.return_value = [0.0] * 6
                camera = Mock()
                color = Mock()
                color.get_data.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
                color.get_timestamp.side_effect = lambda: time.monotonic() * 1000
                camera.aligned_frames.return_value = (None, color, None)
                constructor = Mock()
                modules = {
                    "rtde_receive": types.SimpleNamespace(RTDEReceiveInterface=Mock(return_value=receiver)),
                    "rtde_control": types.SimpleNamespace(RTDEControlInterface=constructor),
                }
                trace_file = Path(directory) / "trace.json"
                parsed = args("--preview-steps", "2", "--trace", str(trace_file), *extra)
                payload = {"cfg": {"task": {"dataset_frequeny": 10}}, "contract": {"frequency_hz": 10}}
                with patch.dict("sys.modules", modules), patch.object(inference, "D435iStream", return_value=camera), \
                     patch.object(inference, "read_controller", return_value={"tcp_offset_m_rad": [0.0]*6}), \
                     patch.object(inference, "ready"), patch.object(inference, "load_policy", return_value=(types.SimpleNamespace(action_horizon=16), payload)), \
                     patch.object(inference, "infer_one", return_value=actions()), patch("builtins.input", return_value="CANCEL"):
                    result = inference.run(parsed)
                self.assertEqual(result, 2 if extra else 0)
                constructor.assert_not_called()
                camera.stop.assert_called_once()
                receiver.disconnect.assert_called_once()
                self.assertEqual(len(json.loads(trace_file.read_text())), 0 if extra else 2)



class TransientCameraErrorTest(unittest.TestCase):
    def test_occasional_camera_timeout_is_retried(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] in (1, 2, 4):
                raise TimeoutError("frame late")
            return sample(time.monotonic())
        stream = ObservationStream(flaky, 50, transient_errors=(TimeoutError,), max_transient_s=1.0)
        stream.start()
        try:
            stream.wait_ready(timeout_s=2.0)
            self.assertIsNone(stream.error)
        finally:
            stream.close()

    def test_persistent_camera_timeout_still_stops(self):
        def broken():
            raise TimeoutError("no frames")
        stream = ObservationStream(broken, 50, transient_errors=(TimeoutError,), max_transient_s=0.2)
        stream.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "no frames"):
                stream.wait_ready(timeout_s=2.0)
        finally:
            stream.close()

    def test_other_errors_are_not_retried(self):
        def broken():
            raise ValueError("robot pose invalid")
        stream = ObservationStream(broken, 50, transient_errors=(TimeoutError,))
        stream.start()
        stream.thread.join(1.0)
        self.assertIsInstance(stream.error, ValueError)



class StartPostureTest(unittest.TestCase):
    DOWN = np.array(inference.GO_HOME_DOWN_IN_BASE) / np.linalg.norm(inference.GO_HOME_DOWN_IN_BASE)
    PIVOT = inference.DEMO_APPROACH_DISTANCE_M

    def go_home_pose(self):
        # G+X along the go_home finger direction (vertical in the world)
        x = self.DOWN; z = np.array([0.0, -1.0, 0.0]); y = np.cross(z, x)
        return np.r_[0.1, -0.4, 0.2, Rotation.from_matrix(np.column_stack([x, y, z])).as_rotvec()]

    def point_on_fingers(self, pose, distance):
        return pose[:3] + Rotation.from_rotvec(pose[3:]).apply([distance, 0.0, 0.0])

    def test_target_matches_demo_gravity_and_keeps_cube_on_finger_axis(self):
        pose = self.go_home_pose()
        target, angle = inference.demo_posture_target(pose, self.DOWN, self.PIVOT)
        gravity = Rotation.from_rotvec(target[3:]).inv().apply(self.DOWN)
        demo = inference.DEMO_START_GRAVITY_IN_G / np.linalg.norm(inference.DEMO_START_GRAVITY_IN_G)
        np.testing.assert_allclose(gravity, demo, atol=1e-6)
        self.assertAlmostEqual(angle, np.degrees(np.arccos(demo[0])), places=3)
        # the cube (pivot) stays at the same world point and on the finger axis
        np.testing.assert_allclose(self.point_on_fingers(target, self.PIVOT),
                                   self.point_on_fingers(pose, self.PIVOT), atol=1e-9)
        # pivot 0 keeps the gripper origin instead
        np.testing.assert_allclose(inference.demo_posture_target(pose, self.DOWN, 0.0)[0][:3], pose[:3], atol=1e-12)
        self.assertLess(inference.demo_posture_target(target, self.DOWN, self.PIVOT)[1], 1e-3)

    def test_tilt_keeps_pivot_fixed_and_reaches_target(self):
        pose = self.go_home_pose()
        correction = inference.demo_posture_correction(pose, self.DOWN)
        target, _ = inference.demo_posture_target(pose, self.DOWN, self.PIVOT)
        state = {"pose": pose.copy()}
        pivots = []
        control, receiver = Mock(), Mock()
        control.initPeriod.return_value = 0
        def servo(command, *_):
            state["pose"] = np.asarray(command, dtype=float)     # identity TCP mapping, perfect tracking
            pivots.append(self.point_on_fingers(state["pose"], self.PIVOT))
            return True
        control.servoL.side_effect = servo
        receiver.getActualTCPPose.side_effect = lambda: state["pose"].tolist()
        receiver.isProtectiveStopped.return_value = False
        receiver.isEmergencyStopped.return_value = False
        identity = lambda T: inference.transform_to_pose(T)
        same = lambda p: np.asarray(p, dtype=float)
        with patch.object(inference.time, "monotonic", side_effect=[0.0] + [i * 0.05 for i in range(1, 400)]):
            final = inference.tilt_to_demo_posture(control, receiver, pose, correction, self.PIVOT,
                                                   identity, same, 0.5, args())
        self.assertLess(np.abs(np.array(pivots) - self.point_on_fingers(pose, self.PIVOT)).max(), 1e-9)
        np.testing.assert_allclose(final, target, atol=1e-9)
        control.servoStop.assert_called_once()



class GripperReadRetryTest(unittest.TestCase):
    def test_dropped_packet_is_retried_other_errors_are_not(self):
        gripper = Mock()
        gripper.read_width_m.side_effect = [RuntimeError("read motor 1: [TxRxResult] There is no status packet!"), 0.05]
        self.assertEqual(inference.read_width_with_retry(gripper), 0.05)
        gripper.read_width_m.side_effect = RuntimeError("read motor 1: [TxRxResult] There is no status packet!")
        with self.assertRaisesRegex(RuntimeError, "TxRxResult"):
            inference.read_width_with_retry(gripper)
        self.assertEqual(gripper.read_width_m.call_count, 2 + 3)
        gripper.read_width_m.side_effect = RuntimeError("current limit exceeded")
        with self.assertRaisesRegex(RuntimeError, "current"):
            inference.read_width_with_retry(gripper)



class FrameCorrectionTest(unittest.TestCase):
    C = inference._homogeneous(inference.DATA_FRAME_CORRECTION)
    DEMO = inference.DEMO_START_GRAVITY_IN_G / np.linalg.norm(inference.DEMO_START_GRAVITY_IN_G)

    def test_vertical_robot_looks_like_demo_start_in_data_frame(self):
        down = np.array([0.3, -0.2, -0.9]); down /= np.linalg.norm(down)
        # robot G with fingers (+X) exactly along gravity
        x = down; y = np.cross([0.0, 0.0, 1.0], x); y /= np.linalg.norm(y); z = np.cross(x, y)
        pose = np.r_[0.1, 0.2, 0.3, Rotation.from_matrix(np.column_stack([x, y, z])).as_rotvec()]
        data = inference.robot_to_data_pose(pose, self.C)
        np.testing.assert_allclose(data[:3], pose[:3])
        np.testing.assert_allclose(Rotation.from_rotvec(data[3:]).inv().apply(down), self.DEMO, atol=1e-9)

    def test_actions_round_trip_consistently(self):
        pose = np.r_[0.1, 0.2, 0.3, 0.4, -0.5, 0.6]
        rel = np.r_[0.02, -0.01, 0.005, Rotation.from_rotvec([0.05, -0.02, 0.1]).as_rotvec()]
        a10 = np.r_[mat_to_pose10d(pose_to_transform(rel)), 0.05][None]
        robot_actions = inference.data_to_robot_actions(a10, self.C)
        # executing the robot-frame action from the robot pose equals executing the data action in data frame
        robot_target = pose_to_transform(pose) @ inference.pose10d_to_mat(robot_actions[0, :9])
        data_target = pose_to_transform(inference.robot_to_data_pose(pose, self.C)) @ pose_to_transform(rel)
        np.testing.assert_allclose(robot_target @ self.C, data_target, atol=1e-9)
        self.assertEqual(robot_actions[0, 9], 0.05)
        np.testing.assert_allclose(inference.data_to_robot_actions(a10, None), a10)

    def test_auto_tilt_and_frame_correction_are_exclusive(self):
        with self.assertRaises(ValueError):
            args("--auto-tilt", "--frame-correction")
        parsed = inference.build_parser().parse_args(["--robot-ip", "x", "--tcp-offset", *(["0"] * 6),
                                                      "--minimum-base-z-m", "-1", "--maximum-base-z-m", "1"])
        self.assertTrue(parsed.auto_tilt)            # defaults: physical tilt on, software correction off
        self.assertFalse(parsed.frame_correction)
        self.assertTrue(args("--frame-correction", "--no-auto-tilt").frame_correction)


class WorldDepthFloorTest(unittest.TestCase):
    # Fingers (+X_G) point straight down in base Z (test frame with world down = base -Z).
    DOWN = np.r_[np.zeros(3), Rotation.from_euler("y", 90, degrees=True).as_rotvec()]
    # Real start pose from the 2026-10-29 runs; the base is tilted 45 deg.
    REAL = np.array([-0.2905, -0.4491, -0.1131, 0.7275, 1.7643, -1.7607])
    GO_HOME_DOWN = np.array([-0.7071, 0.0, -0.7071]) / np.linalg.norm([-0.7071, 0.0, -0.7071])

    def test_sideways_offset_cannot_go_deeper_than_the_floor(self):
        rot = Rotation.from_rotvec(self.REAL[3:])
        fingers, side = rot.apply([1, 0, 0]), rot.apply([0, 1, 0])
        floor = inference.world_depth(self.REAL, self.GO_HOME_DOWN) + 0.05
        for sideways in (0.0, -0.008, 0.008):          # 15:37 drifted about -8 mm along G Y
            deeper = self.REAL.copy()
            deeper[:3] += 0.08 * fingers + sideways * side
            clamped, active = inference.apply_depth_floor(deeper, floor, self.GO_HOME_DOWN)
            self.assertTrue(active)
            self.assertAlmostEqual(inference.world_depth(clamped, self.GO_HOME_DOWN), floor, places=12)
            step = clamped[:3] - deeper[:3]
            np.testing.assert_allclose(np.cross(step, self.GO_HOME_DOWN), 0, atol=1e-12)   # lifted straight up
            np.testing.assert_allclose(clamped[3:], deeper[3:])
        unchanged, active = inference.apply_depth_floor(self.REAL, floor, self.GO_HOME_DOWN)
        self.assertFalse(active)
        np.testing.assert_array_equal(unchanged, self.REAL)
        self.assertFalse(inference.apply_depth_floor(self.REAL, None, self.GO_HOME_DOWN)[1])

    def test_tilted_gripper_is_limited_by_its_lower_fingertip(self):
        # 15:58: centre on the floor, gripper tilted ~6.7 deg, one fingertip 5.8 mm deeper.
        tilted = self.REAL.copy()
        tilted[3:] = (Rotation.from_rotvec(self.REAL[3:]) * Rotation.from_euler("z", 7, degrees=True)).as_rotvec()
        side = Rotation.from_rotvec(tilted[3:]).apply([0, 1, 0])
        tip_extra = 0.05 * abs(side @ self.GO_HOME_DOWN)
        self.assertGreater(tip_extra, 0.005)
        floor = inference.world_depth(tilted, self.GO_HOME_DOWN)           # centre exactly on the floor
        clamped, active = inference.apply_depth_floor(tilted, floor, self.GO_HOME_DOWN, 0.05)
        self.assertTrue(active)
        self.assertAlmostEqual(inference.world_depth(clamped, self.GO_HOME_DOWN, 0.05), floor, places=12)
        self.assertAlmostEqual(inference.world_depth(clamped, self.GO_HOME_DOWN), floor - tip_extra, places=12)
        level, active = inference.apply_depth_floor(self.REAL, inference.world_depth(self.REAL, self.GO_HOME_DOWN)
                                                    + 0.0002, self.GO_HOME_DOWN, 0.05)
        self.assertFalse(active)                     # level gripper: both tips at the centre's depth

    def test_base_z_alone_does_not_bound_depth(self):
        # 15:37: same base Z as the floor, ~8 mm sideways, fingertips 4.5 mm deeper.
        rot = Rotation.from_rotvec(self.REAL[3:])
        fingers, side = rot.apply([1, 0, 0]), rot.apply([0, 1, 0])
        b = self.REAL.copy()
        b[:3] += -0.008 * side
        b[:3] += (self.REAL[2] - b[2]) / fingers[2] * fingers   # back to the same base Z
        self.assertAlmostEqual(b[2], self.REAL[2], places=12)
        extra = inference.world_depth(b, self.GO_HOME_DOWN) - inference.world_depth(self.REAL, self.GO_HOME_DOWN)
        self.assertGreater(abs(extra), 0.004)

    def test_depth_hard_stop_and_arguments(self):
        parsed = args("--world-down-in-base", "0", "0", "-1", "--floor-depth-m", "0.005", "--max-depth-m", "0.008")
        np.testing.assert_allclose(parsed.world_down_unit, [0, 0, -1])
        inference.check_pose_limits(np.array([0, 0, -0.008, 0, 0, 0]), np.zeros(6), parsed)
        with self.assertRaisesRegex(RuntimeError, "Depth protection"):
            inference.check_pose_limits(np.array([0, 0, -0.0081, 0, 0, 0]), np.zeros(6), parsed)
        for bad in (("--floor-depth-m", "0.005"), ("--max-depth-m", "0.008"),
                    ("--floor-depth-m", "0.005", "--max-depth-m", "0.0052"),
                    ("--fingertip-half-span-m", "0.2")):
            with self.assertRaises(ValueError):
                args(*bad)

    def test_switch_gripper_arguments(self):
        parsed = args("--gripper-mode", "switch")
        self.assertEqual((parsed.grip_close_below_m, parsed.grip_open_above_m), (0.088, 0.06))
        self.assertEqual(args().gripper_mode, "position")          # old behaviour stays the default
        for bad in (("--grip-open-above-m", "0.09"), ("--grip-close-below-m", "0.2"),
                    ("--grip-close-below-at-floor-m", "0.093"),                       # needs the soft floor
                    ("--floor-depth-m", "0.005", "--max-depth-m", "0.008", "--grip-close-below-at-floor-m", "0.08")):
            with self.assertRaises(ValueError):
                args("--gripper-mode", "switch", *bad)
        ok = args("--gripper-mode", "switch", "--floor-depth-m", "0.005", "--max-depth-m", "0.008",
                  "--grip-close-below-at-floor-m", "0.093")
        self.assertEqual(ok.grip_close_below_at_floor_m, 0.093)

    def test_servo_holds_floor_while_gripper_keeps_following(self):
        start = self.DOWN.copy()
        state = {"pose": start.copy()}
        # The observed pose follows the simulated arm, so every replan keeps descending.
        observations = ObservationStream(lambda: sample(time.monotonic(), state["pose"].copy()), 10.0)
        observations.start()
        self.addCleanup(observations.close)
        observations.wait_ready()
        worker = PredictionWorker(observations, lambda _: actions(), 10.0)
        worker.start()
        self.addCleanup(worker.close)
        control, receiver = Mock(), Mock()
        def servo(command, *_):
            state["pose"] = np.array(command)
            return True
        control.initPeriod.side_effect = time.monotonic
        control.servoL.side_effect = servo
        control.waitPeriod.side_effect = lambda s: time.sleep(max(0, s + 0.008 - time.monotonic()))
        receiver.getActualTCPPose.side_effect = lambda: state["pose"].tolist()
        receiver.getTimestamp.side_effect = time.monotonic
        receiver.isProtectiveStopped.return_value = False
        receiver.isEmergencyStopped.return_value = False
        widths, trace = [], []
        floor_calls = []
        inference.execute_policy_servo(
            args("--max-duration-s", "1.5", "--world-down-in-base", "0", "0", "-1",
                 "--floor-depth-m", "0.005", "--max-depth-m", "0.008"),
            control, receiver, worker, start, 0.05, transform_to_pose, lambda pose: pose,
            widths.append, trace, 10, on_floor=lambda: floor_calls.append(1))
        self.assertEqual(len(floor_calls), sum(row["floor_active"] for row in trace))   # gripper told about the floor
        down = np.array([0.0, 0.0, -1.0])
        depth = np.array([inference.world_depth(np.array(row["target_g_m_rad"]), down, 0.05) for row in trace])
        self.assertLessEqual(depth.max(), 0.005 + 1e-9)       # lower fingertip never below the floor
        self.assertGreater(depth.max(), 0.0049)                # it did reach the floor
        self.assertTrue(any(row["floor_active"] for row in trace))
        self.assertTrue(all("actual_world_depth_m" in row for row in trace))
        self.assertTrue(all("observation_width_m" in row for row in trace if "policy_step" in row))
        self.assertEqual(len(widths), len(trace))             # gripper commands continue at the floor


class GripperHoldTest(unittest.TestCase):
    @staticmethod
    def robot():
        control, receiver = Mock(), Mock()
        state = {"pose": np.zeros(6)}
        def servo(command, *_):
            state["pose"] = np.array(command)
            return True
        control.initPeriod.side_effect = time.monotonic
        control.servoL.side_effect = servo
        control.waitPeriod.side_effect = lambda s: time.sleep(max(0, s + 0.008 - time.monotonic()))
        receiver.getActualTCPPose.side_effect = lambda: state["pose"].tolist()
        receiver.getTimestamp.side_effect = time.monotonic
        receiver.isProtectiveStopped.return_value = False
        receiver.isEmergencyStopped.return_value = False
        return control, receiver

    def test_arm_holds_and_ignores_slow_inference_while_gripping(self):
        control, receiver = self.robot()
        worker = Mock()
        now = time.monotonic()
        # 2 s old prediction: would normally stop with "stale" after 1 s.
        worker.latest.return_value = Prediction(0, (sample(now-2.1), sample(now-2)), actions(), now-2, now-1.9)
        trace = []
        inference.execute_policy_servo(args("--max-duration-s", "0.6"), control, receiver, worker, np.zeros(6),
                                       0.05, transform_to_pose, lambda pose: pose, lambda w: None, trace, 10,
                                       gripper_busy=lambda: True)
        moved = np.array([row["controller_target_active_tcp_m_rad"] for row in trace])
        np.testing.assert_allclose(moved, 0, atol=1e-12)          # held in place
        self.assertTrue(all(row["gripper_hold"] for row in trace))

    def test_resume_needs_a_prediction_observed_after_the_grip(self):
        control, receiver = self.robot()
        worker = Mock()
        now = time.monotonic()
        worker.latest.return_value = Prediction(0, (sample(now-0.2), sample(now-0.1)), actions(), now-0.1, now)
        busy_until = now + 0.3
        trace = []
        with self.assertRaisesRegex(RuntimeError, "No fresh policy prediction"):
            inference.execute_policy_servo(args("--max-duration-s", "5"), control, receiver, worker, np.zeros(6),
                                           0.05, transform_to_pose, lambda pose: pose, lambda w: None, trace, 10,
                                           gripper_busy=lambda: time.monotonic() < busy_until)
        moved = np.array([row["controller_target_active_tcp_m_rad"] for row in trace])
        np.testing.assert_allclose(moved, 0, atol=1e-12)          # never executed the pre-grip prediction

    def test_resume_with_fresh_predictions(self):
        observations = ObservationStream(lambda: sample(time.monotonic()), 10.0)
        observations.start()
        self.addCleanup(observations.close)
        observations.wait_ready()
        worker = PredictionWorker(observations, lambda _: actions(), 10.0)
        worker.start()
        self.addCleanup(worker.close)
        control, receiver = self.robot()
        start = time.monotonic()
        trace = []
        sent = []
        def set_width(width):
            sent.append((time.monotonic() - start, width))
        inference.execute_policy_servo(args("--max-duration-s", "1.5"), control, receiver, worker, np.zeros(6),
                                       0.9, transform_to_pose, lambda pose: pose, set_width, trace, 10,
                                       gripper_busy=lambda: time.monotonic() - start < 0.5)
        held = [row for row in trace if row["gripper_hold"]]
        # While holding only "no decision" (None) reaches the gripper, so the stale 0.9 m
        # start width is never used; after resuming widths blend from the measured 0.05.
        real = [(t, w) for t, w in sent if w is not None]
        self.assertEqual(len(sent), len(trace))
        self.assertEqual(len(real), len(trace) - len(held))
        self.assertTrue(all(t >= 0.5 for t, _ in real))
        self.assertTrue(all(w < 0.06 for _, w in real))
        free = [row for row in trace if not row["gripper_hold"]]
        self.assertTrue(held and free)
        self.assertGreater(max(row["controller_target_active_tcp_m_rad"][0] for row in free), 0.0001)


if __name__ == "__main__":
    unittest.main()
