import time
from math import pi, radians
from typing import Optional

import math3d as m3d
import numpy as np
import transforms3d.affines
from scipy.spatial.transform import Rotation, Slerp

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

HMOE_X_R = 0.2
HMOE_X_L = -0.2
HOME_Y = -0.6
HOME_Z = 0.10
LEFT_ROBOT_IP = "192.168.1.101"
RIGHT_ROBOT_IP = "192.168.1.102"

# servoL 死区滞回系数: 已伺服时误差需回落到 0.5x 死区以下才 servoStop,
# 避免慢速运动时在 servoL/servoStop 之间反复切换产生的极限环抖动
_SERVO_DEADBAND_HYSTERESIS_RATIO = 0.5


def build_home_transform(ip: str = RIGHT_ROBOT_IP) -> m3d.Transform:
    pose = m3d.Transform()
    if ip == RIGHT_ROBOT_IP:
        pose.pos.x = HMOE_X_R
    elif ip == LEFT_ROBOT_IP:
        pose.pos.x = HMOE_X_L
    pose.pos.y = HOME_Y
    pose.pos.z = HOME_Z
    pose.orient.rotate_yb(-pi)
    return pose


class Robot:
    def __init__(
        self,
        ip: str = RIGHT_ROBOT_IP,
        rotation_y_deg: float = 45,
        rtde_frequency: float = 200.0,
        ur_cap_port: int = 50002,
        blocking: bool = True,
    ):
        super().__init__()
        self._ip = ip
        self._tf_world_base = m3d.Transform()
        self._tf_world_base.orient.rotate_yb(radians(rotation_y_deg))
        self._rtde_r = RTDEReceive(ip, rtde_frequency)
        self._rtde_c = RTDEControl(
            ip,
            rtde_frequency,
            RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_UPLOAD_SCRIPT,
            ur_cap_port,
        )
        self._blocking = blocking  # 新增：保存阻塞标志
        self._servo_active = False  # servoL 是否处于激活状态 (死区滞回用)

    def set_pose(self, tf_world_target: m3d.Transform, *args, **kwargs) -> None:
        """Set an absolute pose through the replay-style smooth servo path."""
        tf_base_target = self._tf_world_base.inverse * tf_world_target
        matrix = np.asarray(tf_base_target.matrix, dtype=float)
        target = np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_rotvec()]
        current_tf = self.get_pose_raw()
        if current_tf is None:
            raise RuntimeError("Unable to read current UR5 pose")
        current_matrix = np.asarray(current_tf.matrix, dtype=float)
        current = np.r_[current_matrix[:3, 3], Rotation.from_matrix(current_matrix[:3, :3]).as_rotvec()]
        vel = float(kwargs.get("vel", 0.1))
        self._smooth_servo_pose(current, target, vel=vel)

    def _smooth_servo_pose(self, current, target, vel=0.1, frequency=125.0,
                           lookahead=0.1, gain=300.0):
        current = np.asarray(current, dtype=float)
        target = np.asarray(target, dtype=float)
        duration = max(1.0, float(np.linalg.norm(target[:3] - current[:3])) / max(vel, 1e-6))
        rotations = Slerp([0.0, duration], Rotation.from_rotvec(np.stack([current[3:], target[3:]])))
        period = 1.0 / frequency
        started = time.monotonic()
        iteration = 0
        try:
            while True:
                elapsed = min(time.monotonic() - started, duration)
                fraction = elapsed / duration
                command = np.empty(6)
                command[:3] = current[:3] * (1.0 - fraction) + target[:3] * fraction
                command[3:] = rotations([elapsed]).as_rotvec()[0]
                if not self._rtde_c.servoL(command.tolist(), 0.0, 0.0, period, lookahead, gain):
                    raise RuntimeError("UR5 rejected servoL pose command")
                if elapsed >= duration:
                    return
                time.sleep(max(0.0, started + (iteration + 1) * period - time.monotonic()))
                iteration += 1
        finally:
            self._rtde_c.servoStop(2.0)

    def get_pose_raw(self) -> Optional[m3d.Transform]:
        if not self._rtde_r.isConnected():
            self._rtde_r.reconnect()
        pose_raw = self._rtde_r.getActualTCPPose()
        if len(pose_raw) != 6:
            print("ERROR: Invalid output from getActualTCPPose(): ", pose_raw)
            return None
        position = pose_raw[:3]
        orientation = Rotation.from_rotvec(pose_raw[3:]).as_matrix()
        pose = transforms3d.affines.compose(position, orientation, np.ones(3))
        return m3d.Transform(pose)

    def get_pose(self) -> m3d.Transform:
        tf_base_target = self.get_pose_raw()
        return self._tf_world_base * tf_base_target

    @staticmethod
    def _xyz_quat_wxyz(
        tf: m3d.Transform,
    ) -> tuple[float, float, float, float, float, float, float]:
        """把世界系位姿变换转成 (x, y, z, qw, qx, qy, qz) 元组。"""
        pos = tf.pos
        rot_mat = tf.orient.matrix
        quat_xyzw = Rotation.from_matrix(
            rot_mat
        ).as_quat()  # SciPy 默认返回 (x, y, z, w)
        return (
            pos.x,
            pos.y,
            pos.z,
            quat_xyzw[3],
            quat_xyzw[0],
            quat_xyzw[1],
            quat_xyzw[2],
        )

    def get_pose_xyz_quat(
        self,
    ) -> tuple[float, float, float, float, float, float, float]:
        """
        获取当前TCP在世界坐标系下的位姿。
        返回格式: (x, y, z, qw, qx, qy, qz)
        """
        return self._xyz_quat_wxyz(self.get_pose())

    def get_pose_snapshot(self):
        """
        一次 RTDE 读取同时得到: base 系原始位姿 / 世界系位姿 / 世界系 (x, y, z, qw, qx, qy, qz)。

        供 100Hz 控制回路每周期调用一次, 替代"伺服前读一次 + 发布时再读一次"的多往返模式。
        RTDE 不可用时返回 (None, None, None)。
        """
        pose_raw = self.get_pose_raw()
        if pose_raw is None:
            return None, None, None
        pose_world = self._tf_world_base * pose_raw
        return pose_raw, pose_world, self._xyz_quat_wxyz(pose_world)

    def get_timestamp(self):
        return self._rtde_r.getTimestamp()

    @staticmethod
    def init_control_period() -> float:
        """控制周期起始时刻 (time.perf_counter, 秒)。"""
        return time.perf_counter()

    @staticmethod
    def wait_control_period(period_start: float, interval: float) -> None:
        """固定节拍等待: 睡眠到 period_start + interval。

        - 分段睡眠 (≤2ms) 以便及时响应退出/命令;
        - 已超时立即返回, 不做追赶补偿 (不累积漂移)。
        """
        deadline = period_start + interval
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.002))

    def move_tool_speed_xyz_wxwywz(
        self,
        x: float = 0,
        y: float = 0,
        z: float = 0,
        wx: float = 0,
        wy: float = 0,
        wz: float = 0,
        acc: float = 0.1,
    ):
        pose_raw = self.get_pose_raw()
        if pose_raw is None:
            return

        rot_base_tcp = np.asarray(pose_raw.orient.matrix, dtype=float)
        linear_tcp = np.asarray([x, y, z], dtype=float)
        angular_tcp = np.asarray([wx, wy, wz], dtype=float)

        linear_base = rot_base_tcp @ linear_tcp
        angular_base = rot_base_tcp @ angular_tcp

        self._rtde_c.speedL(
            [
                float(linear_base[0]),
                float(linear_base[1]),
                float(linear_base[2]),
                float(angular_base[0]),
                float(angular_base[1]),
                float(angular_base[2]),
            ],
            acceleration=acc,
        )

    def follow_abs_xyz_wxyz(
        self,
        x: float,
        y: float,
        z: float,
        qw: float,
        qx: float,
        qy: float,
        qz: float,
        kp: float = 2.0,
        acc: float = 0.1,
        vel: float = 0.10,
        rot_kp: float = 2.0,
        rot_vel: float = 1.0,
        pos_deadband: float = 0.002,
        rot_deadband: float = 0.02,
    ) -> None:
        current_pose = self.get_pose()
        current_position_world = np.asarray(
            [current_pose.pos.x, current_pose.pos.y, current_pose.pos.z],
            dtype=float,
        )
        target_position_world = np.asarray([x, y, z], dtype=float)
        position_error_world = target_position_world - current_position_world

        current_rotation_world = Rotation.from_matrix(
            np.asarray(current_pose.orient.matrix, dtype=float)
        )
        target_rotation_world = Rotation.from_quat([qx, qy, qz, qw])
        rotation_error_world = (
            target_rotation_world * current_rotation_world.inv()
        ).as_rotvec()

        if (
            np.linalg.norm(position_error_world) <= pos_deadband
            and np.linalg.norm(rotation_error_world) <= rot_deadband
        ):
            self._rtde_c.jogStop()
            return

        linear_speed_world = np.clip(position_error_world * kp, -vel, vel)
        angular_speed_world = np.clip(
            rotation_error_world * rot_kp,
            -rot_vel,
            rot_vel,
        )

        rot_world_base = np.asarray(self._tf_world_base.orient.matrix, dtype=float)
        linear_speed_base = rot_world_base.T @ linear_speed_world
        angular_speed_base = rot_world_base.T @ angular_speed_world

        self._rtde_c.jogStart(
            [
                float(linear_speed_base[0] * 1000.0),
                float(linear_speed_base[1] * 1000.0),
                float(linear_speed_base[2] * 1000.0),
                float(angular_speed_base[0]),
                float(angular_speed_base[1]),
                float(angular_speed_base[2]),
            ],
            self._rtde_c.FEATURE_BASE,
            acc=acc,
        )

    def servo_abs_xyz_wxyz(
        self,
        x: float,
        y: float,
        z: float,
        qw: float,
        qx: float,
        qy: float,
        qz: float,
        time_s: float,
        lookahead_time: float = 0.1,
        gain: float = 300.0,
        pos_deadband: float = 0.001,
        rot_deadband: float = 0.01,
        current_pose: Optional[m3d.Transform] = None,
    ) -> None:
        if current_pose is None:
            current_pose = self.get_pose()
        current_position_world = np.asarray(
            [current_pose.pos.x, current_pose.pos.y, current_pose.pos.z],
            dtype=float,
        )
        target_position_world = np.asarray([x, y, z], dtype=float)
        position_error_world = target_position_world - current_position_world

        current_rotation_world = Rotation.from_matrix(
            np.asarray(current_pose.orient.matrix, dtype=float)
        )
        target_rotation_world = Rotation.from_quat([qx, qy, qz, qw])
        rotation_error_world = (
            target_rotation_world * current_rotation_world.inv()
        ).as_rotvec()

        position_error_norm = np.linalg.norm(position_error_world)
        rotation_error_norm = np.linalg.norm(rotation_error_world)
        # 滞回: 未伺服时误差超死区才启动 servoL; 已伺服时误差回落到
        # 0.5x 死区以下才 servoStop, 避免慢速运动时的极限环抖动
        stop_threshold_pos = pos_deadband * _SERVO_DEADBAND_HYSTERESIS_RATIO
        stop_threshold_rot = rot_deadband * _SERVO_DEADBAND_HYSTERESIS_RATIO
        if self._servo_active:
            if (
                position_error_norm <= stop_threshold_pos
                and rotation_error_norm <= stop_threshold_rot
            ):
                self._servo_active = False
                self._rtde_c.servoStop()
                return
        elif (
            position_error_norm <= pos_deadband
            and rotation_error_norm <= rot_deadband
        ):
            # 已停止且误差在死区内: 保持停止, 无需发指令
            return

        self._servo_active = True

        target_world = m3d.Transform()
        target_world.pos.x = x
        target_world.pos.y = y
        target_world.pos.z = z
        target_world.orient = m3d.Orientation(
            Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        )

        target_base = self._tf_world_base.inverse * target_world
        pose = target_base.matrix
        position = pose[:3, 3].A.flatten()
        rotvec = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
        pose_ur = list(position) + list(rotvec)

        self._rtde_c.servoL(
            pose_ur,
            0.0,
            0.0,
            time_s,
            lookahead_time,
            gain,
        )

    def stop_tool_speed(self, acc: float = 10.0) -> None:
        self._servo_active = False  # 复位死区滞回状态
        try:
            self._rtde_c.servoStop(acc)
        except Exception:
            pass
        try:
            self._rtde_c.jogStop()
        except Exception:
            pass
        try:
            self._rtde_c.speedStop(acc)
        except Exception:
            self._rtde_c.stopL(acc)

    def emergency_stop(self) -> None:
        """请求 UR 触发保护停机(protective stop): 电机抱闸保持,
        需示教器人工恢复。先软减速当前运动再触发硬停。"""
        self._servo_active = False  # 复位死区滞回状态
        # 软减速: 立刻终止 servo/jog/speed 当前模式
        try:
            self._rtde_c.servoStop(10.0)
        except Exception:
            pass
        try:
            self._rtde_c.jogStop()
        except Exception:
            pass
        try:
            self._rtde_c.speedStop(10.0)
        except Exception:
            pass
        # 触发 UR 保护停机
        try:
            self._rtde_c.triggerProtectiveStop()
        except Exception:
            pass

    def move_tool_xyz(
        self,
        x: float = 0,
        y: float = 0,
        z: float = 0,
        acc: float = 0.1,
        vel: float = 0.1,
    ) -> None:
        pose = self.get_pose()
        tool_pose_change = m3d.Transform()
        tool_pose_change.pos.x = x
        tool_pose_change.pos.y = y
        tool_pose_change.pos.z = z
        pose = pose * tool_pose_change
        self.set_pose(pose, acc=acc, vel=vel)

    def move_tool_ypr(
        self,
        yaw: float = 0,
        pitch: float = 0,
        roll: float = 0,
        acc: float = 0.1,
        vel: float = 0.1,
    ) -> None:
        pose = self.get_pose()
        pose.orient.rotate_zt(yaw)
        pose.orient.rotate_xt(pitch)
        pose.orient.rotate_yt(roll)
        self.set_pose(pose, acc=acc, vel=vel)

    def move_tool_rpy(
        self,
        roll: float = 0,
        pitch: float = 0,
        yaw: float = 0,
        acc: float = 0.1,
        vel: float = 0.1,
    ) -> None:
        pose = self.get_pose()
        pose.orient.rotate_yt(roll)
        pose.orient.rotate_xt(pitch)
        pose.orient.rotate_zt(yaw)
        self.set_pose(pose, acc=acc, vel=vel)

    def move_tool_xyzypr(
        self,
        x: float = 0,
        y: float = 0,
        z: float = 0,
        pitch: float = 0,
        roll: float = 0,
        yaw: float = 0,
        acc: float = 0.1,
        vel: float = 0.1,
    ) -> None:
        pose = self.get_pose()
        pose.orient.rotate_zt(yaw)
        pose.orient.rotate_xt(pitch)
        pose.orient.rotate_yt(roll)
        tool_pose_change = m3d.Transform()
        tool_pose_change.pos.x = x
        tool_pose_change.pos.y = y
        tool_pose_change.pos.z = z
        pose = pose * tool_pose_change
        self.set_pose(pose, acc=acc, vel=vel)

    def move_abs_xyz_quat(
        self,
        x: float,
        y: float,
        z: float,
        qw: float,
        qx: float,
        qy: float,
        qz: float,
        acc: float = 0.1,
        vel: float = 0.1,
        asynchronous: Optional[bool] = None,
    ) -> None:
        """
        在世界坐标系下，将末端执行器移动到指定的绝对位置和四元数姿态。

        参数:
            x, y, z: 目标位置 (m)
            qw, qx, qy, qz: 目标姿态四元数 (w, x, y, z 顺序)
            acc: 加速度 (m/s²)
            vel: 速度 (m/s)
        """
        # 构造世界坐标系下的目标位姿
        pose = m3d.Transform()
        pose.pos.x = x
        pose.pos.y = y
        pose.pos.z = z

        # 四元数转旋转矩阵 (scipy 需要 (x, y, z, w) 顺序)
        quat_xyzw = [qx, qy, qz, qw]
        rot_mat = Rotation.from_quat(quat_xyzw).as_matrix()
        pose.orient = m3d.Orientation(rot_mat)
        # pose.orient.rotate_zt(pi)
        # 调用 set_pose 执行运动
        if asynchronous is None:
            self.set_pose(pose, acc=acc, vel=vel)
        else:
            self.set_pose(pose, acc=acc, vel=vel, asynchronous=asynchronous)

    # ==============================================

    def go_home(self, acc: float = 0.1, vel: float = 0.1) -> None:
        pose = build_home_transform(self._ip)
        self._servo_active = False
        current = self.get_pose_raw()
        target_base = self._tf_world_base.inverse * pose
        target_matrix = target_base.matrix
        target = np.r_[
            np.asarray(target_matrix[:3, 3]).reshape(3),
            Rotation.from_matrix(np.asarray(target_matrix[:3, :3])).as_rotvec(),
        ]
        start = np.asarray(current.matrix[:3, 3]).reshape(3)
        start_pose = np.r_[start, Rotation.from_matrix(np.asarray(current.matrix[:3, :3])).as_rotvec()]
        duration = max(1.0, float(np.linalg.norm(target[:3] - start_pose[:3])) / max(vel, 1e-6))
        period = 1.0 / 125.0
        rotations = Slerp([0.0, duration], Rotation.from_rotvec(np.stack([start_pose[3:], target[3:]])))
        started = time.monotonic()
        iteration = 0
        try:
            while True:
                elapsed = min(time.monotonic() - started, duration)
                fraction = elapsed / duration
                command = np.empty(6)
                command[:3] = start_pose[:3] * (1.0 - fraction) + target[:3] * fraction
                command[3:] = rotations([elapsed]).as_rotvec()[0]
                if not self._rtde_c.servoL(command.tolist(), 0.0, 0.0, period, 0.1, 300.0):
                    raise RuntimeError("UR5 rejected servoL while returning Home")
                if elapsed >= duration:
                    break
                time.sleep(max(0.0, started + (iteration + 1) * period - time.monotonic()))
                iteration += 1
        finally:
            self._rtde_c.servoStop(2.0)

    def reupload(self):
        self._rtde_c.reuploadScript()

    def close(self) -> None:
        try:
            self._rtde_c.stopScript()
        except Exception:
            pass
        try:
            self._rtde_c.disconnect()
        except Exception:
            pass
        try:
            self._rtde_r.disconnect()
        except Exception:
            pass

    def isready(self):
        return self._rtde_c.isProgramRunning()
