"""连接左壁上的 UR5，并移动到世界坐标系 Home 点。

左壁安装使用与右壁对称的底座变换，但 TCP 姿态保持右壁原来的目标姿态。
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from rtde_control import RTDEControlInterface


# 左臂/左壁配置。项目约定左臂 IP 为 192.168.1.101。
LEFT_ROBOT_IP = "192.168.1.101"
LEFT_HOME_WORLD = (-0.45, 0.1, 0.07)

# 左壁相对右壁镜像安装；单位为度。
LEFT_MOUNT_Y_DEG = -45.0

# Home 坐标系相对原世界坐标系绕 Z 轴逆时针旋转。
WORLD_FRAME_RZ_DEG = 90.0

# Home 时让 TCP 的 +Z 轴严格朝世界坐标系 -Z 方向。
TCP_WORLD_ROTATION_Y_DEG = 180.0
SPEED = 0.1  # m/s
ACCELERATION = 0.1  # m/s²
CONTROL_FREQUENCY_HZ = 125.0
LOOKAHEAD_TIME_S = 0.1
SERVO_GAIN = 300.0

def home_pose_in_robot_base(home_world, mount_y_deg):
    """把旋转后世界坐标系中的 Home TCP 位姿换算到 UR Base 系。"""
    world_home = np.asarray(home_world, dtype=float)
    world_frame_rotation = Rotation.from_euler("z", WORLD_FRAME_RZ_DEG, degrees=True)
    robot_mount_rotation = Rotation.from_euler("y", mount_y_deg, degrees=True)

    # 新坐标系相对旧世界系旋转 +90°，再将旧世界系坐标换到 UR Base。
    base_position = robot_mount_rotation.inv().apply(
        world_frame_rotation.apply(world_home)
    )

    # 目标 TCP 姿态先在旋转后的坐标系定义，再换到 UR Base 系。
    tcp_in_rotated_world = Rotation.from_euler(
        "y", TCP_WORLD_ROTATION_Y_DEG, degrees=True
    )
    tcp_in_old_world = world_frame_rotation * tcp_in_rotated_world
    tcp_in_base = robot_mount_rotation.inv() * tcp_in_old_world
    return np.r_[base_position, tcp_in_base.as_rotvec()].tolist()


def smooth_servo_to_pose(control, receiver, target_pose, frequency_hz=CONTROL_FREQUENCY_HZ,
                         speed=SPEED, acceleration=ACCELERATION,
                         lookahead_time=LOOKAHEAD_TIME_S, gain=SERVO_GAIN):
    """Move to a UR pose with the same fixed-rate servoL contract as replay.

    UR poses use metres plus rotation vectors. Position is linearly interpolated
    and orientation uses Slerp so no Euler/quaternion convention is introduced.
    """
    target = np.asarray(target_pose, dtype=np.float64)
    current = np.asarray(receiver.getActualTCPPose(), dtype=np.float64)
    if current.shape != (6,) or target.shape != (6,) or not np.isfinite(current).all() or not np.isfinite(target).all():
        raise RuntimeError("UR5 returned an invalid TCP pose")
    distance = float(np.linalg.norm(target[:3] - current[:3]))
    angle = float((Rotation.from_rotvec(current[3:]).inv() * Rotation.from_rotvec(target[3:])).magnitude())
    duration = max(1.0, distance / max(speed, 1e-6), angle / max(1.0, acceleration))
    times = np.array([0.0, duration])
    rotations = Slerp(times, Rotation.from_rotvec(np.stack([current[3:], target[3:]])))
    period = 1.0 / frequency_hz
    start = time.monotonic()
    try:
        iteration = 0
        while True:
            elapsed = min(time.monotonic() - start, duration)
            fraction = elapsed / duration
            pose = np.empty(6, dtype=np.float64)
            pose[:3] = current[:3] * (1.0 - fraction) + target[:3] * fraction
            pose[3:] = rotations([elapsed]).as_rotvec()[0]
            if not control.servoL(pose.tolist(), 0.0, 0.0, period, lookahead_time, gain):
                raise RuntimeError("UR5 rejected servoL while returning Home")
            if elapsed >= duration:
                break
            deadline = start + (iteration + 1) * period
            time.sleep(max(0.0, deadline - time.monotonic()))
            iteration += 1
    finally:
        control.servoStop(2.0)


def main():
    arms = (
        ("左臂", LEFT_ROBOT_IP, LEFT_HOME_WORLD, LEFT_MOUNT_Y_DEG),
    )
    connected = []
    try:
        for name, ip, home, mount in arms:
            print(f"连接{name}: {ip}")
            robot = RTDEControlInterface(ip)
            connected.append(robot)
            print(f"{name}回 Home: {home}")
            receiver = __import__("rtde_receive", fromlist=["RTDEReceiveInterface"]).RTDEReceiveInterface(
                ip, CONTROL_FREQUENCY_HZ
            )
            smooth_servo_to_pose(robot, receiver, home_pose_in_robot_base(home, mount))
            receiver.disconnect()
    finally:
        for robot in connected:
            robot.stopScript()
            robot.disconnect()


if __name__ == "__main__":
    main()
