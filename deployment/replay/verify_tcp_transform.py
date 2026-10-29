"""Slow, explicit UR5 tests using the computer-side flange-to-gripper calibration."""
from __future__ import annotations

import argparse
import json
import math
import socket
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# Permit direct execution from this directory while resolving shared helpers
# and calibration files from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrate_robot_tcp import transform

ROOT = PROJECT_ROOT
AXES = {"x": np.array([1., 0., 0.]), "y": np.array([0., 1., 0.]), "z": np.array([0., 0., 1.])}


def pose(matrix):
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_rotvec()]


def load_calibration(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    fg = transform(data["tcp_offset_m_rad"])
    stored = np.asarray(data["T_F_G"], dtype=float)
    if stored.shape != (4, 4) or not np.allclose(fg, stored, atol=1e-8, rtol=0):
        raise ValueError("标定中的 T_F_G 和 tcp_offset_m_rad 不一致")
    probe = np.asarray(data.get("probe_offset_in_G_m", [0.08, 0, 0]), dtype=float)
    if probe.shape != (3,) or not np.isfinite(probe).all():
        raise ValueError("探针偏置无效")
    gp = np.eye(4)
    gp[:3, 3] = probe
    return fg, gp


def gripper_from_active(bh, fg, fh):
    """B: base, F: flange, G: gripper, H: controller's active TCP."""
    return bh @ np.linalg.inv(fh) @ fg


def active_from_gripper(bg, fg, fh):
    return bg @ np.linalg.inv(fg) @ fh


def read_controller(ip):
    """Read active offset/program/joint state on passive port 30012, without uploading a script."""
    def exact(sock, n):
        result = bytearray()
        while len(result) < n:
            chunk = sock.recv(n - len(result))
            if not chunk:
                raise RuntimeError("机器人状态连接断开")
            result.extend(chunk)
        return bytes(result)

    with socket.create_connection((ip, 30012), timeout=4) as sock:
        for _ in range(100):
            length = struct.unpack("!I", exact(sock, 4))[0]
            if not 5 <= length <= 1000000:
                raise RuntimeError("状态数据长度无效")
            packet = exact(sock, length - 4)
            if packet[0] != 16:
                continue
            state, pos = {}, 1
            while pos < len(packet):
                size, kind = struct.unpack_from("!IB", packet, pos)
                if size < 5 or pos + size > len(packet):
                    raise RuntimeError("状态子包无效")
                payload = packet[pos + 5:pos + size]
                if kind == 0 and len(payload) >= 17:
                    values = struct.unpack_from("!Q???????BB", payload)
                    state.update(dict(zip(("timestamp", "connected", "enabled", "power_on", "emergency_stopped",
                        "protective_stopped", "program_running", "program_paused", "robot_mode", "control_mode"), values)))
                elif kind == 1 and len(payload) == 246:
                    state["joint_speeds_rad_s"] = [struct.unpack_from("!d", payload, j * 41 + 16)[0] for j in range(6)]
                elif kind == 4 and len(payload) >= 96:
                    values = struct.unpack_from("!12d", payload)
                    state["actual_tcp_pose_m_rad"] = list(values[:6])
                    state["tcp_offset_m_rad"] = list(values[6:])
                pos += size
            if all(k in state for k in ("tcp_offset_m_rad", "joint_speeds_rad_s", "program_running")):
                for key in ("tcp_offset_m_rad", "actual_tcp_pose_m_rad", "joint_speeds_rad_s"):
                    if not np.isfinite(state[key]).all():
                        raise RuntimeError("控制器返回非有限值")
                return state
    raise RuntimeError("无法读取当前 TCP 偏置")


def ready(state):
    if state["program_running"] or state["program_paused"]:
        raise RuntimeError("机器人有运行或暂停的程序，请先在示教器停止该程序")
    if state["emergency_stopped"] or state["protective_stopped"]:
        raise RuntimeError("机器人处于急停或保护停止")
    if not state["power_on"] or state["robot_mode"] != 7:
        raise RuntimeError("机器人尚未上电就绪")
    if max(abs(v) for v in state["joint_speeds_rad_s"]) > 0.001:
        raise RuntimeError("机器人正在运动，请停稳后重试")


def path_segment(bg, fg, fh, gp, kind, axis, step, angle, pivot, speed, omega):
    gc = gp if kind == "rotation" and pivot == "probe" else np.eye(4)
    start = bg @ gc
    delta = np.eye(4)
    if kind == "translation":
        delta[:3, 3] = AXES[axis] * step
    else:
        delta[:3, :3] = Rotation.from_rotvec(AXES[axis] * angle).as_matrix()
    end = start @ delta
    rotations = Slerp([0, 1], Rotation.from_matrix(np.stack([start[:3, :3], end[:3, :3]])))
    ci = np.linalg.inv(gc)

    def at(fraction):
        bc = np.eye(4)
        bc[:3, :3] = rotations([fraction]).as_matrix()[0]
        bc[:3, 3] = start[:3, 3] * (1-fraction) + end[:3, 3] * fraction
        target_g = bc @ ci
        return target_g, active_from_gripper(target_g, fg, fh)

    # Smoothstep has maximum slope 1.5; include the offset lever arms so that
    # flange, active TCP, G and probe all follow low-speed target trajectories.
    radius = max(np.linalg.norm((fg @ gc)[:3, 3]),
                 np.linalg.norm((np.linalg.inv(fh) @ fg @ gc)[:3, 3]),
                 np.linalg.norm((np.linalg.inv(gp) @ gc)[:3, 3]), np.linalg.norm(gc[:3, 3]))
    duration = max(2., 1.5 * step / speed) if kind == "translation" else max(2., 1.5 * angle / omega, 1.5 * radius * angle / speed)
    return at, duration, gc


def print_plan(bg, fg, fh, gp, args):
    print("控制器实际激活 TCP 偏置 (m/rad):", pose(fh).round(9).tolist())
    print("软件计算的夹爪 G 基座位姿 (m/rad):", pose(bg).round(9).tolist())
    print("探针尖基座位置 (m):", (bg @ gp)[:3, 3].round(9).tolist())
    print("夹爪正轴在基座坐标系中的方向:")
    for axis in args.axes:
        print(f"  +{axis.upper()}_G:", (bg[:3, :3] @ AXES[axis]).round(6).tolist())
    rows = []
    kinds = ["translation", "rotation"] if args.test == "both" else [args.test]
    for kind in kinds:
        for axis in args.axes:
            at, duration, _ = path_segment(bg, fg, fh, gp, kind, axis, args.step_mm / 1000,
                math.radians(args.angle_deg), args.pivot, args.speed_mm_s / 1000, math.radians(args.angular_speed_deg_s))
            target_g, target_h = at(1)
            backend = "moveL" if kind == "translation" else "servoL"
            if kind == "translation":
                distance = np.linalg.norm(target_h[:3, 3] - active_from_gripper(bg, fg, fh)[:3, 3])
                speed = args.speed_mm_s / 1000
                duration = distance / speed + 2*speed / .01
            label = f"{kind} +{axis.upper()}_G"
            print(f"{label} ({backend}): 单程约 {duration:.1f} s, 控制器目标", pose(target_h).round(9).tolist())
            rows.append({"test": label, "backend": backend, "duration_s": float(duration), "target_G_m_rad": pose(target_g).tolist(),
                         "target_active_tcp_m_rad": pose(target_h).tolist()})
    return rows


def check_live(receiver, control, fh, desired_g, fg, stale):
    if not receiver.isConnected() or not control.isConnected():
        raise RuntimeError("RTDE 连接断开")
    if receiver.isProtectiveStopped() or receiver.isEmergencyStopped():
        raise RuntimeError("机器人触发安全停止")
    stamp = receiver.getTimestamp()
    if stamp != stale[0]:
        stale[:] = [stamp, time.monotonic()]
    if time.monotonic() - stale[1] > 0.5:
        raise RuntimeError("RTDE 状态超过 0.5 秒未更新")
    bh = transform(receiver.getActualTCPPose())
    bg = gripper_from_active(bh, fg, fh)
    error = np.linalg.norm(bg[:3, 3] - desired_g[:3, 3])
    angle = Rotation.from_matrix(desired_g[:3, :3].T @ bg[:3, :3]).magnitude()
    if error > 0.005 or angle > math.radians(3):
        raise RuntimeError(f"跟随误差过大：{error*1000:.2f} mm / {math.degrees(angle):.2f} deg")
    velocity = np.asarray(receiver.getActualTCPSpeed())
    if not np.isfinite(velocity).all() or np.linalg.norm(velocity[:3]) > 0.01 or np.linalg.norm(velocity[3:]) > 0.10:
        raise RuntimeError("实测 TCP 速度超过验证限值")
    if not np.isfinite(receiver.getActualQd()).all() or max(abs(v) for v in receiver.getActualQd()) > 0.15:
        raise RuntimeError("实测关节速度超过验证限值")
    return bg


def run_translation_leg(receiver, control, fh, fg, at, speed, reverse=False):
    """Use go_home's moveL command, asynchronously so Ctrl+C and monitoring work."""
    if not control.isProgramRunning():
        raise RuntimeError("RTDE 控制脚本未运行，未发送运动指令")
    desired_g, desired_h = at(0 if reverse else 1)
    start_g = gripper_from_active(transform(receiver.getActualTCPPose()), fg, fh)
    vector = desired_g[:3, 3] - start_g[:3, 3]
    distance = float(np.linalg.norm(vector))
    if distance < 1e-6:
        return start_g
    unit = vector / distance
    duration = distance / speed + 2 * speed / 0.01
    timeout = max(10., duration * 4 + 5)
    stale = [receiver.getTimestamp(), time.monotonic()]
    begin = time.monotonic()
    next_log = begin
    stable = 0
    print(f"[运动] moveL {'返回' if reverse else '去程'}：{distance*1000:.3f} mm，"
          f"速度 {speed*1000:.3f} mm/s，加速度 10 mm/s²", flush=True)
    print("[运动] 目标（当前激活 TCP，m/rad）:", pose(desired_h).round(9).tolist(), flush=True)
    try:
        if not control.moveL(pose(desired_h).tolist(), speed, 0.01, asynchronous=True):
            raise RuntimeError("moveL 返回失败")
        while True:
            now = time.monotonic()
            current_g = gripper_from_active(transform(receiver.getActualTCPPose()), fg, fh)
            displacement = current_g[:3, 3] - start_g[:3, 3]
            along = float(displacement @ unit)
            # Check against the nearest point on the planned line, rather than
            # the endpoint (which may still be 10 mm away at the start).
            reference = desired_g.copy()
            reference[:3, 3] = start_g[:3, 3] + np.clip(along, 0, distance) * unit
            current_g = check_live(receiver, control, fh, reference, fg, stale)
            if not control.isProgramRunning():
                raise RuntimeError("moveL 期间 RTDE 控制脚本停止")
            if along < -0.002 or along > distance + 0.002:
                raise RuntimeError("实际位移偏离规划运动范围")
            error = float(np.linalg.norm(current_g[:3, 3] - desired_g[:3, 3]))
            velocity = np.asarray(receiver.getActualTCPSpeed())
            if now >= next_log:
                print(f"[运动] {now-begin:.1f} 秒；已沿目标方向移动 {along*1000:.3f} mm / {distance*1000:.3f} mm；"
                      f"剩余 {error*1000:.3f} mm；速度缩放 {receiver.getSpeedScaling():.3f}", flush=True)
                next_log = now + 1
            # Do not infer execution from an accepted command or speed_scaling;
            # require actual endpoint arrival and stationary feedback.
            angular_error = Rotation.from_matrix(desired_g[:3, :3].T @ current_g[:3, :3]).magnitude()
            settled = (error < 0.0001 and angular_error < math.radians(.2)
                       and np.linalg.norm(velocity[:3]) < 0.00015 and np.linalg.norm(velocity[3:]) < 0.002)
            stable = stable + 1 if settled else 0
            if stable >= 5:
                return current_g
            if now - begin > timeout:
                raise RuntimeError(f"moveL 超时：实测移动 {along*1000:.3f} mm，未到达目标；"
                                   "请检查示教器速度滑块、暂停状态和控制器日志")
            time.sleep(.02)
    except BaseException:
        # An asynchronous move must be stopped before propagating Ctrl+C/errors.
        try:
            control.stopL(0.1)
        except Exception as exc:
            print(f"停止 moveL 时提示：{exc}", flush=True)
        raise


def run_leg(receiver, control, fh, fg, at, duration, reverse=False, kind="rotation", speed=.001):
    if kind == "translation":
        return run_translation_leg(receiver, control, fh, fg, at, speed, reverse)
    if not control.isProgramRunning():
        raise RuntimeError("RTDE 控制脚本未运行，未发送运动指令")
    scaling = receiver.getSpeedScaling()
    period = 1 / 125
    steps = max(1, math.ceil(duration / period))
    stale = [receiver.getTimestamp(), time.monotonic()]
    start_g = gripper_from_active(transform(receiver.getActualTCPPose()), fg, fh)
    print(f"[运动] {'返回' if reverse else '去程'}开始，单程约 {duration:.1f} 秒，速度缩放 {scaling:.3f}", flush=True)
    for i in range(steps + 1 + 63):
        tick = control.initPeriod()
        s = min(i / steps, 1.)
        blend = s*s*(3-2*s)
        desired_g, desired_h = at(1-blend if reverse else blend)
        actual_g = check_live(receiver, control, fh, desired_g, fg, stale)
        if not control.isProgramRunning():
            raise RuntimeError("运动期间 RTDE 控制脚本停止")
        if i % 125 == 0:
            actual_mm = np.linalg.norm(actual_g[:3, 3] - start_g[:3, 3])*1000
            actual_deg = math.degrees(Rotation.from_matrix(start_g[:3, :3].T @ actual_g[:3, :3]).magnitude())
            print(f"[运动] {i*period:.1f}/{duration:.1f} 秒；实测位移 {actual_mm:.3f} mm，转角 {actual_deg:.3f} deg", flush=True)
        # servoL speed/acceleration arguments are unused in ur-rtde. Slow speed
        # comes from the timed interpolated pose trajectory, not these arguments.
        if not control.servoL(pose(desired_h).tolist(), 0.001, 0.01, period, 0.15, 200):
            raise RuntimeError("servoL 返回失败")
        control.waitPeriod(tick)
    if not control.servoStop(0.1):
        raise RuntimeError("servoStop 返回失败")
    time.sleep(0.2)
    return gripper_from_active(transform(receiver.getActualTCPPose()), fg, fh)


def main():
    # Report progress even when stdout is redirected through deploy.py or tee.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.1.102")
    parser.add_argument("--calibration", type=Path, default=ROOT / "calibration/tcp_gripper_01.json")
    parser.add_argument("--test", choices=["translation", "rotation", "both"], default="translation")
    parser.add_argument("--axes", nargs="+", choices=list(AXES), default=list(AXES))
    parser.add_argument("--pivot", choices=["gripper", "probe"], default="gripper")
    parser.add_argument("--step-mm", type=float, default=3.)
    parser.add_argument("--angle-deg", type=float, default=2.)
    parser.add_argument("--speed-mm-s", type=float, default=1.)
    parser.add_argument("--angular-speed-deg-s", type=float, default=0.5)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--offline-pose", nargs=6, type=float, help="离线预览：控制器当前 TCP 位姿，m/rad")
    parser.add_argument("--offline-active-tcp", nargs=6, type=float, default=[0.]*6)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    for name, upper in (("step_mm", 10), ("angle_deg", 5), ("speed_mm_s", 2), ("angular_speed_deg_s", 1)):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 < value <= upper:
            parser.error(f"--{name.replace('_', '-')} 必须在 (0, {upper}] 内")
    if args.enable_motion and args.offline_pose is not None:
        parser.error("离线预览不能同时开启运动")
    if args.report and args.report.exists():
        parser.error("报告已存在，请换一个路径")
    fg, gp = load_calibration(args.calibration)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "calibration": str(args.calibration.resolve()),
              "robot_ip": args.robot_ip, "mode": "motion" if args.enable_motion else "preview",
              "physical_geometry_verified": False, "completed": False, "legs": []}
    receiver = control = None
    report_stream = None

    def checkpoint(stage):
        report["stage"] = stage
        print(f"[阶段] {stage}", flush=True)
        if report_stream is not None:
            report_stream.seek(0)
            json.dump(report, report_stream, ensure_ascii=False, indent=2, allow_nan=False)
            report_stream.truncate()
            report_stream.flush()

    try:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            report_stream = args.report.open("x", encoding="utf-8")
        checkpoint("读取控制器状态 / 离线位姿")
        if args.offline_pose is not None:
            fh = transform(args.offline_active_tcp)
            bh = transform(args.offline_pose)
        else:
            state = read_controller(args.robot_ip)
            report["initial_controller_state"] = state
            fh, bh = transform(state["tcp_offset_m_rad"]), transform(state["actual_tcp_pose_m_rad"])
        bg = gripper_from_active(bh, fg, fh)
        report["preview"] = print_plan(bg, fg, fh, gp, args)
        checkpoint("预览完成")
        if not args.enable_motion:
            print("只读预览结束。开启实际测试需加 --enable-motion，并在终端输入 RUN。")
            report["completed"] = True
            return 0
        ready(state)
        print("平移测试：观察实际夹爪轴方向；旋转测试：观察所选原点是否保持固定。")
        print("探针旋转测试请脱离工装接触；Ctrl+C 停止，停止后不会自动返回起点。")
        if input("确认已留出空间，输入 RUN 连接控制接口（连接后还需回车才移动） > ").strip() != "RUN":
            print("已取消")
            return 0
        # Check again after operator input, before uploading the control script.
        checkpoint("RUN 已确认，重新检查机器人状态")
        state = read_controller(args.robot_ip)
        ready(state)
        if not np.allclose(transform(state["tcp_offset_m_rad"]), fh, atol=1e-8, rtol=0):
            raise RuntimeError("预览后控制器 TCP 偏置发生变化")
        from rtde_receive import RTDEReceiveInterface
        from rtde_control import RTDEControlInterface
        checkpoint("建立 RTDEReceive 接收连接")
        receiver = RTDEReceiveInterface(args.robot_ip, 125.)
        checkpoint("建立 RTDEControl 控制连接并上传控制脚本")
        control = RTDEControlInterface(args.robot_ip, 125.,
            RTDEControlInterface.FLAG_UPLOAD_SCRIPT | RTDEControlInterface.FLAG_VERBOSE)
        checkpoint("控制连接已建立，检查脚本状态")
        if not control.isProgramRunning():
            raise RuntimeError("RTDEControl 已连接，但控制脚本未运行")
        report["speed_scaling"] = float(receiver.getSpeedScaling())
        report["target_speed_fraction"] = float(receiver.getTargetSpeedFraction())
        print("当前速度缩放:", report["speed_scaling"], "；目标速度比例:", report["target_speed_fraction"],
              "（静止/同步阶段的瞬时缩放可能为零，不据此拒绝运动）")
        checkpoint("读取激活 TCP 偏置")
        if not np.allclose(transform(control.getTCPOffset()), fh, atol=1e-8, rtol=0):
            raise RuntimeError("建立控制连接后 TCP 偏置发生变化")
        kinds = ["translation", "rotation"] if args.test == "both" else [args.test]
        for kind in kinds:
            for axis in args.axes:
                checkpoint(f"等待回车启动 {kind} +{axis.upper()}_G")
                if input(f"回车测试 {kind} +{axis.upper()}_G；q 结束 > ").strip().lower() == "q":
                    return 0
                checkpoint(f"开始准备 {kind} +{axis.upper()}_G")
                if not np.allclose(transform(control.getTCPOffset()), fh, atol=1e-8, rtol=0):
                    raise RuntimeError("控制器 TCP 偏置发生变化")
                bh = transform(receiver.getActualTCPPose())
                bg = gripper_from_active(bh, fg, fh)
                if np.linalg.norm(receiver.getActualTCPSpeed()) > 0.001:
                    raise RuntimeError("机器人未停稳")
                at, duration, gc = path_segment(bg, fg, fh, gp, kind, axis, args.step_mm / 1000,
                    math.radians(args.angle_deg), args.pivot, args.speed_mm_s / 1000, math.radians(args.angular_speed_deg_s))
                checkpoint("检查规划路径：可达性及控制器安全限位")
                for index, fraction in enumerate(np.linspace(0, 1, 21), 1):
                    print(f"[路径检查] {index}/21", flush=True)
                    if not control.isPoseWithinSafetyLimits(pose(at(fraction)[1]).tolist()):
                        raise RuntimeError("规划路径采样点不可达或超出控制器安全限制")
                for reverse in (False, True):
                    if reverse and input("观察实际方向/原点后，回车慢速返回；q 原地结束 > ").strip().lower() == "q":
                        return 0
                    leg_start_g = gripper_from_active(transform(receiver.getActualTCPPose()), fg, fh)
                    checkpoint(f"执行 {kind} +{axis.upper()}_G {'返回' if reverse else '去程'}")
                    backend = "moveL" if kind == "translation" else "servoL"
                    actual_g = run_leg(receiver, control, fh, fg, at, duration, reverse,
                                       kind=kind, speed=args.speed_mm_s / 1000)
                    desired_g = at(0 if reverse else 1)[0]
                    actual_point, desired_point = actual_g @ gc, desired_g @ gc
                    error_mm = np.linalg.norm(actual_point[:3, 3]-desired_point[:3, 3])*1000
                    rotation_deg = math.degrees(Rotation.from_matrix(desired_g[:3, :3].T @ actual_g[:3, :3]).magnitude())
                    displacement_mm = (actual_g[:3, 3]-leg_start_g[:3, 3])*1000
                    print("实测夹爪位移（基座 XYZ，mm）:", displacement_mm.round(3).tolist(),
                          f"；总位移 {np.linalg.norm(displacement_mm):.3f} mm")
                    print(f"软件重建的终点误差: {error_mm:.3f} mm / {rotation_deg:.3f} deg（几何精度还需实际观察/测量）")
                    report["legs"].append({"test": kind, "backend": backend, "axis": axis, "pivot": args.pivot, "return": reverse,
                        "start_G_m_rad": pose(leg_start_g).tolist(), "displacement_base_mm": displacement_mm.tolist(),
                        "actual_G_m_rad": pose(actual_g).tolist(), "expected_G_m_rad": pose(desired_g).tolist(),
                        "endpoint_error_mm": float(error_mm), "endpoint_rotation_error_deg": rotation_deg})
                    checkpoint("当前运动段完成")
                    if error_mm > 1. or rotation_deg > 0.5:
                        raise RuntimeError("终点误差超过 1 mm / 0.5 deg，请检查后再继续")
        report["completed"] = True
        return 0
    except (KeyboardInterrupt, EOFError):
        report["error"] = "operator_stop"
        report["error_stage"] = report.get("stage")
        print("\n测试已停止；不自动返回起点。")
        return 130
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        report["error"] = str(exc)
        report["error_stage"] = report.get("stage")
        print(f"错误：{exc}")
        return 1
    finally:
        checkpoint("结束控制及保存报告")
        if control is not None:
            for operation in (lambda: control.stopL(0.1), lambda: control.servoStop(0.1), control.stopScript, control.disconnect):
                try:
                    operation()
                except Exception as exc:
                    print(f"结束控制时提示：{exc}")
        if receiver is not None:
            receiver.disconnect()
        if report_stream is not None:
            checkpoint("已结束")
            report_stream.close()
            print(f"报告已保存：{args.report}")


if __name__ == "__main__":
    raise SystemExit(main())
