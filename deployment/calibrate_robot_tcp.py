"""Read-only UR fixed-point sampling and offline flange-to-G TCP calibration."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def transform(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("位姿必须是 6 个有限数值，单位 m/rad")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never replace a previous measurement or calibration.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_samples(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    poses = np.asarray(data["flange_poses_m_rad"], dtype=float)
    if poses.ndim != 2 or poses.shape[1:] != (6,) or len(poses) < 4:
        raise ValueError("至少需要 4 个法兰位姿样本，建议 6～10 个")
    return np.stack([transform(pose) for pose in poses])


def spread_report(points):
    center = points.mean(axis=0)
    errors = np.linalg.norm(points - center, axis=1)
    return {
        "fixed_point_base_m": center.tolist(),
        "rms_mm": float(np.sqrt(np.mean(errors**2)) * 1000),
        "max_mm": float(errors.max() * 1000),
        "per_sample_mm": (errors * 1000).tolist(),
    }


def solve_position(transforms):
    if len(transforms) < 4:
        raise ValueError("至少需要 4 个样本")
    rotations = transforms[:, :3, :3]
    positions = transforms[:, :3, 3]
    matrix = np.concatenate(
        [rotations, np.broadcast_to(-np.eye(3), rotations.shape)], axis=2
    ).reshape(-1, 6)
    solution, _, rank, singular = np.linalg.lstsq(matrix, -positions.reshape(-1), rcond=None)
    condition = float(singular[0] / singular[-1]) if rank == 6 else float("inf")
    if rank < 6 or condition > 100:
        raise ValueError("姿态变化不足或接近退化：请绕不同轴倾斜采样，不能只平移或只绕一个轴旋转")
    offset = solution[:3]
    points = np.einsum("nij,j->ni", rotations, offset) + positions
    return offset, {"condition_number": condition, **spread_report(points)}


def collect(args):
    from rtde_receive import RTDEReceiveInterface

    path = Path(args.output)
    if path.exists():
        raise ValueError(f"文件已存在，请换一个输出路径：{path}")
    inverse_tcp = np.linalg.inv(transform(args.active_tcp))
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_ip": args.robot_ip,
        "active_tcp_offset_m_rad": args.active_tcp,
        "reference": args.reference,
        "tcp_poses_m_rad": [],
        "flange_poses_m_rad": [],
    }
    receiver = None
    print("只读取位姿，不发送运动、freedrive 或 TCP 设置命令。")
    print("请核实 --active-tcp 与控制器激活值一致；采样期间不得改变它。")
    print("用现有手动控制对点，停稳后回车采样；u 撤销上一点；q 保存退出。")
    print("每个样本必须用同一工具参考点对准同一个固定点，建议 6～10 个不同姿态。")
    try:
        receiver = RTDEReceiveInterface(args.robot_ip, 125.0)
        while True:
            command = input(f"已采 {len(payload['flange_poses_m_rad'])} 点 > ").strip().lower()
            if command == "q":
                break
            if command == "u":
                if payload["flange_poses_m_rad"]:
                    payload["flange_poses_m_rad"].pop()
                    payload["tcp_poses_m_rad"].pop()
                continue
            if command:
                print("请输入回车、u 或 q")
                continue
            readings, timestamps = [], []
            for _ in range(25):
                if not receiver.isConnected():
                    raise RuntimeError("机器人接收连接已断开")
                timestamps.append(receiver.getTimestamp())
                readings.append(transform(receiver.getActualTCPPose()))
                time.sleep(0.02)
            if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
                print("拒绝样本：机器人时间戳未持续递增，请检查连接")
                continue
            readings = np.stack(readings)
            positions = readings[:, :3, 3]
            rotations = Rotation.from_matrix(readings[:, :3, :3])
            mean_rotation = rotations.mean()
            translation_drift = np.linalg.norm(positions - positions.mean(axis=0), axis=1).max()
            rotation_drift = (mean_rotation.inv() * rotations).magnitude().max()
            if translation_drift > 0.0003 or rotation_drift > np.deg2rad(0.2):
                print("拒绝样本：机械臂尚未停稳，请停稳后重新采样")
                continue
            tcp_pose = np.r_[positions.mean(axis=0), mean_rotation.as_rotvec()]
            flange = transform(tcp_pose) @ inverse_tcp
            flange_pose = np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_rotvec()]
            payload["tcp_poses_m_rad"].append(tcp_pose.tolist())
            payload["flange_poses_m_rad"].append(flange_pose.tolist())
            print("已记录法兰位姿：", np.round(flange_pose, 6).tolist())
    except (KeyboardInterrupt, EOFError):
        print("\n结束采样，保存已有点。")
    finally:
        if receiver is not None:
            receiver.disconnect()
        if payload["flange_poses_m_rad"]:
            write_new(path, payload)
            print(f"原始样本已保存：{path}")


def solve(args):
    samples = load_samples(args.samples)
    offset, report = solve_position(samples)
    result = {
        "schema_version": 1,
        "source_samples": str(Path(args.samples).resolve()),
        "translation_m": offset.tolist(),
        "fit": report,
        "fit_within_tolerance": report["max_mm"] <= args.tolerance_mm,
        "tolerance_mm": args.tolerance_mm,
        "rotation_status": "not_calibrated_by_fixed_point",
        "independent_validation_required": True,
    }
    if args.g_rotvec is not None:
        pose = np.r_[offset, args.g_rotvec]
        result.update({
            "rotation_status": "user_supplied_requires_direction_verification",
            "tcp_offset_m_rad": pose.tolist(),
            "T_F_G": transform(pose).tolist(),
        })
    write_new(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("位置求解已保存；尚未设置机器人 TCP。对点不能求方向，也不能证明 G 点选取正确。")
    if args.g_rotvec is None:
        print("尚缺 G 相对法兰的旋转向量，不能将位置结果直接当成完整 TCP。")
    return 0 if result["fit_within_tolerance"] else 2


def validate(args):
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    if Path(calibration["source_samples"]).resolve() == Path(args.samples).resolve():
        raise ValueError("独立验证必须使用另一组新采样，不能复用拟合文件")
    offset = np.asarray(calibration["translation_m"], dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("无效的位置标定")
    samples = load_samples(args.samples)
    solve_position(samples)  # Reject validation sets with unobservable orientations too.
    points = np.einsum("nij,j->ni", samples[:, :3, :3], offset) + samples[:, :3, 3]
    report = spread_report(points)
    report.update({
        "calibration": str(Path(args.calibration).resolve()),
        "validation_samples": str(Path(args.samples).resolve()),
        "tolerance_mm": args.tolerance_mm,
        "position_within_tolerance": report["max_mm"] <= args.tolerance_mm,
        "orientation_verified": False,
    })
    write_new(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("只验证固定参考点的位置一致性；轴方向和开合重复性需另外验证。")
    return 0 if report["position_within_tolerance"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("collect", help="只读采集：手动对点后按回车")
    sample.add_argument("--robot-ip", required=True)
    sample.add_argument("--active-tcp", nargs=6, type=float, required=True,
                        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                        help="当前激活的法兰到 TCP 偏置，m/rad；确认法兰为 TCP 时填六个零")
    sample.add_argument("--reference", required=True, help="记录 G 参考点、夹爪状态及保持方式")
    sample.add_argument("--output", required=True)
    sample.set_defaults(run=collect)
    fit = commands.add_parser("solve", help="离线求解 TCP 位置")
    fit.add_argument("--samples", required=True)
    fit.add_argument("--g-rotvec", nargs=3, type=float,
                     help="独立确定的 G 相对法兰旋转向量，rad；不是欧拉角")
    fit.add_argument("--output", required=True)
    fit.add_argument("--tolerance-mm", type=float, default=2.0)
    fit.set_defaults(run=solve)
    check = commands.add_parser("validate", help="用另一组对点样本验证位置")
    check.add_argument("--samples", required=True)
    check.add_argument("--calibration", required=True)
    check.add_argument("--output", required=True)
    check.add_argument("--tolerance-mm", type=float, default=2.0)
    check.set_defaults(run=validate)
    args = parser.parse_args()
    if hasattr(args, "tolerance_mm") and (not np.isfinite(args.tolerance_mm) or args.tolerance_mm <= 0):
        parser.error("--tolerance-mm 必须是有限正数")
    try:
        return args.run(args) or 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
