from dynamixel_sdk import *
import time
import sys
import msvcrt

#夹紧：1号角度变小 2号角度变大
DEVICENAME = "COM6"

BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

SERVO_IDS = [1, 2]

ADDR_PRESENT_POSITION = 132
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PROFILE_VELOCITY = 112

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
POSITION_MODE = 3

#设置电机转速
PROFILE_VELOCITY = 30
MOVE_TIMEOUT = 15.0


def check_result(packet_handler, comm_result, dxl_error):
    if comm_result != COMM_SUCCESS:
        print(packet_handler.getTxRxResult(comm_result))
        return False

    if dxl_error != 0:
        print(packet_handler.getRxPacketError(dxl_error))
        return False

    return True


def signed_32(value):
    if value >= 0x80000000:
        return value - 0x100000000
    return value


def read_servo(packet_handler, port_handler, dxl_id):
    position, result, error = packet_handler.read4ByteTxRx(
        port_handler,
        dxl_id,
        ADDR_PRESENT_POSITION
    )

    if not check_result(packet_handler, result, error):
        return None

    return signed_32(position)


def write_position(packet_handler, port_handler, dxl_id, angle):
    angle = float(angle)
    if not 0.0 <= angle <= 360.0:
        raise ValueError("角度必须在 0 到 360 度之间")

    position = round(angle * 4095.0 / 360.0)
    position = min(position, 4095)
    result, error = packet_handler.write4ByteTxRx(
        port_handler, dxl_id, ADDR_GOAL_POSITION, position
    )
    return check_result(packet_handler, result, error)


def set_torque(packet_handler, port_handler, dxl_id, enable):
    result, error = packet_handler.write1ByteTxRx(
        port_handler, dxl_id, ADDR_TORQUE_ENABLE, enable
    )
    return check_result(packet_handler, result, error)


def setup_position_mode(packet_handler, port_handler):
    for dxl_id in SERVO_IDS:
        if not set_torque(packet_handler, port_handler, dxl_id, TORQUE_DISABLE):
            return False

        result, error = packet_handler.write1ByteTxRx(
            port_handler, dxl_id, ADDR_OPERATING_MODE, POSITION_MODE
        )
        if not check_result(packet_handler, result, error):
            return False

        result, error = packet_handler.write4ByteTxRx(
            port_handler, dxl_id, ADDR_PROFILE_VELOCITY, PROFILE_VELOCITY
        )
        if not check_result(packet_handler, result, error):
            return False

    return True


def move_to_angle(packet_handler, port_handler, dxl_id, angle):
    """临时加力矩转到目标角度，到位后立即释放力矩。"""
    target = float(angle)
    if not 0.0 <= target <= 360.0:
        raise ValueError("角度必须在 0 到 360 度之间")

    if not set_torque(packet_handler, port_handler, dxl_id, TORQUE_ENABLE):
        return False
    try:
        if not write_position(packet_handler, port_handler, dxl_id, target):
            return False

        deadline = time.time() + MOVE_TIMEOUT
        while time.time() < deadline:
            position = read_servo(packet_handler, port_handler, dxl_id)
            if position is None:
                return False
            current = (position % 4096) * 360.0 / 4096.0
            error = abs((current - target + 180.0) % 360.0 - 180.0)
            if error <= 2.0:
                print(f"\nID={dxl_id} 已到达 {current:.0f}°，释放扭矩")
                return True
            time.sleep(0.05)

        print(f"\nID={dxl_id} 到位超时，释放扭矩")
        return False
    finally:
        set_torque(packet_handler, port_handler, dxl_id, TORQUE_DISABLE)


def move_by_angle(packet_handler, port_handler, dxl_id, delta):
    """Rotate by delta degrees from the current encoder position."""
    delta = float(delta)
    position = read_servo(packet_handler, port_handler, dxl_id)
    if position is None:
        return False
    current = (position % 4096) * 360.0 / 4096.0
    target = (current + delta) % 360.0
    return move_to_angle(packet_handler, port_handler, dxl_id, target)


def move_both_opposite(packet_handler, port_handler, degrees):
    """Move both motors by equal, opposite relative angles."""
    degrees = float(degrees)
    current = {}
    for dxl_id in SERVO_IDS:
        position = read_servo(packet_handler, port_handler, dxl_id)
        if position is None:
            return False
        current[dxl_id] = (position % 4096) * 360.0 / 4096.0
    targets = {
        SERVO_IDS[0]: (current[SERVO_IDS[0]] + degrees) % 360.0,
        SERVO_IDS[1]: (current[SERVO_IDS[1]] - degrees) % 360.0,
    }
    enabled = []
    try:
        for dxl_id in SERVO_IDS:
            if not set_torque(packet_handler, port_handler, dxl_id, TORQUE_ENABLE):
                return False
            enabled.append(dxl_id)
        for dxl_id in SERVO_IDS:
            if not write_position(packet_handler, port_handler, dxl_id, targets[dxl_id]):
                return False
        deadline = time.time() + MOVE_TIMEOUT
        while time.time() < deadline:
            reached = True
            for dxl_id in SERVO_IDS:
                position = read_servo(packet_handler, port_handler, dxl_id)
                if position is None:
                    return False
                angle = (position % 4096) * 360.0 / 4096.0
                error = abs((angle - targets[dxl_id] + 180.0) % 360.0 - 180.0)
                reached = reached and error <= 2.0
            if reached:
                print("\n两台电机已按相反方向到达目标，释放扭矩")
                return True
            time.sleep(0.05)
        print("\n两台电机到位超时，释放扭矩")
        return False
    finally:
        for dxl_id in enabled:
            set_torque(packet_handler, port_handler, dxl_id, TORQUE_DISABLE)


def move_both_to_angles(packet_handler, port_handler, angle_1, angle_2):
    """Move both motors simultaneously to absolute angles."""
    targets = {SERVO_IDS[0]: float(angle_1), SERVO_IDS[1]: float(angle_2)}
    if any(not 0.0 <= angle <= 360.0 for angle in targets.values()):
        raise ValueError("角度必须在 0 到 360 度之间")
    enabled = []
    try:
        for dxl_id in SERVO_IDS:
            if not set_torque(packet_handler, port_handler, dxl_id, TORQUE_ENABLE):
                return False
            enabled.append(dxl_id)
        for dxl_id in SERVO_IDS:
            if not write_position(packet_handler, port_handler, dxl_id, targets[dxl_id]):
                return False
        deadline = time.time() + MOVE_TIMEOUT
        while time.time() < deadline:
            reached = True
            for dxl_id in SERVO_IDS:
                position = read_servo(packet_handler, port_handler, dxl_id)
                if position is None:
                    return False
                angle = (position % 4096) * 360.0 / 4096.0
                error = abs((angle - targets[dxl_id] + 180.0) % 360.0 - 180.0)
                reached = reached and error <= 2.0
            if reached:
                print("\n两台电机已同时到达指定角度，释放扭矩")
                return True
            time.sleep(0.05)
        print("\n两台电机到位超时，释放扭矩")
        return False
    finally:
        for dxl_id in enabled:
            set_torque(packet_handler, port_handler, dxl_id, TORQUE_DISABLE)


def main():
    port_handler = PortHandler(DEVICENAME)
    packet_handler = PacketHandler(PROTOCOL_VERSION)

    if not port_handler.openPort():
        print("无法打开串口：", DEVICENAME)
        sys.exit(1)

    if not port_handler.setBaudRate(BAUDRATE):
        print("无法设置波特率：", BAUDRATE)
        port_handler.closePort()
        sys.exit(1)

    print("编码器读取程序已启动")
    print("输入 'ID 角度' 控制舵机；'b 角度' 让两台电机反向转动；'both 角度1 角度2' 同时回到指定角度；输入 q 退出")
    print()

    try:
        if not setup_position_mode(packet_handler, port_handler):
            print("设置位置模式失败")
            return

        while True:
            output = []

            for dxl_id in SERVO_IDS:
                data = read_servo(
                    packet_handler,
                    port_handler,
                    dxl_id
                )

                if data is None:
                    continue

                angle = (data % 4096) * 360.0 / 4096.0

                output.append(
                    f"ID={dxl_id} | 角度={angle:5.0f}°"
                )

            print("\r" + "    ".join(output), end="")

            # 非阻塞读取命令，避免影响角度刷新
            if msvcrt.kbhit():
                command = input("\n请输入目标（ID 角度、b 角度，或 both 角度1 角度2）：").strip().lower()
                if command == "q":
                    break
                try:
                    parts = command.split()
                    # Accept both "b 30" and the compact form "b30".
                    if len(parts) == 1 and parts[0].startswith("b") and parts[0] != "both":
                        parts = ["b", parts[0][1:]]
                    if len(parts) == 3 and parts[0] in ("both", "sync"):
                        move_both_to_angles(packet_handler, port_handler, parts[1], parts[2])
                        continue
                    if len(parts) == 2 and parts[0] in ("b", "both"):
                        if not parts[1]:
                            raise ValueError("请输入转动角度")
                        move_both_opposite(packet_handler, port_handler, parts[1])
                        continue
                    if len(parts) == 2:
                        dxl_id_text, angle_text = parts
                        mode = "a"
                    elif len(parts) == 3:
                        dxl_id_text, mode, angle_text = parts
                        if mode not in ("a", "r"):
                            raise ValueError("mode must be a or r")
                    else:
                        raise ValueError("format: ID angle or ID abs/rel angle")
                    dxl_id = int(dxl_id_text)
                    if dxl_id not in SERVO_IDS:
                        raise ValueError("无效舵机 ID")
                    if mode == "r": 
                        move_by_angle(packet_handler, port_handler, dxl_id, angle_text)
                    else:
                        move_to_angle(packet_handler, port_handler, dxl_id, angle_text)
                except ValueError as exc:
                    print(f"输入错误：{exc}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n程序退出")

    finally:
        for dxl_id in SERVO_IDS:
            packet_handler.write1ByteTxRx(
                port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
            )
        port_handler.closePort()


if __name__ == "__main__":
    main()
