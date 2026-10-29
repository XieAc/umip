from dynamixel_sdk import *
import time
import sys
import msvcrt

DEVICENAME = "COM6"

BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

SERVO_IDS = [1, 2]
COUNTS_PER_REV = 4096
ADDR_PRESENT_POSITION = 132
initial_positions = {}

# 两个电机的夹紧方向
# 如果方向相反，将其中一个改为 -1
MOTOR_DIRECTION = {
    1: 1,
    2: -1,
}

# 夹紧速度
GRIP_SPEED = 30

# 松开速度
RELEASE_SPEED = 30

# 内部电流阈值
# 该数值需要通过实际测试调整
CURRENT_THRESHOLD = 5

# 低速阈值
VELOCITY_THRESHOLD = 25

# 判定电机无法继续旋转（堵转）所需的连续时间。
# 只有两个电机都持续高电流且低速达到该时间后才停止运动并保持扭矩。
STALL_TIME = 1

# 最大夹紧时间，防止没有检测到电流时持续运行（仅作兜底保护）
MAX_GRIP_TIME = 5.0


ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_VELOCITY = 104
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# 速度控制模式
VELOCITY_MODE = 1


def check_result(packet_handler, comm_result, dxl_error):
    if comm_result != COMM_SUCCESS:
        print(
            "\n通信错误：",
            packet_handler.getTxRxResult(comm_result)
        )
        return False

    if dxl_error != 0:
        print(
            "\n舵机错误：",
            packet_handler.getRxPacketError(dxl_error)
        )
        return False

    return True


def signed_16(value):
    if value >= 0x8000:
        return value - 0x10000
    return value


def signed_32(value):
    if value >= 0x80000000:
        return value - 0x100000000
    return value


def write_torque(packet_handler, port_handler, dxl_id, enable):
    result, error = packet_handler.write1ByteTxRx(
        port_handler,
        dxl_id,
        ADDR_TORQUE_ENABLE,
        enable
    )

    return check_result(packet_handler, result, error)


def set_velocity(packet_handler, port_handler, dxl_id, velocity):
    velocity = int(velocity)
    velocity_u32 = velocity & 0xFFFFFFFF

    result, error = packet_handler.write4ByteTxRx(
        port_handler,
        dxl_id,
        ADDR_GOAL_VELOCITY,
        velocity_u32
    )

    return check_result(packet_handler, result, error)


def set_velocity_mode(packet_handler, port_handler, dxl_id):
    # 切换工作模式前必须关闭扭矩
    if not write_torque(
        packet_handler,
        port_handler,
        dxl_id,
        TORQUE_DISABLE
    ):
        return False

    result, error = packet_handler.write1ByteTxRx(
        port_handler,
        dxl_id,
        ADDR_OPERATING_MODE,
        VELOCITY_MODE
    )

    return check_result(packet_handler, result, error)


def read_current_velocity(packet_handler, port_handler, dxl_id):
    current, result, error = packet_handler.read2ByteTxRx(
        port_handler,
        dxl_id,
        ADDR_PRESENT_CURRENT
    )

    if not check_result(packet_handler, result, error):
        return None

    velocity, result, error = packet_handler.read4ByteTxRx(
        port_handler,
        dxl_id,
        ADDR_PRESENT_VELOCITY
    )

    if not check_result(packet_handler, result, error):
        return None

    return signed_16(current), signed_32(velocity)


def read_position(packet_handler, port_handler, dxl_id):
    position, result, error = packet_handler.read4ByteTxRx(
        port_handler, dxl_id, ADDR_PRESENT_POSITION
    )
    if not check_result(packet_handler, result, error):
        return None
    return position


def nearest_30_degree_position(position):
    """Return the encoder count for the nearest 30-degree absolute angle."""
    angle_step_count = COUNTS_PER_REV / 12.0
    angle_index = int((position % COUNTS_PER_REV) / angle_step_count + 0.5) % 12
    return int(round(angle_index * angle_step_count)) % COUNTS_PER_REV


def show_positions(packet_handler, port_handler):
    values = []
    for dxl_id in SERVO_IDS:
        position = read_position(packet_handler, port_handler, dxl_id)
        if position is None:
            continue
        zero = initial_positions.get(dxl_id, position)
        delta = signed_32(position - zero)
        values.append(
            f"ID={dxl_id} angle={position * 360.0 / COUNTS_PER_REV:.1f}deg "
            f"delta={delta * 360.0 / COUNTS_PER_REV:+.1f}deg"
        )
    if values:
        print("\r" + " | ".join(values), end="")


def read_command_with_display(packet_handler, port_handler):
    """Wait for a command while continuously refreshing motor positions."""
    command = ""
    print("\n操作说明: g=夹紧  r=松开并关闭扭矩  s=停止但保持扭矩  q=退出")
    print("请输入指令: ", end="", flush=True)
    last_update = 0.0
    while True:
        now = time.time()
        if now - last_update >= 0.1:
            show_positions(packet_handler, port_handler)
            last_update = now
        if msvcrt.kbhit():
            key = msvcrt.getwch()
            if key in ("\r", "\n"):
                print()
                return command.strip().lower()
            if key == "\b":
                if command:
                    command = command[:-1]
                    print("\b \b", end="", flush=True)
            elif key.isprintable():
                command += key
                print(key, end="", flush=True)
        time.sleep(0.01)


def stop_motion_keep_torque(packet_handler, port_handler):
    """
    停止运动，但不关闭扭矩。
    这样舵机仍然保持输出力矩。
    """
    for dxl_id in SERVO_IDS:
        set_velocity(
            packet_handler,
            port_handler,
            dxl_id,
            0
        )

    print("\n已停止运动，保持扭矩夹紧")


def stop_and_disable(packet_handler, port_handler):
    """
    停止电机并关闭扭矩。
    """
    for dxl_id in SERVO_IDS:
        set_velocity(
            packet_handler,
            port_handler,
            dxl_id,
            0
        )

    time.sleep(0.2)

    for dxl_id in SERVO_IDS:
        write_torque(
            packet_handler,
            port_handler,
            dxl_id,
            TORQUE_DISABLE
        )


def grip(packet_handler, port_handler):
    """
    夹紧动作：
    两台电机均检测到低速并持续 STALL_TIME 后，
    停止速度并保持扭矩，锁定当前夹紧位置，直到收到松开命令。
    """

    print("开始夹紧")

    start_time = time.time()
    contact_start = None

    # Re-enable torque after a previous release disabled it.
    for dxl_id in SERVO_IDS:
        if not write_torque(packet_handler, port_handler, dxl_id, TORQUE_ENABLE):
            print(f"ID={dxl_id} torque enable failed")
            return

    # 松开完成后扭矩会被关闭，再次夹紧前需要重新启用。
    for dxl_id in SERVO_IDS:
        if not write_torque(
            packet_handler,
            port_handler,
            dxl_id,
            TORQUE_ENABLE
        ):
            print(f"ID={dxl_id} 启用扭矩失败")
            return

    for dxl_id in SERVO_IDS:
        direction = MOTOR_DIRECTION[dxl_id]

        set_velocity(
            packet_handler,
            port_handler,
            dxl_id,
            direction * GRIP_SPEED
        )

    while True:
        elapsed = time.time() - start_time

        if elapsed > MAX_GRIP_TIME:
            print("\n达到最大夹紧时间，执行保护停止")
            stop_motion_keep_torque(
                packet_handler,
                port_handler
            )
            return

        all_stopped = True
        status_text = []

        for dxl_id in SERVO_IDS:
            data = read_current_velocity(
                packet_handler,
                port_handler,
                dxl_id
            )

            if data is None:
                all_stopped = False
                continue

            current, velocity = data

            # 电流取绝对值，速度也取绝对值
            current_abs = abs(current)
            velocity_abs = abs(velocity)

            status_text.append(
                f"ID={dxl_id} "
                f"I={current_abs} "
                f"V={velocity_abs}"
            )

            # 只要任一电机仍在转动，就继续夹紧；两台都停转后开始计时
            if velocity_abs > VELOCITY_THRESHOLD:
                all_stopped = False

        print(
            "\r" + " | ".join(status_text),
            end=""
        )
        show_positions(packet_handler, port_handler)

        if all_stopped:
            if contact_start is None:
                contact_start = time.time()

            if time.time() - contact_start >= STALL_TIME:
                print()
                print("检测到电机停转已持续 1 秒，锁定当前位置；按 r 松开")
                stop_motion_keep_torque(
                    packet_handler,
                    port_handler
                )
                return
        else:
            contact_start = None

        time.sleep(0.03)


def release(packet_handler, port_handler):
    """
    松开夹爪。
    实际方向可能需要根据机械安装调整。
    """

    print("开始松开")

    for dxl_id in SERVO_IDS:
        direction = MOTOR_DIRECTION[dxl_id]

        set_velocity(
            packet_handler,
            port_handler,
            dxl_id,
            -direction * RELEASE_SPEED
        )

    release_start = time.time()
    while time.time() - release_start < 1.0:
        show_positions(packet_handler, port_handler)
        time.sleep(0.03)
    print()

    for dxl_id in SERVO_IDS:
        set_velocity(
            packet_handler,
            port_handler,
            dxl_id,
            0
        )

    # 松开完成后关闭扭矩，让电机完全卸力。
    time.sleep(0.1)

    for dxl_id in SERVO_IDS:
        write_torque(
            packet_handler,
            port_handler,
            dxl_id,
            TORQUE_DISABLE
        )

    print("松开动作完成，电机扭矩已关闭")


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

    print("串口连接成功")

    try:
        # 设置速度模式
        for dxl_id in SERVO_IDS:
            if not set_velocity_mode(
                packet_handler,
                port_handler,
                dxl_id
            ):
                print(f"ID={dxl_id} 设置速度模式失败")
                return

        # 打开扭矩
        for dxl_id in SERVO_IDS:
            write_torque(
                packet_handler,
                port_handler,
                dxl_id,
                TORQUE_ENABLE
            )

        # Record the startup position as the reference for relative movement.
        for dxl_id in SERVO_IDS:
            position = read_position(packet_handler, port_handler, dxl_id)
            if position is not None:
                initial_positions[dxl_id] = nearest_30_degree_position(position)
        print("\n已记录程序启动时角度为相对零点")

        print()
        print("控制菜单：")
        print("  g：夹紧")
        print("  r：松开")
        print("  s：停止但保持扭矩")
        print("  q：停止并退出")

        while True:
            command = read_command_with_display(
                packet_handler,
                port_handler
            )

            if command == "g":
                grip(
                    packet_handler,
                    port_handler
                )

            elif command == "r":
                release(
                    packet_handler,
                    port_handler
                )

            elif command == "s":
                stop_motion_keep_torque(
                    packet_handler,
                    port_handler
                )

            elif command == "q":
                break

            else:
                print("无效命令")

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C")

    finally:
        stop_and_disable(
            packet_handler,
            port_handler
        )

        port_handler.closePort()
        print("串口已关闭")


if __name__ == "__main__":
    main()
