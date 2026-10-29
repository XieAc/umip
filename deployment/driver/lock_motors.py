"""Hold all configured motors at their current positions."""

from dynamixel_sdk import *
import msvcrt
import sys
import time


DEVICENAME = "COM6"
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0
SERVO_IDS = [1, 2]

ADDR_PRESENT_POSITION = 132
ADDR_OPERATING_MOD1E = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
POSITION_MODE = 3
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0


def check_result(packet_handler, comm_result, dxl_error):
    if comm_result != COMM_SUCCESS:
        print(packet_handler.getTxRxResult(comm_result))
        return False
    if dxl_error != 0:
        print(packet_handler.getRxPacketError(dxl_error))
        return False
    return True


def read_position(packet_handler, port_handler, dxl_id):
    position, result, error = packet_handler.read4ByteTxRx(
        port_handler, dxl_id, ADDR_PRESENT_POSITION
    )
    if not check_result(packet_handler, result, error):
        return None
    return position


def write_torque(packet_handler, port_handler, dxl_id, enabled):
    result, error = packet_handler.write1ByteTxRx(
        port_handler, dxl_id, ADDR_TORQUE_ENABLE, enabled
    )
    return check_result(packet_handler, result, error)


def main():
    port_handler = PortHandler(DEVICENAME)
    packet_handler = PacketHandler(PROTOCOL_VERSION)
    locked_ids = []

    if not port_handler.openPort():
        print("无法打开串口：", DEVICENAME)
        sys.exit(1)
    if not port_handler.setBaudRate(BAUDRATE):
        print("无法设置波特率：", BAUDRATE)
        port_handler.closePort()
        sys.exit(1)

    try:
        for dxl_id in SERVO_IDS:
            if not write_torque(packet_handler, port_handler, dxl_id, TORQUE_DISABLE):
                return
            result, error = packet_handler.write1ByteTxRx(
                port_handler, dxl_id, ADDR_OPERATING_MODE, POSITION_MODE
            )
            if not check_result(packet_handler, result, error):
                return
            position = read_position(packet_handler, port_handler, dxl_id)
            if position is None:
                return
            result, error = packet_handler.write4ByteTxRx(
                port_handler, dxl_id, ADDR_GOAL_POSITION, position
            )
            if not check_result(packet_handler, result, error):
                return
            if not write_torque(packet_handler, port_handler, dxl_id, TORQUE_ENABLE):
                return
            locked_ids.append(dxl_id)

        print("电机已锁死，按 q 或 Ctrl+C 退出并释放扭矩。")
        while True:
            if msvcrt.kbhit() and msvcrt.getwch().lower() == "q":
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        for dxl_id in locked_ids:
            write_torque(packet_handler, port_handler, dxl_id, TORQUE_DISABLE)
        port_handler.closePort()
        print("\n已释放扭矩，程序退出。")


if __name__ == "__main__":
    main()
