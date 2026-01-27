"""
小板 PnP -> 串口 UART 桥接节点

功能：
- 订阅 /small_board_pose/pose（小板位姿）
- 计算 left_mm / up_mm（单位：mm）
- 按 serial_dispose 的最小协议格式发送：
  [0xFA][ID][left_mm(float32)][up_mm(float32)]

说明：
- 只发送左右/上下两个量，不发送前后/角度
- 数据单位是 mm，float32，小端序（<ff）
- 若置信度/重投影误差不达标，则不发送（依赖 MCU 侧超时保护）
"""

from __future__ import annotations

import struct
import time
from typing import Optional
import math
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Float32, Float64

# 复用你们电控组的串口模板（serial_dispose）
from get_dispose_serial.myserial import AsyncSerial_t


class SmallBoardSerialBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("small_board_serial_bridge")

        # --- 串口参数（与电控组对齐） ---
        self.declare_parameter("port", "/dev/serial_ch340")
        self.declare_parameter("baudrate", 115200)
        # 帧头 + 帧 ID
        self.declare_parameter("out_first_frame", 0xFA)
        self.declare_parameter("out_frame_id", 0xB1)

        # --- 订阅话题（默认小板节点输出） ---
        self.declare_parameter("pose_topic", "/small_board_pose/pose")
        self.declare_parameter("confidence_topic", "/small_board_pose/confidence")
        self.declare_parameter("reproj_mean_topic", "/small_board_pose/reproj_error_mean_px")

        # --- 质量门控（默认不限制，可根据实际需要设阈值） ---
        self.declare_parameter("min_confidence", 0.0)
        self.declare_parameter("max_mean_reproj_px", 0.0)

        # --- 输出符号控制（若电控方向相反可翻转） ---
        self.declare_parameter("invert_left", False)
        self.declare_parameter("invert_up", False)

        # 日志节流（秒）
        self.declare_parameter("log_interval_sec", 0.1)
        # ACK 接收（可选）：MCU 回传一个字节表示“已收到”
        self.declare_parameter("ack_enable", True)
        self.declare_parameter("ack_byte", 0xEE)

        self._last_conf: Optional[float] = None
        self._last_mean: Optional[float] = None
        self._last_log = 0.0
        self._last_ack_log = 0.0
        self._ack_count = 0

        # 初始化串口（异步读写）
        port = str(self.get_parameter("port").value)
        baudrate = int(self.get_parameter("baudrate").value)
        self.serial = AsyncSerial_t(port=port, baudrate=baudrate)
        # 启动监听（即使当前不需要 MCU 回传，也能自动完成连接管理）
        self.serial.startListening(lambda data: self._on_serial_rx(data))

        # 订阅小板位姿 + 质量指标
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("pose_topic").value),
            self._on_pose,
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("confidence_topic").value),
            self._on_confidence,
            10,
        )
        self.create_subscription(
            Float64,
            str(self.get_parameter("reproj_mean_topic").value),
            self._on_reproj_mean,
            10,
        )

        self.get_logger().info(f"Serial bridge started on {port}@{baudrate}")

    # --- 串口接收回调（目前仅打印，后续可扩展 MCU->PC 的回传协议） ---
    def _on_serial_rx(self, data: bytes) -> None:
        if not bool(self.get_parameter("ack_enable").value):
            return
        ack = int(self.get_parameter("ack_byte").value) & 0xFF
        if bytes([ack]) in data:
            self._ack_count += 1
            self._throttle_ack_log(f"MCU ACK received (0x{ack:02X}), count={self._ack_count}")

    def _on_confidence(self, msg: Float32) -> None:
        self._last_conf = float(msg.data)

    def _on_reproj_mean(self, msg: Float64) -> None:
        self._last_mean = float(msg.data)

    def _quality_ok(self) -> bool:
        min_conf = float(self.get_parameter("min_confidence").value)
        max_mean = float(self.get_parameter("max_mean_reproj_px").value)

        if min_conf > 0.0:
            if self._last_conf is None or self._last_conf < min_conf:
                return False
        if max_mean > 0.0:
            if self._last_mean is None or self._last_mean > max_mean:
                return False
        return True

    def _on_pose(self, msg: PoseStamped) -> None:
        # 坐标约定（相机光学坐标系）：
        # x: 右, y: 下, z: 前
        # 我们需要 left/up：
        #left_mm = -x_mm, up_mm = -y_mm
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)

        left_mm = -x * 1000.0
        up_mm = -y * 1000.0

        #矛杆与charuco码原点的偏移量 
        x_mgpianyi=0
        y_mgpianyi=0
        #夹爪与摄像头中心的偏移量
        x_jzpianyi=0
        y_jzpianyi=0
        #摄像头与夹爪的角度偏移（横向与纵向）
        alpha=0
        seita=0
        #水平吗偏移计算
        left_mm = left_mm + x_mgpianyi-x_jzpianyi
        up_mm = up_mm+y_mgpianyi-y_jzpianyi
        #角度偏移计算
        left_mm = left_mm-math.sin(alpha)-10
        up_mm= up_mm-math.sin(seita)+10


        if bool(self.get_parameter("invert_left").value):
            left_mm = -left_mm
        
        if bool(self.get_parameter("invert_up").value):
            up_mm = -up_mm

        # 质量门控：不满足就不发送（依赖 MCU 超时保护）
        if not self._quality_ok():
            self._throttle_log("Skip TX (quality gate)")
            return

        # 打包：SOF + ID + <ff (left_mm, up_mm)
        sof = int(self.get_parameter("out_first_frame").value) & 0xFF
        fid = int(self.get_parameter("out_frame_id").value) & 0xFF
        payload = struct.pack("<ff", float(left_mm), float(up_mm))
        frame = bytes([sof, fid]) + payload
        self.serial.write(frame)
        self._throttle_log(f"TX left_mm={left_mm:+.3f} up_mm={up_mm:+.3f}")
    def _throttle_log(self, text: str) -> None:
        self.get_logger().info(text)

    def _throttle_ack_log(self, text: str) -> None:
        interval = float(self.get_parameter("log_interval_sec").value)
        now = time.time()
        if now - self._last_ack_log >= interval:
            self._last_ack_log = now
            self.get_logger().info(text)


def main() -> None:
    rclpy.init()
    node = SmallBoardSerialBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
