#!/usr/bin/env python3
from __future__ import annotations

import os
import struct
import sys

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

# Allow importing local modules from the same folder no matter where script is launched.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from arucopnp import HighPrecisionPoseEstimator
from myserial import AsyncSerial_t


# ---- 固定参数（按你的现场直接改这里） ----
IMAGE_TOPIC = "/hik_camera/image"
DRAW_RESULT_TOPIC = "/arucopnp/draw_result"
SERIAL_PORT = "/dev/serial_qh"
SERIAL_BAUD = 230400

class ArucoPnpSerialNode(Node):
    def __init__(self) -> None:
        super().__init__("arucopnp_serial_node")
        K=np.array(
          [ [1322.8397832601186, 0.0, 623.2118921253351],
            [0.0, 1338.418629358359, 509.76668739080276],
            [0.0, 0.0, 1.0]]     )
        D=np.array([-0.131611456394353, 0.9770122703554055, -0.0037105334423302764, -0.0033209523486226952, -2.6406900561614206])
        self._estimator = HighPrecisionPoseEstimator(K=K,D=D)
        self._bridge = CvBridge()

        self._draw_pub = self.create_publisher(Image, DRAW_RESULT_TOPIC, qos_profile_sensor_data)
        self._img_sub = self.create_subscription(Image, IMAGE_TOPIC, self._on_image, qos_profile_sensor_data)
        self._serial = AsyncSerial_t(SERIAL_PORT, SERIAL_BAUD)

    @staticmethod
    def _to_spear_vision_frame(tvec: np.ndarray) -> bytes:
        """Convert solvePnP tvec(m) to spear_vision UART frame."""
        t = np.asarray(tvec, dtype=np.float64).reshape(-1)
        x_m, y_m = float(t[0]), float(t[1])
        #检测是否为Nan如果是Nan则返回全0数据
        if np.isnan(x_m) or np.isnan(y_m):
            x_m, y_m = 0.0, 0.0
        # Follow spear_vision convention:
        # left_mm = -x * 1000, up_mm = -y * 1000
        left_mm = -x_m * 1000.0
        up_mm = -y_m * 1000.0
        print(f"left_mm: {left_mm:.1f}, up_mm: {up_mm:.1f}")
        payload = struct.pack("<ff", left_mm, up_mm)
        return bytes([0xFA, 0xB1]) + payload

    def _on_image(self, msg: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        result = self._estimator.on_image(frame)
        rvec, tvec = (None, None) if result is None else result

        out_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        out_msg.header = msg.header
        self._draw_pub.publish(out_msg)

        if rvec is None or tvec is None:
            return

        frame = self._to_spear_vision_frame(tvec)
        self._serial.write(frame)
        # self.get_logger().info(f"TX {frame.hex()}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArucoPnpSerialNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
