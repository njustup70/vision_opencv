#!/usr/bin/env python3
from __future__ import annotations

import os
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
IMAGE_TOPIC = "/hik_camera/image_raw"
DRAW_RESULT_TOPIC = "/arucopnp/draw_result"
SERIAL_PORT = "/dev/serial_qh"
SERIAL_BAUD = 230400


class ArucoPnpSerialNode(Node):
    def __init__(self) -> None:
        super().__init__("arucopnp_serial_node")

        self._estimator = HighPrecisionPoseEstimator()
        self._bridge = CvBridge()

        self._draw_pub = self.create_publisher(Image, DRAW_RESULT_TOPIC, qos_profile_sensor_data)
        self._img_sub = self.create_subscription(Image, IMAGE_TOPIC, self._on_image, qos_profile_sensor_data)
        self._serial = AsyncSerial_t(SERIAL_PORT, SERIAL_BAUD)

        self.get_logger().info(f"image_topic={IMAGE_TOPIC}")
        self.get_logger().info(f"draw_result_topic={DRAW_RESULT_TOPIC}")
        self.get_logger().info(f"serial={SERIAL_PORT} @ {SERIAL_BAUD}")

    def _on_image(self, msg: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        result = self._estimator.on_image(frame)
        rvec, tvec = (None, None) if result is None else result

        out_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        out_msg.header = msg.header
        self._draw_pub.publish(out_msg)

        if rvec is None or tvec is None:
            return

        rv = np.array(rvec, dtype=np.float64).reshape(3)
        tv = np.array(tvec, dtype=np.float64).reshape(3)
        # 发串口

        # self._serial.write(payload.encode("ascii"))


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
