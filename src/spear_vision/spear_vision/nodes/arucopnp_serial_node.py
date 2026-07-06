#!/usr/bin/env python3
"""
ArUco PnP + 串口 ROS 2 节点
从 spear/ros2_arucopnp_serial_node.py 迁入 spear_vision 包

订阅话题:
  - /hik_camera/image  (sensor_msgs/Image)

发布话题:
  - /arucopnp/draw_result  (sensor_msgs/Image)      带标注的检测画面
  - /arucopnp/offset_mm    (geometry_msgs/PointStamped)  偏移量 (mm)

串口发送:
  - 帧格式: 0xFA 0xB1 + left_mm(f32) + up_mm(f32)
"""

import struct

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from spear_vision.core.high_precision_pose_estimator import HighPrecisionPoseEstimator
from spear_vision.utils.async_serial import AsyncSerial_t

# ---- 固定参数 ----
IMAGE_TOPIC = "/hik_camera/image_raw"
DRAW_RESULT_TOPIC = "/arucopnp/draw_result"
OFFSET_MM_TOPIC = "/arucopnp/offset_mm"
SERIAL_PORT = "/dev/ch340"
SERIAL_BAUD = 921600


class ArucoPnpSerialNode(Node):
    def __init__(self) -> None:
        super().__init__("arucopnp_serial_node")

        K = np.array(
            [[1322.8397832601186, 0.0, 623.2118921253351],
             [0.0, 1338.418629358359, 509.76668739080276],
             [0.0, 0.0, 1.0]])
        D = np.array([-0.131611456394353, 0.9770122703554055,
                      -0.0037105334423302764, -0.0033209523486226952,
                      -2.6406900561614206])

        self._estimator = HighPrecisionPoseEstimator(K=K, D=D)
        self._bridge = CvBridge()

        self._draw_pub = self.create_publisher(Image, DRAW_RESULT_TOPIC,
                                                qos_profile_sensor_data)
        self._offset_pub = self.create_publisher(PointStamped, OFFSET_MM_TOPIC, 10)
        self._img_sub = self.create_subscription(Image, IMAGE_TOPIC,
                                                  self._on_image, 1)
        self._serial = AsyncSerial_t(SERIAL_PORT, SERIAL_BAUD)

    @staticmethod
    def _compute_offsets_mm(tvec: np.ndarray) -> tuple[float, float]:
        t = np.asarray(tvec, dtype=np.float64).reshape(-1)
        x_m, y_m = float(t[0]), float(t[1])
        if np.isnan(x_m) or np.isnan(y_m):
            return 0.0, 0.0
        left_mm = -x_m * 1000.0
        up_mm = -y_m * 1000.0
        return left_mm, up_mm

    def _on_image(self, msg: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        result = self._estimator.on_image(frame)
        rvec, tvec = (None, None) if result is None else result

        out_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        out_msg.header = msg.header
        self._draw_pub.publish(out_msg)

        if rvec is None or tvec is None:
            return

        left_mm, up_mm = self._compute_offsets_mm(tvec)
        print(f"left_mm: {left_mm:.1f}, up_mm: {up_mm:.1f}")

        if abs(left_mm) > 1e-6 and abs(up_mm) > 1e-6:
            offset_msg = PointStamped()
            offset_msg.header = msg.header
            offset_msg.point.x = left_mm
            offset_msg.point.y = up_mm
            offset_msg.point.z = 0.0
            self._offset_pub.publish(offset_msg)

            payload = struct.pack("<ff", left_mm, up_mm)
            frame_bytes = bytes([0xFA, 0xB1]) + payload
            self._serial.write(frame_bytes)


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
