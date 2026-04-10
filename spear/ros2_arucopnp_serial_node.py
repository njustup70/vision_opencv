#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

# Allow importing local modules from the same folder no matter where script is launched.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from arucopnp import HighPrecisionPoseEstimator
from signal_filter import OffsetSmoother


# ---- 固定参数（按你的现场直接改这里） ----
IMAGE_TOPIC = "/hik_camera/image_raw"
DRAW_RESULT_TOPIC = "/arucopnp/draw_result"
OFFSET_MM_TOPIC = "/arucopnp/offset_mm"
COMMAND_TOPIC = "/update_exec_req"
START_COMMAND = "spear_build"
STOP_COMMAND = "stop"
# 对方真正对接点相对 ChArUco 原点的外参（单位：mm，默认全 0）
TARGET_POINT_X_MM = 0.0
TARGET_POINT_Y_MM = 0.0
TARGET_POINT_Z_MM = 0.0
# 我方真正对接参考点相对相机原点的外参（单位：mm，默认全 0）
OUR_REF_X_MM = 0.0
OUR_REF_Y_MM = 0.0
OUR_REF_Z_MM = 0.0
# 外参修正量（单位：mm），作用在滤波后的 left/up 偏移上
LEFT_OFFSET_MM = 0.0
UP_OFFSET_MM = 0.0
YAW_OFFSET_DEG = 0.0
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
        self._offset_pub = self.create_publisher(PointStamped, OFFSET_MM_TOPIC, 10)
        self._img_sub = self.create_subscription(Image, IMAGE_TOPIC, self._on_image, 1)
        self._cmd_sub = self.create_subscription(String, COMMAND_TOPIC, self._on_exec_request, 10)
        self._enabled = False

        # One Euro Filter + 死区平滑器
        # min_cutoff: 静止平滑强度 (越小越平稳, 但响应略慢)
        # beta:       动态跟随系数 (越大快速移动时越灵敏)
        # dead_zone_mm: 死区阈值, 小于此值视为静止不发送
        self._smoother = OffsetSmoother(min_cutoff=0.5, beta=0.007, dead_zone_mm=0.3)
        self.get_logger().info(
            f"Waiting for {COMMAND_TOPIC} == '{START_COMMAND}' before spear alignment starts."
        )

    def _on_exec_request(self, msg: String) -> None:
        command = msg.data.strip()
        if command == START_COMMAND:
            self._enabled = True
            self._smoother.reset()
            self.get_logger().info("Spear alignment enabled.")
            return
        if command == STOP_COMMAND:
            self._enabled = False
            self._smoother.reset()
            self.get_logger().info("Spear alignment disabled.")

    @staticmethod
    def _compute_alignment_error_mm(
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute left/up/forward alignment error from full pose."""
        t = np.asarray(tvec, dtype=np.float64).reshape(-1)
        r = np.asarray(rvec, dtype=np.float64).reshape(-1)
        if t.size < 3 or r.size < 3:
            return 0.0, 0.0, 0.0
        if not np.all(np.isfinite(t[:3])) or not np.all(np.isfinite(r[:3])):
            return 0.0, 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(r[:3].reshape(3, 1))

        target_point_in_board_m = np.array(
            [TARGET_POINT_X_MM, TARGET_POINT_Y_MM, TARGET_POINT_Z_MM],
            dtype=np.float64,
        ) / 1000.0
        our_ref_in_camera_m = np.array(
            [OUR_REF_X_MM, OUR_REF_Y_MM, OUR_REF_Z_MM],
            dtype=np.float64,
        ) / 1000.0

        target_point_in_camera_m = rotation_matrix @ target_point_in_board_m + t[:3]
        alignment_error_m = target_point_in_camera_m - our_ref_in_camera_m

        left_mm = -float(alignment_error_m[0]) * 1000.0
        up_mm = -float(alignment_error_m[1]) * 1000.0
        forward_mm = float(alignment_error_m[2]) * 1000.0
        return left_mm, up_mm, forward_mm

    def _on_image(self, msg: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if not self._enabled:
            out_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            out_msg.header = msg.header
            self._draw_pub.publish(out_msg)
            return

        result = self._estimator.on_image(frame)
        rvec, tvec = (None, None) if result is None else result

        out_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        out_msg.header = msg.header
        self._draw_pub.publish(out_msg)

        if rvec is None or tvec is None:
            return

        raw_left, raw_up, raw_forward = self._compute_alignment_error_mm(rvec, tvec)

        # One Euro Filter 平滑 + 死区抑制
        left_mm, up_mm = self._smoother.update(raw_left, raw_up)
        left_mm += LEFT_OFFSET_MM
        up_mm += UP_OFFSET_MM
    
        print(
            f"raw=({raw_left:.1f}, {raw_up:.1f}, {raw_forward:.1f})  "
            f"filtered=({left_mm:.1f}, {up_mm:.1f})"
        )

        # 发布 ROS topic (发布滤波后的值)
        if abs(raw_left) > 1e-6 or abs(raw_up) > 1e-6:
            offset_msg = PointStamped()
            offset_msg.header = msg.header
            offset_msg.point.x = left_mm
            offset_msg.point.y = up_mm
            offset_msg.point.z = 0.0
            self._offset_pub.publish(offset_msg)


    
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
