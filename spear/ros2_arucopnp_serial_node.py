#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,CompressedImage
from std_msgs.msg import String

# 允许脚本从任意目录启动时，都能导入当前文件夹里的本地模块。
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
# R1 矛杆中心点相对 ChArUco 原点的外参（单位：mm，默认全 0）
R1_ROD_CENTER_X_MM = 0.0
R1_ROD_CENTER_Y_MM = 0.0
R1_ROD_CENTER_Z_MM = 0.0
# R1 矛杆轴线方向在 ChArUco 坐标系里的向量。
# 只看方向，长度无所谓；按现场标定结果填。
R1_ROD_AXIS_X_IN_BOARD = 0.0
R1_ROD_AXIS_Y_IN_BOARD = 0.0
R1_ROD_AXIS_Z_IN_BOARD = 1.0
# R2 矛头中心/入口中心相对相机原点的外参（单位：mm，默认全 0）
R2_HEAD_CENTER_X_MM = 0.0
R2_HEAD_CENTER_Y_MM = 0.0
R2_HEAD_CENTER_Z_MM = 0.0
# R2 矛头入口平面的法向/矛头轴线方向在相机坐标系里的向量。
# 如果相机光轴和矛头轴线平行，保持 [0, 0, 1]。
R2_HEAD_AXIS_X_IN_CAMERA = 0.0
R2_HEAD_AXIS_Y_IN_CAMERA = 0.0
R2_HEAD_AXIS_Z_IN_CAMERA = 1.0
# True：用“矛杆轴线”和“矛头入口平面”的交点计算 left/up。
# False：只用矛杆中心点和矛头中心点做点对点平移误差。
USE_AXIS_INTERSECTION_COMPENSATION = True
# 外参修正量（单位：mm），作用在滤波后的 left/up 偏移上
LEFT_OFFSET_MM = -60.0
# UP_OFFSET_MM = -200.0
UP_RESULT= -45.0
# 只用于 yaw 调试显示的零点修正；left/up 补偿由轴线和平面交点计算得到。
YAW_OFFSET_DEG = 0.0


class ArucoPnpSerialNode(Node):
    def __init__(self) -> None:
        super().__init__("arucopnp_serial_node")
        K=np.array(
          [ [1301.1167695971926, 0.0, 618.8649503224167],
            [0.0, 1300.52230424669, 525.5565334404847],
            [0.0, 0.0, 1.0]]     )
        D=np.array([-0.0457043987, -0.2172562790, -0.002088440089, 0.000944754920863791,1.3830427124470757])
        self._estimator = HighPrecisionPoseEstimator(K=K,D=D)
        self._bridge = CvBridge()

        self._draw_pub = self.create_publisher(CompressedImage, DRAW_RESULT_TOPIC, qos_profile_sensor_data)
        self._offset_pub = self.create_publisher(PointStamped, OFFSET_MM_TOPIC, 10)
        self._img_sub = self.create_subscription(Image, IMAGE_TOPIC, self._on_image, 1)
        self._cmd_sub = self.create_subscription(String, COMMAND_TOPIC, self._on_exec_request, 10)
        self._enabled = False

        # One Euro Filter + 死区平滑器
        # min_cutoff: 静止平滑强度 (越小越平稳, 但响应略慢)
        # beta:       动态跟随系数 (越大快速移动时越灵敏)
        # dead_zone_mm: 死区阈值, 小于此值视为静止不发送
        self._smoother = OffsetSmoother(min_cutoff=0.5, beta=0.007, dead_zone_mm=0.3)
        # self.get_logger().info(
        #     f"等待 {COMMAND_TOPIC} == '{START_COMMAND}' 后开始矛杆对接。"
        # )

    def _on_exec_request(self, msg: String) -> None:
        command = msg.data.strip()
        if command == START_COMMAND:
            self._enabled = True
            self._smoother.reset()
            # self.get_logger().info("矛杆对接已启用。")
            return
        if command == STOP_COMMAND:
            self._enabled = False
            self._smoother.reset()
            # self.get_logger().info("矛杆对接已停止。")

    @staticmethod
    def _compute_alignment_error_mm(
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """计算矛杆对接需要下发的 left/up 偏差。

        默认模式：只计算 R1 矛杆中心点和 R2 矛头中心点之间的点对点误差。
        轴线模式：把 R1 矛杆轴线投影到 R2 矛头入口平面上，
        这样小 yaw 偏差造成的落点变化会折算进水平/竖直移动命令。
        """
        t = np.asarray(tvec, dtype=np.float64).reshape(-1)
        r = np.asarray(rvec, dtype=np.float64).reshape(-1)
        if t.size < 3 or r.size < 3:
            return 0.0, 0.0, 0.0, 0.0
        if not np.all(np.isfinite(t[:3])) or not np.all(np.isfinite(r[:3])):
            return 0.0, 0.0, 0.0, 0.0

        board_to_camera_r, _ = cv2.Rodrigues(r[:3].reshape(3, 1))

        rod_center_in_board_m = np.array(
            [R1_ROD_CENTER_X_MM, R1_ROD_CENTER_Y_MM, R1_ROD_CENTER_Z_MM],
            dtype=np.float64,
        ) / 1000.0
        head_center_in_camera_m = np.array(
            [R2_HEAD_CENTER_X_MM, R2_HEAD_CENTER_Y_MM, R2_HEAD_CENTER_Z_MM],
            dtype=np.float64,
        ) / 1000.0

        rod_center_in_camera_m = board_to_camera_r @ rod_center_in_board_m + t[:3]
        command_point_in_camera_m = rod_center_in_camera_m
        rod_axis_in_camera = None
        head_axis_in_camera = None
        yaw_deg = 0.0

        if USE_AXIS_INTERSECTION_COMPENSATION:
            rod_axis_in_board = np.array(
                [
                    R1_ROD_AXIS_X_IN_BOARD,
                    R1_ROD_AXIS_Y_IN_BOARD,
                    R1_ROD_AXIS_Z_IN_BOARD,
                ],
                dtype=np.float64,
            )
            head_axis_in_camera = np.array(
                [
                    R2_HEAD_AXIS_X_IN_CAMERA,
                    R2_HEAD_AXIS_Y_IN_CAMERA,
                    R2_HEAD_AXIS_Z_IN_CAMERA,
                ],
                dtype=np.float64,
            )
            rod_axis_norm = float(np.linalg.norm(rod_axis_in_board))
            head_axis_norm = float(np.linalg.norm(head_axis_in_camera))

            if rod_axis_norm > 1e-9 and head_axis_norm > 1e-9:
                rod_axis_in_camera = board_to_camera_r @ (rod_axis_in_board / rod_axis_norm)
                head_axis_in_camera = head_axis_in_camera / head_axis_norm
                denom = float(np.dot(rod_axis_in_camera, head_axis_in_camera))

                if abs(denom) > 1e-6:
                    scale = float(
                        np.dot(
                            head_center_in_camera_m - rod_center_in_camera_m,
                            head_axis_in_camera,
                        )
                        / denom
                    )
                    command_point_in_camera_m = rod_center_in_camera_m + scale * rod_axis_in_camera

                yaw_deg = ArucoPnpSerialNode._horizontal_angle_deg(
                    rod_axis_in_camera,
                    head_axis_in_camera,
                ) - YAW_OFFSET_DEG

        alignment_error_m = command_point_in_camera_m - head_center_in_camera_m

        left_mm = -float(alignment_error_m[0]) * 1000.0
        up_mm = -float(alignment_error_m[1]) * 1000.0
        if head_axis_in_camera is None:
            forward_mm = float(alignment_error_m[2]) * 1000.0
        else:
            forward_mm = float(
                np.dot(rod_center_in_camera_m - head_center_in_camera_m, head_axis_in_camera)
            ) * 1000.0
        return left_mm, up_mm, forward_mm, yaw_deg

    @staticmethod
    def _horizontal_angle_deg(v: np.ndarray, ref: np.ndarray) -> float:
        """计算相机 XZ 水平面上，从 ref 到 v 的有符号夹角。"""
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        ref = np.asarray(ref, dtype=np.float64).reshape(-1)
        if v.size < 3 or ref.size < 3:
            return 0.0
        vx, vz = float(v[0]), float(v[2])
        rx, rz = float(ref[0]), float(ref[2])
        if math.hypot(vx, vz) < 1e-9 or math.hypot(rx, rz) < 1e-9:
            return 0.0
        return math.degrees(math.atan2(vx * rz - vz * rx, vx * rx + vz * rz))

    def _on_image(self, msg: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # print(f"收到图像 {frame.shape[1]}x{frame.shape[0]} @ {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}")
        if not self._enabled:
            out_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            out_msg.header = msg.header
            self._draw_pub.publish(out_msg)
            return

        result = self._estimator.on_image(frame)
        rvec, tvec = (None, None) if result is None else result
        #转化成CompressedImage格式发布
        out_msg = self._bridge.cv2_to_compressed_imgmsg(frame, dst_format="jpeg")
        out_msg.header = msg.header
        
        self._draw_pub.publish(out_msg)

        if rvec is None or tvec is None:
            return

        raw_left, raw_up, raw_forward, raw_yaw = self._compute_alignment_error_mm(rvec, tvec)

        # One Euro Filter 平滑 + 死区抑制
        left_mm, up_mm = self._smoother.update(raw_left, raw_up)
        left_mm += LEFT_OFFSET_MM
        # up_mm += UP_OFFSET_MM
        up_mm=UP_RESULT

        # print(
        #     f"raw=({raw_left:.1f}, {raw_up:.1f}, {raw_forward:.1f})  "
        #     f"yaw={raw_yaw:+.2f}deg  "
        #     f"filtered=({left_mm:.1f}, {up_mm:.1f})"
        # )

        # 发布 ROS topic（发布滤波后的值）
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
