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
from sensor_msgs.msg import Image
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
LEFT_OFFSET_MM = -60.0
# UP_OFFSET_MM = -200.0
UP_RESULT= -45.0
# 只用于 yaw 调试显示的零点修正；left/up 补偿由轴线和平面交点计算得到。
# 外参修正量（单位：mm），作用在滤波后的 left/up 偏移上


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

        self._draw_pub = self.create_publisher(Image, DRAW_RESULT_TOPIC, qos_profile_sensor_data)
        self._offset_pub = self.create_publisher(PointStamped, OFFSET_MM_TOPIC, 10)
        self._img_sub = self.create_subscription(Image, IMAGE_TOPIC, self._on_image, 1)
        self._cmd_sub = self.create_subscription(String, COMMAND_TOPIC, self._on_exec_request, 10)
        self._enabled = True

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
        tvec: np.ndarray,board_pitch_deg=-21.0
    ) -> tuple[float, float, float, float]:

        """
        计算矛杆对接的 left/up 偏差以及 Yaw 补偿角。
        
        参数:
        tvec, rvec: solvePnP 输出的标定板外参
        board_pitch_deg: 标定板相对于矛杆的物理 Pitch 角度 (通常为 30 或 -30，取决于安装)
        
        返回:
        left_mm: 向左移动的毫米数
        up_mm: 向上移动的毫米数
        yaw_deg: 对接机构需要旋转的 Yaw 角度 (度)
        """
        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        r = np.asarray(rvec, dtype=np.float64).reshape(3)
        if t.size < 3 or r.size < 3:
            return 0.0, 0.0, 0.0, 0.0
        if not np.all(np.isfinite(t[:3])) or not np.all(np.isfinite(r[:3])):
            return 0.0, 0.0, 0.0, 0.0
        # 1. 获取标定板到相机的旋转矩阵 R_cam_board
        R_cam_board, _ = cv2.Rodrigues(r)
        
        
        #求绕当前x旋转n度
        # 2. 从旋转矩阵中提取欧拉角 (假设旋转顺序为 Z-Y-X)
        # R[2, 1] 和 R[2, 2] 用于计算绕 X 轴的角度
        # R[2, 0] 用于计算绕 Y 轴的角度
    
        # 绕 X 轴旋转角度 (Pitch)
        #绕当前y旋转30度
        R=R_cam_board
        # R=R_cam_board*np.array([[1, 0, 0], [0, math.cos(math.radians(-board_pitch_deg)), -math.sin(math.radians(-board_pitch_deg))], [0, math.sin(math.radians(-board_pitch_deg)), math.cos(math.radians(-board_pitch_deg))]])  
        theta_x = np.arctan2(R[2, 1], R[2, 2])
        
        # 绕 Y 轴旋转角度 (Yaw)
        # 注意: 这里根据 R[2,0] 计算，具体正负号取决于你定义的"正对"方向
        theta_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        
        # 绕 Z 轴旋转角度 (Roll) - 如果不需要可以忽略
        theta_z = np.arctan2(R[1, 0], R[0, 0])
        
        # 3. 弧度转角度
        angle_x = np.degrees(theta_x)
        angle_y = np.degrees(theta_y)
        up_mm=float(t[1]) * 1000.0
        left_mm=float(t[0]+0.375*math.sin(theta_x)) * 1000.0
        yaw_deg=angle_x
        # print(f"tvec={t},left_mm={left_mm:.2f},yaw={angle_x:.2f}")
        return -left_mm, up_mm, yaw_deg,yaw_deg
        print(f"Pitch={angle_x:.2f}deg, Yaw={angle_y:.2f}deg, Roll={np.degrees(theta_z):.2f}deg")
        # 2. 定义从“矛杆”到“标定板”的逆向补偿旋转 (剥离 30度 Pitch)
        # 注意：这里的正负号取决于标定板是上仰还是下俯，如果在实机上 X 轴偏了，将 30 改为 -30 即可
        theta = math.radians(board_pitch_deg)
        R_board_spear = np.array([
            [1, 0, 0],
            [0, math.cos(theta), -math.sin(theta)],
            [0, math.sin(theta),  math.cos(theta)]
        ])
        
        # 3. 计算矛杆在相机坐标系下的真实旋转矩阵
        R_cam_spear = R_cam_board @ R_board_spear
        
        # 4. 提取矛杆的真实轴线方向 (即相机坐标系下的矛杆 Z 轴分量)
        rod_axis = R_cam_spear[:, 2]
        
        # 【校验机制】: 因为相机和矛杆都没有 Pitch，所以它们都在水平面上。
        # 此时 rod_axis 的 Y 轴分量 rod_axis[1] 在理论上应该非常接近 0。
        # 如果实际跑出来 rod_axis[1] 很大，说明 board_pitch_deg 的正负号填反了，或者机械安装有较大误差。
        
        # 5. 射线求交：补偿由于 Distance 和 Yaw 造成的 X/Y 偏移
        if abs(rod_axis[2]) > 1e-9:
            # 将 t (标定板原点) 沿着 spear_axis 延伸到相机的 Z=0 平面
            command_point = t - (t[2] / rod_axis[2]) * rod_axis
        else:
            command_point = t
        print(f"t={t}, rod_axis={rod_axis}, command_point={command_point}")    
        left_mm = -float(command_point[0]) * 1000.0
        up_mm = -float(command_point[1]) * 1000.0
        
        # 6. 计算 Yaw 角度补偿
        # 提取矛杆 Z 轴在相机 XZ 平面上的投影夹角
        # R_cam_spear[0, 2] 是 X 方向分量，R_cam_spear[2, 2] 是 Z 方向分量
        yaw_rad = math.atan2(R_cam_spear[0, 2], R_cam_spear[2, 2])
        yaw_deg = math.degrees(yaw_rad)
        
        return left_mm, up_mm, yaw_deg,0.0


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

        out_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
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
            offset_msg.point.z = raw_yaw
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
