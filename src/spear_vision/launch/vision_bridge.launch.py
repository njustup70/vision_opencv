"""
同时启动 YOLO 检测 + ArUco PnP 串口桥接。

功能：
- YOLO2topic: USB 摄像头 YOLO 推理，发布 class_name 和标注图像
- ros2_arucopnp_serial_node: 海康相机 ArUco 位姿估计 + 串口下发偏移
"""

import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    # 脚本路径（相对于 vision_opencv 仓库根目录）
    # __file__ → launch/ → spear_vision/ → src/ → vision_opencv/
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    yolo_script = os.path.join(repo_root, "cv_lib", "YOLO2topic.py")
    spear_script = os.path.join(repo_root, "spear", "ros2_arucopnp_serial_node.py")

    return LaunchDescription(
        [
            # ---- YOLO 检测节点 ----
            ExecuteProcess(
                cmd=["python3", yolo_script],
                name="YOLO_recog_node",
                output="screen",
                shell=False,
            ),
            # ---- ArUco PnP 串口桥接节点 ----
            ExecuteProcess(
                cmd=["python3", spear_script],
                name="arucopnp_serial_node",
                output="screen",
                shell=False,
            ),
        ]
    )
