"""
同时启动 YOLO 检测 + ArUco PnP 串口 两个节点

用法:
    ros2 launch spear_vision launch_vision.launch.py
"""

import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # launch 文件位于 <ws>/src/spear_vision/launch/launch_vision.launch.py
    # 往上 4 层 → workspace 根
    ws_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    model_path = os.path.join(ws_root, "1.20.pt")

    # ---- YOLO 目标检测节点 ----
    yolo_node = Node(
        package="spear_vision",
        executable="yolo_detection",
        name="YOLO_recog_node",
        output="screen",
        parameters=[{"model_path": model_path}],
    )

    # ---- ArUco PnP + 串口节点 ----
    aruco_node = Node(
        package="spear_vision",
        executable="arucopnp_serial",
        name="arucopnp_serial_node",
        output="screen",
    )

    return LaunchDescription([
        yolo_node,
        aruco_node,
    ])
