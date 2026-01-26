"""
启动：小板 PnP + 串口桥接（一键模式）

功能：
- small_board_pose: 计算小板位姿
- small_board_serial_bridge: 把 left/up(mm) 发给 STM32（UART）
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory("spear_vision")

    # 小板配置 + 相机内参（优先工作空间）
    workspace_cfg = os.path.expanduser("~/CHaruco/hik_ws/src/spear_vision/config/small_board.yaml")
    default_cfg = workspace_cfg if os.path.exists(workspace_cfg) else os.path.join(pkg_share, "config", "small_board.yaml")

    workspace_cam = os.path.expanduser("~/CHaruco/hik_ws/src/spear_vision/config/camera.yaml")
    default_cam = workspace_cam if os.path.exists(workspace_cam) else ""

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path", default_value=default_cfg),
            DeclareLaunchArgument("camera_calibration_yaml", default_value=default_cam),
            DeclareLaunchArgument("show_opencv_window", default_value="true"),
            DeclareLaunchArgument("opencv_window_name", default_value="small_board_view"),
            # 串口参数
            DeclareLaunchArgument("port", default_value="/dev/serial_ch340"),
            DeclareLaunchArgument("baudrate", default_value="115200"),
            DeclareLaunchArgument("out_first_frame", default_value="250"),  # 0xFA
            DeclareLaunchArgument("out_frame_id", default_value="177"),  # 0xB1
            DeclareLaunchArgument("min_confidence", default_value="0.0"),
            DeclareLaunchArgument("max_mean_reproj_px", default_value="0.0"),
            DeclareLaunchArgument("invert_left", default_value="false"),
            DeclareLaunchArgument("invert_up", default_value="false"),
            # 小板位姿节点
            Node(
                package="spear_vision",
                executable="small_board_pose",
                name="small_board_pose",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "config_path": LaunchConfiguration("config_path"),
                        "camera_calibration_yaml": LaunchConfiguration("camera_calibration_yaml"),
                        "show_opencv_window": ParameterValue(LaunchConfiguration("show_opencv_window"), value_type=bool),
                        "opencv_window_name": LaunchConfiguration("opencv_window_name"),
                    }
                ],
            ),
            # 串口桥接节点
            Node(
                package="spear_vision",
                executable="small_board_serial_bridge",
                name="small_board_serial_bridge",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "port": LaunchConfiguration("port"),
                        "baudrate": LaunchConfiguration("baudrate"),
                        "out_first_frame": LaunchConfiguration("out_first_frame"),
                        "out_frame_id": LaunchConfiguration("out_frame_id"),
                        "min_confidence": LaunchConfiguration("min_confidence"),
                        "max_mean_reproj_px": LaunchConfiguration("max_mean_reproj_px"),
                        "invert_left": ParameterValue(LaunchConfiguration("invert_left"), value_type=bool),
                        "invert_up": ParameterValue(LaunchConfiguration("invert_up"), value_type=bool),
                    }
                ],
            ),
        ]
    )

