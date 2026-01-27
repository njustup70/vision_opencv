"""
启动：小 ChArUco 板（5cm x 5cm）位姿估计（small_board_pose）

用途：
- 完全以“小板”为基准输出 camera_T_small_board；
- 弹 OpenCV 窗口可视化：marker/角点、置信度、重投影误差、以及 (left/up/forward) mm 偏移。
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

    workspace_cfg = os.path.expanduser("~/CHaruco/hik_ws/src/spear_vision/config/small_board.yaml")
    default_config = workspace_cfg if os.path.exists(workspace_cfg) else os.path.join(pkg_share, "config", "small_board.yaml")

    # 默认优先使用工作空间里的 camera.yaml（内参标定输出）
    workspace_cam = os.path.expanduser("~/CHaruco/hik_ws/src/spear_vision/config/camera.yaml")
    default_cam = workspace_cam if os.path.exists(workspace_cam) else ""

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path", default_value=default_config),
            DeclareLaunchArgument("camera_calibration_yaml", default_value=default_cam),
            DeclareLaunchArgument("show_opencv_window", default_value="true"),
            DeclareLaunchArgument("opencv_window_name", default_value="small_board_view"),
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
        ]
    )

