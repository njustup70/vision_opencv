"""
启动：单板位姿估计（board_pose）

默认读取 `spear_vision/config/board.yaml`，可通过 launch 参数覆盖：
- config_path: board.yaml 的路径
- camera_calibration_yaml: 相机内参 YAML（推荐）
- show_opencv_window: 是否弹出 OpenCV 视野窗口（默认开启）
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
    workspace_cfg = os.path.expanduser("~/CHaruco/hik_ws/src/spear_vision/config/board.yaml")
    if os.path.exists(workspace_cfg):
        default_config = workspace_cfg
    else:
        default_config = os.path.join(pkg_share, "config", "board.yaml")

    # 默认优先使用工作空间里的 camera.yaml，减少你每次 launch 都要手填路径的负担。
    workspace_cam = os.path.expanduser("~/CHaruco/hik_ws/src/spear_vision/config/camera.yaml")
    default_cam = workspace_cam if os.path.exists(workspace_cam) else ""

    # 注意：board_pose 节点会从 config_path 读取 topic/板参数/gating/pnp 等
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path", default_value=default_config),
            DeclareLaunchArgument("camera_calibration_yaml", default_value=default_cam),
            DeclareLaunchArgument("show_opencv_window", default_value="true"),
            Node(
                package="spear_vision",
                executable="board_pose",
                name="board_pose",
                output="screen",
                parameters=[
                    {
                        "config_path": LaunchConfiguration("config_path"),
                        "camera_calibration_yaml": LaunchConfiguration("camera_calibration_yaml"),
                        "show_opencv_window": ParameterValue(
                            LaunchConfiguration("show_opencv_window"), value_type=bool
                        ),
                    }
                ],
            ),
        ]
    )
