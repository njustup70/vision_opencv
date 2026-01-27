"""
启动：ChArUco 相机内参标定（charuco_calib）

默认读取 `spear_vision/config/board.yaml` 的 charuco 参数（棋盘规格与字典）。

额外说明：
- 为了“开箱即用”地看到标定视野，本 launch 支持直接让节点弹出 OpenCV 窗口：
  `ros2 launch spear_vision charuco_calibration.launch.py show_opencv_window:=true`
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
    default_config = os.path.join(pkg_share, "config", "board.yaml")
    workspace_cfg_dir = os.path.expanduser("~/CHaruco/hik_ws/src/spear_vision/config")
    if os.path.isdir(workspace_cfg_dir):
        default_output = os.path.join(workspace_cfg_dir, "camera.yaml")
    else:
        default_output = os.path.expanduser("~/charuco_camera.yaml")

    # output_yaml_path 默认写到 ~/charuco_camera.yaml
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path", default_value=default_config),
            DeclareLaunchArgument("output_yaml_path", default_value=default_output),
            # 让标定节点直接弹出 OpenCV 窗口显示视野（不依赖 rqt_image_view/rviz）
            DeclareLaunchArgument("show_opencv_window", default_value="true"),
            DeclareLaunchArgument("opencv_window_name", default_value="charuco_calib"),
            # 自动标定/自动退出（满足“采满 100 帧自动保存并退出”的流程）
            DeclareLaunchArgument("target_samples", default_value="100"),
            # 采样步长：每 N 帧取 1 帧（默认 20，给你时间改变棋盘姿态）
            DeclareLaunchArgument("sample_stride", default_value="20"),
            DeclareLaunchArgument("auto_calibrate_on_target", default_value="true"),
            DeclareLaunchArgument("auto_exit_after_calibration", default_value="true"),
            Node(
                package="spear_vision",
                executable="charuco_calib",
                name="charuco_calibration",
                output="screen",
                parameters=[
                    {
                        "config_path": LaunchConfiguration("config_path"),
                        "output_yaml_path": LaunchConfiguration("output_yaml_path"),
                        "show_opencv_window": ParameterValue(
                            LaunchConfiguration("show_opencv_window"), value_type=bool
                        ),
                        "opencv_window_name": LaunchConfiguration("opencv_window_name"),
                        "target_samples": LaunchConfiguration("target_samples"),
                        "sample_stride": LaunchConfiguration("sample_stride"),
                        "auto_calibrate_on_target": ParameterValue(
                            LaunchConfiguration("auto_calibrate_on_target"), value_type=bool
                        ),
                        "auto_exit_after_calibration": ParameterValue(
                            LaunchConfiguration("auto_exit_after_calibration"), value_type=bool
                        ),
                    }
                ],
            ),
        ]
    )
