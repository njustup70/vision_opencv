"""
启动：双板 + tip 位姿估计（spear_tip）

默认读取 `spear_vision/config/spear_tip.yaml`：
- primary_charuco: 大板（倾斜放置）
- secondary_charuco: 小 5x5 板（贴矛头平面）
- tip.offset_m / tip.rpy_deg: tip 相对小板的固定关系

说明：
- 现在外参标定与运行被拆为两个节点；
- 你可以通过 mode 选择启动哪个节点：
  * mode=calibrate -> spear_tip_calib
  * mode=run       -> spear_tip_run
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory("spear_vision")
    workspace_cfg = os.path.expanduser("~/CHaruco/hik_ws/src/spear_vision/config/spear_tip.yaml")
    if os.path.exists(workspace_cfg):
        default_config = workspace_cfg
    else:
        default_config = os.path.join(pkg_share, "config", "spear_tip.yaml")

    # camera_calibration_yaml 推荐填写为标定导出的 YAML（保证 PnP 的尺度正确）
    # 默认优先使用工作空间里的 camera.yaml，减少你每次 launch 都要手填路径的负担。
    workspace_cam = os.path.expanduser("~/CHaruco/hik_ws/src/spear_vision/config/camera.yaml")
    default_cam = workspace_cam if os.path.exists(workspace_cam) else ""
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path", default_value=default_config),
            DeclareLaunchArgument("camera_calibration_yaml", default_value=default_cam),
            DeclareLaunchArgument("mode", default_value="calibrate"),
            # 运行时可关闭可视化（OpenCV 弹窗 + debug_image）
            DeclareLaunchArgument("show_opencv_window", default_value="true"),
            Node(
                package="spear_vision",
                executable="spear_tip_calib",
                name="spear_tip_calib",
                output="screen",
                condition=IfCondition(
                    PythonExpression(["'", LaunchConfiguration("mode"), "' == 'calibrate'"])
                ),
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
            Node(
                package="spear_vision",
                executable="spear_tip_run",
                name="spear_tip_run",
                output="screen",
                condition=IfCondition(
                    PythonExpression(["'", LaunchConfiguration("mode"), "' == 'run'"])
                ),
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
