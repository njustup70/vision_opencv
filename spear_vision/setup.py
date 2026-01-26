from glob import glob

from setuptools import find_packages, setup

package_name = "spear_vision"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # ament_python 固定约定：resource 文件用于 ament index 识别该包
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        # 安装 package.xml 到 share/<pkg> 下（ROS2 查找依赖/元信息用）
        (f"share/{package_name}", ["package.xml"]),
        # 安装配置文件与 launch 文件，便于 ros2 launch / 参数加载
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@example.com",
    description="ChArUco/ArUco board pose estimation for ROS 2.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # 单板位姿：camera_T_board（优先 ChArUco，fallback ArUco Board）
            "board_pose = spear_vision.nodes.board_pose_node:main",
            # 小板位姿：camera_T_small_board（只用小板，输出 left/up/forward mm）
            "small_board_pose = spear_vision.nodes.small_board_pose_node:main",
            # 小板位姿 -> 串口桥接（仅发送 left/up mm）
            "small_board_serial_bridge = spear_vision.nodes.small_board_serial_bridge_node:main",
            # ChArUco 内参标定：导出 camera_info YAML
            "charuco_calib = spear_vision.nodes.charuco_calibration_node:main",
            # 双板 + tip：camera_T_tip（用于矛头对接）
            "spear_tip = spear_vision.nodes.spear_tip_node:main",
            # 外参标定专用（大板+小板）
            "spear_tip_calib = spear_vision.nodes.spear_tip_calib_node:main",
            # 运行测量专用（只看大板）
            "spear_tip_run = spear_vision.nodes.spear_tip_run_node:main",
        ],
    },
)
