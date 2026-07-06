import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessExit


def generate_launch_description():
    # launch 文件在 <repo>/src/spear_vision/launch/，往上 3 层到项目根
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

    # ---- YOLO 检测节点 ----
    # 必须在 project_root 运行（脚本内部读 cv_lib/USB_capture.yaml）
    yolo_node = ExecuteProcess(
        cmd=["python3", os.path.join(project_root, "cv_lib", "YOLO2topic.py")],
        cwd=project_root,
        name="YOLO_recog_node",
        output="screen",
        emulate_tty=True,
    )

    # ---- ArUco PnP + 串口节点 ----
    # 必须在 spear/ 目录运行（依赖本地模块 arucopnp / myserial）
    spear_dir = os.path.join(project_root, "spear")
    aruco_node = ExecuteProcess(
        cmd=["python3", os.path.join(spear_dir, "ros2_arucopnp_serial_node.py")],
        cwd=spear_dir,
        name="arucopnp_serial_node",
        output="screen",
        emulate_tty=True,
    )

    # 节点退出时打印日志
    yolo_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=yolo_node,
            on_exit=[LogInfo(msg="YOLO 节点已退出")],
        )
    )
    aruco_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=aruco_node,
            on_exit=[LogInfo(msg="ArUco PnP 节点已退出")],
        )
    )

    return LaunchDescription([
        yolo_node,
        aruco_node,
        yolo_exit_handler,
        aruco_exit_handler,
    ])
