"""
标定结果“自动复用”辅助（spear_vision）

你的目标是：内参标定一次后，后续 PnP 节点无需再手动复制/粘贴 YAML 路径。

实现思路（不改变既有对外接口）：
- `charuco_calibration_node` 每次保存内参 YAML 后，额外写入一个“指针文件”，记录最新的 YAML 路径；
- `board_pose_node` / `spear_tip_node` 若参数 `camera_calibration_yaml` 为空，则尝试读取该指针文件，
  自动加载“最近一次标定结果”。

这样你只需要标定时指定 output_yaml_path（或用默认），PnP 节点会自动复用。
"""

from __future__ import annotations

import os
from typing import Optional


def _ros_home() -> str:
    # ROS_HOME：ROS 生态里常用的运行时目录，默认 ~/.ros
    return os.environ.get("ROS_HOME", os.path.expanduser("~/.ros"))


def _store_dir() -> str:
    return os.path.join(_ros_home(), "spear_vision")


def last_calibration_path_file() -> str:
    # 记录“最近一次内参 YAML 路径”的指针文件
    return os.path.join(_store_dir(), "last_camera_calibration_path.txt")


def write_last_calibration_path(yaml_path: str) -> None:
    # 写入指针文件（覆盖更新）
    expanded = os.path.expanduser(str(yaml_path))
    os.makedirs(_store_dir(), exist_ok=True)
    with open(last_calibration_path_file(), "w", encoding="utf-8") as f:
        f.write(expanded.strip() + "\n")


def read_last_calibration_path() -> Optional[str]:
    # 读取指针文件，返回 YAML 路径（若不存在/无效则返回 None）
    path_file = last_calibration_path_file()
    if not os.path.exists(path_file):
        return None
    try:
        with open(path_file, "r", encoding="utf-8") as f:
            p = (f.read() or "").strip()
    except Exception:
        return None
    if not p:
        return None
    p = os.path.expanduser(p)
    if not os.path.exists(p):
        return None
    return p

