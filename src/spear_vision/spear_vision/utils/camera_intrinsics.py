"""
相机内参解析（spear_vision）

本包所有 PnP / 重投影误差计算，都依赖相机内参（K + D）。

注意：
- 你当前的海康驱动 `hik_camera_ros2` 发布的 `CameraInfo` 只填了 header，
  K/D/R/P 为空，因此运行时必须：
  1) 先用 `charuco_calibration_node` 标定生成 YAML，或
  2) 使用其它标定工具生成 camera_info YAML，再由节点加载。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from sensor_msgs.msg import CameraInfo

from .yaml_io import load_yaml


@dataclass(frozen=True)
class CameraIntrinsics:
    # camera_matrix: 3x3 相机内参矩阵
    # dist_coeffs: 畸变参数（常见为 5 个：k1,k2,p1,p2,k3；也可能更多）
    camera_matrix: np.ndarray  # (3, 3)
    dist_coeffs: np.ndarray  # (N, 1)
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    def is_valid(self) -> bool:
        # 最基本的有效性检查：fx/fy 必须为正
        if self.camera_matrix.shape != (3, 3):
            return False
        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        if fx <= 0.0 or fy <= 0.0:
            return False
        return True


def _as_np_float64_matrix(data: Any, shape: tuple[int, ...]) -> np.ndarray:
    # YAML 里通常是 list[float]，这里统一转为 float64 并 reshape
    arr = np.array(data, dtype=np.float64)
    return arr.reshape(shape)


def intrinsics_from_camera_info(msg: CameraInfo) -> Optional[CameraIntrinsics]:
    # ROS CameraInfo.k 是 3x3 展平数组（row-major）
    k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    if fx <= 0.0 or fy <= 0.0:
        # 常见于驱动没有发布有效内参：保持返回 None，让上层走 YAML 或等待有效 CameraInfo
        return None

    # ROS CameraInfo.d 是变长数组，reshape 为 (N,1) 便于 OpenCV 接口调用
    d = np.array(msg.d, dtype=np.float64).reshape(-1, 1)
    return CameraIntrinsics(
        camera_matrix=k,
        dist_coeffs=d,
        image_width=int(msg.width) if msg.width else None,
        image_height=int(msg.height) if msg.height else None,
    )


def intrinsics_from_yaml(path: str) -> CameraIntrinsics:
    # 兼容 ROS camera_calibration 工具链常见 YAML 格式：
    # camera_matrix: {rows, cols, data:[...]}
    # distortion_coefficients: {rows, cols, data:[...]}
    data = load_yaml(path)
    if "camera_matrix" not in data or "distortion_coefficients" not in data:
        raise ValueError("Invalid camera calibration YAML (missing camera_matrix/distortion_coefficients).")

    cm = data["camera_matrix"]
    dc = data["distortion_coefficients"]
    k = _as_np_float64_matrix(cm["data"], (3, 3))
    d = np.array(dc["data"], dtype=np.float64).reshape(-1, 1)
    return CameraIntrinsics(
        camera_matrix=k,
        dist_coeffs=d,
        image_width=int(data.get("image_width")) if data.get("image_width") else None,
        image_height=int(data.get("image_height")) if data.get("image_height") else None,
    )
