"""
位姿平滑滤波（与 ROS 解耦）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from spear_vision.utils.tf_utils import (
    Rt,
    matrix_to_quaternion,
    quaternion_nlerp,
    quaternion_to_matrix,
    rodrigues_to_matrix,
)


@dataclass
class PoseLowPassFilter:
    # 低通滤波（保存内部状态）
    _prev: Optional[Rt] = None

    def reset(self) -> None:
        self._prev = None

    @property
    def last(self) -> Optional[Rt]:
        return self._prev

    def update(self, rt: Rt, alpha: float, rotation_mode: str = "rvec") -> Rt:
        # rotation_mode:
        # - "rvec": 保持原行为（对 rvec 线性插值）
        # - "quat": 使用四元数 nlerp（更严谨）
        # 防御性：避免 NaN/Inf 位姿进入滤波器导致后续一直输出 NaN（表现为 TF/坐标轴消失）
        def _is_valid(x: Rt) -> bool:
            return bool(np.isfinite(np.array(x.rvec, dtype=np.float64)).all() and np.isfinite(np.array(x.tvec, dtype=np.float64)).all())

        if self._prev is None:
            if _is_valid(rt):
                self._prev = rt
            return rt

        if not _is_valid(self._prev):
            # 兜底：若内部状态已坏，直接用当前输入重置
            self._prev = rt if _is_valid(rt) else None
            return rt

        if not _is_valid(rt):
            # 输入无效：保持上一帧（宁可“保持最后一次有效姿态”，也不要把 NaN 传播出去）
            return self._prev

        a = float(alpha)
        if rotation_mode == "quat":
            q0 = matrix_to_quaternion(rodrigues_to_matrix(self._prev.rvec))
            q1 = matrix_to_quaternion(rodrigues_to_matrix(rt.rvec))
            q = quaternion_nlerp(q0, q1, 1.0 - a)
            rmat = quaternion_to_matrix(q[0], q[1], q[2], q[3])
            import cv2

            rvec, _ = cv2.Rodrigues(rmat)
            rvec = np.array(rvec, dtype=np.float64).reshape(3, 1)
        else:
            rvec = a * self._prev.rvec + (1.0 - a) * rt.rvec
            rvec = np.array(rvec, dtype=np.float64).reshape(3, 1)

        tvec = a * self._prev.tvec + (1.0 - a) * rt.tvec
        tvec = np.array(tvec, dtype=np.float64).reshape(3, 1)

        self._prev = Rt(rvec=rvec, tvec=tvec)
        return self._prev
