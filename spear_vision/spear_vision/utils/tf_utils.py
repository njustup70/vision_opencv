"""
位姿/变换工具（spear_vision）

约定：
- rvec/tvec 与 OpenCV 一致：rvec 为 Rodrigues 旋转向量（3x1），tvec 为平移（3x1）。
- Rt 表示一个刚体变换：从“源坐标系”到“目标坐标系”的变换。
  例如：若 solvePnP 得到 rvec/tvec 表示 board 在 camera 下的位姿，
  这里可以把它理解为 camera_T_board（父=camera，子=board）。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Rt:
    # rvec/tvec shape 统一为 (3,1)，避免 OpenCV/NumPy 混用时出现 (3,) 或 (1,3) 的坑
    rvec: np.ndarray  # (3, 1)
    tvec: np.ndarray  # (3, 1)


def rpy_deg_to_rvec(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    # 将 roll/pitch/yaw（度）转换为 Rodrigues rvec。
    # 这里使用常见的 ZYX（yaw-pitch-roll）组合形式构造旋转矩阵。
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rmat = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    import cv2

    rvec, _ = cv2.Rodrigues(rmat)
    return np.array(rvec, dtype=np.float64).reshape(3, 1)


def rvec_tvec_to_rt(rvec: np.ndarray, tvec: np.ndarray) -> Rt:
    # 强制 reshape，保证后续矩阵运算维度一致
    return Rt(
        rvec=np.array(rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.array(tvec, dtype=np.float64).reshape(3, 1),
    )


def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    # Rodrigues 向量 -> 3x3 旋转矩阵
    import cv2

    rmat, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64).reshape(3, 1))
    return rmat


def rmat_to_rpy_deg(rmat: np.ndarray) -> tuple[float, float, float]:
    # 旋转矩阵 -> roll/pitch/yaw（度），与 rpy_deg_to_rvec 的 ZYX 约定一致
    r = np.array(rmat, dtype=np.float64).reshape(3, 3)
    yaw = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    pitch = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    roll = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    return float(roll), float(pitch), float(yaw)


def matrix_to_quaternion(rmat: np.ndarray) -> tuple[float, float, float, float]:
    # 旋转矩阵 -> 四元数（x,y,z,w）
    # 这里实现了一个不依赖外部库的转换，并在末尾做归一化。
    r = np.array(rmat, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))

    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    else:
        if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = (1.0 + r[0, 0] - r[1, 1] - r[2, 2]) ** 0.5 * 2.0
            qw = (r[2, 1] - r[1, 2]) / s
            qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s
            qz = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = (1.0 + r[1, 1] - r[0, 0] - r[2, 2]) ** 0.5 * 2.0
            qw = (r[0, 2] - r[2, 0]) / s
            qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s
            qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = (1.0 + r[2, 2] - r[0, 0] - r[1, 1]) ** 0.5 * 2.0
            qw = (r[1, 0] - r[0, 1]) / s
            qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s
            qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    # 四元数（x,y,z,w）-> 旋转矩阵（3x3）
    x = float(qx)
    y = float(qy)
    z = float(qz)
    w = float(qw)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n

    xx = x * x * s
    yy = y * y * s
    zz = z * z * s
    xy = x * y * s
    xz = x * z * s
    yz = y * z * s
    wx = w * x * s
    wy = w * y * s
    wz = w * z * s

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def quaternion_nlerp(
    q0: tuple[float, float, float, float],
    q1: tuple[float, float, float, float],
    t: float,
) -> tuple[float, float, float, float]:
    # 归一化线性插值（nlerp）：
    # - 相比 slerp 更快，足够用于实时位姿平滑；
    # - 若点积为负，翻转一边四元数，避免走长弧导致跳变。
    a = np.array(q0, dtype=np.float64)
    b = np.array(q1, dtype=np.float64)
    if float(np.dot(a, b)) < 0.0:
        b = -b
    q = (1.0 - t) * a + t * b
    q /= np.linalg.norm(q) + 1e-12
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def rotation_angle_deg(q0: tuple[float, float, float, float], q1: tuple[float, float, float, float]) -> float:
    # 两个四元数之间的夹角（度），用于离群样本判断
    a = np.array(q0, dtype=np.float64).reshape(4)
    b = np.array(q1, dtype=np.float64).reshape(4)
    dot = float(np.dot(a, b))
    dot = float(max(-1.0, min(1.0, abs(dot))))  # abs：q 与 -q 等价
    ang = 2.0 * math.degrees(math.acos(dot))
    return float(ang)


def rt_to_matrix(rt: Rt) -> np.ndarray:
    # Rt -> 4x4 齐次变换矩阵
    rmat = rodrigues_to_matrix(rt.rvec)
    t = np.array(rt.tvec, dtype=np.float64).reshape(3, 1)
    tmat = np.eye(4, dtype=np.float64)
    tmat[:3, :3] = rmat
    tmat[:3, 3:] = t
    return tmat


def matrix_to_rt(tmat: np.ndarray) -> Rt:
    # 4x4 齐次矩阵 -> Rt
    import cv2

    t = np.array(tmat[:3, 3], dtype=np.float64).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(np.array(tmat[:3, :3], dtype=np.float64).reshape(3, 3))
    return Rt(rvec=np.array(rvec, dtype=np.float64).reshape(3, 1), tvec=t)


def invert_rt(rt: Rt) -> Rt:
    # 变换求逆：T^-1 = [R^T, -R^T t]
    rmat = rodrigues_to_matrix(rt.rvec)
    r_inv = rmat.T
    t_inv = -r_inv @ rt.tvec
    import cv2

    rvec_inv, _ = cv2.Rodrigues(r_inv)
    return Rt(rvec=np.array(rvec_inv, dtype=np.float64).reshape(3, 1), tvec=t_inv)


def compose_rt(a: Rt, b: Rt) -> Rt:
    # 变换复合：T = A * B
    # 例如：camera_T_tip = camera_T_board * board_T_tip
    ma = rt_to_matrix(a)
    mb = rt_to_matrix(b)
    return matrix_to_rt(ma @ mb)
