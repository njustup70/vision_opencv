"""
外参标定（primary_T_secondary）核心逻辑（与 ROS 解耦）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from spear_vision.utils.tf_utils import (
    Rt,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rodrigues_to_matrix,
    rotation_angle_deg,
    rmat_to_rpy_deg,
)


@dataclass(frozen=True)
class CalibrationStats:
    num_samples: int
    num_kept: int
    t_std: Optional[np.ndarray]
    rpy_deg: Optional[tuple[float, float, float]]


class ExtrinsicCalibrator:
    def __init__(self) -> None:
        self._samples: list[Rt] = []
        self._finalized: bool = False
        self._final_rt: Optional[Rt] = None
        self._final_stats: Optional[CalibrationStats] = None

    def add_sample(
        self,
        rt: Rt,
        conf_primary: float,
        conf_secondary: float,
        frame_index: int,
        stride: int,
        min_conf_p: float,
        min_conf_s: float,
    ) -> bool:
        # 返回值：是否成功加入样本
        if self._finalized:
            return False
        if int(stride) > 1 and (int(frame_index) % int(stride)) != 0:
            return False
        if float(conf_primary) < float(min_conf_p) or float(conf_secondary) < float(min_conf_s):
            return False
        self._samples.append(rt)
        return True

    def maybe_finalize(
        self,
        required_samples: int,
        outlier_translation_m: float,
        outlier_rotation_deg: float,
    ) -> tuple[bool, Optional[Rt], CalibrationStats]:
        # 若样本数达到 required_samples，则计算并固化外参
        if self._finalized and self._final_rt is not None and self._final_stats is not None:
            return True, self._final_rt, self._final_stats

        total = len(self._samples)
        if total < int(required_samples):
            stats = CalibrationStats(num_samples=total, num_kept=total, t_std=None, rpy_deg=None)
            return False, None, stats

        samples_used: list[Rt] = list(self._samples)

        # 先用“全体均值”作为中心（与旧实现一致）
        mean_rt = self._mean_rt(samples_used)
        kept = samples_used

        outlier_t = float(outlier_translation_m)
        outlier_r = float(outlier_rotation_deg)
        if outlier_t > 0.0 or outlier_r > 0.0:
            # 风险点：
            # - 如果用“包含离群点的均值”做中心，极端情况下均值会被拉偏，
            #   导致“正常点反而全被剔除”的灾难性结果（单个大离群点就足够触发）。
            # 处理策略（不改变阈值语义，只改变中心的鲁棒性）：
            # 1) 先按 mean_rt 做一次剔除（保持旧逻辑优先）；
            # 2) 若保留样本太少（<3），则退化到更鲁棒的中心：
            #    - 平移中心用 median（对单点离群不敏感）
            #    - 旋转中心用“中位角距离最小”的样本四元数（类似 1-step robust）
            kept = self._filter_outliers(samples_used, mean_rt, outlier_t, outlier_r)

            if len(kept) < 3:
                t_ref, q_ref = self._robust_reference(samples_used)
                kept2 = self._filter_outliers_by_ref(samples_used, t_ref, q_ref, outlier_t, outlier_r)
                if len(kept2) >= 3:
                    kept = kept2

            # 若剔除后仍然 <3，则说明阈值过严/数据分布异常：不做剔除，避免 num_kept=0
            if len(kept) < 3:
                kept = samples_used

            if len(kept) >= 3 and len(kept) < len(samples_used):
                mean_rt = self._mean_rt(kept)

        stats = self._make_stats(total=len(samples_used), kept=kept, mean_rt=mean_rt)
        self._finalized = True
        self._final_rt = mean_rt
        self._final_stats = stats
        return True, mean_rt, stats

    def reset(self) -> None:
        self._samples.clear()
        self._finalized = False
        self._final_rt = None
        self._final_stats = None

    def status(self) -> CalibrationStats:
        # 返回当前状态（未 finalize 时用现有样本统计）
        total = len(self._samples)
        if total == 0:
            return CalibrationStats(num_samples=0, num_kept=0, t_std=None, rpy_deg=None)
        mean_rt = self._mean_rt(self._samples)
        return self._make_stats(total=total, kept=self._samples, mean_rt=mean_rt)

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @staticmethod
    def _mean_rt(samples: list[Rt]) -> Rt:
        # 平移取均值；旋转用四元数符号对齐后求和归一化
        if not samples:
            raise ValueError("No samples.")

        ts = np.stack([np.array(s.tvec, dtype=np.float64).reshape(3) for s in samples], axis=0)
        t_mean = np.mean(ts, axis=0).reshape(3, 1)

        q_list = []
        for s in samples:
            q = matrix_to_quaternion(rodrigues_to_matrix(s.rvec))  # (x,y,z,w)
            q_list.append(np.array(q, dtype=np.float64).reshape(4))
        q0 = q_list[0]
        acc = np.zeros(4, dtype=np.float64)
        for q in q_list:
            if float(np.dot(q0, q)) < 0.0:
                q = -q
            acc += q
        acc /= np.linalg.norm(acc) + 1e-12
        rmat = quaternion_to_matrix(float(acc[0]), float(acc[1]), float(acc[2]), float(acc[3]))

        import cv2

        rvec, _ = cv2.Rodrigues(rmat)
        return Rt(rvec=np.array(rvec, dtype=np.float64).reshape(3, 1), tvec=np.array(t_mean, dtype=np.float64).reshape(3, 1))

    @staticmethod
    def _filter_outliers(samples: list[Rt], center: Rt, outlier_t: float, outlier_r: float) -> list[Rt]:
        # 按中心位姿做一次离群剔除
        t_ref = np.array(center.tvec, dtype=np.float64).reshape(3, 1)
        q_ref = matrix_to_quaternion(rodrigues_to_matrix(center.rvec))
        return ExtrinsicCalibrator._filter_outliers_by_ref(samples, t_ref, q_ref, outlier_t, outlier_r)

    @staticmethod
    def _filter_outliers_by_ref(
        samples: list[Rt],
        t_ref: np.ndarray,
        q_ref: tuple[float, float, float, float],
        outlier_t: float,
        outlier_r: float,
    ) -> list[Rt]:
        kept: list[Rt] = []
        for s in samples:
            dt = float(np.linalg.norm(np.array(s.tvec, dtype=np.float64).reshape(3, 1) - t_ref))
            q = matrix_to_quaternion(rodrigues_to_matrix(s.rvec))
            dr = float(rotation_angle_deg(q_ref, q))
            ok_t = (outlier_t <= 0.0) or (dt <= outlier_t)
            ok_r = (outlier_r <= 0.0) or (dr <= outlier_r)
            if ok_t and ok_r:
                kept.append(s)
        return kept

    @staticmethod
    def _robust_reference(samples: list[Rt]) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        # 构造一个更鲁棒的参考中心（用于“均值被离群点拉偏”的场景）
        ts = np.stack([np.array(s.tvec, dtype=np.float64).reshape(3) for s in samples], axis=0)
        t_med = np.median(ts, axis=0).reshape(3, 1)

        # 旋转参考：选择“与其它样本的中位角距离最小”的那一个样本作为中心
        qs = [matrix_to_quaternion(rodrigues_to_matrix(s.rvec)) for s in samples]
        if not qs:
            return t_med, (0.0, 0.0, 0.0, 1.0)

        best_i = 0
        best_med = float("inf")
        for i, qi in enumerate(qs):
            d = [rotation_angle_deg(qi, qj) for qj in qs]
            med = float(np.median(np.array(d, dtype=np.float64))) if d else float("inf")
            if med < best_med:
                best_med = med
                best_i = i

        return t_med, qs[best_i]

    @staticmethod
    def _make_stats(total: int, kept: list[Rt], mean_rt: Rt) -> CalibrationStats:
        if not kept:
            return CalibrationStats(num_samples=total, num_kept=0, t_std=None, rpy_deg=None)
        ts = np.stack([np.array(s.tvec, dtype=np.float64).reshape(3) for s in kept], axis=0)
        std_t = np.std(ts, axis=0)
        roll, pitch, yaw = rmat_to_rpy_deg(rodrigues_to_matrix(mean_rt.rvec))
        return CalibrationStats(
            num_samples=int(total),
            num_kept=int(len(kept)),
            t_std=np.array(std_t, dtype=np.float64).reshape(3),
            rpy_deg=(float(roll), float(pitch), float(yaw)),
        )
