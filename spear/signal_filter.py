"""自适应信号滤波器 —— 用于平滑 PnP 偏移量, 消除电机抖动。

核心算法: One Euro Filter (1€ Filter)
  - 信号静止时 → 大幅平滑, 彻底消除抖动
  - 信号快速移动时 → 减小平滑, 保证响应速度
  - 延迟远低于滑动平均 / 中值滤波 / 卡尔曼滤波

参考论文: Casiez, Roussel & Vogel, "1€ Filter: A Simple Speed-based Low-pass Filter
          for Noisy Input in Interactive Systems", CHI 2012.

使用示例:
    smoother = OffsetSmoother()           # 使用默认参数
    left_mm, up_mm = smoother.update(left_mm, up_mm)
    if smoother.should_send:              # 死区判断
        serial.write(...)

Author: Antigravity AI
Date: 2026-03-30
"""

from __future__ import annotations

import math
import time


class OneEuroFilter:
    """单通道 One Euro 自适应低通滤波器。

    Parameters
    ----------
    min_cutoff : float
        最小截止频率 (Hz)。值越小, 静止时平滑越强。
        推荐 0.3 ~ 1.0, 默认 0.5。
    beta : float
        速度系数。值越大, 快速运动时滤波越弱(跟随性越好)。
        推荐 0.001 ~ 0.05, 默认 0.007。
    d_cutoff : float
        速度信号的截止频率, 通常固定 1.0 即可。
    """

    def __init__(
        self,
        min_cutoff: float = 0.5,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        # 内部状态
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def reset(self) -> None:
        """重置滤波器状态(例如丢失目标后重新检测到时调用)。"""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _smoothing_factor(t_e: float, cutoff: float) -> float:
        """计算低通滤波的平滑因子 alpha。"""
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def __call__(self, x: float, t: float | None = None) -> float:
        """输入原始值, 返回滤波后的值。

        Parameters
        ----------
        x : float
            当前原始测量值。
        t : float, optional
            当前时间戳 (秒)。如果不传, 则自动使用 time.monotonic()。
        """
        if t is None:
            t = time.monotonic()

        # 第一次调用 —— 直接输出, 无法计算速度
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x

        # 帧间时间
        t_e = t - self._t_prev
        if t_e <= 0:
            t_e = 1e-6  # 防止除零

        # 1) 先对速度做低通滤波
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # 2) 根据速度自适应调整截止频率
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # 3) 对信号本身做低通滤波
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev

        # 保存状态
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t

        return x_hat


class OffsetSmoother:
    """双通道偏移量平滑器 (left_mm, up_mm), 包含 One Euro Filter + 死区。

    Parameters
    ----------
    min_cutoff : float
        One Euro Filter 最小截止频率, 越小静止时越平滑。默认 0.5。
    beta : float
        速度系数, 越大跟随快速运动越灵敏。默认 0.007。
    dead_zone_mm : float
        死区阈值 (毫米)。滤波后 |offset| < dead_zone 的分量置零,
        避免电机持续做微量修正。默认 0.3 mm。
    lost_timeout_s : float
        目标丢失超时 (秒)。超过此时间没有新数据视为目标丢失,
        下次检测到时重置滤波器。默认 0.5 秒。
    """

    def __init__(
        self,
        min_cutoff: float = 0.5,
        beta: float = 0.007,
        dead_zone_mm: float = 0.3,
        lost_timeout_s: float = 0.5,
    ) -> None:
        self._filter_x = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._filter_y = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.dead_zone_mm = dead_zone_mm
        self.lost_timeout_s = lost_timeout_s

        self._last_update_t: float | None = None
        self.should_send: bool = False  # 上次 update 后是否应发送串口

        # 滤波后的值 (供外部读取)
        self.filtered_left_mm: float = 0.0
        self.filtered_up_mm: float = 0.0

    def reset(self) -> None:
        """手动重置所有状态。"""
        self._filter_x.reset()
        self._filter_y.reset()
        self._last_update_t = None
        self.should_send = False
        self.filtered_left_mm = 0.0
        self.filtered_up_mm = 0.0

    def update(self, left_mm: float, up_mm: float) -> tuple[float, float]:
        """输入原始偏移量, 返回滤波后的偏移量。

        同时更新 self.should_send 标志:
          - True  → 至少一个轴超出死区, 应发送串口指令
          - False → 两个轴都在死区内, 无需发送

        Parameters
        ----------
        left_mm : float
            原始 left 偏移 (mm)。
        up_mm : float
            原始 up 偏移 (mm)。

        Returns
        -------
        (filtered_left_mm, filtered_up_mm)
        """
        now = time.monotonic()

        # 如果距离上次更新超时 → 目标可能曾丢失, 重置滤波器
        if self._last_update_t is not None:
            if (now - self._last_update_t) > self.lost_timeout_s:
                self._filter_x.reset()
                self._filter_y.reset()

        self._last_update_t = now

        # One Euro Filter 平滑
        fx = self._filter_x(left_mm, now)
        fy = self._filter_y(up_mm, now)

        # 死区处理
        if abs(fx) < self.dead_zone_mm:
            fx = 0.0
        if abs(fy) < self.dead_zone_mm:
            fy = 0.0

        self.filtered_left_mm = fx
        self.filtered_up_mm = fy
        self.should_send = (abs(fx) > 1e-6) or (abs(fy) > 1e-6)

        return fx, fy
