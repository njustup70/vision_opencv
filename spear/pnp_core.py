from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# 本文件只做单 marker 的 PnP：
# 1) 构造 marker 的 3D 角点
# 2) 根据 2D 像素角点解算 rvec/tvec
# 3) 计算重投影误差与 left/up 偏移


@dataclass(frozen=True)
class PnPResult:
    ok: bool
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    mean_reproj_px: Optional[float] = None
    max_reproj_px: Optional[float] = None


def build_single_marker_object_points(marker_size_m: float) -> np.ndarray:
    # 建立 marker 局部坐标系下的四个角点（Z=0 平面，原点在 marker 中心）。
    half = float(marker_size_m) / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def compute_reprojection_errors_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    # 用当前 rvec/tvec 把 3D 点重新投影回图像，计算每个点的像素误差。
    obj = np.array(object_points, dtype=np.float64).reshape(-1, 1, 3)
    img = np.array(image_points, dtype=np.float64).reshape(-1, 1, 2)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
    err = proj.reshape(-1, 2) - img.reshape(-1, 2)
    return np.linalg.norm(err, axis=1)


def _method_to_flag(method: str) -> int:
    # 支持 "IPPE_SQUARE" 和 "SOLVEPNP_IPPE_SQUARE" 两种写法。
    if not method:
        return int(cv2.SOLVEPNP_ITERATIVE)
    if not method.startswith("SOLVEPNP_"):
        method = "SOLVEPNP_" + method
    if not hasattr(cv2, method):
        raise ValueError(f"Unknown solvePnP method: {method}")
    return int(getattr(cv2, method))


def _solve_candidates(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    flag: int,
) -> list[tuple[float, float, np.ndarray, np.ndarray]]:
    # solvePnPGeneric 可能返回多组解（平面目标常见），这里全部取出并打分。
    try:
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=int(flag),
        )
    except Exception:
        return []

    if int(n) <= 0 or not rvecs or not tvecs:
        return []

    # 候选元素: (mean_err, max_err, rvec, tvec)
    out: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    for i in range(len(rvecs)):
        rvec = np.array(rvecs[i], dtype=np.float64).reshape(3, 1)
        tvec = np.array(tvecs[i], dtype=np.float64).reshape(3, 1)
        if not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
            continue
        errs = compute_reprojection_errors_px(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec)
        if errs.size == 0 or not np.isfinite(errs).all():
            continue
        mean_err = float(np.mean(errs))
        max_err = float(np.max(errs))
        out.append((mean_err, max_err, rvec, tvec))
    return out


def solve_single_marker_pose(
    marker_corners_xy: np.ndarray,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    prefer: str = "SOLVEPNP_IPPE_SQUARE",
    fallback: str = "SOLVEPNP_ITERATIVE",
    refine_lm: bool = True,
) -> PnPResult:
    # 主入口：输入单个 marker 的 4 个像素角点，输出最优位姿。
    img = np.array(marker_corners_xy, dtype=np.float64).reshape(-1, 2)
    if img.shape[0] != 4:
        return PnPResult(ok=False)

    obj = build_single_marker_object_points(marker_size_m).reshape(-1, 3)
    prefer_flag = _method_to_flag(prefer)
    fallback_flag = _method_to_flag(fallback)

    # 先用 prefer 方法求解，失败再 fallback。
    cands = _solve_candidates(obj, img, camera_matrix, dist_coeffs, prefer_flag)
    if not cands:
        cands = _solve_candidates(obj, img, camera_matrix, dist_coeffs, fallback_flag)
    if not cands:
        return PnPResult(ok=False)

    # 选择平均重投影误差最小的候选解。
    cands.sort(key=lambda x: x[0])
    mean_err, max_err, rvec, tvec = cands[0]

    if refine_lm and hasattr(cv2, "solvePnPRefineLM"):
        # 用 LM 做一次局部细化，通常能再降低重投影误差。
        try:
            r2, t2 = cv2.solvePnPRefineLM(obj, img, camera_matrix, dist_coeffs, rvec, tvec)
            r2 = np.array(r2, dtype=np.float64).reshape(3, 1)
            t2 = np.array(t2, dtype=np.float64).reshape(3, 1)
            if np.isfinite(r2).all() and np.isfinite(t2).all():
                rvec, tvec = r2, t2
                errs = compute_reprojection_errors_px(obj, img, camera_matrix, dist_coeffs, rvec, tvec)
                if errs.size > 0 and np.isfinite(errs).all():
                    mean_err = float(np.mean(errs))
                    max_err = float(np.max(errs))
        except Exception:
            pass

    if not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
        return PnPResult(ok=False)

    return PnPResult(
        ok=True,
        rvec=rvec,
        tvec=tvec,
        mean_reproj_px=float(mean_err),
        max_reproj_px=float(max_err),
    )


def compute_left_up_mm(tvec: np.ndarray) -> tuple[float, float]:
    # 与现有串口定义保持一致：
    # left_mm = -x*1000, up_mm = -y*1000
    t = np.array(tvec, dtype=np.float64).reshape(3)
    x_m = float(t[0])
    y_m = float(t[1])
    left_mm = -x_m * 1000.0
    up_mm = -y_m * 1000.0
    return left_mm, up_mm


def draw_pose_overlay(
    frame_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    axis_length_m: float,
    left_mm: float,
    up_mm: float,
    mean_reproj_px: Optional[float],
    max_reproj_px: Optional[float],
) -> np.ndarray:
    # 画坐标轴和关键数值，便于现场调试（偏移量 + reproj 质量）。
    out = frame_bgr
    cv2.drawFrameAxes(out, camera_matrix, dist_coeffs, rvec, tvec, float(axis_length_m), 2)

    line1 = f"left={left_mm:.1f}mm up={up_mm:.1f}mm"
    if mean_reproj_px is None or max_reproj_px is None:
        line2 = "reproj=n/a"
    else:
        line2 = f"reproj mean={mean_reproj_px:.3f}px max={max_reproj_px:.3f}px"

    cv2.putText(out, line1, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(out, line2, (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return out
