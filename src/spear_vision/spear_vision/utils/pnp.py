"""
PnP 求解与重投影误差计算（spear_vision）

核心思想：
- 由于 IPPE/EPnP 等方法可能返回多解（尤其是近平面目标），这里使用 solvePnPGeneric，
  对每个候选解计算重投影误差，选取“平均重投影误差最小”的解；
- 如果 prefer 方法失败，再 fallback 到更通用的 ITERATIVE；
- 可选地使用 solvePnPRefineLM 做一次 LM 细化，进一步压低重投影误差。

单位说明：
- object_points 使用米（m）时，输出 tvec 也是米（与 board 物理尺寸一致）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from spear_vision.utils.tf_utils import matrix_to_quaternion, rodrigues_to_matrix, rotation_angle_deg


def method_to_cv2_flag(method: str) -> int:
    # 支持 "SOLVEPNP_IPPE" 或 "IPPE" 两种写法
    if not method:
        return cv2.SOLVEPNP_ITERATIVE
    if not method.startswith("SOLVEPNP_"):
        method = "SOLVEPNP_" + method
    if not hasattr(cv2, method):
        raise ValueError(f"Unknown solvePnP method '{method}'")
    return int(getattr(cv2, method))


@dataclass(frozen=True)
class PnPResult:
    ok: bool
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    mean_reproj_px: Optional[float] = None
    max_reproj_px: Optional[float] = None
    num_points: int = 0


def compute_reprojection_errors_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    # 将 3D 点投影到图像上，再与检测到的 2D 点比较，得到像素级误差（px）
    obj = np.array(object_points, dtype=np.float64).reshape(-1, 1, 3)
    img = np.array(image_points, dtype=np.float64).reshape(-1, 1, 2)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
    err = proj.reshape(-1, 2) - img.reshape(-1, 2)
    return np.linalg.norm(err, axis=1)


def solve_pnp_best(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    prefer: str,
    fallback: str,
    refine_lm: bool = True,
    prior_rvec: Optional[np.ndarray] = None,
    prior_tvec: Optional[np.ndarray] = None,
    continuity_reproj_ratio: float = 1.05,
    continuity_translation_m: float = 0.005,
    continuity_rotation_deg: float = 5.0,
    enforce_positive_z: bool = False,
) -> PnPResult:
    # 统一输入形状：obj=(N,3), img=(N,2)
    obj = np.array(object_points, dtype=np.float64).reshape(-1, 3)
    img = np.array(image_points, dtype=np.float64).reshape(-1, 2)
    if obj.shape[0] < 4:
        # 一般 PnP 至少需要 4 个非共线点（更少时稳定性很差）
        return PnPResult(ok=False, num_points=int(obj.shape[0]))

    def _candidates(flags: int) -> list[tuple[float, float, np.ndarray, np.ndarray]]:
        # solvePnPGeneric：允许返回多组 (rvec,tvec) 候选解
        try:
            n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, camera_matrix, dist_coeffs, flags=int(flags))
        except Exception:
            return []
        if int(n) <= 0 or not rvecs or not tvecs:
            return []

        out: list[tuple[float, float, np.ndarray, np.ndarray]] = []
        for i in range(len(rvecs)):
            rvec = np.array(rvecs[i], dtype=np.float64).reshape(3, 1)
            tvec = np.array(tvecs[i], dtype=np.float64).reshape(3, 1)
            # 防御性：某些 OpenCV/输入组合下可能返回 NaN/Inf（会导致 TF/可视化“时有时无”）
            if not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
                continue
            errs = compute_reprojection_errors_px(obj, img, camera_matrix, dist_coeffs, rvec, tvec)
            if errs.size and not np.isfinite(errs).all():
                continue
            mean_err = float(np.mean(errs)) if errs.size else float("inf")
            max_err = float(np.max(errs)) if errs.size else float("inf")
            if not (np.isfinite(mean_err) and np.isfinite(max_err)):
                continue
            out.append((mean_err, max_err, rvec, tvec))
        return out

    prefer_flag = method_to_cv2_flag(prefer)
    fallback_flag = method_to_cv2_flag(fallback)

    cands = _candidates(prefer_flag)
    if not cands:
        cands = _candidates(fallback_flag)
    if not cands:
        return PnPResult(ok=False, num_points=int(obj.shape[0]))

    # 按平均重投影误差排序（与旧实现保持一致：默认取 mean_err 最小的解）
    cands.sort(key=lambda x: x[0])

    # 物理约束（可选）：目标应在相机前方（tvec.z > 0）
    if bool(enforce_positive_z):
        pos = [c for c in cands if float(c[3][2]) > 0.0]
        if pos:
            cands = pos

    best_mean, best_max, best_rvec, best_tvec = cands[0]

    # 风险 D 规避：平面 PnP 的二义性（例如 IPPE）在某些姿态下会出现两解误差非常接近，
    # 仅按 reprojection error 选解可能偶发“翻转/大跳”。
    # 这里在“误差接近”的情况下，引入连续性约束：优先选择与上一帧更接近的候选解。
    if prior_rvec is not None and prior_tvec is not None and len(cands) >= 2:
        try:
            best_err = float(best_mean)
            ratio = float(max(1.0, continuity_reproj_ratio))
            near = [c for c in cands if float(c[0]) <= best_err * ratio]
            if len(near) >= 2:
                q_prior = matrix_to_quaternion(rodrigues_to_matrix(prior_rvec))
                t_prior = np.array(prior_tvec, dtype=np.float64).reshape(3, 1)
                t_scale = float(max(1e-9, continuity_translation_m))
                r_scale = float(max(1e-9, continuity_rotation_deg))

                best = None
                best_score = float("inf")
                for mean_err, max_err, rvec, tvec in near:
                    dt = float(np.linalg.norm(np.array(tvec, dtype=np.float64).reshape(3, 1) - t_prior))
                    q = matrix_to_quaternion(rodrigues_to_matrix(rvec))
                    dr = float(rotation_angle_deg(q_prior, q))
                    score = (dt / t_scale) + (dr / r_scale)
                    if score < best_score - 1e-12 or (abs(score - best_score) <= 1e-12 and float(mean_err) < float(best_mean)):
                        best_score = score
                        best = (float(mean_err), float(max_err), rvec, tvec)
                if best is not None:
                    best_mean, best_max, best_rvec, best_tvec = best
        except Exception:
            pass

    rvec, tvec = best_rvec, best_tvec
    if refine_lm and hasattr(cv2, "solvePnPRefineLM"):
        # LM 细化：通常能进一步降低重投影误差（对亚毫米级稳定性很关键）
        try:
            r2, t2 = cv2.solvePnPRefineLM(obj, img, camera_matrix, dist_coeffs, rvec, tvec)
            r2 = np.array(r2, dtype=np.float64).reshape(3, 1)
            t2 = np.array(t2, dtype=np.float64).reshape(3, 1)
            if np.isfinite(r2).all() and np.isfinite(t2).all():
                rvec, tvec = r2, t2
        except Exception:
            pass

    if not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
        return PnPResult(ok=False, num_points=int(obj.shape[0]))

    errs = compute_reprojection_errors_px(obj, img, camera_matrix, dist_coeffs, rvec, tvec)
    if errs.size and not np.isfinite(errs).all():
        return PnPResult(ok=False, num_points=int(obj.shape[0]))
    mean_err = float(np.mean(errs)) if errs.size else None
    max_err = float(np.max(errs)) if errs.size else None
    if mean_err is not None and not np.isfinite(mean_err):
        return PnPResult(ok=False, num_points=int(obj.shape[0]))
    if max_err is not None and not np.isfinite(max_err):
        return PnPResult(ok=False, num_points=int(obj.shape[0]))
    return PnPResult(
        ok=True,
        rvec=rvec,
        tvec=tvec,
        mean_reproj_px=mean_err,
        max_reproj_px=max_err,
        num_points=int(obj.shape[0]),
    )
