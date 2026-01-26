"""
Board 位姿估计核心逻辑（与 ROS 解耦）

用途：
- 复用到 spear_tip_node / board_pose_node；
- 统一 ChArUco / ArUco Board 的 PnP、门控与置信度逻辑；
- 便于单元测试与维护。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from spear_vision.utils.camera_intrinsics import CameraIntrinsics
from spear_vision.utils.opencv_aruco import (
    MarkerDetection,
    filter_markers_for_board,
    get_board_ids,
    get_board_obj_points,
    get_charuco_chessboard_corners,
    interpolate_charuco_corners,
    refine_charuco_corners_subpix,
)
from spear_vision.utils.pnp import PnPResult, solve_pnp_best
from spear_vision.utils.tf_utils import Rt


@dataclass(frozen=True)
class BoardSpec:
    # 与棋盘相关的静态配置（尺寸/字典/ID 区间/门控阈值等）
    name: str
    frame: str
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    dictionary: str
    ids_start: int
    min_charuco_corners: int
    fallback_enable: bool
    fallback_min_markers: int
    use_refine_detected_markers: bool


@dataclass(frozen=True)
class GateSpec:
    # 重投影误差门控（gating）
    max_mean_reproj_px: float
    max_max_reproj_px: float
    min_border_px: int


@dataclass(frozen=True)
class PoseEstimate:
    # 单帧位姿估计结果
    ok: bool
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    mean_reproj_px: Optional[float] = None
    max_reproj_px: Optional[float] = None
    confidence: float = 0.0
    method: str = ""
    used: str = ""  # "charuco" | "aruco_board"
    num_points: int = 0


@dataclass(frozen=True)
class MethodNames:
    # 不同节点的“方法/失败原因”字符串命名略有差异，这里做可配置化
    charuco_near_border: str = "charuco_near_border"
    charuco_insufficient: str = "charuco_insufficient"
    aruco_none: str = "aruco_none"
    aruco_too_few: str = "aruco_too_few"
    aruco_no_matches: str = "aruco_no_matches"
    aruco_near_border: str = "aruco_near_border"
    pnp_failed: str = "pnp_failed"
    reproj_gate: str = "reproj_gate"
    pnp_ok: str = "pnp_ok"


def _is_near_border(points_xy: np.ndarray, width: int, height: int, border_px: int) -> bool:
    # 边缘门控：点过于靠边时更容易受畸变/遮挡/裁剪影响导致跳变
    if border_px <= 0:
        return False
    pts = np.array(points_xy, dtype=np.float64).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return bool(
        np.any(x < border_px)
        or np.any(x > (width - 1 - border_px))
        or np.any(y < border_px)
        or np.any(y > (height - 1 - border_px))
    )


class BoardPoseEstimator:
    def __init__(self, method_names: Optional[MethodNames] = None) -> None:
        self._method = method_names or MethodNames()

    def estimate(
        self,
        gray: np.ndarray,
        intrinsics: CameraIntrinsics,
        board,
        detection: MarkerDetection,
        spec: BoardSpec,
        gate: GateSpec,
        image_size: Optional[tuple[int, int]] = None,
        pnp_prefer: str = "SOLVEPNP_IPPE",
        pnp_fallback: str = "SOLVEPNP_ITERATIVE",
        pnp_refine_lm: bool = True,
        prior_rt: Optional[Rt] = None,
    ) -> PoseEstimate:
        # 估计单块板位姿（优先 ChArUco，失败 fallback ArUco Board）
        if image_size is None:
            h, w = gray.shape[:2]
        else:
            w, h = int(image_size[0]), int(image_size[1])

        min_charuco = int(spec.min_charuco_corners)

        # 0) 先按 board.ids 过滤 marker（避免画面里其它 marker 干扰插值/PnP）
        #    注意：即使上层已经 refineDetectedMarkers，这里再过滤一次也基本不增开销，
        #    但能显著降低“同字典其它 marker”对 ChArUco 插值的影响（更稳）。
        corners_b, ids_b = filter_markers_for_board(detection.corners, detection.ids, board)

        # 1) ChArUco 内角点（精度更高）
        num, charuco_corners, charuco_ids = interpolate_charuco_corners(gray, board, corners_b, ids_b)
        if charuco_corners is not None and charuco_ids is not None and int(num) >= 4:
            charuco_corners = refine_charuco_corners_subpix(gray, charuco_corners, win_size=3)
            img_pts = charuco_corners.reshape(-1, 2)
            if _is_near_border(img_pts, w, h, gate.min_border_px):
                return PoseEstimate(ok=False, used="charuco", method=self._method.charuco_near_border)

            chess = get_charuco_chessboard_corners(board)
            obj_pts = chess[np.array(charuco_ids, dtype=np.int32).reshape(-1)]
            res = solve_pnp_best(
                obj_pts,
                img_pts,
                intrinsics.camera_matrix,
                intrinsics.dist_coeffs,
                prefer=pnp_prefer,
                fallback=pnp_fallback,
                refine_lm=pnp_refine_lm,
                # 风险 D 规避：使用“上一帧位姿”作为连续性约束，抑制平面二义性导致的翻转/跳变
                prior_rvec=(prior_rt.rvec if prior_rt is not None else None),
                prior_tvec=(prior_rt.tvec if prior_rt is not None else None),
                enforce_positive_z=True,
            )
            est = self._gate_and_score(res, used="charuco", gate=gate, max_possible=(spec.squares_x - 1) * (spec.squares_y - 1))
            if est.ok:
                # 角点不足时降低置信度（但仍允许输出）
                if int(num) < min_charuco and min_charuco > 0:
                    corner_factor = float(max(0.0, min(1.0, float(num) / float(min_charuco))))
                    est = PoseEstimate(
                        ok=est.ok,
                        rvec=est.rvec,
                        tvec=est.tvec,
                        mean_reproj_px=est.mean_reproj_px,
                        max_reproj_px=est.max_reproj_px,
                        confidence=float(est.confidence) * corner_factor,
                        method=f"{est.method}_low_charuco_corners({int(num)}/{min_charuco})",
                        used=est.used,
                        num_points=est.num_points,
                    )
            return est

        # 2) fallback：ArUco Board（更容易成功）
        if not bool(spec.fallback_enable):
            return PoseEstimate(ok=False, used="charuco", method=self._method.charuco_insufficient)

        if ids_b is None or len(ids_b) == 0:
            return PoseEstimate(ok=False, used="aruco_board", method=self._method.aruco_none)
        if len(ids_b) < int(spec.fallback_min_markers):
            return PoseEstimate(ok=False, used="aruco_board", method=self._method.aruco_too_few)

        board_ids = get_board_ids(board)
        obj_points = get_board_obj_points(board)
        if board_ids.size == 0 or not obj_points:
            return PoseEstimate(ok=False, used="aruco_board", method=self._method.aruco_no_matches)

        id_to_index = {int(mid): i for i, mid in enumerate(board_ids.reshape(-1).tolist())}
        obj_pts = []
        img_pts = []
        for i, mid in enumerate(np.array(ids_b, dtype=np.int32).reshape(-1).tolist()):
            if int(mid) not in id_to_index:
                continue
            board_i = id_to_index[int(mid)]
            obj_corners = np.array(obj_points[board_i], dtype=np.float64).reshape(-1, 3)
            img_corners = np.array(corners_b[i], dtype=np.float64).reshape(-1, 2)
            obj_pts.append(obj_corners)
            img_pts.append(img_corners)
        if not obj_pts:
            return PoseEstimate(ok=False, used="aruco_board", method=self._method.aruco_no_matches)

        obj_pts_arr = np.vstack(obj_pts)
        img_pts_arr = np.vstack(img_pts)
        if _is_near_border(img_pts_arr, w, h, gate.min_border_px):
            return PoseEstimate(ok=False, used="aruco_board", method=self._method.aruco_near_border)

        res = solve_pnp_best(
            obj_pts_arr,
            img_pts_arr,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            prefer=pnp_prefer,
            fallback=pnp_fallback,
            refine_lm=pnp_refine_lm,
            prior_rvec=(prior_rt.rvec if prior_rt is not None else None),
            prior_tvec=(prior_rt.tvec if prior_rt is not None else None),
            enforce_positive_z=True,
        )
        return self._gate_and_score(res, used="aruco_board", gate=gate, max_possible=(spec.squares_x - 1) * (spec.squares_y - 1))

    def _gate_and_score(self, res: PnPResult, used: str, gate: GateSpec, max_possible: int) -> PoseEstimate:
        # 重投影误差门控
        if not res.ok or res.rvec is None or res.tvec is None:
            return PoseEstimate(ok=False, method=self._method.pnp_failed, used=used, num_points=res.num_points)

        mean_err = float(res.mean_reproj_px or 1e9)
        max_err = float(res.max_reproj_px or 1e9)
        # 防御性：若出现 NaN/Inf，必须当作失败处理，否则会发布 NaN TF/位姿（表现为坐标系“时有时无”）
        if not (np.isfinite(mean_err) and np.isfinite(max_err) and np.isfinite(res.rvec).all() and np.isfinite(res.tvec).all()):
            return PoseEstimate(ok=False, method=self._method.pnp_failed, used=used, num_points=res.num_points)
        if mean_err > gate.max_mean_reproj_px or max_err > gate.max_max_reproj_px:
            return PoseEstimate(
                ok=False,
                rvec=res.rvec,
                tvec=res.tvec,
                mean_reproj_px=mean_err,
                max_reproj_px=max_err,
                method=self._method.reproj_gate,
                used=used,
                num_points=res.num_points,
            )

        points_ratio = min(1.0, float(res.num_points) / float(max(max_possible, 1)))
        base = 0.7 if used == "charuco" else 0.4
        conf = base + 0.3 * points_ratio
        conf *= math.exp(-mean_err / max(gate.max_mean_reproj_px, 1e-6))
        conf = float(max(0.0, min(1.0, conf)))

        return PoseEstimate(
            ok=True,
            rvec=res.rvec,
            tvec=res.tvec,
            mean_reproj_px=mean_err,
            max_reproj_px=max_err,
            confidence=conf,
            method=self._method.pnp_ok,
            used=used,
            num_points=res.num_points,
        )
