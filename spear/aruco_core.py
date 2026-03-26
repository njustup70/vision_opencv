from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# 本文件只做 ArUco 检测相关工作：
# 1) 创建 detector
# 2) 在一帧图像中找目标 ID
# 3) 提供可视化叠加（画框/ID/中心点）


def _require_aruco() -> None:
    # 某些 OpenCV 构建没有 aruco 模块，这里提前报错更容易定位环境问题。
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is required. Install OpenCV with aruco support.")


def resolve_dictionary(name: str):
    # 允许传 "6X6_250" 或 "DICT_6X6_250"，统一解析为 OpenCV 字典对象。
    _require_aruco()
    if not name:
        raise ValueError("dictionary name is empty")
    if not name.startswith("DICT_"):
        name = "DICT_" + name
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown aruco dictionary: {name}")
    dict_id = getattr(cv2.aruco, name)
    getter = getattr(cv2.aruco, "getPredefinedDictionary", None)
    if getter is not None:
        return getter(dict_id)
    legacy_getter = getattr(cv2.aruco, "Dictionary_get", None)
    if legacy_getter is None:
        raise RuntimeError("No aruco dictionary getter API found")
    return legacy_getter(dict_id)


def create_detector(dictionary_name: str):
    # 返回 (字典对象, 参数对象, detector对象/None)，外部循环复用，避免每帧重复创建。
    _require_aruco()
    aruco_dict = resolve_dictionary(dictionary_name)
    creator = getattr(cv2.aruco, "DetectorParameters_create", None)
    if creator is not None:
        params = creator()
    else:
        params = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    else:
        detector = None
    return aruco_dict, params, detector


@dataclass(frozen=True)
class TargetDetection:
    # found=False: 本帧没找到目标ID
    # found=True: corners 为目标 marker 四个角点(顺序与 OpenCV 检测输出一致)
    found: bool
    marker_id: Optional[int] = None
    corners: Optional[np.ndarray] = None  # shape: (4,2)
    center_xy: Optional[tuple[int, int]] = None
    all_corners: Optional[list[np.ndarray]] = None
    all_ids: Optional[np.ndarray] = None


def detect_target(gray: np.ndarray, aruco_dict, params, detector, target_marker_id: int) -> TargetDetection:
    # 输入灰度图，输出“目标ID的单个 marker 检测结果”。
    # 注意：本函数会保留 all_ids/all_corners，便于上层做调试可视化。
    _require_aruco()
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        legacy_detect = getattr(cv2.aruco, "detectMarkers", None)
        if legacy_detect is None:
            raise RuntimeError("No ArUco detection API found")
        corners, ids, _ = legacy_detect(gray, aruco_dict, parameters=params)

    if ids is None or len(ids) == 0:
        return TargetDetection(found=False, all_corners=corners, all_ids=ids)

    # OpenCV ids 可能是 (N,1)，这里拉平成 (N,) 便于查找目标 ID。
    ids_flat = np.array(ids, dtype=np.int32).reshape(-1)
    target_idx = None
    for i, mid in enumerate(ids_flat.tolist()):
        if int(mid) == int(target_marker_id):
            target_idx = i
            break

    if target_idx is None:
        return TargetDetection(found=False, all_corners=corners, all_ids=ids)

    # 单个 marker 角点标准形状是 (4,2)：左上、右上、右下、左下。
    c = np.array(corners[target_idx], dtype=np.float64).reshape(4, 2)
    center = c.mean(axis=0)
    center_xy = (int(round(float(center[0]))), int(round(float(center[1]))))
    return TargetDetection(
        found=True,
        marker_id=int(target_marker_id),
        corners=c,
        center_xy=center_xy,
        all_corners=corners,
        all_ids=ids,
    )


def draw_aruco_overlay(frame_bgr: np.ndarray, detection: TargetDetection) -> np.ndarray:
    # 先画全部检测到的 marker，再高亮目标 marker。
    out = frame_bgr
    if detection.all_ids is not None and detection.all_corners is not None and len(detection.all_ids) > 0:
        cv2.aruco.drawDetectedMarkers(out, detection.all_corners, detection.all_ids)

    if detection.found and detection.corners is not None and detection.marker_id is not None:
        pts = detection.corners.astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        if detection.center_xy is not None:
            cv2.circle(out, detection.center_xy, 4, (0, 0, 255), -1)
            cv2.putText(
                out,
                f"id={detection.marker_id}",
                (detection.center_xy[0] + 6, detection.center_xy[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return out
