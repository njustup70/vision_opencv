"""
运行时环境检查（spear_vision）

目的：
- 风险 B：ROS2 的 cv_bridge 常链接系统 OpenCV；如果同时 pip 装了另一个 OpenCV wheel，
  可能出现 ABI/版本冲突（运行时崩溃/符号找不到/行为异常）。
- 风险 A：OpenCV 4.7+ aruco Python API 发生过破坏性变化，这里打印一下“当前走的是新/旧 API”，
  便于现场定位问题。
"""

from __future__ import annotations

from typing import Any


def warn_opencv_environment(logger: Any) -> None:
    # logger: rclpy logger 或任何提供 info/warn 方法的对象
    try:
        import cv2

        cv2_ver = getattr(cv2, "__version__", "unknown")
        cv2_path = getattr(cv2, "__file__", "")
        has_aruco = hasattr(cv2, "aruco")
        aruco_mode = "none"
        if has_aruco:
            aruco_mode = "ArucoDetector" if hasattr(cv2.aruco, "ArucoDetector") else "legacy"

        if hasattr(logger, "info"):
            logger.info(f"OpenCV {cv2_ver} ({cv2_path}) aruco_api={aruco_mode}")
    except Exception:
        return

    # cv_bridge 与 cv2 的来源不一致时给出强提醒
    try:
        import cv2
        import cv_bridge

        cv2_path = getattr(cv2, "__file__", "") or ""
        cvb_path = getattr(cv_bridge, "__file__", "") or ""
        if not cv2_path or not cvb_path:
            return

        cv2_is_pip = ("site-packages" in cv2_path) and ("dist-packages" not in cv2_path)
        cv2_is_apt = ("dist-packages" in cv2_path) or cv2_path.startswith("/usr/")
        cvb_is_pip = ("site-packages" in cvb_path) and ("dist-packages" not in cvb_path)
        cvb_is_apt = ("dist-packages" in cvb_path) or cvb_path.startswith("/usr/")

        if cv2_is_pip and cvb_is_apt and hasattr(logger, "warn"):
            logger.warn(
                "Detected pip OpenCV (cv2) with system/apt cv_bridge. This can cause ABI/version mismatch crashes. "
                "Recommendation: use system OpenCV only (apt python3-opencv) OR rebuild cv_bridge against the same OpenCV."
            )
        if cv2_is_apt and cvb_is_pip and hasattr(logger, "warn"):
            logger.warn(
                "Detected system/apt OpenCV (cv2) with pip cv_bridge. This can cause ABI/version mismatch crashes. "
                "Recommendation: keep cv_bridge and OpenCV from the same source/toolchain."
            )
    except Exception:
        return

