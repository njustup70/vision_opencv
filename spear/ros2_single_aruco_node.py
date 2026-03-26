#!/usr/bin/env python3
from __future__ import annotations

import struct
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from aruco_core import create_detector, detect_target, draw_aruco_overlay
from pnp_core import compute_left_up_mm, draw_pose_overlay, solve_single_marker_pose

try:
    import serial  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    serial = None


# Fixed configuration (no ROS2 parameters by default)
IMAGE_TOPIC = "/hik_camera/image_raw"
CAMERA_INFO_TOPIC = "/hik_camera/camera_info"
ARUCO_DICTIONARY = "DICT_6X6_250"
TARGET_MARKER_ID = 100
MARKER_SIZE_M = 0.008
PNP_PREFER = "SOLVEPNP_IPPE_SQUARE"
PNP_FALLBACK = "SOLVEPNP_ITERATIVE"
PNP_REFINE_LM = True

SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200
SERIAL_TIMEOUT_SEC = 0.01
SERIAL_SOF = 0xFA
SERIAL_FRAME_ID = 0xB1

SHOW_WINDOW = True
WINDOW_NAME = "single_aruco_view"
AXIS_LENGTH_M = 0.02


class SingleArucoNode(Node):
    def __init__(self) -> None:
        super().__init__("single_aruco_node")
        self._bridge = CvBridge()

        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        self._warn_t_no_intrinsics = 0.0

        self._aruco_dict, self._aruco_params, self._aruco_detector = create_detector(ARUCO_DICTIONARY)
        self._serial = self._try_open_serial()

        self._sub_cam = self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self._on_camera_info, qos_profile_sensor_data)
        self._sub_img = self.create_subscription(Image, IMAGE_TOPIC, self._on_image, qos_profile_sensor_data)

        self.get_logger().info(
            f"Started. image={IMAGE_TOPIC} camera_info={CAMERA_INFO_TOPIC} marker_id={TARGET_MARKER_ID} size={MARKER_SIZE_M}m"
        )

    def _try_open_serial(self):
        if serial is None:
            self.get_logger().warning("pyserial not available, serial output disabled")
            return None
        try:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=SERIAL_TIMEOUT_SEC)
            self.get_logger().info(f"Serial opened: {SERIAL_PORT} @ {SERIAL_BAUD}")
            return ser
        except Exception as exc:
            self.get_logger().warning(f"Serial open failed ({SERIAL_PORT}): {exc}. Continue without serial output.")
            return None

    def _on_camera_info(self, msg: CameraInfo) -> None:
        k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        d = np.array(msg.d, dtype=np.float64).reshape(-1, 1)
        if not np.isfinite(k).all() or not np.isfinite(d).all():
            return
        self._camera_matrix = k
        self._dist_coeffs = d

    def _on_image(self, msg: Image) -> None:
        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        det = detect_target(gray, self._aruco_dict, self._aruco_params, self._aruco_detector, TARGET_MARKER_ID)
        vis = draw_aruco_overlay(bgr, det)

        if self._camera_matrix is None or self._dist_coeffs is None:
            now = time.time()
            if now - self._warn_t_no_intrinsics > 1.0:
                self.get_logger().warning("No camera intrinsics yet (waiting for camera_info)")
                self._warn_t_no_intrinsics = now
            self._show(vis)
            return

        if not det.found or det.corners is None:
            self._show(vis)
            return

        pnp = solve_single_marker_pose(
            marker_corners_xy=det.corners,
            marker_size_m=MARKER_SIZE_M,
            camera_matrix=self._camera_matrix,
            dist_coeffs=self._dist_coeffs,
            prefer=PNP_PREFER,
            fallback=PNP_FALLBACK,
            refine_lm=PNP_REFINE_LM,
        )
        if not pnp.ok or pnp.rvec is None or pnp.tvec is None:
            self._show(vis)
            return

        left_mm, up_mm = compute_left_up_mm(pnp.tvec)
        self._send_serial(left_mm, up_mm)

        vis = draw_pose_overlay(
            frame_bgr=vis,
            camera_matrix=self._camera_matrix,
            dist_coeffs=self._dist_coeffs,
            rvec=pnp.rvec,
            tvec=pnp.tvec,
            axis_length_m=AXIS_LENGTH_M,
            left_mm=left_mm,
            up_mm=up_mm,
            mean_reproj_px=pnp.mean_reproj_px,
            max_reproj_px=pnp.max_reproj_px,
        )

        self.get_logger().info(
            f"id={TARGET_MARKER_ID} left={left_mm:.1f}mm up={up_mm:.1f}mm "
            f"reproj=({0.0 if pnp.mean_reproj_px is None else pnp.mean_reproj_px:.3f},"
            f"{0.0 if pnp.max_reproj_px is None else pnp.max_reproj_px:.3f})"
        )
        self._show(vis)

    def _send_serial(self, left_mm: float, up_mm: float) -> None:
        if self._serial is None:
            return
        try:
            frame = struct.pack("<BBff", int(SERIAL_SOF), int(SERIAL_FRAME_ID), float(left_mm), float(up_mm))
            self._serial.write(frame)
        except Exception as exc:
            self.get_logger().warning(f"Serial write failed: {exc}")

    def _show(self, vis_bgr: np.ndarray) -> None:
        if not SHOW_WINDOW:
            return
        cv2.imshow(WINDOW_NAME, vis_bgr)
        cv2.waitKey(1)

    def destroy_node(self) -> None:
        try:
            if self._serial is not None:
                self._serial.close()
        except Exception:
            pass
        if SHOW_WINDOW:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = SingleArucoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
