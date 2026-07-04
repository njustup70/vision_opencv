#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import statistics
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image


@dataclass
class RoundResult:
    index: int
    rms: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    samples: int
    image_size: tuple[int, int]
    yaml_path: str


def _aruco_dict(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("This OpenCV build has no cv2.aruco module. Install opencv-contrib support.")
    if not name.startswith("DICT_"):
        name = "DICT_" + name
    dict_id = getattr(cv2.aruco, name)
    # OpenCV >= 4.7: getPredefinedDictionary; < 4.7: Dictionary_get
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)


def _set_board_ids(board, ids_start: int) -> None:
    if ids_start == 0:
        return
    ids = None
    if hasattr(board, "getIds"):
        ids = board.getIds()
    elif hasattr(board, "ids"):
        ids = board.ids
    if ids is None:
        raise RuntimeError("This OpenCV CharucoBoard API does not expose marker ids.")

    desired = (np.arange(len(np.asarray(ids).reshape(-1)), dtype=np.int32) + ids_start).reshape(-1, 1)
    if hasattr(board, "setIds"):
        board.setIds(desired)
    elif hasattr(board, "ids"):
        board.ids = desired
    else:
        raise RuntimeError("This OpenCV CharucoBoard API does not allow marker id assignment.")


def _create_charuco_board(cfg: dict):
    dictionary = _aruco_dict(str(cfg["dictionary"]))
    squares_x = int(cfg["squares_x"])
    squares_y = int(cfg["squares_y"])
    square_length_m = float(cfg["square_length_m"])
    marker_length_m = float(cfg["marker_length_m"])
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, square_length_m, marker_length_m, dictionary
        )
    else:
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length_m, marker_length_m, dictionary
        )
    _set_board_ids(board, int(cfg.get("ids_start", 0)))
    return board, dictionary


def _detector_params():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    return cv2.aruco.DetectorParameters()


def _detect_markers(gray, dictionary, params):
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


def _image_msg_to_gray(msg: Image) -> np.ndarray:
    encoding = msg.encoding.lower()
    width = int(msg.width)
    height = int(msg.height)
    step = int(msg.step)
    data = memoryview(msg.data)

    if encoding in ("mono8", "8uc1") or encoding.startswith("bayer_"):
        row = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
        return np.ascontiguousarray(row[:, :width])

    if encoding in ("mono16", "16uc1"):
        row_elems = step // 2
        row = np.frombuffer(data, dtype=np.uint16).reshape(height, row_elems)
        mono16 = np.ascontiguousarray(row[:, :width])
        return cv2.convertScaleAbs(mono16, alpha=255.0 / max(float(mono16.max()), 1.0))

    channel_map = {
        "bgr8": (3, None),
        "rgb8": (3, cv2.COLOR_RGB2GRAY),
        "bgra8": (4, cv2.COLOR_BGRA2GRAY),
        "rgba8": (4, cv2.COLOR_RGBA2GRAY),
    }
    if encoding in channel_map:
        channels, code = channel_map[encoding]
        row = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
        img = np.ascontiguousarray(row[:, : width * channels].reshape(height, width, channels))
        if code is None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(img, code)

    raise ValueError(f"Unsupported image encoding for calibration: {msg.encoding}")


def _cv2_to_image_msg(img: np.ndarray, encoding: str, header) -> Image:
    msg = Image()
    msg.header = header
    msg.height = int(img.shape[0])
    msg.width = int(img.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    channels = 1 if img.ndim == 2 else int(img.shape[2])
    msg.step = int(msg.width * channels * img.dtype.itemsize)
    msg.data = np.ascontiguousarray(img).tobytes()
    return msg


def _refine_markers(gray, board, corners, ids, rejected):
    if ids is None or len(ids) == 0:
        return corners, ids, rejected
    try:
        refined = cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejected)
    except Exception:
        return corners, ids, rejected
    if len(refined) >= 3:
        return refined[0], refined[1], refined[2]
    return corners, ids, rejected


def _interpolate_charuco(gray, board, corners, ids):
    if ids is None or len(ids) == 0:
        return 0, None, None
    return cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)


def _calibrate_charuco(corners_list, ids_list, board, image_size):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 120, 1e-7)
    if hasattr(cv2.aruco, "calibrateCameraCharucoExtended"):
        (
            rms,
            camera_matrix,
            dist_coeffs,
            _rvecs,
            _tvecs,
            _std_int,
            _std_ext,
            _per_view,
        ) = cv2.aruco.calibrateCameraCharucoExtended(
            corners_list,
            ids_list,
            board,
            image_size,
            None,
            None,
            flags=0,
            criteria=criteria,
        )
        return float(rms), camera_matrix, dist_coeffs
    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.aruco.calibrateCameraCharuco(
        corners_list,
        ids_list,
        board,
        image_size,
        None,
        None,
        flags=0,
        criteria=criteria,
    )
    return float(rms), camera_matrix, dist_coeffs


def _camera_yaml(camera_name, image_size, camera_matrix, dist_coeffs, rms, samples, extra=None):
    w, h = image_size
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    d = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    out = {
        "image_width": int(w),
        "image_height": int(h),
        "camera_name": str(camera_name),
        "camera_matrix": {"rows": 3, "cols": 3, "data": k.reshape(-1).tolist()},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": int(d.size), "data": d.tolist()},
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": [
                float(k[0, 0]),
                0.0,
                float(k[0, 2]),
                0.0,
                0.0,
                float(k[1, 1]),
                float(k[1, 2]),
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
        },
        "charuco_calibration": {
            "rms_reprojection_px": float(rms),
            "num_samples": int(samples),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    if extra:
        out["charuco_calibration"].update(extra)
    return out


def _save_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


class OptimizedCharucoCalib(Node):
    def __init__(self, cfg: dict):
        super().__init__("optimized_charuco_calibration")
        self.cfg = cfg
        self.board, self.dictionary = _create_charuco_board(cfg["board"])
        self.detector_params = _detector_params()

        cal = cfg["calibration"]
        self.rounds = int(cal["rounds"])
        self.samples_per_round = int(cal["samples_per_round"])
        self.sample_stride = int(cal["sample_stride"])
        self.process_hz = float(cal.get("process_hz", 5.0))
        self.min_charuco_corners = int(cal["min_charuco_corners"])
        self.max_samples_per_round = int(cal["max_samples_per_round"])
        self.settle_seconds = float(cal["settle_seconds_between_rounds"])
        self.show_window = bool(cal["show_opencv_window"])
        if self.show_window and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            self.get_logger().warn("No desktop DISPLAY/WAYLAND_DISPLAY found; OpenCV preview window disabled.")
            self.show_window = False
        self.publish_debug_image = bool(cal.get("publish_debug_image", True))
        self.output_dir = os.path.expanduser(str(cal["output_dir"]))
        self.camera_name = str(cal["camera_name"])
        self.save_average_yaml = os.path.expanduser(str(cal["save_average_yaml"]))
        self.save_best_yaml = os.path.expanduser(str(cal["save_best_yaml"]))
        self.save_joint_yaml = os.path.expanduser(
            str(cal.get("save_joint_yaml", os.path.join(self.output_dir, "camera_joint.yaml")))
        )

        self.frame_index = 0
        self.current_round = 1
        self.round_corners: list[np.ndarray] = []
        self.round_ids: list[np.ndarray] = []
        self.all_corners: list[np.ndarray] = []
        self.all_ids: list[np.ndarray] = []
        self.image_size: tuple[int, int] | None = None
        self.results: list[RoundResult] = []
        self.done = False
        self.round_started_at = time.monotonic()
        self.last_header = None

        image_topic = str(cfg["topics"]["image"])
        self.image_topic = image_topic
        self.image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.image_sub = None
        self.waiting_for_image = False
        self.image_request_started_at = 0.0
        self.process_timer = self.create_timer(1.0 / max(self.process_hz, 0.1), self.request_image)
        self.debug_pub = (
            self.create_publisher(Image, "~/debug_image", qos_profile_sensor_data)
            if self.publish_debug_image
            else None
        )
        self.get_logger().info(f"Subscribing image: {image_topic}")
        self.get_logger().info(
            f"Plan: {self.rounds} rounds x {self.samples_per_round} samples, "
            f"stride={self.sample_stride}, process_hz={self.process_hz:.1f}, "
            f"min_charuco_corners={self.min_charuco_corners}"
        )

    def request_image(self) -> None:
        if self.done:
            return
        if time.monotonic() - self.round_started_at < self.settle_seconds:
            return
        if len(self.round_corners) >= self.max_samples_per_round:
            return
        if self.waiting_for_image:
            if time.monotonic() - self.image_request_started_at > 2.0:
                self.destroy_image_subscription()
            else:
                return

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.on_image_once, self.image_qos
        )
        self.waiting_for_image = True
        self.image_request_started_at = time.monotonic()

    def destroy_image_subscription(self) -> None:
        if self.image_sub is not None:
            try:
                self.destroy_subscription(self.image_sub)
            except Exception:
                pass
        self.image_sub = None
        self.waiting_for_image = False

    def on_image_once(self, msg: Image) -> None:
        self.destroy_image_subscription()
        self.frame_index += 1
        self.process_image(msg)

    def process_image(self, msg: Image) -> None:
        if self.done:
            return
        if time.monotonic() - self.round_started_at < self.settle_seconds:
            return
        if len(self.round_corners) >= self.max_samples_per_round:
            return

        try:
            gray = _image_msg_to_gray(msg)
        except Exception as exc:
            self.get_logger().error(f"Failed to convert image message: {exc}")
            return
        self.last_header = msg.header
        h, w = gray.shape[:2]
        if self.image_size is None:
            self.image_size = (w, h)
        elif self.image_size != (w, h):
            self.get_logger().error(
                f"Image size changed from {self.image_size} to {(w, h)}. "
                "Stop the node and restart calibration with a fixed camera resolution."
            )
            self.done = True
            threading.Thread(target=self._shutdown_soon, daemon=True).start()
            return

        corners, ids, rejected = _detect_markers(gray, self.dictionary, self.detector_params)
        corners, ids, rejected = _refine_markers(gray, self.board, corners, ids, rejected)
        num, charuco_corners, charuco_ids = _interpolate_charuco(gray, self.board, corners, ids)
        num = int(num)
        if charuco_corners is None or charuco_ids is None or num < self.min_charuco_corners:
            self._show(gray, corners, ids, None, None, num)
            return

        cv2.cornerSubPix(
            gray,
            charuco_corners,
            (3, 3),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        )
        self.round_corners.append(charuco_corners)
        self.round_ids.append(charuco_ids)
        self.get_logger().info(
            f"round {self.current_round}/{self.rounds}: "
            f"sample {len(self.round_corners)}/{self.samples_per_round}, charuco={num}"
        )
        self._show(gray, corners, ids, charuco_corners, charuco_ids, num)

        if len(self.round_corners) >= self.samples_per_round:
            self._finish_round()

    def _make_debug_image(self, gray, marker_corners, marker_ids, charuco_corners, charuco_ids, num):
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if marker_ids is not None and len(marker_ids) > 0:
            try:
                cv2.aruco.drawDetectedMarkers(bgr, marker_corners, marker_ids)
            except Exception:
                pass
        if charuco_corners is not None and charuco_ids is not None:
            try:
                cv2.aruco.drawDetectedCornersCharuco(bgr, charuco_corners, charuco_ids, (0, 0, 255))
            except Exception:
                pass
        text = (
            f"round={self.current_round}/{self.rounds} "
            f"samples={len(self.round_corners)}/{self.samples_per_round} "
            f"charuco={num}/{self.min_charuco_corners}"
        )
        cv2.putText(bgr, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return bgr

    def _show(self, gray, marker_corners, marker_ids, charuco_corners, charuco_ids, num):
        bgr = self._make_debug_image(gray, marker_corners, marker_ids, charuco_corners, charuco_ids, num)
        if self.debug_pub is not None:
            try:
                self.debug_pub.publish(_cv2_to_image_msg(bgr, "bgr8", self.last_header))
            except Exception:
                pass
        if not self.show_window:
            return
        cv2.imshow("optimized_charuco_calibration", bgr)
        cv2.waitKey(1)

    def _finish_round(self):
        if self.image_size is None:
            return
        self.get_logger().info(f"Calibrating round {self.current_round}...")
        try:
            rms, k, d = _calibrate_charuco(
                self.round_corners, self.round_ids, self.board, self.image_size
            )
        except Exception as exc:
            self.get_logger().error(
                f"Round {self.current_round} calibration failed: {exc}. "
                "This round will be collected again; move the board through more varied poses."
            )
            self.round_corners.clear()
            self.round_ids.clear()
            self.round_started_at = time.monotonic()
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        yaml_path = os.path.join(
            self.output_dir, f"camera_round_{self.current_round:02d}_{ts}.yaml"
        )
        _save_yaml(
            yaml_path,
            _camera_yaml(
                self.camera_name,
                self.image_size,
                k,
                d,
                rms,
                len(self.round_corners),
                extra={"round": int(self.current_round)},
            ),
        )
        self.results.append(
            RoundResult(
                self.current_round,
                rms,
                np.asarray(k, dtype=np.float64),
                np.asarray(d, dtype=np.float64).reshape(-1),
                len(self.round_corners),
                self.image_size,
                yaml_path,
            )
        )
        self.all_corners.extend(self.round_corners)
        self.all_ids.extend(self.round_ids)
        self.get_logger().info(
            f"Saved round {self.current_round}: rms={rms:.6f}px path={yaml_path}"
        )

        if self.current_round >= self.rounds:
            self._finish_all()
            return

        self.current_round += 1
        self.round_corners.clear()
        self.round_ids.clear()
        self.round_started_at = time.monotonic()
        self.get_logger().info(
            f"Move the board to a new pose. Next round starts after {self.settle_seconds:.1f}s."
        )

    def _finish_all(self):
        best = min(self.results, key=lambda r: r.rms)
        joint_yaml = ""
        try:
            joint_rms, joint_k, joint_d = _calibrate_charuco(
                self.all_corners, self.all_ids, self.board, best.image_size
            )
            _save_yaml(
                self.save_joint_yaml,
                _camera_yaml(
                    self.camera_name,
                    best.image_size,
                    joint_k,
                    joint_d,
                    joint_rms,
                    len(self.all_corners),
                    extra={
                        "selected": "joint_all_samples",
                        "rounds": len(self.results),
                        "round_yaml_paths": [r.yaml_path for r in self.results],
                    },
                ),
            )
            joint_yaml = self.save_joint_yaml
            self.get_logger().info(f"Saved joint calibration: {self.save_joint_yaml}")
        except Exception as exc:
            self.get_logger().error(f"Joint calibration failed: {exc}")

        _save_yaml(
            self.save_best_yaml,
            _camera_yaml(
                self.camera_name,
                best.image_size,
                best.camera_matrix,
                best.dist_coeffs,
                best.rms,
                best.samples,
                extra={
                    "selected": "best_round",
                    "best_round": int(best.index),
                    "source_yaml": best.yaml_path,
                },
            ),
        )

        image_size = best.image_size
        ks = np.stack([r.camera_matrix.reshape(3, 3) for r in self.results], axis=0)
        max_d = max(r.dist_coeffs.size for r in self.results)
        ds = []
        for r in self.results:
            d = np.zeros((max_d,), dtype=np.float64)
            d[: r.dist_coeffs.size] = r.dist_coeffs
            ds.append(d)
        d_stack = np.stack(ds, axis=0)
        weights = np.asarray([1.0 / max(r.rms, 1e-9) for r in self.results], dtype=np.float64)
        weights /= np.sum(weights)
        avg_k = np.sum(ks * weights[:, None, None], axis=0)
        avg_d = np.sum(d_stack * weights[:, None], axis=0)
        rms_mean = statistics.mean([r.rms for r in self.results])

        _save_yaml(
            self.save_average_yaml,
            _camera_yaml(
                self.camera_name,
                image_size,
                avg_k,
                avg_d,
                rms_mean,
                sum(r.samples for r in self.results),
                extra={
                    "selected": "weighted_average",
                    "rounds": len(self.results),
                    "round_rms": [float(r.rms) for r in self.results],
                    "round_yaml_paths": [r.yaml_path for r in self.results],
                    "weights": weights.tolist(),
                },
            ),
        )

        summary_path = os.path.join(self.output_dir, "summary.yaml")
        _save_yaml(
            summary_path,
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "recommended_yaml": self.save_joint_yaml if joint_yaml else self.save_best_yaml,
                "joint_yaml": joint_yaml,
                "best_yaml": self.save_best_yaml,
                "average_yaml": self.save_average_yaml,
                "rounds": [
                    {
                        "round": r.index,
                        "rms_reprojection_px": float(r.rms),
                        "samples": r.samples,
                        "yaml": r.yaml_path,
                    }
                    for r in self.results
                ],
            },
        )
        if joint_yaml:
            self.get_logger().info(f"Recommended calibration for PnP: {joint_yaml}")
        self.get_logger().info(f"Saved best calibration: {self.save_best_yaml}")
        self.get_logger().info(f"Saved averaged calibration: {self.save_average_yaml}")
        self.get_logger().info(f"Saved summary: {summary_path}")
        self.done = True
        threading.Thread(target=self._shutdown_soon, daemon=True).start()

    def _shutdown_soon(self):
        time.sleep(0.5)
        if self.show_window:
            try:
                cv2.destroyWindow("optimized_charuco_calibration")
            except Exception:
                cv2.destroyAllWindows()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rclpy.init()
    node = OptimizedCharucoCalib(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
