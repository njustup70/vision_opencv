"""
小 ChArUco 板位姿估计节点（small_board_pose_node）

目标：
- 只用“小板”（5cm x 5cm 的 5x5 ChArUco）做 PnP；
- 读取相机内参 YAML（camera.yaml），输出 camera_T_small_board；
- 可选：等待字符串控制话题收到 "spear" 后再开始计算；
- 发布左右/上下偏移量到 ROS2 结果话题，方便主逻辑直接订阅；
- 运行时弹出 OpenCV 窗口可视化：
  1) 检测到的 ArUco ID、角点、ChArUco 角点；
  2) 重投影误差 mean/max（px）与 confidence；
  3) 摄像头相对小板原点的平移（left/up/forward，单位 mm，精确到 0.001mm）。

坐标/符号约定（常见 optical frame）：
- OpenCV solvePnP 输出的 tvec 是“相机坐标系下，物体原点的位置”：
  x：向右；y：向下；z：向前（离相机更远）。
- 你更关心“左/上”偏移，因此这里显示：
  left_mm  = -x_mm
  up_mm    = -y_mm
  forward_mm = +z_mm
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, Float32MultiArray, Float64, String
from tf2_ros import TransformBroadcaster

from spear_vision.core.board_pose_estimator import BoardPoseEstimator, BoardSpec, GateSpec, PoseEstimate
from spear_vision.core.pose_filter import PoseLowPassFilter
from spear_vision.utils.calibration_store import read_last_calibration_path
from spear_vision.utils.camera_intrinsics import CameraIntrinsics, intrinsics_from_camera_info, intrinsics_from_yaml
from spear_vision.utils.opencv_aruco import (
    create_charuco_board,
    create_detector_parameters,
    detect_markers,
    draw_marker_corners,
    draw_marker_ids,
    interpolate_charuco_corners,
    refine_markers,
    resolve_dictionary,
)
from spear_vision.utils.ros_conversions import rt_to_pose_stamped, rt_to_transform_stamped
from spear_vision.utils.runtime_checks import warn_opencv_environment
from spear_vision.utils.tf_utils import rvec_tvec_to_rt
from spear_vision.utils.yaml_io import load_yaml


class SmallBoardPoseNode(Node):
    def __init__(self) -> None:
        super().__init__("small_board_pose")
        warn_opencv_environment(self.get_logger())

        # --- 参数区：支持通过 config_path YAML 一键配置（small_board.yaml） ---
        self.declare_parameter("config_path", "")
        self.declare_parameter("camera_calibration_yaml", "")
        self.declare_parameter("prefer_camera_info", True)

        self.declare_parameter("image_topic", "/hik_camera/image_raw")
        self.declare_parameter("camera_info_topic", "")

        # TF frame_id/child_frame_id
        self.declare_parameter("camera_frame", "")
        self.declare_parameter("board_frame", "spear_small_board_frame")

        self.declare_parameter("publish_tf", True)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("publish_pose", True)
        self.declare_parameter("publish_offsets", True)

        # 控制话题：可选要求先收到 start_command_value 才开始算
        self.declare_parameter("require_start_command", False)
        self.declare_parameter("command_topic", "~/command")
        self.declare_parameter("start_command_value", "spear")
        self.declare_parameter("stop_command_value", "stop")
        self.declare_parameter("offset_topic", "~/offset_mm")

        # 可视化开关（默认开启）
        self.declare_parameter("show_opencv_window", True)
        self.declare_parameter("opencv_window_name", "small_board_view")

        self.declare_parameter("smoothing_alpha", 0.8)
        self.declare_parameter("smoothing_rotation_mode", "rvec")
        self.declare_parameter("debug_axis_length_m", 0.03)

        self.declare_parameter("aruco_refine_detected_markers", True)
        self.declare_parameter("aruco_fallback_enable", True)
        self.declare_parameter("aruco_fallback_min_markers", 1)

        self.declare_parameter("pnp_prefer", "SOLVEPNP_IPPE")
        self.declare_parameter("pnp_fallback", "SOLVEPNP_ITERATIVE")
        self.declare_parameter("pnp_refine_lm", True)

        # 小板默认门控参数（可由 YAML 覆盖）
        self.declare_parameter("min_charuco_corners", 10)
        self.declare_parameter("max_mean_reproj_px", 0.6)
        self.declare_parameter("max_max_reproj_px", 1.5)
        self.declare_parameter("min_border_px", 10)

        self._bridge = CvBridge()
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._pose_enabled = not bool(self.get_parameter("require_start_command").value)

        self._pose_estimator = BoardPoseEstimator()
        self._pose_filter = PoseLowPassFilter()

        # 1) 读取 YAML 配置（可覆盖 topic/板参数/gating/pnp）
        # 2) 初始化 board 与检测参数
        # 3) 加载内参（推荐）
        # 4) 初始化 ROS 通信
        self._load_config_if_any()
        self._init_board()
        self._load_intrinsics_if_any()
        self._init_ros()

    def _load_config_if_any(self) -> None:
        config_path = self.get_parameter("config_path").get_parameter_value().string_value
        if not config_path:
            return

        data = load_yaml(config_path)
        topics = data.get("topics", {})
        frames = data.get("frames", {})
        charuco = data.get("charuco", {})
        gating = data.get("gating", {})
        pnp = data.get("pnp", {})
        fallback = data.get("aruco_board_fallback", {})

        self._maybe_set_param_from_dict("image_topic", topics, "image")
        self._maybe_set_param_from_dict("camera_info_topic", topics, "camera_info")

        self._maybe_set_param_from_dict("camera_frame", frames, "camera_frame")
        self._maybe_set_param_from_dict("board_frame", frames, "board_frame")

        self._cfg_dictionary = str(charuco.get("dictionary", "DICT_6X6_250"))
        self._cfg_squares_x = int(charuco.get("squares_x", 5))
        self._cfg_squares_y = int(charuco.get("squares_y", 5))
        self._cfg_square_length_m = float(charuco.get("square_length_m", 0.01))
        self._cfg_marker_length_m = float(charuco.get("marker_length_m", 0.008))
        self._cfg_ids_start = int(charuco.get("ids_start", 100))

        self._maybe_set_param_from_dict("min_charuco_corners", gating, "min_charuco_corners")
        self._maybe_set_param_from_dict("max_mean_reproj_px", gating, "max_mean_reproj_px")
        self._maybe_set_param_from_dict("max_max_reproj_px", gating, "max_max_reproj_px")
        self._maybe_set_param_from_dict("min_border_px", gating, "min_border_px")

        self._maybe_set_param_from_dict("pnp_prefer", pnp, "prefer")
        self._maybe_set_param_from_dict("pnp_fallback", pnp, "fallback")

        self._maybe_set_param_from_dict("aruco_fallback_enable", fallback, "enable")
        self._maybe_set_param_from_dict("aruco_fallback_min_markers", fallback, "min_markers")
        self._maybe_set_param_from_dict("aruco_refine_detected_markers", fallback, "use_refine_detected_markers")

        self.get_logger().info(f"Loaded config: {config_path}")

    def _maybe_set_param_from_dict(self, param_name: str, data: dict[str, Any], key: str) -> None:
        if key not in data:
            return
        try:
            self.set_parameters([rclpy.parameter.Parameter(param_name, value=data[key])])
        except Exception:
            pass

    def _init_board(self) -> None:
        dictionary_name = getattr(self, "_cfg_dictionary", None) or "DICT_6X6_250"
        squares_x = getattr(self, "_cfg_squares_x", None) or 5
        squares_y = getattr(self, "_cfg_squares_y", None) or 5
        square_length_m = getattr(self, "_cfg_square_length_m", None) or 0.01
        marker_length_m = getattr(self, "_cfg_marker_length_m", None) or 0.008
        ids_start = getattr(self, "_cfg_ids_start", None) or 100

        self._board_dictionary_name = str(dictionary_name)
        self._board_squares_x = int(squares_x)
        self._board_squares_y = int(squares_y)
        self._board_square_length_m = float(square_length_m)
        self._board_marker_length_m = float(marker_length_m)
        self._board_ids_start = int(ids_start)

        dictionary = resolve_dictionary(dictionary_name)
        self._board = create_charuco_board(
            squares_x=int(squares_x),
            squares_y=int(squares_y),
            square_length_m=float(square_length_m),
            marker_length_m=float(marker_length_m),
            dictionary=dictionary,
            ids_start=int(ids_start),
        )
        self._dictionary = dictionary
        self._detector_params = create_detector_parameters()

        self.get_logger().info(
            "Small board: %dx%d square=%.3fm marker=%.3fm dict=%s ids_start=%d"
            % (
                int(squares_x),
                int(squares_y),
                float(square_length_m),
                float(marker_length_m),
                str(dictionary_name),
                int(ids_start),
            )
        )

    def _load_intrinsics_if_any(self) -> None:
        # 读取相机内参（优先顺序）：
        # 1) 参数 camera_calibration_yaml
        # 2) 最近一次 charuco_calib 写入的指针文件
        # 3) 工作空间固定路径 ~/CHaruco/hik_ws/src/spear_vision/config/camera.yaml
        path = self.get_parameter("camera_calibration_yaml").get_parameter_value().string_value
        if not path:
            auto_path = read_last_calibration_path()
            if auto_path:
                path = auto_path
                self.get_logger().info(f"camera_calibration_yaml is empty; auto-loading from last calibration: {path}")
            else:
                ws_default = os.path.expanduser("~/CHaruco/hik_ws/src/spear_vision/config/camera.yaml")
                if os.path.exists(ws_default):
                    path = ws_default
                    self.get_logger().info(f"camera_calibration_yaml is empty; falling back to workspace camera.yaml: {path}")
                else:
                    return

        try:
            self._intrinsics = intrinsics_from_yaml(path)
            self.get_logger().info(f"Loaded camera intrinsics from YAML: {path}")
        except Exception as exc:
            self.get_logger().error(f"Failed to load camera intrinsics YAML: {exc}")

    def _init_ros(self) -> None:
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value

        camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        derived_info = image_topic.rstrip("/") + "/camera_info"
        if not camera_info_topic or camera_info_topic.strip() in ("/hik_camera/camera_info",):
            camera_info_topic = derived_info

        self._sub_img = self.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)
        self._sub_info = self.create_subscription(
            CameraInfo, camera_info_topic, self._on_camera_info, qos_profile_sensor_data
        )
        self.get_logger().info(f"Subscribing image: {image_topic}")
        self.get_logger().info(f"Subscribing camera_info: {camera_info_topic}")

        self._pose_pub = self.create_publisher(PoseStamped, "~/pose", 10)
        self._debug_pub = self.create_publisher(Image, "~/debug_image", qos_profile_sensor_data)
        self._err_mean_pub = self.create_publisher(Float64, "~/reproj_error_mean_px", 10)
        self._err_max_pub = self.create_publisher(Float64, "~/reproj_error_max_px", 10)
        self._confidence_pub = self.create_publisher(Float32, "~/confidence", 10)
        self._method_pub = self.create_publisher(String, "~/method", 10)
        offset_topic = str(self.get_parameter("offset_topic").value).strip() or "~/offset_mm"
        self._offset_pub = self.create_publisher(Float32MultiArray, offset_topic, 10)

        command_topic = str(self.get_parameter("command_topic").value).strip()
        self._command_sub = None
        if command_topic:
            self._command_sub = self.create_subscription(String, command_topic, self._on_command, 10)
            if bool(self.get_parameter("require_start_command").value):
                start_cmd = str(self.get_parameter("start_command_value").value).strip()
                self.get_logger().info(f"Waiting for start command '{start_cmd}' on {command_topic}")

        self._tf_broadcaster = TransformBroadcaster(self)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        # 如果驱动发布了有效 CameraInfo，则可以热更新（但你的 hik_camera_ros2 很可能为空）
        if not bool(self.get_parameter("prefer_camera_info").value):
            return
        intr = intrinsics_from_camera_info(msg)
        if intr is None or not intr.is_valid():
            return
        self._intrinsics = intr

    def _on_command(self, msg: String) -> None:
        if not bool(self.get_parameter("require_start_command").value):
            return

        data = msg.data.strip()
        start_cmd = str(self.get_parameter("start_command_value").value).strip()
        stop_cmd = str(self.get_parameter("stop_command_value").value).strip()

        if data == start_cmd:
            if not self._pose_enabled:
                self.get_logger().info(f"Received start command '{data}', pose estimation enabled")
            self._pose_enabled = True
            return

        if stop_cmd and data == stop_cmd:
            if self._pose_enabled:
                self.get_logger().info(f"Received stop command '{data}', pose estimation paused")
            self._pose_enabled = False

    def _on_image(self, msg: Image) -> None:
        if bool(self.get_parameter("require_start_command").value) and not self._pose_enabled:
            return

        gray = self._bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

        # 1) ArUco marker 检测
        detection = detect_markers(gray, self._dictionary, self._detector_params)
        if bool(self.get_parameter("aruco_refine_detected_markers").value):
            # 2) refineDetectedMarkers：利用 board 布局找回漏检 marker
            if self._intrinsics is None:
                detection = refine_markers(gray, self._board, detection, None, None)
            else:
                detection = refine_markers(
                    gray,
                    self._board,
                    detection,
                    self._intrinsics.camera_matrix,
                    self._intrinsics.dist_coeffs,
                )
        # 3) 位姿估计（优先 charuco → fallback aruco_board）
        estimate = self._estimate_pose(gray, detection)
        self._publish_outputs(msg, gray, detection, estimate)

    def _estimate_pose(self, gray: np.ndarray, detection) -> PoseEstimate:
        h, w = gray.shape[:2]
        if self._intrinsics is None or not self._intrinsics.is_valid():
            return PoseEstimate(ok=False, method="no_intrinsics")

        spec = BoardSpec(
            name="small_board",
            frame=str(self.get_parameter("board_frame").value),
            squares_x=int(self._board_squares_x),
            squares_y=int(self._board_squares_y),
            square_length_m=float(self._board_square_length_m),
            marker_length_m=float(self._board_marker_length_m),
            dictionary=str(self._board_dictionary_name),
            ids_start=int(self._board_ids_start),
            min_charuco_corners=int(self.get_parameter("min_charuco_corners").value),
            fallback_enable=bool(self.get_parameter("aruco_fallback_enable").value),
            fallback_min_markers=int(self.get_parameter("aruco_fallback_min_markers").value),
            use_refine_detected_markers=bool(self.get_parameter("aruco_refine_detected_markers").value),
        )
        gate = GateSpec(
            max_mean_reproj_px=float(self.get_parameter("max_mean_reproj_px").value),
            max_max_reproj_px=float(self.get_parameter("max_max_reproj_px").value),
            min_border_px=int(self.get_parameter("min_border_px").value),
        )

        prefer = str(self.get_parameter("pnp_prefer").value)
        fallback = str(self.get_parameter("pnp_fallback").value)
        refine_lm = bool(self.get_parameter("pnp_refine_lm").value)

        est = self._pose_estimator.estimate(
            gray,
            self._intrinsics,
            self._board,
            detection,
            spec,
            gate,
            image_size=(w, h),
            pnp_prefer=prefer,
            pnp_fallback=fallback,
            pnp_refine_lm=refine_lm,
            prior_rt=self._pose_filter.last,
        )

        if not est.ok or est.rvec is None or est.tvec is None:
            return est

        alpha = float(self.get_parameter("smoothing_alpha").value)
        mode = str(self.get_parameter("smoothing_rotation_mode").value)
        rt = self._pose_filter.update(rvec_tvec_to_rt(est.rvec, est.tvec), alpha, rotation_mode=mode)
        return replace(est, rvec=rt.rvec, tvec=rt.tvec)

    @staticmethod
    def _tvec_to_offsets_mm(tvec: np.ndarray) -> Optional[tuple[float, float, float]]:
        t = np.array(tvec, dtype=np.float64).reshape(3)
        if not np.isfinite(t).all():
            return None

        x_mm = float(t[0] * 1000.0)
        y_mm = float(t[1] * 1000.0)
        z_mm = float(t[2] * 1000.0)
        return (-x_mm, -y_mm, z_mm)

    def _publish_outputs(self, msg: Image, gray: np.ndarray, detection, est: PoseEstimate) -> None:
        stamp = msg.header.stamp
        camera_frame = str(self.get_parameter("camera_frame").value).strip() or msg.header.frame_id
        board_frame = str(self.get_parameter("board_frame").value).strip()
        offsets_mm: Optional[tuple[float, float, float]] = None

        if est.ok and est.rvec is not None and est.tvec is not None:
            # 防御性：若 rvec/tvec 出现 NaN/Inf，则不要发布 Pose/TF（否则你会看到坐标系“时有时无”）
            if not (
                np.isfinite(np.array(est.rvec, dtype=np.float64)).all()
                and np.isfinite(np.array(est.tvec, dtype=np.float64)).all()
            ):
                est = PoseEstimate(ok=False, method="invalid_pose_nan", used=est.used)
            else:
                rt = rvec_tvec_to_rt(est.rvec, est.tvec)
                if bool(self.get_parameter("publish_pose").value):
                    self._pose_pub.publish(rt_to_pose_stamped(rt, stamp, camera_frame))

                if bool(self.get_parameter("publish_tf").value):
                    self._tf_broadcaster.sendTransform(rt_to_transform_stamped(rt, stamp, camera_frame, board_frame))

                if est.mean_reproj_px is not None:
                    self._err_mean_pub.publish(Float64(data=float(est.mean_reproj_px)))
                if est.max_reproj_px is not None:
                    self._err_max_pub.publish(Float64(data=float(est.max_reproj_px)))
                self._confidence_pub.publish(Float32(data=float(est.confidence)))
                self._method_pub.publish(String(data=f"{est.used}:{est.method}"))
                offsets_mm = self._tvec_to_offsets_mm(est.tvec)
                if bool(self.get_parameter("publish_offsets").value) and offsets_mm is not None:
                    left_mm, up_mm, _ = offsets_mm
                    self._offset_pub.publish(Float32MultiArray(data=[left_mm, up_mm]))

        if not bool(self.get_parameter("publish_debug_image").value):
            return

        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if detection.ids is not None and len(detection.ids) > 0:
            try:
                cv2.aruco.drawDetectedMarkers(bgr, detection.corners, detection.ids)
            except Exception:
                pass
            # 标出每个 marker 的 ID + 四角点，便于你肉眼判断“是否在飘”
            try:
                draw_marker_ids(bgr, detection.corners, detection.ids)
                draw_marker_corners(bgr, detection.corners)
            except Exception:
                pass

        num, cc, ci = interpolate_charuco_corners(gray, self._board, detection.corners, detection.ids)
        if cc is not None and ci is not None and int(num) > 0:
            try:
                cv2.aruco.drawDetectedCornersCharuco(bgr, cc, ci, (0, 0, 255))
            except Exception:
                pass

        if est.ok and est.rvec is not None and est.tvec is not None and self._intrinsics is not None:
            try:
                cv2.drawFrameAxes(
                    bgr,
                    self._intrinsics.camera_matrix,
                    self._intrinsics.dist_coeffs,
                    est.rvec,
                    est.tvec,
                    float(self.get_parameter("debug_axis_length_m").value),
                )
            except Exception:
                pass

            mean_px = float(est.mean_reproj_px or 0.0)
            max_px = float(est.max_reproj_px or 0.0)
            conf = float(est.confidence)

            cv2.putText(
                bgr,
                f"{est.used} charuco={int(num)} mean={mean_px:.3f}px max={max_px:.3f}px conf={conf:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # 输出平移（mm，精确到 0.001mm）
            if offsets_mm is None:
                cv2.putText(
                    bgr,
                    "invalid pose (tvec has NaN/Inf) -> TF suppressed",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                self._debug_pub.publish(self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8"))
                if bool(self.get_parameter("show_opencv_window").value):
                    try:
                        name = str(self.get_parameter("opencv_window_name").value) or "small_board_view"
                        cv2.imshow(name, bgr)
                        cv2.waitKey(1)
                    except Exception as exc:
                        self.get_logger().warn(f"OpenCV window disabled (GUI not available): {exc}")
                        self.set_parameters([rclpy.parameter.Parameter("show_opencv_window", value=False)])
                return
            left_mm, up_mm, forward_mm = offsets_mm

            cv2.putText(
                bgr,
                f"offset(mm): left={left_mm:+.3f} up={up_mm:+.3f} forward={forward_mm:+.3f}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                bgr,
                f"no pose ({est.method}) charuco={int(num)} markers={0 if detection.ids is None else int(len(detection.ids))}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        self._debug_pub.publish(self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8"))

        # 可选：直接弹出 OpenCV 窗口（默认开启）
        if bool(self.get_parameter("show_opencv_window").value):
            try:
                name = str(self.get_parameter("opencv_window_name").value) or "small_board_view"
                cv2.imshow(name, bgr)
                cv2.waitKey(1)
            except Exception as exc:
                self.get_logger().warn(f"OpenCV window disabled (GUI not available): {exc}")
                self.set_parameters([rclpy.parameter.Parameter("show_opencv_window", value=False)])


def main() -> None:
    rclpy.init()
    node = SmallBoardPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
