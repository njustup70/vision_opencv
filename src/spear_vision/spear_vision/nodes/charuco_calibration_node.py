"""
ChArUco 相机内参标定节点（spear_vision）

功能：
- 订阅单路图像（mono8/bgr8 均可，内部转 mono8）；
- 检测 ChArUco board，累计多帧、多姿态样本；
- 通过 `cv2.aruco.calibrateCameraCharucoExtended` 求解相机内参（K、D）；
- 导出为 ROS 常用 camera_info YAML 格式，供后续 PnP 直接加载使用。

使用建议（面向亚毫米稳定性）：
1) 拍摄多角度/多距离/多位置的样本（覆盖成像区域），避免只在中心采样；
2) `min_charuco_corners` 不要太低，角点越多，标定越稳；
3) `sample_stride` 用于降采样，避免重复帧过多；`max_samples` 控制上限；
4) 标定完成后，运行位姿节点时建议“优先使用 YAML 内参”，
   因为你的驱动当前 `CameraInfo` 可能为空内参。
"""

from __future__ import annotations

import os
import threading
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String

from spear_vision.utils.opencv_aruco import (
    MarkerDetection,
    create_charuco_board,
    create_detector_parameters,
    detect_markers,
    interpolate_charuco_corners,
    refine_charuco_corners_subpix,
    refine_markers,
    resolve_dictionary,
)
from spear_vision.utils.calibration_store import write_last_calibration_path
from spear_vision.utils.runtime_checks import warn_opencv_environment
from spear_vision.utils.yaml_io import load_yaml, save_yaml


class CharucoCalibrationNode(Node):
    def __init__(self) -> None:
        super().__init__("charuco_calibration_node")
        warn_opencv_environment(self.get_logger())

        # --- 参数区：既可直接 ros2 param/launch 传入，也可通过 config_path 的 YAML 覆盖 ---
        self.declare_parameter("config_path", "")
        self.declare_parameter("image_topic", "/hik_camera/image_raw")
        self.declare_parameter("dictionary", "DICT_6X6_250")
        self.declare_parameter("squares_x", 5)
        self.declare_parameter("squares_y", 5)
        self.declare_parameter("square_length_m", 0.03)
        self.declare_parameter("marker_length_m", 0.024)
        self.declare_parameter("ids_start", 0)
        self.declare_parameter("min_charuco_corners", 12)
        # 采样步长：每 N 帧采 1 帧样本（N 越大，你越有时间改变棋盘姿态）
        self.declare_parameter("sample_stride", 20)
        # “目标样本数”：仅用于进度显示与建议，不会阻止你提前触发 calibrate_now（但样本太少会报错）
        # 标定目标帧数：达到后自动触发求解并退出（可通过参数关闭）
        self.declare_parameter("target_samples", 100)
        self.declare_parameter("auto_calibrate_on_target", True)
        self.declare_parameter("auto_exit_after_calibration", True)
        self.declare_parameter("max_samples", 200)
        # 默认把内参保存到工作空间（若路径不存在则退回到 ~/charuco_camera.yaml）
        workspace_cfg_dir = os.path.expanduser("~/CHaruco/hik_ws/src/spear_vision/config")
        if os.path.isdir(workspace_cfg_dir):
            default_out = os.path.join(workspace_cfg_dir, "camera.yaml")
        else:
            default_out = os.path.expanduser("~/charuco_camera.yaml")
        self.declare_parameter("output_yaml_path", default_out)
        self.declare_parameter("camera_name", "hik_camera")
        self.declare_parameter("publish_debug_image", True)
        # 直接弹出 OpenCV 窗口显示标定视野（不依赖 RViz/rqt_image_view）
        # 注意：无桌面/无 DISPLAY 的环境会失败，此时会自动降级为只发布 ~/debug_image
        self.declare_parameter("show_opencv_window", True)
        self.declare_parameter("opencv_window_name", "charuco_calibration")
        self.declare_parameter("calibrate_now", False)
        self.declare_parameter("reset_samples", False)

        self._bridge = CvBridge()
        self._frame_index = 0
        self._frames_received = 0
        self._last_marker_count = 0
        self._last_charuco_count = 0
        self._calibrated = False
        self._calibrating = False
        self._calib_thread: threading.Thread | None = None
        self._shutdown_requested = False

        # 累计样本：每个样本由 (charuco_corners, charuco_ids) 组成
        # 注意：同一帧里并不是所有角点都能插值成功，所以 ids 是变长集合
        self._charuco_corners_list: list[np.ndarray] = []
        self._charuco_ids_list: list[np.ndarray] = []
        self._image_size: tuple[int, int] | None = None  # (w, h)

        self._load_config_if_any()
        self._init_board()

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self._sub = self.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribing image: {image_topic}")

        # debug_image：叠加 marker/charuco 角点可视化，方便现场调参/看漏检
        self._debug_pub = self.create_publisher(Image, "~/debug_image", qos_profile_sensor_data)
        # 进度/状态：方便命令行或 rqt_plot 查看（不依赖 GUI）
        self._pub_frames = self.create_publisher(Int32, "~/frames_received", 10)
        self._pub_samples = self.create_publisher(Int32, "~/samples_accepted", 10)
        self._pub_status = self.create_publisher(String, "~/status", 10)
        # calibrate_now/reset_samples 通过动态参数触发（避免写服务/动作，操作更直接）
        self._param_cb = self.add_on_set_parameters_callback(self._on_params)

    def _load_config_if_any(self) -> None:
        config_path = self.get_parameter("config_path").get_parameter_value().string_value
        if not config_path:
            return

        data = load_yaml(config_path)
        charuco = data.get("charuco", {})
        topics = data.get("topics", {})

        self._maybe_set_param_from_dict("image_topic", topics, "image")
        self._maybe_set_param_from_dict("dictionary", charuco, "dictionary")
        self._maybe_set_param_from_dict("squares_x", charuco, "squares_x")
        self._maybe_set_param_from_dict("squares_y", charuco, "squares_y")
        self._maybe_set_param_from_dict("square_length_m", charuco, "square_length_m")
        self._maybe_set_param_from_dict("marker_length_m", charuco, "marker_length_m")
        self._maybe_set_param_from_dict("ids_start", charuco, "ids_start")

        gating = data.get("gating", {})
        self._maybe_set_param_from_dict("min_charuco_corners", gating, "min_charuco_corners")
        # 可选：让 YAML 配置里直接写 target_samples
        self._maybe_set_param_from_dict("target_samples", gating, "target_samples")
        # 可选：让 YAML 配置里直接写 sample_stride（兼容用户“每 20 帧取 1 帧”的习惯）
        self._maybe_set_param_from_dict("sample_stride", gating, "sample_stride")

        self.get_logger().info(f"Loaded config: {config_path}")

    def _maybe_set_param_from_dict(self, param_name: str, data: dict[str, Any], key: str) -> None:
        if key not in data:
            return
        try:
            # 这里用 set_parameters 统一把 YAML 写入 node 参数，后续逻辑只读参数即可
            self.set_parameters([rclpy.parameter.Parameter(param_name, value=data[key])])
        except Exception:
            pass

    def _init_board(self) -> None:
        dictionary_name = self.get_parameter("dictionary").get_parameter_value().string_value
        squares_x = int(self.get_parameter("squares_x").value)
        squares_y = int(self.get_parameter("squares_y").value)
        square_length_m = float(self.get_parameter("square_length_m").value)
        marker_length_m = float(self.get_parameter("marker_length_m").value)
        ids_start = int(self.get_parameter("ids_start").value)

        dictionary = resolve_dictionary(dictionary_name)
        self._board = create_charuco_board(
            squares_x=squares_x,
            squares_y=squares_y,
            square_length_m=square_length_m,
            marker_length_m=marker_length_m,
            dictionary=dictionary,
            ids_start=ids_start,
        )
        self._dictionary = dictionary
        self._detector_params = create_detector_parameters()

        self.get_logger().info(
            "ChArUco board: %dx%d square=%.3fm marker=%.3fm dict=%s ids_start=%d"
            % (squares_x, squares_y, square_length_m, marker_length_m, dictionary_name, ids_start)
        )

    def _on_params(self, params: list[rclpy.parameter.Parameter]) -> SetParametersResult:
        # 用“参数回调”实现两个按钮：
        # - reset_samples: 清空样本缓存
        # - calibrate_now: 立即用当前样本计算并保存内参
        reset = False
        calibrate = False
        for p in params:
            if p.name == "reset_samples" and bool(p.value):
                reset = True
            if p.name == "calibrate_now" and bool(p.value):
                calibrate = True

        if reset:
            self._charuco_corners_list.clear()
            self._charuco_ids_list.clear()
            self._image_size = None
            self._frames_received = 0
            self._last_marker_count = 0
            self._last_charuco_count = 0
            self.get_logger().info("Samples reset.")

        if calibrate:
            try:
                self._start_calibration_async()
            except Exception as exc:
                self.get_logger().error(f"Calibration failed: {exc}")
                return SetParametersResult(successful=False, reason=str(exc))

        return SetParametersResult(successful=True)

    def _on_image(self, msg: Image) -> None:
        # 采样降频：只取每 stride 帧里的 1 帧，减少重复样本
        self._frame_index += 1
        self._frames_received += 1
        self._pub_frames.publish(Int32(data=int(self._frames_received)))
        stride = int(self.get_parameter("sample_stride").value)
        if stride > 1 and (self._frame_index % stride) != 0:
            return

        max_samples = int(self.get_parameter("max_samples").value)
        if len(self._charuco_corners_list) >= max_samples:
            return

        # 标定只需要灰度图；bgr8 会在 cv_bridge 内部转为 mono8
        gray = self._bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        h, w = gray.shape[:2]
        if self._image_size is None:
            self._image_size = (w, h)

        # 若正在标定（耗时计算），保持画面刷新但不再追加样本，避免 GUI 无响应
        if self._calibrating:
            dummy = MarkerDetection(corners=[], ids=None, rejected=[])
            self._publish_debug(gray, dummy, None, None)
            return

        # 1) 先做 ArUco marker 检测
        detection = detect_markers(gray, self._dictionary, self._detector_params)
        self._last_marker_count = int(len(detection.ids)) if detection.ids is not None else 0
        # 2) 利用 board 布局 refine（尝试找回漏检 marker）
        # 标定阶段不强依赖相机内参，因此 camera_matrix/dist_coeffs 传 None 也可工作
        detection = refine_markers(gray, self._board, detection, None, None)

        # 3) 由 marker 推断 ChArUco 内角点（用于高精度标定）
        num, charuco_corners, charuco_ids = interpolate_charuco_corners(
            gray, self._board, detection.corners, detection.ids
        )
        self._last_charuco_count = int(num)
        min_corners = int(self.get_parameter("min_charuco_corners").value)
        if charuco_corners is None or charuco_ids is None or num < min_corners:
            self._publish_debug(gray, detection, None, None)
            return

        # 4) 亚像素细化角点（对内参精度影响非常大）
        charuco_corners = refine_charuco_corners_subpix(gray, charuco_corners, win_size=3)

        self._charuco_corners_list.append(charuco_corners)
        self._charuco_ids_list.append(charuco_ids)

        self._pub_samples.publish(Int32(data=int(len(self._charuco_corners_list))))

        target = int(self.get_parameter("target_samples").value)
        self.get_logger().info(
            f"Sample accepted: {len(self._charuco_corners_list)}/{target} (max={max_samples}, corners={int(num)})"
        )
        self._publish_debug(gray, detection, charuco_corners, charuco_ids)

        # 达到目标样本数后，自动标定并退出（可关闭）
        if (
            not self._calibrated
            and not self._calibrating
            and bool(self.get_parameter("auto_calibrate_on_target").value)
            and len(self._charuco_corners_list) >= target
        ):
            try:
                self._start_calibration_async()
            except Exception as exc:
                self.get_logger().error(f"Auto calibration failed: {exc}")

    def _publish_debug(self, gray, detection, charuco_corners, charuco_ids) -> None:
        if not bool(self.get_parameter("publish_debug_image").value):
            return
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if detection.ids is not None and len(detection.ids) > 0:
            try:
                cv2.aruco.drawDetectedMarkers(bgr, detection.corners, detection.ids)
            except Exception:
                pass
        if charuco_corners is not None and charuco_ids is not None:
            try:
                cv2.aruco.drawDetectedCornersCharuco(bgr, charuco_corners, charuco_ids, (0, 0, 255))
            except Exception:
                pass
        # 进度信息直接叠加到 debug 图像上（“标定视野可视化”）
        # - accepted/target/max：已采样/目标/上限
        # - last(markers,charuco)/min：上一帧检测情况（用于判断是不是“没看到板/角点太少/反光模糊”）
        accepted = int(len(self._charuco_corners_list))
        target = int(self.get_parameter("target_samples").value)
        max_samples = int(self.get_parameter("max_samples").value)
        min_corners = int(self.get_parameter("min_charuco_corners").value)
        stride = int(self.get_parameter("sample_stride").value)

        lines = [
            f"frames_received={int(self._frames_received)} stride={stride}",
            f"accepted={accepted} target={target} max={max_samples}",
            f"last: markers={int(self._last_marker_count)} charuco={int(self._last_charuco_count)}/{min_corners}",
            "trigger: ros2 param set /charuco_calibration calibrate_now true",
        ]
        if self._calibrating:
            lines.insert(0, "status=calibrating... (please wait)")
        y = 28
        for line in lines:
            cv2.putText(bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 24

        # 同时发布一条 status 字符串（便于命令行查看）
        self._pub_status.publish(
            String(
                data=(
                    f"frames={int(self._frames_received)} accepted={accepted}/{target} "
                    f"last_markers={int(self._last_marker_count)} last_charuco={int(self._last_charuco_count)}/{min_corners}"
                )
            )
        )

        # 可选：直接弹 OpenCV 窗口，省去单独打开 rqt_image_view
        if bool(self.get_parameter("show_opencv_window").value):
            try:
                name = str(self.get_parameter("opencv_window_name").value) or "charuco_calibration"
                cv2.imshow(name, bgr)
                cv2.waitKey(1)
            except Exception as exc:
                # GUI 不可用时自动降级（只发布 debug image，不让节点崩）
                self.get_logger().warn(f"OpenCV window disabled (GUI not available): {exc}")
                self.set_parameters([rclpy.parameter.Parameter("show_opencv_window", value=False)])

        self._debug_pub.publish(self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8"))

    def _run_calibration_and_save(self) -> None:
        # 计算内参前要先收到图像并积累足够样本
        if self._image_size is None:
            raise RuntimeError("No image received yet.")
        # 依然保留一个“硬下限”，避免样本太少导致结果不稳定/不可解
        if len(self._charuco_corners_list) < 10:
            raise RuntimeError("Not enough samples (need >= 10).")

        # 提示：如果你设置了 target_samples，更推荐达到 target 再标定（精度更稳）
        target = int(self.get_parameter("target_samples").value)
        if target > 10 and len(self._charuco_corners_list) < target:
            self.get_logger().warn(
                f"Calibrating with only {len(self._charuco_corners_list)}/{target} accepted samples; "
                "intrinsics may be less stable. Consider collecting more frames."
            )

        w, h = self._image_size
        flags = 0
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        # 风险 A 兼容：OpenCV 不同版本可能只有 calibrateCameraCharuco 或 calibrateCameraCharucoExtended
        if hasattr(cv2.aruco, "calibrateCameraCharucoExtended"):
            # calibrateCameraCharucoExtended 会返回：
            # - rms：总体 RMS 重投影误差（像素）
            # - camera_matrix / dist_coeffs：我们关心的内参结果
            # - per_view：每个样本视角的误差，可用于剔除坏帧（后续可扩展）
            (
                rms,
                camera_matrix,
                dist_coeffs,
                rvecs,
                tvecs,
                _std_int,
                _std_ext,
                _per_view,
            ) = cv2.aruco.calibrateCameraCharucoExtended(
                self._charuco_corners_list,
                self._charuco_ids_list,
                self._board,
                (w, h),
                None,
                None,
                flags=flags,
                criteria=criteria,
            )
        elif hasattr(cv2.aruco, "calibrateCameraCharuco"):
            # 旧/精简接口：缺少 std/per_view，但足够导出 K/D
            rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                self._charuco_corners_list,
                self._charuco_ids_list,
                self._board,
                (w, h),
                None,
                None,
                flags=flags,
                criteria=criteria,
            )
        else:
            raise RuntimeError("OpenCV aruco does not provide calibrateCameraCharuco* API.")

        self.get_logger().info(f"Calibration RMS reprojection error: {float(rms):.6f} px")

        output_path = self.get_parameter("output_yaml_path").get_parameter_value().string_value
        camera_name = self.get_parameter("camera_name").get_parameter_value().string_value
        dist = np.array(dist_coeffs, dtype=np.float64).reshape(-1).tolist()
        k = np.array(camera_matrix, dtype=np.float64).reshape(3, 3).reshape(-1).tolist()

        # 输出为 ROS camera_calibration 常见 YAML 结构
        # 这样后续可直接被本包/其它节点读取并填充到 CameraInfo
        yaml_out = {
            "image_width": int(w),
            "image_height": int(h),
            "camera_name": camera_name,
            "camera_matrix": {"rows": 3, "cols": 3, "data": k},
            "distortion_model": "plumb_bob",
            "distortion_coefficients": {"rows": 1, "cols": int(len(dist)), "data": dist},
            "rectification_matrix": {"rows": 3, "cols": 3, "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]},
            "projection_matrix": {
                "rows": 3,
                "cols": 4,
                "data": [
                    float(camera_matrix[0, 0]),
                    0.0,
                    float(camera_matrix[0, 2]),
                    0.0,
                    0.0,
                    float(camera_matrix[1, 1]),
                    float(camera_matrix[1, 2]),
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
            },
            "spear_vision": {
                "rms_reprojection_px": float(rms),
                "num_samples": int(len(self._charuco_corners_list)),
            },
        }

        save_yaml(output_path, yaml_out)
        self.get_logger().info(f"Saved camera calibration YAML: {output_path}")
        # 打印关键内参，便于你在终端记录/核对
        self.get_logger().info(f"Camera matrix K:\n{np.array(camera_matrix, dtype=np.float64)}")
        self.get_logger().info(f"Distortion coeffs D: {np.array(dist_coeffs, dtype=np.float64).reshape(-1).tolist()}")
        self.get_logger().info(
            f"Image size: {int(w)}x{int(h)}, samples={int(len(self._charuco_corners_list))}"
        )

    def _start_calibration_async(self) -> None:
        # 避免重复触发
        if self._calibrating or self._calibrated:
            return
        self._calibrating = True
        self.get_logger().info("Calibration started (async). Please wait...")

        def _worker() -> None:
            try:
                self._run_calibration_and_save()
                self._calibrated = True
                if bool(self.get_parameter("auto_exit_after_calibration").value):
                    self._request_shutdown()
            except Exception as exc:
                self.get_logger().error(f"Calibration failed: {exc}")
            finally:
                self._calibrating = False

        self._calib_thread = threading.Thread(target=_worker, daemon=True)
        self._calib_thread.start()

    def _request_shutdown(self) -> None:
        # 只请求一次，避免多次触发
        if self._shutdown_requested:
            return
        self._shutdown_requested = True

        def _do_shutdown():
            self.get_logger().info("Calibration done. Shutting down node.")
            try:
                if bool(self.get_parameter("show_opencv_window").value):
                    name = str(self.get_parameter("opencv_window_name").value) or "charuco_calibration"
                    try:
                        cv2.destroyWindow(name)
                    except Exception:
                        cv2.destroyAllWindows()
                self.destroy_node()
            finally:
                rclpy.shutdown()

        # 用一个短定时器延迟，确保日志先刷新
        self.create_timer(0.2, _do_shutdown)
        # 额外写入“最近一次标定结果路径”，便于后续 PnP 节点自动复用，无需手动传参
        try:
            write_last_calibration_path(output_path)
            self.get_logger().info("Updated last calibration pointer for auto-loading by PnP nodes.")
        except Exception as exc:
            self.get_logger().warn(f"Failed to write last calibration pointer: {exc}")


def main() -> None:
    rclpy.init()
    node = CharucoCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
