"""
双 ChArUco 板（大板 + 小 5x5 板）联合位姿估计节点（spear_vision）

动机（对应你的工艺流程）：
1) 大板（primary）用于提供“更稳、更准”的空间基准，但在正对相机时 PnP 容易退化/不稳；
   因此你会让大板倾斜 5~10 度来改善几何条件。
2) 小板（secondary，5x5 ChArUco）贴在矛头同一平面上，用于替代“单个小 ArUco”：
   - 多角点统计更稳、抗模糊/反光更强；
   - 即使部分遮挡，也更容易获得可用的角点/marker 来求解位姿。
3) 同一帧同时看到 primary+secondary 时，可实时计算 primary_T_secondary（两板之间外参）。
   同时，你还可以把这个外参保存成 YAML，后续 secondary 暂时看不到时也能用：
     camera_T_secondary = camera_T_primary * primary_T_secondary
4) 最终把“矛头 tip”作为 secondary 坐标系下的固定偏移（tip_offset_m / tip_rpy_deg）进行组合，
   得到 camera_T_tip，实现相机→矛头位姿输出。

坐标约定：
- 所有 Pose/TF 输出都是 camera_T_*（父=相机，子=board/tip）。
- primary_to_secondary 话题输出的是 primary_T_secondary（父=primary，子=secondary）。
- 物理单位统一为米（m）。

强烈建议：
- primary 与 secondary 的 marker ID 区间不要重叠（通过 ids_start 错开），否则会互相误匹配导致跳变。
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
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, Float64, String
from tf2_ros import TransformBroadcaster

from spear_vision.core.board_pose_estimator import BoardPoseEstimator, BoardSpec, GateSpec, MethodNames, PoseEstimate
from spear_vision.core.extrinsic_calibrator import ExtrinsicCalibrator
from spear_vision.core.pose_filter import PoseLowPassFilter
from spear_vision.utils.camera_intrinsics import CameraIntrinsics, intrinsics_from_camera_info, intrinsics_from_yaml
from spear_vision.utils.opencv_aruco import (
    MarkerDetection,
    board_marker_id_set,
    create_charuco_board,
    create_detector_parameters,
    detect_markers,
    draw_marker_corners,
    draw_marker_ids,
    filter_markers_for_board,
    interpolate_charuco_corners,
    refine_markers,
    resolve_dictionary,
)
from spear_vision.utils.ros_conversions import rt_to_pose_stamped, rt_to_transform_stamped
from spear_vision.utils.tf_utils import Rt, compose_rt, invert_rt, rmat_to_rpy_deg, rodrigues_to_matrix, rpy_deg_to_rvec, rvec_tvec_to_rt
from spear_vision.utils.runtime_checks import warn_opencv_environment
from spear_vision.utils.calibration_store import read_last_calibration_path
from spear_vision.utils.yaml_io import load_yaml, save_yaml


class SpearTipNode(Node):
    def __init__(self, node_name: str = "spear_tip_node") -> None:
        super().__init__(node_name)
        warn_opencv_environment(self.get_logger())

        # --- 参数区：支持通过 spear_tip.yaml 一键配置 ---
        self.declare_parameter("config_path", "")
        self.declare_parameter("camera_calibration_yaml", "")
        self.declare_parameter("prefer_camera_info", True)

        # 节点工作模式：
        # - calibrate：标定阶段（画面同时包含大板+小板），累积多帧估计 primary_T_secondary，并写入配置文件
        # - run：运行阶段（只检测大板），使用已固化的 primary_T_secondary 推算小板/矛头位姿
        self.declare_parameter("mode", "run")

        # --- 外参固化（primary_T_secondary）多帧标定相关参数 ---
        # 说明：
        # - 标定阶段我们会在“每次两块板都成功出位姿”的前提下采样；
        # - 采样数量越多，均值越稳（更接近你要的“亚毫米稳定”）；
        # - sample_stride 用于降低相邻帧强相关（例如 2 表示每 2 帧取 1 帧样本）。
        # 外参标定目标样本数（达到后自动固化）
        self.declare_parameter("calib_required_samples", 100)
        # 采样步长：每 N 帧采 1 帧样本
        # 你现场反馈 20 帧太慢，这里默认改为 2（约 10Hz 采样），依靠 gating + confidence 控制坏样本。
        self.declare_parameter("calib_sample_stride", 2)
        self.declare_parameter("calib_min_primary_confidence", 0.6)
        self.declare_parameter("calib_min_secondary_confidence", 0.6)
        self.declare_parameter("calib_outlier_translation_m", 0.002)  # 离群样本平移阈值（m），默认 2mm
        self.declare_parameter("calib_outlier_rotation_deg", 2.0)  # 离群样本旋转阈值（deg）
        self.declare_parameter("calib_auto_finalize", True)
        self.declare_parameter("calib_save_to_config", True)
        self.declare_parameter("calib_output_config_yaml", "")  # 为空则覆盖 config_path 指向的文件
        self.declare_parameter("calib_auto_switch_to_run", True)
        # 标定完成后自动退出（用于“标定完就退出”的流程）
        self.declare_parameter("calib_auto_exit", True)
        # 触发型参数：需要用户从 false->true 切换才会再次触发
        self.declare_parameter("calib_reset_samples", False)
        self.declare_parameter("finalize_calibration_now", False)

        self.declare_parameter("image_topic", "/hik_camera/image_raw")
        self.declare_parameter("camera_info_topic", "")

        # 输出坐标系命名（TF child_frame_id / PoseStamped frame_id）
        self.declare_parameter("camera_frame", "")
        self.declare_parameter("primary_frame", "spear_board_frame")
        self.declare_parameter("secondary_frame", "spear_tip_charuco_frame")
        self.declare_parameter("tip_frame", "spear_tip_frame")

        self.declare_parameter("publish_tf", True)
        self.declare_parameter("publish_debug_image", True)
        # 直接弹出 OpenCV 视野窗口（不依赖 rviz/rqt_image_view）
        self.declare_parameter("show_opencv_window", True)
        self.declare_parameter("opencv_window_name", "spear_tip_view")
        self.declare_parameter("smoothing_alpha", 0.8)
        # 旋转平滑方式：默认 rvec 线性插值（保持原行为），可选 "quat"
        self.declare_parameter("smoothing_rotation_mode", "rvec")

        # PnP 方法：默认 prefer=IPPE（对近平面更合适），失败再 fallback=ITERATIVE
        self.declare_parameter("pnp_prefer", "SOLVEPNP_IPPE")
        self.declare_parameter("pnp_fallback", "SOLVEPNP_ITERATIVE")
        self.declare_parameter("pnp_refine_lm", True)

        # 保存/加载 primary->secondary 外参（用于 secondary 丢失时的间接推算）
        self.declare_parameter("save_primary_to_secondary_yaml", "")
        self.declare_parameter("save_primary_to_secondary_now", False)
        self.declare_parameter("load_primary_to_secondary_yaml", "")

        # tip 相对 secondary board 的固定偏移（同一平面情况下通常 dz=0）
        # 你原先“2D 几何加减”得到的偏移量，可以直接填到 tip_offset_m。
        self.declare_parameter("tip_offset_m", [0.0, 0.0, 0.0])
        self.declare_parameter("tip_rpy_deg", [0.0, 0.0, 0.0])

        self._bridge = CvBridge()
        self._intrinsics: Optional[CameraIntrinsics] = None
        # 固化外参：primary_T_secondary（标定阶段估计后写入配置文件；运行阶段用它推算小板/矛头）
        self._primary_to_secondary_rt: Optional[Rt] = None

        # 位姿估计/平滑/标定的“核心逻辑模块”（ROS 无关）
        self._pose_estimator = BoardPoseEstimator(
            method_names=MethodNames(
                charuco_near_border="near_border",
                charuco_insufficient="insufficient_corners",
                aruco_none="too_few_markers",
                aruco_too_few="too_few_markers",
                aruco_no_matches="no_matches",
                aruco_near_border="near_border",
                pnp_failed="pnp_failed",
                reproj_gate="reproj_gate",
                pnp_ok="pnp_ok",
            )
        )
        self._filter_primary = PoseLowPassFilter()
        self._filter_secondary = PoseLowPassFilter()
        self._calibrator = ExtrinsicCalibrator()

        # 标定阶段用到的状态量
        self._frame_index: int = 0
        self._shutdown_requested: bool = False
        # debug 可视化辅助：记录“本帧检测到了多少 marker/ID”，用于快速判断是“检测失败”还是“配置过滤错”
        self._dbg_base_markers: int = 0
        self._dbg_base_ids_preview: str = ""
        self._dbg_primary_markers: int = 0
        self._dbg_primary_ids_preview: str = ""
        self._dbg_secondary_markers: int = 0
        self._dbg_secondary_ids_preview: str = ""

        # 1) 从 YAML 解析两块板配置、门控阈值、tip 偏移等
        # 2) 创建 OpenCV board 对象并检查 ID 是否重叠
        # 3) 加载相机内参（YAML 可选；也可运行时从 CameraInfo 覆盖）
        # 4) 初始化 ROS 通信
        self._primary_spec, self._secondary_spec, self._gate_primary, self._gate_secondary = self._load_config()
        self._init_boards()
        self._load_intrinsics_if_any()
        self._load_primary_to_secondary_if_any()
        self._init_ros()

        self._param_cb = self.add_on_set_parameters_callback(self._on_params)

    def _load_config(self) -> tuple[BoardSpec, BoardSpec, GateSpec, GateSpec]:
        # 读取 spear_tip.yaml（若为空则使用默认参数）
        cfg_path = self.get_parameter("config_path").get_parameter_value().string_value
        self._loaded_config_path = str(cfg_path or "")
        data: dict[str, Any] = {}
        if cfg_path:
            data = load_yaml(cfg_path)
            self.get_logger().info(f"Loaded config: {cfg_path}")

        def _get(d: dict[str, Any], key: str, default):
            return d.get(key, default) if isinstance(d, dict) else default

        # --- 1) 解析“模式/标定策略/已固化外参” ---
        # 这些字段不会影响 board 的 OpenCV 构造，但会影响节点运行流程（校准 or 运行）。
        mode = _get(data, "mode", None)
        if mode is not None:
            self.set_parameters([rclpy.parameter.Parameter("mode", value=str(mode))])

        calib = _get(data, "calibration", {})
        if isinstance(calib, dict):
            # calibration.* -> ROS 参数
            mapping = (
                ("required_samples", "calib_required_samples", int),
                ("sample_stride", "calib_sample_stride", int),
                ("min_primary_confidence", "calib_min_primary_confidence", float),
                ("min_secondary_confidence", "calib_min_secondary_confidence", float),
                ("outlier_translation_m", "calib_outlier_translation_m", float),
                ("outlier_rotation_deg", "calib_outlier_rotation_deg", float),
                ("auto_finalize", "calib_auto_finalize", bool),
                ("save_to_config", "calib_save_to_config", bool),
                ("output_config_yaml", "calib_output_config_yaml", str),
                ("auto_switch_to_run", "calib_auto_switch_to_run", bool),
                ("auto_exit", "calib_auto_exit", bool),
            )
            for src_key, param_name, cast in mapping:
                if src_key in calib:
                    try:
                        self.set_parameters([rclpy.parameter.Parameter(param_name, value=cast(calib[src_key]))])
                    except Exception:
                        self.get_logger().warn(f"Invalid calibration.{src_key}: {calib[src_key]!r}")

        extr = _get(data, "primary_to_secondary", {})
        if isinstance(extr, dict):
            t = extr.get("translation_m")
            rpy = extr.get("rpy_deg")
            if isinstance(t, (list, tuple)) and len(t) == 3 and isinstance(rpy, (list, tuple)) and len(rpy) == 3:
                # 从 config 直接读取已固化外参（如果同时又设置了 load_primary_to_secondary_yaml，则后者会覆盖这里）
                rvec = rpy_deg_to_rvec(float(rpy[0]), float(rpy[1]), float(rpy[2]))
                tvec = np.array([[float(t[0])], [float(t[1])], [float(t[2])]], dtype=np.float64)
                self._primary_to_secondary_rt = Rt(rvec=rvec, tvec=tvec)
                self.get_logger().info("Loaded primary->secondary extrinsic from config (primary_to_secondary.*)")

        # --- 2) 解析 topics/frames/pnp 等通用配置 ---
        topics = _get(data, "topics", {})
        if isinstance(topics, dict):
            if "image" in topics:
                self.set_parameters([rclpy.parameter.Parameter("image_topic", value=str(topics["image"]))])
            if "camera_info" in topics:
                self.set_parameters([rclpy.parameter.Parameter("camera_info_topic", value=str(topics["camera_info"]))])

        frames = _get(data, "frames", {})
        if isinstance(frames, dict):
            for k, param_name in (
                ("camera_frame", "camera_frame"),
                ("primary_frame", "primary_frame"),
                ("secondary_frame", "secondary_frame"),
                ("tip_frame", "tip_frame"),
            ):
                if k in frames:
                    self.set_parameters([rclpy.parameter.Parameter(param_name, value=str(frames[k]))])

        pnp = _get(data, "pnp", {})
        if isinstance(pnp, dict):
            if "prefer" in pnp:
                self.set_parameters([rclpy.parameter.Parameter("pnp_prefer", value=str(pnp["prefer"]))])
            if "fallback" in pnp:
                self.set_parameters([rclpy.parameter.Parameter("pnp_fallback", value=str(pnp["fallback"]))])

        primary_cfg = _get(data, "primary_charuco", _get(data, "charuco", {}))
        secondary_cfg = _get(data, "secondary_charuco", {})

        # primary：大板（倾斜放置用于稳定位姿）
        primary = BoardSpec(
            name="primary",
            frame=str(self.get_parameter("primary_frame").value),
            squares_x=int(_get(primary_cfg, "squares_x", 5)),
            squares_y=int(_get(primary_cfg, "squares_y", 5)),
            square_length_m=float(_get(primary_cfg, "square_length_m", 0.03)),
            marker_length_m=float(_get(primary_cfg, "marker_length_m", 0.024)),
            dictionary=str(_get(primary_cfg, "dictionary", "DICT_6X6_250")),
            ids_start=int(_get(primary_cfg, "ids_start", 0)),
            min_charuco_corners=int(_get(_get(data, "gating_primary", _get(data, "gating", {})), "min_charuco_corners", 12)),
            fallback_enable=bool(_get(_get(data, "aruco_board_fallback", {}), "enable", True)),
            fallback_min_markers=int(_get(_get(data, "aruco_board_fallback", {}), "min_markers", 1)),
            use_refine_detected_markers=bool(_get(_get(data, "aruco_board_fallback", {}), "use_refine_detected_markers", True)),
        )

        # secondary：小板（默认 5x5），建议 ids_start 与大板错开（避免 marker ID 冲突）
        # 小板尺寸请按你实际打印物理尺寸填写，否则 PnP 的 tvec 尺度会错。
        secondary = BoardSpec(
            name="secondary",
            frame=str(self.get_parameter("secondary_frame").value),
            squares_x=int(_get(secondary_cfg, "squares_x", 5)),
            squares_y=int(_get(secondary_cfg, "squares_y", 5)),
            square_length_m=float(_get(secondary_cfg, "square_length_m", 0.01)),
            marker_length_m=float(_get(secondary_cfg, "marker_length_m", 0.008)),
            dictionary=str(_get(secondary_cfg, "dictionary", primary.dictionary)),
            ids_start=int(_get(secondary_cfg, "ids_start", 100)),
            min_charuco_corners=int(_get(_get(data, "gating_secondary", {}), "min_charuco_corners", 6)),
            fallback_enable=bool(_get(_get(data, "secondary_aruco_fallback", {}), "enable", True)),
            fallback_min_markers=int(_get(_get(data, "secondary_aruco_fallback", {}), "min_markers", 1)),
            use_refine_detected_markers=bool(_get(_get(data, "secondary_aruco_fallback", {}), "use_refine_detected_markers", True)),
        )

        gate_p = _get(data, "gating_primary", _get(data, "gating", {}))
        gate_s = _get(data, "gating_secondary", {})
        # 门控阈值：越严格越稳，但会增加“无输出帧”的概率，需要按工况折中
        gate_primary = GateSpec(
            max_mean_reproj_px=float(_get(gate_p, "max_mean_reproj_px", 0.6)),
            max_max_reproj_px=float(_get(gate_p, "max_max_reproj_px", 1.5)),
            min_border_px=int(_get(gate_p, "min_border_px", 20)),
        )
        gate_secondary = GateSpec(
            max_mean_reproj_px=float(_get(gate_s, "max_mean_reproj_px", 0.8)),
            max_max_reproj_px=float(_get(gate_s, "max_max_reproj_px", 2.0)),
            min_border_px=int(_get(gate_s, "min_border_px", 10)),
        )

        tip_cfg = _get(data, "tip", {})
        if isinstance(tip_cfg, dict):
            ofs = tip_cfg.get("offset_m")
            rpy = tip_cfg.get("rpy_deg")
            if isinstance(ofs, (list, tuple)) and len(ofs) == 3:
                self.set_parameters([rclpy.parameter.Parameter("tip_offset_m", value=[float(x) for x in ofs])])
            if isinstance(rpy, (list, tuple)) and len(rpy) == 3:
                self.set_parameters([rclpy.parameter.Parameter("tip_rpy_deg", value=[float(x) for x in rpy])])

        return primary, secondary, gate_primary, gate_secondary

    def _init_boards(self) -> None:
        # ArUco/ChArUco 检测器参数（阈值/角点细化等）
        self._detector_params = create_detector_parameters()

        dict_primary = resolve_dictionary(self._primary_spec.dictionary)
        dict_secondary = resolve_dictionary(self._secondary_spec.dictionary)
        self._dict_primary = dict_primary
        self._dict_secondary = dict_secondary

        self._board_primary = create_charuco_board(
            self._primary_spec.squares_x,
            self._primary_spec.squares_y,
            self._primary_spec.square_length_m,
            self._primary_spec.marker_length_m,
            dict_primary,
            ids_start=self._primary_spec.ids_start,
        )
        self._board_secondary = create_charuco_board(
            self._secondary_spec.squares_x,
            self._secondary_spec.squares_y,
            self._secondary_spec.square_length_m,
            self._secondary_spec.marker_length_m,
            dict_secondary,
            ids_start=self._secondary_spec.ids_start,
        )

        self._primary_ids = board_marker_id_set(self._board_primary)
        self._secondary_ids = board_marker_id_set(self._board_secondary)
        overlap = self._primary_ids.intersection(self._secondary_ids)
        if overlap:
            # 如果两块板 ID 重叠，filter_markers_for_board 仍能区分“属于哪个 board”，
            # 但在 detectMarkers → refine/插值过程中会更容易被误匹配，建议从源头错开。
            self.get_logger().warn(
                f"Primary/secondary marker IDs overlap ({len(overlap)} ids). "
                "请用 ids_start 将两块板的 ID 区间错开，否则会导致误匹配。"
            )

        self.get_logger().info(
            "Primary board: %dx%d square=%.3fm marker=%.3fm dict=%s ids_start=%d"
            % (
                self._primary_spec.squares_x,
                self._primary_spec.squares_y,
                self._primary_spec.square_length_m,
                self._primary_spec.marker_length_m,
                self._primary_spec.dictionary,
                self._primary_spec.ids_start,
            )
        )
        self.get_logger().info(
            "Secondary board: %dx%d square=%.3fm marker=%.3fm dict=%s ids_start=%d"
            % (
                self._secondary_spec.squares_x,
                self._secondary_spec.squares_y,
                self._secondary_spec.square_length_m,
                self._secondary_spec.marker_length_m,
                self._secondary_spec.dictionary,
                self._secondary_spec.ids_start,
            )
        )

    def _load_intrinsics_if_any(self) -> None:
        # 从 YAML 加载内参（推荐做法）
        path = self.get_parameter("camera_calibration_yaml").get_parameter_value().string_value
        if not path:
            # 你提出的“标定一次后自动复用”：若参数未显式指定，则尝试读取最近一次标定结果路径
            auto_path = read_last_calibration_path()
            if auto_path:
                path = auto_path
                self.get_logger().info(f"camera_calibration_yaml is empty; auto-loading from last calibration: {path}")
            else:
                # 兜底：若指针文件不存在/失效，则尝试从工作空间固定位置加载（满足你“都放在工作空间里”的需求）
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

    def _load_primary_to_secondary_if_any(self) -> None:
        # 读取 primary->secondary 外参：
        # 当 secondary 偶发丢失时，可以由 camera_T_primary 推导出 camera_T_secondary。
        path = self.get_parameter("load_primary_to_secondary_yaml").get_parameter_value().string_value
        if not path:
            return
        data = load_yaml(path)
        t = data.get("translation_m")
        rpy = data.get("rpy_deg")
        if not (isinstance(t, (list, tuple)) and len(t) == 3):
            raise ValueError("primary_to_secondary YAML missing translation_m[3]")
        if not (isinstance(rpy, (list, tuple)) and len(rpy) == 3):
            raise ValueError("primary_to_secondary YAML missing rpy_deg[3]")
        rvec = rpy_deg_to_rvec(rpy[0], rpy[1], rpy[2])
        tvec = np.array([[float(t[0])], [float(t[1])], [float(t[2])]], dtype=np.float64)
        self._primary_to_secondary_rt = Rt(rvec=rvec, tvec=tvec)
        self.get_logger().info(f"Loaded primary->secondary extrinsic: {path}")

    def _init_ros(self) -> None:
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        derived_info = image_topic.rstrip("/") + "/camera_info"
        if not camera_info_topic or camera_info_topic.strip() in ("/hik_camera/camera_info",):
            # hik_camera_ros2 使用 image_transport::CameraPublisher，默认 camera_info = image_topic + "/camera_info"
            camera_info_topic = derived_info
        elif camera_info_topic != derived_info:
            self.get_logger().warn(
                f"camera_info_topic='{camera_info_topic}' does not match '{derived_info}'. "
                "If you are using hik_camera_ros2, the default is image_topic + '/camera_info'."
            )

        # sensor_data QoS：保证低延迟视觉处理
        self._sub_img = self.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)
        self._sub_info = self.create_subscription(
            CameraInfo, camera_info_topic, self._on_camera_info, qos_profile_sensor_data
        )
        self.get_logger().info(f"Subscribing image: {image_topic}")
        self.get_logger().info(f"Subscribing camera_info: {camera_info_topic}")

        self._pub_primary = self.create_publisher(PoseStamped, "~/primary_pose", 10)
        self._pub_secondary = self.create_publisher(PoseStamped, "~/secondary_pose", 10)
        self._pub_tip = self.create_publisher(PoseStamped, "~/tip_pose", 10)
        self._pub_primary_to_secondary = self.create_publisher(PoseStamped, "~/primary_to_secondary", 10)

        self._debug_pub = self.create_publisher(Image, "~/debug_image", qos_profile_sensor_data)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._err_primary = self.create_publisher(Float64, "~/primary_reproj_mean_px", 10)
        self._err_secondary = self.create_publisher(Float64, "~/secondary_reproj_mean_px", 10)
        self._confidence_primary = self.create_publisher(Float32, "~/primary_confidence", 10)
        self._confidence_secondary = self.create_publisher(Float32, "~/secondary_confidence", 10)
        self._method = self.create_publisher(String, "~/method", 10)

    def _on_params(self, params: list[rclpy.parameter.Parameter]) -> SetParametersResult:
        # 触发型参数说明：
        # - 这些参数一般由 CLI/GUI 动态设置；
        # - 需要“从 false -> true 的切换”才会再次触发（否则值不变不会触发回调）。
        do_save = False
        do_reset = False
        do_finalize = False
        for p in params:
            if p.name == "save_primary_to_secondary_now" and bool(p.value):
                do_save = True
            if p.name == "calib_reset_samples" and bool(p.value):
                do_reset = True
            if p.name == "finalize_calibration_now" and bool(p.value):
                do_finalize = True

        if do_reset:
            # 清空标定样本，并清除已固化外参（便于重新标定）
            self._calibrator.reset()
            self._primary_to_secondary_rt = None
            self.get_logger().warn("Calibration samples cleared; primary_T_secondary reset.")

        if do_finalize:
            try:
                # 手动 finalize：保持旧行为（只要求 >=3 个样本即可）
                self._finalize_calibration_from_samples(required_samples=3)
            except Exception as exc:
                return SetParametersResult(successful=False, reason=str(exc))

        if do_save:
            try:
                self._save_primary_to_secondary_now()
            except Exception as exc:
                return SetParametersResult(successful=False, reason=str(exc))

        return SetParametersResult(successful=True)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        # 若驱动发布有效内参，可热更新
        if not bool(self.get_parameter("prefer_camera_info").value):
            return
        intr = intrinsics_from_camera_info(msg)
        if intr is None or not intr.is_valid():
            return
        self._intrinsics = intr

    def _on_image(self, msg: Image) -> None:
        # 帧计数：用于标定阶段 sample_stride 采样
        self._frame_index += 1

        # 双板节点同样只需要灰度图；输出 debug_image 时再叠加到 BGR
        gray = self._bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

        # --- 模式分流：calibrate（双板） vs run（只检测大板） ---
        mode = str(self.get_parameter("mode").value).strip().lower()
        if mode not in ("run", "calibrate"):
            mode = "run"
        camera_frame = str(self.get_parameter("camera_frame").value).strip() or msg.header.frame_id
        stamp = msg.header.stamp

        # 没有内参无法做 PnP/外参采样，但仍然可以做 marker/角点可视化，便于你现场定位“为何识别不到”的问题。
        # （你刚遇到的“画面非常流畅、完全不画 marker”的现象，很多时候就是这里没拿到内参导致的。）
        intr_ok = self._intrinsics is not None and self._intrinsics.is_valid()

        if mode == "run":
            # 运行阶段：只检测大板（primary），小板不再出现在画面中，省算力且避免误检干扰
            det_primary = detect_markers(gray, self._dict_primary, self._detector_params)
            self._dbg_base_markers = int(len(det_primary.ids)) if det_primary.ids is not None else 0
            self._dbg_base_ids_preview = (
                ",".join(str(int(x)) for x in np.array(det_primary.ids, dtype=np.int32).reshape(-1).tolist()[:8])
                if det_primary.ids is not None and len(det_primary.ids) > 0
                else ""
            )
            if intr_ok and self._primary_spec.use_refine_detected_markers:
                det_primary = refine_markers(
                    gray,
                    self._board_primary,
                    det_primary,
                    self._intrinsics.camera_matrix,
                    self._intrinsics.dist_coeffs,
                )
            self._dbg_primary_markers = int(len(det_primary.ids)) if det_primary.ids is not None else 0
            self._dbg_primary_ids_preview = (
                ",".join(str(int(x)) for x in np.array(det_primary.ids, dtype=np.int32).reshape(-1).tolist()[:8])
                if det_primary.ids is not None and len(det_primary.ids) > 0
                else ""
            )
            self._dbg_secondary_markers = 0
            self._dbg_secondary_ids_preview = ""

            if not intr_ok:
                # 没内参：只做可视化，不做 PnP/发布（避免输出尺度错误的位姿）
                self._method.publish(String(data="mode=run intrinsics=missing (no PnP)"))
                self._publish_debug(msg, gray, det_primary, None, None, None, None)
                return

            est_primary = self._estimate_with_filter(
                gray,
                self._board_primary,
                det_primary,
                self._primary_spec,
                self._gate_primary,
                self._filter_primary,
            )
            primary_rt = self._publish_pose_and_tf(
                est_primary,
                stamp,
                camera_frame,
                self._primary_spec.frame,
                self._pub_primary,
                self._err_primary,
                self._confidence_primary,
            )

            # tip：运行阶段依赖 “已固化的 primary_T_secondary” 来间接推算
            tip_rt = self._compute_tip_pose(primary_rt, None)
            if tip_rt is not None:
                tip_frame = str(self.get_parameter("tip_frame").value)
                self._pub_tip.publish(rt_to_pose_stamped(tip_rt, stamp, camera_frame))

                if bool(self.get_parameter("publish_tf").value):
                    self._tf_broadcaster.sendTransform(
                        rt_to_transform_stamped(tip_rt, stamp, camera_frame, tip_frame)
                    )

            self._method.publish(String(data=f"mode=run primary={est_primary.used}:{est_primary.method} secondary=skipped"))
            self._publish_debug(msg, gray, det_primary, None, tip_rt, est_primary, None)
            return

        # --- calibrate：双板同时识别，累积多帧估计 primary_T_secondary ---
        # 若两块板用同一个字典，可以只 detectMarkers 一次再按 ID 分流；否则分别 detect
        if self._primary_spec.dictionary == self._secondary_spec.dictionary:
            base_det = detect_markers(gray, self._dict_primary, self._detector_params)
            self._dbg_base_markers = int(len(base_det.ids)) if base_det.ids is not None else 0
            self._dbg_base_ids_preview = (
                ",".join(str(int(x)) for x in np.array(base_det.ids, dtype=np.int32).reshape(-1).tolist()[:8])
                if base_det.ids is not None and len(base_det.ids) > 0
                else ""
            )
            det_primary = MarkerDetection(
                *filter_markers_for_board(base_det.corners, base_det.ids, self._board_primary),
                base_det.rejected,
            )
            det_secondary = MarkerDetection(
                *filter_markers_for_board(base_det.corners, base_det.ids, self._board_secondary),
                base_det.rejected,
            )
        else:
            det_primary = detect_markers(gray, self._dict_primary, self._detector_params)
            det_secondary = detect_markers(gray, self._dict_secondary, self._detector_params)
            self._dbg_base_markers = int(
                (len(det_primary.ids) if det_primary.ids is not None else 0)
                + (len(det_secondary.ids) if det_secondary.ids is not None else 0)
            )
            self._dbg_base_ids_preview = ""

        if intr_ok and self._primary_spec.use_refine_detected_markers:
            det_primary = refine_markers(
                gray,
                self._board_primary,
                det_primary,
                self._intrinsics.camera_matrix,
                self._intrinsics.dist_coeffs,
            )
        if intr_ok and self._secondary_spec.use_refine_detected_markers:
            det_secondary = refine_markers(
                gray,
                self._board_secondary,
                det_secondary,
                self._intrinsics.camera_matrix,
                self._intrinsics.dist_coeffs,
            )
        self._dbg_primary_markers = int(len(det_primary.ids)) if det_primary.ids is not None else 0
        self._dbg_primary_ids_preview = (
            ",".join(str(int(x)) for x in np.array(det_primary.ids, dtype=np.int32).reshape(-1).tolist()[:8])
            if det_primary.ids is not None and len(det_primary.ids) > 0
            else ""
        )
        self._dbg_secondary_markers = int(len(det_secondary.ids)) if det_secondary.ids is not None else 0
        self._dbg_secondary_ids_preview = (
            ",".join(str(int(x)) for x in np.array(det_secondary.ids, dtype=np.int32).reshape(-1).tolist()[:8])
            if det_secondary.ids is not None and len(det_secondary.ids) > 0
            else ""
        )

        if not intr_ok:
            # 没内参：仍然给你看“两块板的 marker/角点是否检测到了”，但不做 PnP/不采样外参
            self._method.publish(String(data="mode=calibrate intrinsics=missing (no PnP/no calib)"))
            self._publish_debug(msg, gray, det_primary, det_secondary, None, None, None)
            return

        est_primary = self._estimate_with_filter(
            gray,
            self._board_primary,
            det_primary,
            self._primary_spec,
            self._gate_primary,
            self._filter_primary,
        )
        est_secondary = self._estimate_with_filter(
            gray,
            self._board_secondary,
            det_secondary,
            self._secondary_spec,
            self._gate_secondary,
            self._filter_secondary,
        )

        primary_rt = self._publish_pose_and_tf(
            est_primary,
            stamp,
            camera_frame,
            self._primary_spec.frame,
            self._pub_primary,
            self._err_primary,
            self._confidence_primary,
        )
        secondary_rt = self._publish_pose_and_tf(
            est_secondary,
            stamp,
            camera_frame,
            self._secondary_spec.frame,
            self._pub_secondary,
            self._err_secondary,
            self._confidence_secondary,
        )

        primary_to_secondary = None
        if primary_rt is not None and secondary_rt is not None:
            # primary_T_secondary = inv(camera_T_primary) * (camera_T_secondary)
            primary_to_secondary = compose_rt(invert_rt(primary_rt), secondary_rt)

            # 发布两板相对位姿（用于现场查看两板的固化关系是否稳定）
            self._pub_primary_to_secondary.publish(rt_to_pose_stamped(primary_to_secondary, stamp, self._primary_spec.frame))

            # 标定采样：把每一帧算出的 primary_T_secondary 纳入样本集合
            self._maybe_collect_calibration_sample(primary_to_secondary, est_primary, est_secondary)

        tip_rt = self._compute_tip_pose(primary_rt, secondary_rt)
        if tip_rt is not None:
            tip_frame = str(self.get_parameter("tip_frame").value)
            self._pub_tip.publish(rt_to_pose_stamped(tip_rt, stamp, camera_frame))

            if bool(self.get_parameter("publish_tf").value):
                self._tf_broadcaster.sendTransform(
                    rt_to_transform_stamped(tip_rt, stamp, camera_frame, tip_frame)
                )

        required = int(self.get_parameter("calib_required_samples").value)
        status = "done" if self._calibrator.finalized else f"{self._calibrator.sample_count}/{required}"
        self._method.publish(
            String(data=f"mode=calibrate calib={status} primary={est_primary.used}:{est_primary.method} secondary={est_secondary.used}:{est_secondary.method}")
        )
        self._publish_debug(msg, gray, det_primary, det_secondary, tip_rt, est_primary, est_secondary)

    def _publish_pose_and_tf(
        self,
        est: PoseEstimate,
        stamp,
        camera_frame: str,
        child_frame: str,
        pub_pose,
        pub_err,
        pub_conf,
    ) -> Optional[Rt]:
        # 发布 PoseStamped + TF，并返回 Rt 供后续组合使用
        if not est.ok or est.rvec is None or est.tvec is None:
            return None

        rt = rvec_tvec_to_rt(est.rvec, est.tvec)
        pub_pose.publish(rt_to_pose_stamped(rt, stamp, camera_frame))

        if est.mean_reproj_px is not None:
            pub_err.publish(Float64(data=float(est.mean_reproj_px)))
        pub_conf.publish(Float32(data=float(est.confidence)))

        if bool(self.get_parameter("publish_tf").value):
            self._tf_broadcaster.sendTransform(rt_to_transform_stamped(rt, stamp, camera_frame, child_frame))

        return rt

    def _estimate_with_filter(
        self,
        gray: np.ndarray,
        board,
        detection: MarkerDetection,
        spec: BoardSpec,
        gate: GateSpec,
        pose_filter: PoseLowPassFilter,
    ) -> PoseEstimate:
        # 估计单板位姿（核心逻辑交给 BoardPoseEstimator），再做平滑滤波
        prefer = str(self.get_parameter("pnp_prefer").value)
        fallback = str(self.get_parameter("pnp_fallback").value)
        refine_lm = bool(self.get_parameter("pnp_refine_lm").value)
        est = self._pose_estimator.estimate(
            gray,
            self._intrinsics,
            board,
            detection,
            spec,
            gate,
            pnp_prefer=prefer,
            pnp_fallback=fallback,
            pnp_refine_lm=refine_lm,
            prior_rt=pose_filter.last,
        )

        if not est.ok or est.rvec is None or est.tvec is None:
            return est

        alpha = float(self.get_parameter("smoothing_alpha").value)
        mode = str(self.get_parameter("smoothing_rotation_mode").value)
        rt = pose_filter.update(rvec_tvec_to_rt(est.rvec, est.tvec), alpha, rotation_mode=mode)
        return replace(est, rvec=rt.rvec, tvec=rt.tvec)

    def _compute_tip_pose(self, primary_rt: Optional[Rt], secondary_rt: Optional[Rt]) -> Optional[Rt]:
        # 优先使用当前帧 secondary board 来推 tip（最直接、误差传播最少）
        if secondary_rt is None:
            # 若 secondary 丢了，但 primary + 已知 primary->secondary 外参存在，则可间接推算
            if primary_rt is None:
                return None
            if self._primary_to_secondary_rt is None:
                return None
            secondary_rt = compose_rt(primary_rt, self._primary_to_secondary_rt)

        tip_offset = list(self.get_parameter("tip_offset_m").value)
        tip_rpy = list(self.get_parameter("tip_rpy_deg").value)
        tip_offset = [float(x) for x in tip_offset]
        tip_rpy = [float(x) for x in tip_rpy]

        tip_rel = Rt(
            rvec=rpy_deg_to_rvec(tip_rpy[0], tip_rpy[1], tip_rpy[2]),
            tvec=np.array([[tip_offset[0]], [tip_offset[1]], [tip_offset[2]]], dtype=np.float64),
        )
        # camera_T_tip = camera_T_secondary * secondary_T_tip
        return compose_rt(secondary_rt, tip_rel)

    def _maybe_collect_calibration_sample(self, primary_to_secondary: Rt, est_primary: PoseEstimate, est_secondary: PoseEstimate) -> None:
        # 标定阶段：把每帧的 primary_T_secondary 作为一个样本加入集合，再做“多帧均值”固化外参
        if self._calibrator.finalized:
            return

        stride = int(self.get_parameter("calib_sample_stride").value)
        min_p = float(self.get_parameter("calib_min_primary_confidence").value)
        min_s = float(self.get_parameter("calib_min_secondary_confidence").value)
        added = self._calibrator.add_sample(
            primary_to_secondary,
            est_primary.confidence,
            est_secondary.confidence,
            frame_index=self._frame_index,
            stride=stride,
            min_conf_p=min_p,
            min_conf_s=min_s,
        )
        if not added:
            return

        required = int(self.get_parameter("calib_required_samples").value)
        n = self._calibrator.sample_count
        if n == 1 or n % 5 == 0:
            self.get_logger().info(f"Calibration samples: {n}/{required}")

        if bool(self.get_parameter("calib_auto_finalize").value) and n >= required:
            self._finalize_calibration_from_samples(required_samples=required)

    def _finalize_calibration_from_samples(self, required_samples: Optional[int] = None) -> None:
        # 用当前累积的样本集合，计算并固化 primary_T_secondary，并（可选）写入配置文件
        if self._calibrator.finalized:
            return

        required = int(required_samples) if required_samples is not None else int(self.get_parameter("calib_required_samples").value)
        outlier_t = float(self.get_parameter("calib_outlier_translation_m").value)
        outlier_r = float(self.get_parameter("calib_outlier_rotation_deg").value)

        finalized, mean_rt, stats = self._calibrator.maybe_finalize(
            required_samples=required,
            outlier_translation_m=outlier_t,
            outlier_rotation_deg=outlier_r,
        )
        if not finalized or mean_rt is None:
            raise RuntimeError(f"Not enough calibration samples: {self._calibrator.sample_count} (need >= {required}).")

        self._primary_to_secondary_rt = mean_rt

        # 打印一下统计量，方便你判断“固化是否足够稳”
        std_t = stats.t_std if stats.t_std is not None else np.zeros(3, dtype=np.float64)
        roll, pitch, yaw = stats.rpy_deg if stats.rpy_deg is not None else (0.0, 0.0, 0.0)
        self.get_logger().info(
            "Calibration finalized: samples=%d, t_mean=[%.4f %.4f %.4f]m, t_std=[%.4f %.4f %.4f]m, rpy=[%.2f %.2f %.2f]deg"
            % (
                stats.num_kept,
                float(mean_rt.tvec[0]),
                float(mean_rt.tvec[1]),
                float(mean_rt.tvec[2]),
                float(std_t[0]),
                float(std_t[1]),
                float(std_t[2]),
                float(roll),
                float(pitch),
                float(yaw),
            )
        )

        if bool(self.get_parameter("calib_save_to_config").value):
            self._write_primary_to_secondary_to_config(mean_rt, num_samples=stats.num_kept)

        if bool(self.get_parameter("calib_auto_switch_to_run").value):
            # 切换到 run：后续就不再检测小板（只看大板），与“临时小板标定后拿掉”的流程一致
            self.set_parameters([rclpy.parameter.Parameter("mode", value="run")])

        # 若要求“标定完成后自动退出”，则在这里触发关机
        if bool(self.get_parameter("calib_auto_exit").value):
            self._request_shutdown()

    def _write_primary_to_secondary_to_config(self, rt: Rt, num_samples: int) -> None:
        # 将 primary_T_secondary 写入 YAML 配置文件（注意：PyYAML safe_dump 会丢失注释）
        out_path = self.get_parameter("calib_output_config_yaml").get_parameter_value().string_value
        if not out_path:
            out_path = self.get_parameter("config_path").get_parameter_value().string_value
        if not out_path:
            raise RuntimeError("No config file path to write (calib_output_config_yaml/config_path are empty).")

        expanded = os.path.expanduser(out_path)
        if os.path.exists(expanded):
            data = load_yaml(expanded)
        else:
            data = {}

        roll, pitch, yaw = rmat_to_rpy_deg(rodrigues_to_matrix(rt.rvec))
        data["primary_to_secondary"] = {
            "translation_m": [float(rt.tvec[0]), float(rt.tvec[1]), float(rt.tvec[2])],
            "rpy_deg": [float(roll), float(pitch), float(yaw)],
            "num_samples": int(num_samples),
            "notes": "primary_T_secondary (auto-generated by spear_tip_node calibration)",
        }
        # 若用户选择“标定后自动切 run”，这里也把文件里的 mode 改成 run，避免下次启动又走标定流程
        if bool(self.get_parameter("calib_auto_switch_to_run").value):
            data["mode"] = "run"

        save_yaml(expanded, data)
        self.get_logger().warn(f"Wrote primary_to_secondary into config YAML (comments may be lost): {expanded}")

    def _save_primary_to_secondary_now(self) -> None:
        # 保存 primary_T_secondary 外参：
        # 建议在两块板同时可见且输出稳定时触发，并尽量让画面里角点数量充足。
        path = self.get_parameter("save_primary_to_secondary_yaml").get_parameter_value().string_value
        if not path:
            raise RuntimeError("save_primary_to_secondary_yaml is empty.")
        if self._filter_primary.last is None or self._filter_secondary.last is None:
            raise RuntimeError("Need both primary and secondary pose available to save extrinsic.")

        rel = compose_rt(invert_rt(self._filter_primary.last), self._filter_secondary.last)
        rmat = rodrigues_to_matrix(rel.rvec)

        # 保存为 RPY（便于人工理解/修改）
        roll, pitch, yaw = rmat_to_rpy_deg(rmat)

        out = {
            "translation_m": [float(rel.tvec[0]), float(rel.tvec[1]), float(rel.tvec[2])],
            "rpy_deg": [float(roll), float(pitch), float(yaw)],
            "notes": "primary_T_secondary (computed from latest poses)",
        }
        save_yaml(path, out)
        self.get_logger().info(f"Saved primary->secondary extrinsic: {path}")

    def _publish_debug(
        self,
        msg: Image,
        gray: np.ndarray,
        det_primary: Optional[MarkerDetection],
        det_secondary: Optional[MarkerDetection],
        tip_rt: Optional[Rt],
        est_primary: Optional[PoseEstimate] = None,
        est_secondary: Optional[PoseEstimate] = None,
    ) -> None:
        # debug_image：叠加 marker 检测、三套坐标轴（primary/secondary/tip），用于现场调参定位问题
        if not bool(self.get_parameter("publish_debug_image").value):
            return
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        intr_ok = self._intrinsics is not None and self._intrinsics.is_valid()

        if det_primary is not None and det_primary.ids is not None and len(det_primary.ids) > 0:
            try:
                cv2.aruco.drawDetectedMarkers(bgr, det_primary.corners, det_primary.ids)
            except Exception:
                pass
            try:
                draw_marker_ids(bgr, det_primary.corners, det_primary.ids)
                draw_marker_corners(bgr, det_primary.corners)
            except Exception:
                pass
        if det_secondary is not None and det_secondary.ids is not None and len(det_secondary.ids) > 0:
            try:
                cv2.aruco.drawDetectedMarkers(bgr, det_secondary.corners, det_secondary.ids)
            except Exception:
                pass
            try:
                draw_marker_ids(bgr, det_secondary.corners, det_secondary.ids)
                draw_marker_corners(bgr, det_secondary.corners)
            except Exception:
                pass

        # 可视化 ChArUco 角点（便于判断“角点数量是否足够/是否稳定”）
        num_p = 0
        num_s = 0
        try:
            if det_primary is not None:
                num_p, cc_p, ci_p = interpolate_charuco_corners(gray, self._board_primary, det_primary.corners, det_primary.ids)
                if cc_p is not None and ci_p is not None and int(num_p) > 0:
                    cv2.aruco.drawDetectedCornersCharuco(bgr, cc_p, ci_p, (0, 0, 255))
        except Exception:
            pass
        try:
            if det_secondary is not None:
                num_s, cc_s, ci_s = interpolate_charuco_corners(gray, self._board_secondary, det_secondary.corners, det_secondary.ids)
                if cc_s is not None and ci_s is not None and int(num_s) > 0:
                    cv2.aruco.drawDetectedCornersCharuco(bgr, cc_s, ci_s, (255, 0, 0))
        except Exception:
                pass

        if est_primary is not None and est_primary.ok and est_primary.rvec is not None and est_primary.tvec is not None:
            if intr_ok:
                cv2.drawFrameAxes(
                    bgr,
                    self._intrinsics.camera_matrix,
                    self._intrinsics.dist_coeffs,
                    est_primary.rvec,
                    est_primary.tvec,
                    0.05,
                )
        if est_secondary is not None and est_secondary.ok and est_secondary.rvec is not None and est_secondary.tvec is not None:
            if intr_ok:
                cv2.drawFrameAxes(
                    bgr,
                    self._intrinsics.camera_matrix,
                    self._intrinsics.dist_coeffs,
                    est_secondary.rvec,
                    est_secondary.tvec,
                    0.03,
                )
        if tip_rt is not None:
            if intr_ok:
                cv2.drawFrameAxes(
                    bgr,
                    self._intrinsics.camera_matrix,
                    self._intrinsics.dist_coeffs,
                    tip_rt.rvec,
                    tip_rt.tvec,
                    0.03,
                )

        # 标定阶段叠加样本进度信息
        try:
            mode = str(self.get_parameter("mode").value).strip().lower()
            if mode == "calibrate":
                required = int(self.get_parameter("calib_required_samples").value)
                accepted = int(self._calibrator.sample_count)
                stride = int(self.get_parameter("calib_sample_stride").value)
                mod = int(self._frame_index % max(stride, 1))
                cv2.putText(
                    bgr,
                    f"calib_samples={accepted}/{required} stride={stride} mod={mod} primary_charuco={int(num_p)} secondary_charuco={int(num_s)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    bgr,
                    f"markers: base={int(self._dbg_base_markers)} primary={int(self._dbg_primary_markers)} secondary={int(self._dbg_secondary_markers)}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                if self._dbg_base_ids_preview:
                    cv2.putText(
                        bgr,
                        f"base_ids: {self._dbg_base_ids_preview}",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
                if self._dbg_primary_ids_preview:
                    cv2.putText(
                        bgr,
                        f"primary_ids: {self._dbg_primary_ids_preview}",
                        (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
                if self._dbg_secondary_ids_preview:
                    cv2.putText(
                        bgr,
                        f"secondary_ids: {self._dbg_secondary_ids_preview}",
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                # 同屏给出“为什么没有采样”的关键诊断：
                # - 必须 primary/secondary 都 PnP 成功且通过 reproj/border gating 才会加入样本；
                # - 还要满足 min_conf 与 sample_stride 才会计数 +1。
                if est_primary is not None:
                    mp = "-" if est_primary.mean_reproj_px is None else f"{float(est_primary.mean_reproj_px):.2f}"
                    cp = f"{float(est_primary.confidence):.2f}"
                    cv2.putText(
                        bgr,
                        f"primary: ok={int(bool(est_primary.ok))} used={est_primary.used} method={est_primary.method} mean={mp} conf={cp}",
                        (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                if est_secondary is not None:
                    ms = "-" if est_secondary.mean_reproj_px is None else f"{float(est_secondary.mean_reproj_px):.2f}"
                    cs = f"{float(est_secondary.confidence):.2f}"
                    cv2.putText(
                        bgr,
                        f"secondary: ok={int(bool(est_secondary.ok))} used={est_secondary.used} method={est_secondary.method} mean={ms} conf={cs}",
                        (10, 175),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
        except Exception:
            pass

        # 内参缺失提示（很多“为什么不识别/不出位姿”的根因都在这里）
        if not intr_ok:
            hint = "NO INTRINSICS: run charuco_calibration -> camera.yaml, then set camera_calibration_yaml"
            cv2.putText(bgr, hint, (10, bgr.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        self._debug_pub.publish(self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8"))

        # 可选：直接弹出 OpenCV 窗口（默认启用）
        if bool(self.get_parameter("show_opencv_window").value):
            try:
                name = str(self.get_parameter("opencv_window_name").value) or "spear_tip_view"
                cv2.imshow(name, bgr)
                cv2.waitKey(1)
            except Exception as exc:
                self.get_logger().warn(f"OpenCV window disabled (GUI not available): {exc}")
                self.set_parameters([rclpy.parameter.Parameter("show_opencv_window", value=False)])

    def _request_shutdown(self) -> None:
        # 避免重复触发
        if self._shutdown_requested:
            return
        self._shutdown_requested = True

        def _do_shutdown():
            self.get_logger().info("Calibration done. Shutting down node.")
            try:
                if bool(self.get_parameter("show_opencv_window").value):
                    name = str(self.get_parameter("opencv_window_name").value) or "spear_tip_view"
                    try:
                        cv2.destroyWindow(name)
                    except Exception:
                        cv2.destroyAllWindows()
                self.destroy_node()
            finally:
                rclpy.shutdown()

        # 延迟一小段时间，确保日志先刷出来
        self.create_timer(0.2, _do_shutdown)


def main() -> None:
    rclpy.init()
    node = SpearTipNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
