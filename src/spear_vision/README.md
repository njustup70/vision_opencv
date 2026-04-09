# spear_vision

`spear_vision` 是一个基于 ROS 2 + OpenCV ArUco/ChArUco 的 Python 视觉包，围绕同一套棋盘检测与 PnP 求解能力，拆出了几条互相关联但可以独立使用的工作链路：

1. `charuco_calib`：相机内参标定，输出 `camera.yaml`。
2. `board_pose`：单块大板位姿估计。
3. `small_board_pose`：小板位姿估计，并输出 `left/up` 偏移。
4. `small_board_serial_bridge`：把小板偏移通过 UART 发给 STM32。
5. `spear_tip_calib` / `spear_tip_run`：双板外参标定与运行期矛尖位姿估计。

这个包的代码结构整体比较清晰：几何求解、滤波、OpenCV 兼容层尽量下沉到 `core/` 与 `utils/`，ROS 节点层主要负责参数、话题、TF、可视化和任务流程。它不是“只会识别一个码板”的最小示例，而是一套面向现场工况的视觉基础设施，重点解决以下问题：

- ChArUco 与 ArUco Board 混合求解。
- 平面目标在 PnP 中的二义性与跳变。
- 多帧外参标定的离群点剔除。
- 识别失败时的可调试性。
- OpenCV 版本差异带来的 API 兼容问题。
- 标定结果的自动复用。

下文会从目录结构、核心算法、各节点职责、配置文件含义、推荐工作流、已知约束几个层面，对这个包做一次“代码对应行为”的深度分析。

## 1. 包的整体定位

从代码上看，`spear_vision` 解决的是一个“从棋盘检测到空间位姿，再到机械执行接口”的完整链路问题，而不是单纯的图像识别。

### 1.1 面向的典型场景

- 用固定尺寸的大 ChArUco 板建立相机到工位的空间基准。
- 用 5cm x 5cm 的小 ChArUco 板做更灵活的局部对位。
- 标定“大板坐标系”与“小板坐标系”的固定关系。
- 将矛尖 `tip` 视作相对小板的固定偏移，从而推导 `camera_T_tip`。
- 把视觉输出进一步桥接到下位机串口协议。

### 1.2 设计上的几个关键取舍

- 优先使用 ChArUco 内角点做 PnP，因为角点精度通常高于 marker 四角。
- 当 ChArUco 内角点不足时，不立即放弃，而是回退到 ArUco Board 进行求解。
- 对重投影误差和边缘区域做门控，宁可丢帧，也不轻易发布不可靠位姿。
- 对连续帧使用低通滤波，降低平面目标的瞬态跳变。
- 对外参标定使用多帧均值与离群剔除，而不是单帧直接固化。
- 在 OpenCV ArUco API 发生变化时，通过兼容层屏蔽新旧版本差异。

## 2. 目录结构与职责划分

当前包目录如下：

```text
spear_vision/
├── config/
│   ├── board.yaml
│   ├── camera.yaml
│   ├── small_board.yaml
│   └── spear_tip.yaml
├── docs/
│   └── serial_bridge_protocol.md
├── launch/
│   ├── board_pose.launch.py
│   ├── charuco_calibration.launch.py
│   ├── small_board_pose.launch.py
│   ├── small_board_serial_bridge.launch.py
│   └── spear_tip.launch.py
├── spear_vision/
│   ├── core/
│   │   ├── board_pose_estimator.py
│   │   ├── extrinsic_calibrator.py
│   │   └── pose_filter.py
│   ├── nodes/
│   │   ├── board_pose_node.py
│   │   ├── charuco_calibration_node.py
│   │   ├── small_board_pose_node.py
│   │   ├── small_board_serial_bridge_node.py
│   │   ├── spear_tip_node.py
│   │   ├── spear_tip_calib_node.py
│   │   └── spear_tip_run_node.py
│   └── utils/
│       ├── calibration_store.py
│       ├── camera_intrinsics.py
│       ├── opencv_aruco.py
│       ├── pnp.py
│       ├── ros_conversions.py
│       ├── runtime_checks.py
│       ├── tf_utils.py
│       └── yaml_io.py
├── test/
│   ├── test_extrinsic_calibrator.py
│   └── test_tf_utils.py
├── package.xml
└── setup.py
```

### 2.1 `nodes/`

`nodes/` 是 ROS 2 入口层，主要负责：

- 声明和读取 ROS 参数。
- 从 YAML 把静态配置映射到参数。
- 订阅图像、`CameraInfo`、控制话题。
- 发布 `PoseStamped`、TF、调试图像、误差、置信度。
- 组织标定模式、运行模式、串口发送等流程。

### 2.2 `core/`

`core/` 是与 ROS 解耦的核心算法层：

- `board_pose_estimator.py`：单块板的统一位姿估计逻辑。
- `extrinsic_calibrator.py`：多帧外参标定与离群点剔除。
- `pose_filter.py`：一阶低通位姿平滑。

这部分是整个包里最值得复用的地方。

### 2.3 `utils/`

`utils/` 是支撑层，负责：

- OpenCV ArUco/ChArUco 新旧 API 兼容。
- PnP 多候选解比较与连续性约束。
- `Rt` 变换组合与四元数计算。
- CameraInfo/YAML 内参解析。
- ROS 消息转换。
- 标定结果自动复用。
- YAML 安全读写。

## 3. 入口、可执行文件与 launch 关系

`setup.py` 中定义了以下 ROS 2 可执行入口：

| 可执行名 | 对应节点 | 主要用途 |
| --- | --- | --- |
| `board_pose` | `board_pose_node.py` | 大板位姿估计 |
| `small_board_pose` | `small_board_pose_node.py` | 小板位姿估计与偏移输出 |
| `small_board_serial_bridge` | `small_board_serial_bridge_node.py` | 串口桥接 |
| `charuco_calib` | `charuco_calibration_node.py` | 内参标定 |
| `spear_tip` | `spear_tip_node.py` | 双板主节点，支持 calibrate/run |
| `spear_tip_calib` | `spear_tip_calib_node.py` | 外参标定专用包装 |
| `spear_tip_run` | `spear_tip_run_node.py` | 运行期 tip 估计包装 |

现有 launch 文件对应关系如下：

| launch 文件 | 启动内容 | 说明 |
| --- | --- | --- |
| `board_pose.launch.py` | `board_pose` | 单板大板定位 |
| `charuco_calibration.launch.py` | `charuco_calib` | 相机内参标定 |
| `small_board_pose.launch.py` | `small_board_pose` | 小板定位 |
| `small_board_serial_bridge.launch.py` | `small_board_pose` + `small_board_serial_bridge` | 小板定位并串口输出 |
| `spear_tip.launch.py` | `spear_tip_calib` 或 `spear_tip_run` | 通过 `mode` 切换标定/运行 |

## 4. 核心算法分析

这一节是理解整个包最关键的部分。

### 4.1 单板位姿估计主链路

无论是 `board_pose`、`small_board_pose`，还是 `spear_tip` 里的 primary/secondary 棋盘，核心流程都由 `BoardPoseEstimator.estimate()` 驱动，步骤基本一致：

1. `detectMarkers` 检测 ArUco marker。
2. 通过 `refineDetectedMarkers` 尝试找回漏检 marker。
3. 按当前 board 的 `ids_start` 和 `board.ids` 过滤 marker，避免画面中其它 marker 干扰。
4. 调用 `interpolateCornersCharuco` 推断 ChArUco 内角点。
5. 对 ChArUco 角点做 `cornerSubPix` 亚像素细化。
6. 对点集做边缘门控，太靠近图像边界则拒绝本帧。
7. 使用 `solve_pnp_best()` 做 PnP 求解。
8. 若 ChArUco 失败，再回退到 ArUco Board 的角点集合做 PnP。
9. 对结果做重投影误差门控。
10. 计算置信度并返回 `PoseEstimate`。

### 4.2 为什么是“优先 ChArUco，失败再回退 ArUco Board”

代码里明确体现了这种层级：

- ChArUco 内角点更稠密、几何约束更强、角点可做亚像素细化，因此更适合高精度定位。
- 当内角点数量不足、插值失败或贴边时，不会直接判定失败，而是尝试用同一块 board 上的 marker 四角继续做 ArUco Board PnP。
- 对于轻度遮挡、光照波动、小范围离焦场景，这种回退策略能显著增加“可输出帧”的比例。

### 4.3 PnP 求解如何避免平面目标跳变

`utils/pnp.py` 的实现比普通 `cv2.solvePnP()` 更稳：

- 优先使用 `solvePnPGeneric` 获取多个候选解。
- 对每个候选解计算像素级重投影误差。
- 默认选择平均重投影误差最小的解。
- 若多个候选解误差非常接近，则引入上一帧位姿作为连续性约束，优先选择与历史更接近的候选解。
- 可选调用 `solvePnPRefineLM` 进一步做 LM 细化。
- 可以强制 `tvec.z > 0`，避免目标被解到相机后方。

这点非常重要，因为平面棋盘在某些姿态下容易出现 PnP 二义性，只按误差最小选解会偶发翻转。

### 4.4 门控策略

门控主要分为三类：

1. 角点数量门控。
2. 重投影误差门控。
3. 图像边缘门控。

对应参数通常出现在 YAML 的 `gating`、`gating_primary`、`gating_secondary` 中：

- `min_charuco_corners`
- `max_mean_reproj_px`
- `max_max_reproj_px`
- `min_border_px`

代码采取的是“先尽量求，再严格筛”的思路：

- ChArUco 角点数量如果略低于阈值，某些情况下仍然允许输出，但会降低置信度。
- 只要重投影误差超过阈值，就直接拒绝该帧。
- 点集贴近图像边缘也直接拒绝，避免畸变区和裁剪区导致抖动。

### 4.5 置信度如何计算

`BoardPoseEstimator._gate_and_score()` 中，置信度是一个启发式量，不是概率学意义上的置信区间。它综合了：

- 是 `charuco` 还是 `aruco_board`。
- 实际参与求解的点数占最大可能点数的比例。
- 平均重投影误差与阈值的关系。

可以把它理解为“当前帧质量评分”，主要用于：

- 串口桥接节点的质量门控。
- 双板外参标定时筛除低质量样本。

### 4.6 位姿平滑

`PoseLowPassFilter` 实现了一阶低通滤波：

- 平移始终线性插值。
- 旋转可以在 `rvec` 线性插值与 `quat` nlerp 两种模式之间选择。
- 默认参数里 `smoothing_alpha=0.8`，意味着输出更偏向上一帧，牺牲部分动态响应换取稳定性。

此外它还显式处理了 `NaN/Inf`，避免坏值进入滤波器后把后续输出全部污染。

### 4.7 双板外参标定

`ExtrinsicCalibrator` 的逻辑也比“简单求平均”更完整：

1. 只有当 primary 和 secondary 都成功求得位姿，才会构造 `primary_T_secondary` 样本。
2. 只有当两块板的置信度都高于阈值，且满足采样步长 `sample_stride`，样本才会被接受。
3. 当样本数达到阈值后，先求一轮均值。
4. 再按平移阈值和旋转阈值剔除离群样本。
5. 如果均值中心被离群点拖偏，会退化到更鲁棒的参考中心重新判断。
6. 最终把保留下来的样本再求均值，固化为 `primary_T_secondary`。

这套逻辑比单帧保存外参稳得多，适合现场标定。

### 4.8 tip 位姿的组合方式

`spear_tip_node.py` 中，tip 位姿不是直接识别出来的，而是组合得到：

- 若当前帧看到了 secondary，则 `camera_T_tip = camera_T_secondary * secondary_T_tip`。
- 若当前帧没看到 secondary，但已有固化的 `primary_T_secondary`，则先算
  `camera_T_secondary = camera_T_primary * primary_T_secondary`，
  再继续算 `camera_T_tip`。

这意味着：

- 标定期需要同时看到大板和小板。
- 运行期只看大板也能输出 tip，只要外参已经固化。

### 4.9 OpenCV 兼容层的价值

`opencv_aruco.py` 是这个包非常有价值的一层封装，它处理了很多“现场一跑就崩”的版本差异问题：

- 旧版 `CharucoBoard_create` 与新版 `CharucoBoard(...)` 构造差异。
- `DetectorParameters_create` 与 `DetectorParameters()` 差异。
- `ArucoDetector` 新旧 API 差异。
- `interpolateCornersCharuco` 在非零 `ids_start` 下可能失败的问题。
- `board.ids`、`getIds()`、`getObjPoints()` 等绑定差异。

如果后续你们升级 OpenCV，这一层是最优先需要回归验证的地方。

## 5. 各节点深度说明

## 5.1 `charuco_calibration_node.py`

用途：

- 采集多帧 ChArUco 图像。
- 使用 `cv2.aruco.calibrateCameraCharucoExtended` 或兼容 API 标定内参。
- 导出 ROS 常见 `camera_info` YAML 格式。

默认输入：

- 图像：`/hik_camera/image_raw`

默认输出：

- `~/debug_image`
- `~/frames_received`
- `~/samples_accepted`
- `~/status`

若通过 `charuco_calibration.launch.py` 启动，节点名默认是 `charuco_calibration`，因此相对话题通常会展开为：

- `/charuco_calibration/debug_image`
- `/charuco_calibration/frames_received`
- `/charuco_calibration/samples_accepted`
- `/charuco_calibration/status`

关键参数：

- `config_path`
- `image_topic`
- `dictionary`
- `squares_x` / `squares_y`
- `square_length_m` / `marker_length_m`
- `ids_start`
- `min_charuco_corners`
- `sample_stride`
- `target_samples`
- `max_samples`
- `output_yaml_path`
- `auto_calibrate_on_target`
- `auto_exit_after_calibration`
- `calibrate_now`
- `reset_samples`

行为特征：

- 只要一帧中的 ChArUco 角点数达到阈值，就会把该帧纳入样本。
- 支持通过动态参数 `calibrate_now=true` 手动触发标定。
- 默认支持达到目标样本数后自动开始标定。
- 默认支持标定后自动退出。
- 支持 OpenCV 窗口和 `~/debug_image` 双通道可视化。

特别注意：

- 标定结果写出为 `camera.yaml`，这个文件是后续所有 PnP 节点的尺度基础。
- 代码设计上希望在标定后把“最近一次内参路径”写入 `~/.ros/spear_vision/last_camera_calibration_path.txt`，以便其它节点自动复用。
- 但当前实现中，`_request_shutdown()` 里用于写这个指针的变量作用域存在问题，自动复用指针不一定总能成功更新。稳妥做法仍然是显式传 `camera_calibration_yaml`。

## 5.2 `board_pose_node.py`

用途：

- 使用单块大板估计 `camera_T_board`。

默认输入：

- 图像：`/hik_camera/image_raw`
- 相机内参：优先 `CameraInfo`，否则尝试 YAML

默认输出：

- `~/pose`
- `~/debug_image`
- `~/reproj_error_mean_px`
- `~/reproj_error_max_px`
- `~/confidence`
- `~/method`
- TF：`camera_frame -> board_frame`

若通过默认 launch 启动，节点名是 `board_pose`，相对话题通常会展开为：

- `/board_pose/pose`
- `/board_pose/debug_image`
- `/board_pose/reproj_error_mean_px`
- `/board_pose/reproj_error_max_px`
- `/board_pose/confidence`
- `/board_pose/method`

关键参数：

- `config_path`
- `camera_calibration_yaml`
- `prefer_camera_info`
- `image_topic`
- `camera_info_topic`
- `camera_frame`
- `board_frame`
- `publish_tf`
- `publish_debug_image`
- `publish_pose`
- `show_opencv_window`
- `smoothing_alpha`
- `smoothing_rotation_mode`
- `use_clahe`
- `aruco_refine_detected_markers`
- `aruco_fallback_enable`
- `pnp_prefer`
- `pnp_fallback`
- `pnp_refine_lm`
- `min_charuco_corners`
- `max_mean_reproj_px`
- `max_max_reproj_px`
- `min_border_px`

行为特征：

- 支持可选 CLAHE 预处理，应对光照不均。
- 在没有有效内参时，会继续发布调试图像，但不会发布位姿。
- `method` 话题会明确说明当前用的是 `charuco` 还是 `aruco_board`，以及失败原因。

## 5.3 `small_board_pose_node.py`

用途：

- 使用 5cm x 5cm 的小 ChArUco 板估计 `camera_T_small_board`。
- 输出 `left/up/forward` 偏移，其中 ROS 话题只发送 `left/up`。

默认输入：

- 图像：`/hik_camera/image_raw`
- 可选控制话题：`/update_exec_req`

默认输出：

- `~/pose`
- `~/debug_image`
- `~/reproj_error_mean_px`
- `~/reproj_error_max_px`
- `~/confidence`
- `~/method`
- `/small_board_pose/offset_mm`
- TF：`camera_frame -> spear_small_board_frame`

相对话题在默认节点名 `small_board_pose` 下通常会展开为：

- `/small_board_pose/pose`
- `/small_board_pose/debug_image`
- `/small_board_pose/reproj_error_mean_px`
- `/small_board_pose/reproj_error_max_px`
- `/small_board_pose/confidence`
- `/small_board_pose/method`

偏移定义：

- `left_mm = -x * 1000`
- `up_mm = -y * 1000`
- `forward_mm = z * 1000`

代码里 `Float32MultiArray` 只发布 `[left_mm, up_mm]`。

控制逻辑：

- 当 `require_start_command=true` 时，节点不会一启动就计算位姿。
- 只有接收到 `command_topic` 上的 `start_command_value` 后才开始输出。
- 接收到 `stop_command_value` 后会暂停计算。

默认控制参数：

- `command_topic = /update_exec_req`
- `start_command_value = spear_build`
- `stop_command_value = stop`

这条链路特别适合做“按命令启动一次视觉定位”的工位流程。

## 5.4 `small_board_serial_bridge_node.py`

用途：

- 订阅小板位姿与质量指标。
- 计算 `left_mm` / `up_mm`。
- 按最小 UART 协议发送给 STM32。

默认订阅：

- `/small_board_pose/pose`
- `/small_board_pose/confidence`
- `/small_board_pose/reproj_error_mean_px`

协议格式：

```text
[SOF][ID][left_mm(float32)][up_mm(float32)]
```

默认参数：

- `port=/dev/serial_ch340`
- `baudrate=115200`
- `out_first_frame=0xFA`
- `out_frame_id=0xB1`

质量门控：

- `min_confidence`
- `max_mean_reproj_px`

只要门控不通过，该帧就不会发给下位机。

非常重要的实现细节：

- 这个节点并不是“纯粹把 `small_board_pose` 的 left/up 原样发出去”。
- 当前实现中存在一段机械补偿/人工修正逻辑：
  - `x_mgpianyi`、`y_mgpianyi`
  - `x_jzpianyi`、`y_jzpianyi`
  - `alpha`、`seita`
  - 以及固定的 `-10` 和 `+10` 偏置
- 由于这些变量目前在代码里都写死在 `_on_pose()` 中，默认行为会使发送值相对视觉原始偏移再发生修正。

其中最需要警惕的是：

- 当 `alpha=0`、`seita=0` 时，代码仍会额外执行
  - `left_mm = left_mm - 10`
  - `up_mm = up_mm + 10`

这意味着串口输出默认带固定偏置，并不等于视觉节点发布的原始偏移。使用前务必确认这是你们实际工艺需要的补偿，而不是遗留调试代码。

## 5.5 `spear_tip_node.py`

这是整个包里最复杂、最“系统级”的节点。

它同时处理：

- 两块棋盘的配置读取。
- 双板检测与单板检测。
- 外参标定与运行模式切换。
- `primary_pose`、`secondary_pose`、`tip_pose` 的发布。
- `primary_T_secondary` 的计算、固化与加载。

### 运行模式

`mode` 支持两种：

- `calibrate`
- `run`

#### `calibrate`

- 同时检测大板与小板。
- 当两块板都成功出位姿时，计算 `primary_T_secondary`。
- 多帧采样后用 `ExtrinsicCalibrator` 固化外参。
- 可选把外参写回 `spear_tip.yaml`。
- 可选自动把 `mode` 改为 `run`。
- 可选自动退出。

#### `run`

- 只检测大板，节省算力。
- 使用已固化的 `primary_T_secondary` 推算 `camera_T_secondary`。
- 再叠加 `tip_offset_m` / `tip_rpy_deg` 推出 `camera_T_tip`。

### 输出

无论是 `calibrate` 还是 `run`，节点都会创建这些发布器：

- `~/primary_pose`
- `~/secondary_pose`
- `~/tip_pose`
- `~/primary_to_secondary`
- `~/debug_image`
- `~/primary_reproj_mean_px`
- `~/secondary_reproj_mean_px`
- `~/primary_confidence`
- `~/secondary_confidence`
- `~/method`

如果使用包装节点：

- `spear_tip_calib` 的相对话题会挂在 `/spear_tip_calib/...`
- `spear_tip_run` 的相对话题会挂在 `/spear_tip_run/...`

### 关键参数

通用参数：

- `config_path`
- `camera_calibration_yaml`
- `prefer_camera_info`
- `image_topic`
- `camera_info_topic`
- `camera_frame`
- `primary_frame`
- `secondary_frame`
- `tip_frame`
- `publish_tf`
- `publish_debug_image`
- `show_opencv_window`
- `smoothing_alpha`
- `smoothing_rotation_mode`
- `pnp_prefer`
- `pnp_fallback`
- `pnp_refine_lm`
- `tip_offset_m`
- `tip_rpy_deg`

标定相关参数：

- `calib_required_samples`
- `calib_sample_stride`
- `calib_min_primary_confidence`
- `calib_min_secondary_confidence`
- `calib_outlier_translation_m`
- `calib_outlier_rotation_deg`
- `calib_auto_finalize`
- `calib_save_to_config`
- `calib_output_config_yaml`
- `calib_auto_switch_to_run`
- `calib_auto_exit`
- `calib_reset_samples`
- `finalize_calibration_now`

外参读写参数：

- `save_primary_to_secondary_yaml`
- `save_primary_to_secondary_now`
- `load_primary_to_secondary_yaml`

### 实现上的几个优点

- primary/secondary 可以使用同一字典，只要 `ids_start` 区间不重叠。
- 当字典相同时，标定模式只检测一次 marker，再按 ID 分流，节省算力。
- 对两块板分别做平滑与门控。
- 支持通过 `method` 话题清晰暴露当前模式和失败原因。
- 调试图里会显示采样进度、每块板的 marker 数、角点数、误差、置信度。

### 需要特别注意的地方

- `primary_to_secondary` 写回 YAML 时，使用的是 `yaml.safe_dump`，原文件注释会丢失。
- 若 `calib_auto_switch_to_run=true`，写回配置时还会把文件中的 `mode` 改成 `run`。
- 如果 primary 与 secondary 的 marker ID 区间重叠，节点会给出警告，但仍然能跑；只是误匹配风险会明显增大。

## 6. 配置文件说明

## 6.1 `config/board.yaml`

用于大板位姿估计与 ChArUco 内参标定。

主要字段：

- `board_outer_size_mm`
- `charuco.*`
- `aruco_board_fallback.*`
- `pnp.*`
- `gating.*`
- `topics.*`
- `frames.*`

其中：

- `charuco.square_length_m` 与 `marker_length_m` 决定真实尺度。
- `ids_start` 默认从 `0` 开始。
- `gating.target_samples` 和 `gating.sample_stride` 会被 `charuco_calibration_node` 直接复用。

## 6.2 `config/small_board.yaml`

用于小板定位。

当前默认配置表示：

- 5 x 5 squares
- 每个 square 为 `0.01m`
- 每个 marker 为 `0.008m`
- 外轮廓为 `50mm x 50mm`
- `ids_start = 100`

把小板 ID 从 `100` 开始，是为了和大板默认的 `0..` 区间错开。

## 6.3 `config/spear_tip.yaml`

这是系统级配置文件，控制双板标定和 tip 估计流程。

最关键的部分有：

- `topics`
- `mode`
- `calibration`
- `primary_to_secondary`
- `frames`
- `primary_charuco`
- `secondary_charuco`
- `gating_primary`
- `gating_secondary`
- `tip`

需要重点理解：

- `primary_to_secondary` 是标定后的核心资产。
- `tip.offset_m` 和 `tip.rpy_deg` 决定了小板坐标系到矛尖坐标系的固定位姿。
- `secondary_charuco` 的实际物理尺寸必须和打印件一致，否则所有 tip 结果的尺度都会错。

## 6.4 `config/camera.yaml`

这是内参标定输出文件，使用 ROS 常见 `camera_info` YAML 结构，包含：

- 图像分辨率
- `camera_matrix`
- `distortion_coefficients`
- `rectification_matrix`
- `projection_matrix`
- `spear_vision.rms_reprojection_px`
- `spear_vision.num_samples`

## 7. 坐标系与单位约定

整个包对坐标系的约定是统一的：

- 所有 `PoseStamped` 与 TF 都遵循 `camera_T_object` 的含义。
- 父坐标系通常是相机光学坐标系。
- 子坐标系是棋盘或 tip。

OpenCV 光学坐标系约定：

- `x` 向右
- `y` 向下
- `z` 向前

因此：

- `board_pose` / `small_board_pose` / `spear_tip` 中的 `tvec` 单位都是米。
- `small_board_pose` 中导出的 `left/up/forward` 单位是毫米。
- `small_board_serial_bridge` 发送的也是毫米。

需要额外记住：

- `primary_to_secondary` 话题发布的是 `primary_T_secondary`，不是 `camera_T_secondary`。

## 8. 推荐工作流

以下流程最符合当前代码设计。

### 8.1 第一步：标定相机内参

以下命令假设工作空间根目录是 `/home/hjw/opencv/vision_opencv`：

```bash
cd /home/hjw/opencv/vision_opencv
colcon build --packages-select spear_vision
source install/setup.bash

ros2 launch spear_vision charuco_calibration.launch.py \
  config_path:=/home/hjw/opencv/vision_opencv/src/spear_vision/config/board.yaml \
  output_yaml_path:=/home/hjw/opencv/vision_opencv/src/spear_vision/config/camera.yaml
```

如果想手动触发标定：

```bash
ros2 param set /charuco_calibration calibrate_now true
```

### 8.2 第二步：验证大板定位

```bash
ros2 launch spear_vision board_pose.launch.py \
  config_path:=/home/hjw/opencv/vision_opencv/src/spear_vision/config/board.yaml \
  camera_calibration_yaml:=/home/hjw/opencv/vision_opencv/src/spear_vision/config/camera.yaml
```

重点观察：

- `debug_image` 是否能稳定看到 marker、内角点和坐标轴。
- `reproj_error_mean_px` 是否长期较小。
- `method` 是否大部分时间是 `charuco:pnp_ok`。

### 8.3 第三步：验证小板定位

```bash
ros2 launch spear_vision small_board_pose.launch.py \
  config_path:=/home/hjw/opencv/vision_opencv/src/spear_vision/config/small_board.yaml \
  camera_calibration_yaml:=/home/hjw/opencv/vision_opencv/src/spear_vision/config/camera.yaml \
  require_start_command:=false
```

如果保留默认的命令触发机制，可以用：

```bash
ros2 topic pub --once /update_exec_req std_msgs/msg/String "{data: spear_build}"
```

暂停时：

```bash
ros2 topic pub --once /update_exec_req std_msgs/msg/String "{data: stop}"
```

### 8.4 第四步：标定双板外参

```bash
ros2 launch spear_vision spear_tip.launch.py \
  mode:=calibrate \
  config_path:=/home/hjw/opencv/vision_opencv/src/spear_vision/config/spear_tip.yaml \
  camera_calibration_yaml:=/home/hjw/opencv/vision_opencv/src/spear_vision/config/camera.yaml
```

标定完成后，`primary_to_secondary` 会被写回配置文件，且在默认配置下 `mode` 会自动切成 `run`。

### 8.5 第五步：运行 tip 位姿估计

```bash
ros2 launch spear_vision spear_tip.launch.py \
  mode:=run \
  config_path:=/home/hjw/opencv/vision_opencv/src/spear_vision/config/spear_tip.yaml \
  camera_calibration_yaml:=/home/hjw/opencv/vision_opencv/src/spear_vision/config/camera.yaml
```

### 8.6 第六步：接入串口桥接

```bash
ros2 launch spear_vision small_board_serial_bridge.launch.py \
  config_path:=/home/hjw/opencv/vision_opencv/src/spear_vision/config/small_board.yaml \
  camera_calibration_yaml:=/home/hjw/opencv/vision_opencv/src/spear_vision/config/camera.yaml \
  port:=/dev/serial_ch340 \
  baudrate:=115200
```

如果要严格按质量门控发送，可以加：

```bash
min_confidence:=0.6 max_mean_reproj_px:=0.8
```

## 9. 默认输入输出速查

### 9.1 `board_pose`

订阅：

- `/hik_camera/image_raw`
- `/hik_camera/image_raw/camera_info` 或参数指定值

发布：

- `/board_pose/pose`
- `/board_pose/debug_image`
- `/board_pose/reproj_error_mean_px`
- `/board_pose/reproj_error_max_px`
- `/board_pose/confidence`
- `/board_pose/method`

TF：

- `camera_frame -> board_frame`

### 9.2 `small_board_pose`

订阅：

- `/hik_camera/image_raw`
- `/hik_camera/image_raw/camera_info`
- `/update_exec_req`

发布：

- `/small_board_pose/pose`
- `/small_board_pose/debug_image`
- `/small_board_pose/reproj_error_mean_px`
- `/small_board_pose/reproj_error_max_px`
- `/small_board_pose/confidence`
- `/small_board_pose/method`
- `/small_board_pose/offset_mm`

TF：

- `camera_frame -> spear_small_board_frame`

### 9.3 `charuco_calibration`

订阅：

- `/hik_camera/image_raw`

发布：

- `/charuco_calibration/debug_image`
- `/charuco_calibration/frames_received`
- `/charuco_calibration/samples_accepted`
- `/charuco_calibration/status`

### 9.4 `spear_tip_calib`

订阅：

- `/hik_camera/image_raw`
- `/hik_camera/image_raw/camera_info`

发布：

- `/spear_tip_calib/primary_pose`
- `/spear_tip_calib/secondary_pose`
- `/spear_tip_calib/tip_pose`
- `/spear_tip_calib/primary_to_secondary`
- `/spear_tip_calib/debug_image`
- `/spear_tip_calib/primary_reproj_mean_px`
- `/spear_tip_calib/secondary_reproj_mean_px`
- `/spear_tip_calib/primary_confidence`
- `/spear_tip_calib/secondary_confidence`
- `/spear_tip_calib/method`

### 9.5 `spear_tip_run`

订阅：

- `/hik_camera/image_raw`
- `/hik_camera/image_raw/camera_info`

发布：

- `/spear_tip_run/primary_pose`
- `/spear_tip_run/tip_pose`
- `/spear_tip_run/debug_image`
- `/spear_tip_run/primary_reproj_mean_px`
- `/spear_tip_run/primary_confidence`
- `/spear_tip_run/method`

## 10. 依赖与运行环境

`package.xml` 中声明的主要依赖有：

- `rclpy`
- `sensor_msgs`
- `geometry_msgs`
- `std_msgs`
- `cv_bridge`
- `tf2_ros`
- `rcl_interfaces`
- `opencv-python`
- `serial_dispose`

其中有几个依赖关系需要特别注意：

### 10.1 OpenCV 必须带 `cv2.aruco`

如果当前 OpenCV 构建没有 `cv2.aruco`，整个包无法工作。

### 10.2 `cv_bridge` 与 OpenCV 来源要一致

`runtime_checks.py` 明确会检查：

- `cv2` 是否来自 pip
- `cv_bridge` 是否来自系统包

如果两者来源不一致，可能出现 ABI 或版本不匹配问题。现场表现通常是：

- 节点启动即崩
- GUI 异常
- 符号找不到
- `cv_bridge` 转图像时异常

### 10.3 `serial_dispose` 是串口桥接的硬依赖

`small_board_serial_bridge_node.py` 直接依赖：

```python
from get_dispose_serial.myserial import AsyncSerial_t
```

如果该模块不可用，串口桥接节点无法启动。

## 11. 已知限制与易踩坑点

这是阅读代码后最值得提前提醒的部分。

### 11.1 内参是前提条件

除了纯调试图像可视化外，所有 PnP 计算都依赖有效相机内参。

如果驱动发出的 `CameraInfo` 是空的，那么必须：

- 手动传 `camera_calibration_yaml`
- 或确保 YAML 路径能被正确自动加载

### 11.2 launch 里存在硬编码工作空间回退路径

多个 launch 文件优先尝试以下路径：

- `~/CHaruco/hik_ws/src/spear_vision/config/board.yaml`
- `~/CHaruco/hik_ws/src/spear_vision/config/small_board.yaml`
- `~/CHaruco/hik_ws/src/spear_vision/config/spear_tip.yaml`
- `~/CHaruco/hik_ws/src/spear_vision/config/camera.yaml`

而当前源码目录是：

- `/home/hjw/opencv/vision_opencv/src/spear_vision`

这意味着如果你不显式传 `config_path` 和 `camera_calibration_yaml`，launch 默认值不一定会指向当前仓库。为了避免歧义，建议在实际使用时总是显式指定这两个参数。

### 11.3 写回 YAML 会丢注释

无论是：

- `charuco_calibration_node` 写 `camera.yaml`
- `spear_tip_node` 写 `primary_to_secondary`

都使用 `yaml.safe_dump`。这会导致模板文件中的注释和人工排版丢失。

### 11.4 双板配置时必须尽量错开 ID 区间

虽然代码能过滤 marker，但如果 primary 与 secondary 的 marker ID 区间重叠，误匹配风险仍会明显上升。最稳妥的方式仍然是：

- primary 用 `ids_start = 0`
- secondary 用 `ids_start = 100`

### 11.5 串口桥接节点有硬编码补偿

这一点前面已经提到，但值得再次强调：

- 当前串口输出不是纯视觉原始值。
- 默认会对 `left/up` 加固定修正。

如果你们后续发现视觉调得很准、下位机却总差一个固定量，优先检查这里。

### 11.6 自动复用最近一次标定结果的机制并不绝对可靠

`board_pose`、`small_board_pose`、`spear_tip` 在 `camera_calibration_yaml` 为空时，会按以下顺序尝试：

1. `~/.ros/spear_vision/last_camera_calibration_path.txt`
2. `~/CHaruco/hik_ws/src/spear_vision/config/camera.yaml`

这个设计思路是好的，但结合当前实现和当前仓库路径，仍然推荐显式传参，而不要完全依赖隐式回退。

## 12. 测试现状

当前 `test/` 目录包含两个单元测试文件：

- `test_extrinsic_calibrator.py`
- `test_tf_utils.py`

覆盖点主要是：

- `ExtrinsicCalibrator` 的离群点剔除逻辑。
- `Rt` 变换组合、求逆与 RPY/四元数转换的一致性。

测试说明了一个事实：这个包的几何基础能力已经开始被抽离成可单测模块，这是一个很好的方向。但节点层的 ROS 话题流、OpenCV 检测行为、YAML 写回逻辑，目前还没有系统化自动测试。

## 13. 从维护者视角看，这个包最值得保留的优点

- 架构分层合理，`core/` 和 `utils/` 的可复用性高。
- 节点日志与 debug image 信息量充足，适合现场排障。
- 多数参数都可以被 YAML 和 launch 参数双重控制，灵活度高。
- 对 OpenCV 版本差异做了比较细致的兼容处理。
- 双板标定到运行的工作流是完整闭环。

## 14. 后续维护建议

如果后续继续演进这个包，优先建议做这些事情：

1. 修复 `charuco_calibration_node.py` 中“最近一次标定结果路径”写回的作用域问题。
2. 把 `small_board_serial_bridge_node.py` 中的机械补偿改成参数，而不是硬编码。
3. 给 `spear_tip_node.py` 增加更多单元测试或回放测试，尤其是配置写回和模式切换。
4. 把 launch 中硬编码的 `~/CHaruco/hik_ws/...` 路径替换成更通用的包内 share 路径或当前工作空间路径策略。
5. 如果未来要长期维护，建议补一份“真实工装尺寸与 ID 规划”文档，和代码一起版本化。

---

如果你只想最快上手，建议按这个顺序理解和验证：

1. 先跑 `charuco_calib`，确认有可靠的 `camera.yaml`。
2. 再跑 `board_pose`，看大板位姿是否稳定。
3. 再跑 `small_board_pose`，看 `offset_mm` 是否符合预期。
4. 需要 tip 时，再用 `spear_tip_calib -> spear_tip_run`。
5. 最后才接 `small_board_serial_bridge`，并确认串口补偿逻辑是否符合你们机械结构。
