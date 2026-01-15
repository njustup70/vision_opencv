# DepthCamera
深度相机相关的功能实现与测试脚本，主要基于 ROS2 话题进行采集、深度/彩色对齐、点云投影与平面拟合等操作。

## 依赖与话题
- 依赖：ROS2(`rclpy`)、`sensor_msgs`、`cv_bridge`、`image_geometry`、`numpy`、`opencv-python`、`open3d`（部分脚本）、`ultralytics`（YOLO脚本）
- 常用话题：
  - `/camera/depth/image_raw`
  - `/camera/color/image_raw`
  - `/camera/depth/camera_info`
  - `/camera/color/camera_info`
  - `/camera/depth/points`（已弃用，仅测试脚本涉及）

## 文件说明
### DepthCamera.py
- 深度相机核心封装：`DepthCamera`、`DepthCamNode`、`pix_to_cam`
- `loadCameraInfo` 支持从话题或本地 YAML 加载内参和深度到彩色外参

### PlaneDepToSpace.py
- 功能：识别 ROI 内主体深度峰值并计算质心像素与相机坐标
- 注意：深度图与彩色图叠加时需对齐尺寸（如 1280x720 -> 848x480）

### plan_PC_fit.py
- 局部平面拟合与区域生长：`region_growing_plane`、`fit_plane_from_depth`
- 依赖 Open3D

### get_cam_xangle_point.py
- 单点平面法向量测量（相机俯角）

### get_cam_xangle_average.py
- 多点采样后求平均法向量（含异常点剔除逻辑）
- 结果写入 `DepthCamera/xangle.txt`，需手动更新 `attitude_info.yaml`

### average_nor_from_file.py
- 从 `xangle.txt` 重新计算平均法向量（可调阈值）

### get_a_image.py
- 读取一帧 `/camera/color/image_raw` 作为标定/点选输入

### get_point_loc.py
- 变形恢复测试：点选 4 个角点，恢复 ROI 并显示 3D 坐标
- 依赖 `deform_restore`

### restore_YOLO.py
- 基于变形恢复的 ROI 预处理 + YOLO 目标检测

### check_spearhead.py
- 矛头检测：将点云划分为 6 个子区域并统计点数

## 配置文件
- `color_camera_info.yaml`：彩色相机内参
- `depth_camera_info.yaml`：深度相机内参
- `depth_to_color_info.yaml`：深度到彩色外参
- `attitude_info.yaml`：相机姿态/俯角配置

## 另
 - 所有的测试代码已删，直接跑需要切分支